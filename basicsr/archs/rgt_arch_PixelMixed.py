import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.nn import functional as F

from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import timm
import math
import numpy as np
import random
from basicsr.utils.registry import ARCH_REGISTRY


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return x1 * x2


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x
        # return(x+x.mean(dim=1,keepdim=True))*0.5

class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class PixelMixer(nn.Module):
    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(PixelMixer, self).__init__()

        assert planes % 5 == 0

        self.planes = planes
        self.mix_margin = mix_margin  # 像素的偏移量

        # 创建一个 2D mask，形状为 (planes, 1, mix_margin * 2 + 1, mix_margin * 2 + 1)
        self.mask = nn.Parameter(torch.zeros((self.planes, 1, mix_margin * 2 + 1, mix_margin * 2 + 1)),
                                 requires_grad=False)

        # 设置 mask 中的值
        self.mask[3::5, 0, 0, mix_margin] = 1.
        self.mask[2::5, 0, -1, mix_margin] = 1.
        self.mask[1::5, 0, mix_margin, 0] = 1.
        self.mask[0::5, 0, mix_margin, -1] = 1.
        self.mask[4::5, 0, mix_margin, mix_margin] = 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin
        # 将输入从 (B, H, W, C) 转置为 (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # 添加 2D 环形填充：(padding_left, padding_right, padding_top, padding_bottom)
        x = F.pad(x, pad=(m, m, m, m), mode='circular')
        # 进行 2D 卷积操作
        x = F.conv2d(input=x,
                     weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
                     dilation=(1, 1), groups=self.planes)
        # 将输出从 (B, C, H, W) 转置回 (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
        self.token_mixer = PixelMixer(planes=dim)
        self.norm = nn.BatchNorm2d(dim)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.zz = nn.Conv2d(dim*2,dim,kernel_size=3, stride=1, padding=1)
        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C
        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C    Size([1,64,64,90])
        x_s = self.token_mixer(x) - x
        x_s = x_s.permute(0, 3, 2, 1)
        x = x.permute(0, 3, 2, 1)            # 1,64,64,90   ---->1,90,64,64
        x_s = self.norm(x_s)
                                               #重排为 B H,W C
        x_s = torch.cat((x,x_s),dim=1)
                                                  #Size([1,64,64,180])

                                                   #con2D默认卷积为B，C W H,因此对tensor重新排列
        x = self.zz(x_s)                                           #调整维度为初始dim
        x = x.permute(0, 3, 2, 1)                                 #重排为 B H,W C
        return x


class L_SA(nn.Module):
    # The implementation builds on CAT code https://github.com/zhengchen1999/CAT/blob/main/basicsr/archs/cat_arch.py
    def __init__(self, dim, num_heads,
                 split_size=[2,4], shift_size=[1,2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., idx=0, reso=64, rs_id=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                WindowAttention(
                    dim//2, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)

            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None

            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1) # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for V-Shift
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1) # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """

        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)


        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
            qkv = qkv.view(3, B, _H, _W, C)
            # H-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C//2)
            # V-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C//2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))

            else:
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # H-Rwin
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)

        # mix
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + lcm

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RG_SA(nn.Module):
    """
    Recursive-Generalization Self-Attention (RG-SA).
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        c_ratio (float): channel adjustment factor.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5):
        super(RG_SA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.cr = int(dim * c_ratio) # scaled channel dimension

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        # RGM
        self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(self.cr),
            nn.GELU())
        # CA
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        # CPE
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        _scale = 1

        # reduction
        _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        if self.training:
            _time = max(int(math.log(H//4, 4)), int(math.log(W//4, 4)))
        else:
            _time = max(int(math.log(H//16, 4)), int(math.log(W//16, 4)))
            if _time < 2: _time = 2 # testing _time must equal or larger than training _time (2)

        _scale = 4 ** _time

        # Recursion xT
        for _ in range(_time):
            _x = self.reduction1(_x)

        _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C')
        _x = self.norm_act(_x)

        # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
        q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)

        # corss-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # CPE
        # v_shape=(B, H, N', C//H)
        v = v + self.cpe(v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)).view(B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# class Transformer(nn.Module):
#     def __init__(self, backbone, pretrained=None):
#         super().__init__()
#         self.encoder = getattr(pvt_v2, backbone)()
#         if pretrained:
#             checkpoint = torch.load('pvt_v2_b3.pth', map_location='cpu')
#             if 'model' in checkpoint:
#                 checkpoint_model = checkpoint['model']
#             else:
#                 checkpoint_model = checkpoint
#             state_dict = self.encoder.state_dict()
#             for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
#                 if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#                     print(f"Removing key {k} from pretrained checkpoint")
#                     del checkpoint_model[k]
#             self.encoder.load_state_dict(checkpoint_model, strict=False)
#
#     def forward(self, x):
#         # Define forward pass logic if needed
#         return self.encoder(x)
# def Encoder():
#     model = Transformer('pvt_v2_b3', pretrained=True)
#     return model.encoder
# class RG_SA(nn.Module):
#     """
#     Recursive-Generalization Self-Attention (RG-SA).
#     Args:
#         dim (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#         c_ratio (float): channel adjustment factor.
#     """
#
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5, backbone='pvt_v2_b3', pretrained=True):
#         super(RG_SA, self).__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#
#         self.cr = int(dim * c_ratio)  # scaled channel dimension
#
#         # Initialize Transformer with PVT backbone
#         self.transformer = Transformer(backbone, pretrained)
#
#         # self.scale = qk_scale or head_dim ** -0.5
#         self.scale = qk_scale or (head_dim * c_ratio) ** -0.5
#
#         # RGM
#         self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#         self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
#         self.norm_act = nn.Sequential(
#             nn.LayerNorm(self.cr),
#             nn.GELU())
#         # CA
#         self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
#         self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
#         self.v = nn.Linear(self.cr, dim, bias=qkv_bias)
#
#         # CPE
#         self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#         self.proj = nn.Linear(dim, dim)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#
#         _scale = 1
#
#         # Use Transformer to encode input x
#         _x = self.transformer(x)
#
#         _x = _x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
#
#         if self.training:
#             _time = max(int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))
#         else:
#             _time = max(int(math.log(H // 16, 4)), int(math.log(W // 16, 4)))
#             if _time < 2:
#                 _time = 2  # testing _time must equal or larger than training _time (2)
#
#         _scale = 4 ** _time
#
#         # Recursion xT
#         for _ in range(_time):
#             _x = self.reduction1(_x)
#
#         _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C')
#         _x = self.norm_act(_x)
#
#         # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
#         q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
#         k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
#         v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
#
#         # corss-attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         # CPE
#         v = v + self.cpe(
#             v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)).view(
#             B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, idx=0,
                 rs_id=0, split_size=[2,4], shift_size=[1,2], reso=64, c_ratio=0.5, layerscale_value=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if idx % 2 == 0:
            self.attn = L_SA(
                dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                drop=drop, idx=idx, reso=reso, rs_id=rs_id
            )
        else:
            self.attn = RG_SA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, c_ratio=c_ratio
            )
        # self.lsk_factor = LSKblock(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

        # HAI
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_size):
        H, W = x_size

        res = x

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # HAI
        x = x + (res * self.gamma)

        return x
# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
#                  attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, idx=0,
#                  rs_id=0, split_size=[2,4], shift_size=[1,2], reso=64, c_ratio=0.5, layerscale_value=1e-4):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         if idx % 2 == 0:
#             self.attn = L_SA(
#                 dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
#                 drop=drop, idx=idx, reso=reso, rs_id=rs_id
#             )
#         else:
#             self.attn = RG_SA(
#                 dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
#                 proj_drop=drop, c_ratio=c_ratio
#             )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
#         self.norm2 = norm_layer(dim)
#
#         self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
#
#     def forward(self, x, x_size):
#         B, C, H, W = x.size()
#         res = x
#
#         x = self.norm1(x)
#         x = x + self.drop_path(self.attn(x, H, W))
#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
#         x = x + (res * self.gamma)
#
#         return x


class ResidualGroup(nn.Module):

    def __init__(   self,
                    dim,
                    reso,
                    num_heads,
                    mlp_ratio=4.,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_paths=None,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    depth=2,
                    use_chk=False,
                    resi_connection='1conv',
                    rs_id=0,
                    split_size=[8,8],
                    c_ratio = 0.5):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                idx = i,
                rs_id = rs_id,
                split_size = split_size,
                shift_size = [split_size[0]//2, split_size[1]//2],
                c_ratio = c_ratio,
                )for i in range(depth)])


        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size):
        """
        Input:

        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x
# class ResidualGroup(nn.Module):
#
#     def __init__(self, dim, reso, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_paths=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  depth=2, use_chk=False, resi_connection='1conv', rs_id=0, split_size=[8,8],
#                  c_ratio=0.5):
#         super().__init__()
#         self.use_chk = use_chk
#         self.reso = reso
#
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_paths[i],
#                 act_layer=act_layer,
#                 norm_layer=norm_layer,
#                 idx=i,
#                 rs_id=rs_id,
#                 split_size=split_size,
#                 shift_size=[split_size[0] // 2, split_size[1] // 2],
#                 c_ratio=c_ratio,
#             ) for i in range(depth)])
#
#         if resi_connection == '1conv':
#             self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             self.conv = nn.Sequential(
#                 nn.Conv2d(dim, dim // 4, 3, 1, 1),
#                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
#                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 nn.Conv2d(dim // 4, dim, 3, 1, 1))
#
#     def forward(self, x, x_size):
#         """
#         Input:
#
#         Output: x: (B, H*W, C)
#         """
#         H, W = x_size
#         res = x
#         block_outputs = []
#
#         for blk in self.blocks:
#             if self.use_chk:
#                 x = checkpoint.checkpoint(blk, x, x_size)
#             else:
#                 x = blk(x, x_size)
#             block_outputs.append(x)
#         x = torch.cat(block_outputs, dim=1)  # Concatenate all block outputs along the channel dimension
#         x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
#         x = self.conv(x)
#         x = rearrange(x, "b c h w -> b (h w) c")
#         x = res + x
#
#         return x


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

@ARCH_REGISTRY.register()
class RGT(nn.Module):

    def __init__(self,
                img_size=64,
                in_chans=3,
                embed_dim=180,
                depth=[2,2,2,2],
                num_heads=[2,2,2,2],
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                use_chk=False,
                upscale=2,
                img_range=1.,
                resi_connection='1conv',
                split_size=[8,8],
                c_ratio=0.5,
                **kwargs):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads

        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rs_id=i,
                split_size = split_size,
                c_ratio = c_ratio
                )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, Reconstruction ------------------------- #
        self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        """
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x


if __name__ == '__main__':
    upscale = 1
    height = 62
    width = 66
    model = RGT(
        upscale=2,
        in_chans=3,
        img_size=64,
        img_range=1.,
        depth=[6,6,6,6,6,6],
        embed_dim=180,
        num_heads=[6,6,6,6,6,6],
        mlp_ratio=2,
        resi_connection='1conv',
        split_size=[8, 8],
        upsampler='pixelshuffle').cuda()
    # print(model)
    print(height, width)

    x = torch.randn((1, 3, height, width)).cuda()
    x = model(x)
    print(x.shape)



# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# from torch import Tensor
# from torch.nn import functional as F
#
# from timm.models.layers import DropPath, trunc_normal_
# from einops.layers.torch import Rearrange
# from einops import rearrange, repeat
# import timm
# import math
# import numpy as np
# import random
# from basicsr.utils.registry import ARCH_REGISTRY
#
#
# def img2windows(img, H_sp, W_sp):
#     """
#     Input: Image (B, C, H, W)
#     Output: Window Partition (B', N, C)
#     """
#     B, C, H, W = img.shape
#     img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
#     img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
#     return img_perm
#
#
# def windows2img(img_splits_hw, H_sp, W_sp, H, W):
#     """
#     Input: Window Partition (B', N, C)
#     Output: Image (B, H, W, C)
#     """
#     B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
#
#     img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
#     img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return img
#
#
# class Gate(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv
#
#     def forward(self, x, H, W):
#         # Split
#         x1, x2 = x.chunk(2, dim = -1)
#         B, N, C = x.shape
#         x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()
#
#         return x1 * x2
#
#
# class MLP(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.sg = Gate(hidden_features//2)
#         self.fc2 = nn.Linear(hidden_features//2, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x, H, W):
#         """
#         Input: x: (B, H*W, C), H, W
#         Output: x: (B, H*W, C)
#         """
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#
#         x = self.sg(x, H, W)
#         x = self.drop(x)
#
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class DynamicPosBias(nn.Module):
#     # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
#     """ Dynamic Relative Position Bias.
#     Args:
#         dim (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         residual (bool):  If True, use residual strage to connect conv.
#     """
#     def __init__(self, dim, num_heads, residual):
#         super().__init__()
#         self.residual = residual
#         self.num_heads = num_heads
#         self.pos_dim = dim // 4
#         self.pos_proj = nn.Linear(2, self.pos_dim)
#         self.pos1 = nn.Sequential(
#             nn.LayerNorm(self.pos_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.pos_dim, self.pos_dim),
#         )
#         self.pos2 = nn.Sequential(
#             nn.LayerNorm(self.pos_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.pos_dim, self.pos_dim)
#         )
#         self.pos3 = nn.Sequential(
#             nn.LayerNorm(self.pos_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.pos_dim, self.num_heads)
#         )
#     def forward(self, biases):
#         if self.residual:
#             pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
#             pos = pos + self.pos1(pos)
#             pos = pos + self.pos2(pos)
#             pos = self.pos3(pos)
#         else:
#             pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
#         return pos
#
# class PixelMixer(nn.Module):
#     def __init__(self, planes: int, mix_margin: int = 1) -> None:
#         super(PixelMixer, self).__init__()
#
#         assert planes % 5 == 0
#
#         self.planes = planes
#         self.mix_margin = mix_margin  # 像素的偏移量
#         self.mask = nn.Parameter(torch.zeros((self.planes, 1, mix_margin * 2 + 1, mix_margin * 2 + 1)),
#                                  requires_grad=False)
#
#         self.mask[3::5, 0, 0, mix_margin] = 1.
#         self.mask[2::5, 0, -1, mix_margin] = 1.
#         self.mask[1::5, 0, mix_margin, 0] = 1.
#         self.mask[0::5, 0, mix_margin, -1] = 1.
#         self.mask[4::5, 0, mix_margin, mix_margin] = 1.
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         m = self.mix_margin
#         x = F.conv2d(input=F.pad(x, pad=(m, m, m, m), mode='circular'),
#                      weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
#                      dilation=(1, 1), groups=self.planes)
#         return x
#
# class SWSA(nn.Module):
#     def __init__(self, dim: int,
#                  num_heads: int,
#                  attn_layer: list = None,
#                  proj_layer: list = None,
#                  window_list: tuple = ((8, 8),),
#                  shift_list: tuple = None,
#                  return_attns: bool = False,
#                  ) -> None:
#         super(SWSA, self).__init__()
#
#         self.dim = dim
#         self.num_heads = num_heads
#         self.return_attns = return_attns
#
#         self.window_list = window_list
#         if shift_list is not None:
#             assert len(shift_list) == len(window_list)
#             self.shift_list = shift_list
#         else:
#             self.shift_list = ((0, 0),) * len(window_list)
#
#         self.attn = nn.Sequential(*attn_layer if attn_layer is not None else [nn.Identity()])
#         self.proj = nn.Sequential(*proj_layer if proj_layer is not None else [nn.Identity()])
#
#     @staticmethod
#     def check_image_size(x: torch.Tensor, window_size: tuple) -> torch.Tensor:
#         _, _, h, w = x.size()
#         windows_num_h = math.ceil(h / window_size[0])
#         windows_num_w = math.ceil(w / window_size[1])
#         mod_pad_h = windows_num_h * window_size[0] - h
#         mod_pad_w = windows_num_w * window_size[1] - w
#         return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor or tuple:
#         qkv = self.attn(x)
#         _, C, _, _ = qkv.size()
#
#         qkv_list = torch.split(qkv, [C // len(self.window_list)] * len(self.window_list), dim=1)
#
#         output_list = list()
#         if self.return_attns:
#             attn_list = list()
#
#         for attn_slice, window_size, shift_size in zip(qkv_list, self.window_list, self.shift_list):
#             _, _, h, w = attn_slice.size()
#             attn_slice = self.check_image_size(attn_slice, window_size)
#
#             if shift_size != (0, 0):
#                 attn_slice = torch.roll(attn_slice, shifts=shift_size, dims=(2, 3))
#
#             _, _, H, W = attn_slice.size()
#             q, v = rearrange(attn_slice, 'b (qv head c) (nh ws1) (nw ws2) -> qv (b head nh nw) (ws1 ws2) c',
#                              qv=2, head=self.num_heads,
#                              ws1=window_size[0], ws2=window_size[1])
#             attn = (q @ q.transpose(-2, -1))
#             attn = F.softmax(attn, dim=-1)
#             if self.return_attns:
#                 attn_list.append(attn.reshape(self.num_heads, -1,
#                                               window_size[0] * window_size[1],
#                                               window_size[0] * window_size[1]))
#             output = rearrange(attn @ v, '(b head nh nw) (ws1 ws2) c -> b (head c) (nh ws1) (nw ws2)',
#                                head=self.num_heads,
#                                nh=H // window_size[0], nw=W // window_size[1],
#                                ws1=window_size[0], ws2=window_size[1])
#
#             if shift_size != (0, 0):
#                 output = torch.roll(output, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
#
#             output_list.append(output[:, :, :h, :w])
#
#         output = self.proj(torch.cat(output_list, dim=1))
#
#         if self.return_attns:
#             return output, attn_list
#         else:
#             return output
#
# # class WindowAttention(nn.Module):
# #     def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
# #         super().__init__()
# #         self.dim = dim
# #         self.dim_out = dim_out or dim
# #         self.split_size = split_size
# #         self.num_heads = num_heads
# #         self.idx = idx
# #         self.position_bias = position_bias
# #
# #         head_dim = dim // num_heads
# #         self.scale = qk_scale or head_dim ** -0.5
# #
# #         if idx == 0:
# #             H_sp, W_sp = self.split_size[0], self.split_size[1]
# #         elif idx == 1:
# #             W_sp, H_sp = self.split_size[0], self.split_size[1]
# #         else:
# #             print ("ERROR MODE", idx)
# #             exit(0)
# #         self.H_sp = H_sp
# #         self.W_sp = W_sp
# #
# #         if self.position_bias:
# #             self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
# #             # generate mother-set
# #             position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
# #             position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
# #             biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
# #             biases = biases.flatten(1).transpose(0, 1).contiguous().float()
# #             self.register_buffer('rpe_biases', biases)
# #
# #             # get pair-wise relative position index for each token inside the window
# #             coords_h = torch.arange(self.H_sp)
# #             coords_w = torch.arange(self.W_sp)
# #             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
# #             coords_flatten = torch.flatten(coords, 1)
# #             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
# #             relative_coords = relative_coords.permute(1, 2, 0).contiguous()
# #             relative_coords[:, :, 0] += self.H_sp - 1
# #             relative_coords[:, :, 1] += self.W_sp - 1
# #             relative_coords[:, :, 0] *= 2 * self.W_sp - 1
# #             relative_position_index = relative_coords.sum(-1)
# #             self.register_buffer('relative_position_index', relative_position_index)
# #
# #         self.attn_drop = nn.Dropout(attn_drop)
# #
# #     def im2win(self, x, H, W):
# #         B, N, C = x.shape
# #         x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
# #         x = img2windows(x, self.H_sp, self.W_sp)
# #         x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
# #         return x
# #
# #     def forward(self, qkv, H, W, mask=None):
# #         """
# #         Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
# #         Output: x (B, H, W, C)
# #         """
# #         q,k,v = qkv[0], qkv[1], qkv[2]
# #
# #         B, L, C = q.shape
# #         assert L == H * W, "flatten img_tokens has wrong size"
# #
# #         # partition the q,k,v, image to window
# #         q = self.im2win(q, H, W)
# #         k = self.im2win(k, H, W)
# #         v = self.im2win(v, H, W)
# #
# #         q = q * self.scale
# #         attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
# #
# #         # calculate drpe
# #         if self.position_bias:
# #             pos = self.pos(self.rpe_biases)
# #             # select position bias
# #             relative_position_bias = pos[self.relative_position_index.view(-1)].view(
# #                 self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
# #             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
# #             attn = attn + relative_position_bias.unsqueeze(0)
# #
# #         N = attn.shape[3]
# #
# #         # use mask for shift window
# #         if mask is not None:
# #             nW = mask.shape[0]
# #             attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
# #             attn = attn.view(-1, self.num_heads, N, N)
# #
# #         attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
# #         attn = self.attn_drop(attn)
# #
# #         x = (attn @ v)
# #         x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C
# #
# #         # merge the window, window to image
# #         x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C
# #
# #         return x
#
# class WindowAttention(nn.Module):
#     def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
#         super().__init__()
#         self.dim = dim
#         self.dim_out = dim_out or dim
#         self.split_size = split_size
#         self.num_heads = num_heads
#         self.idx = idx
#         self.position_bias = position_bias
#
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         if idx == 0:
#             H_sp, W_sp = self.split_size[0], self.split_size[1]
#         elif idx == 1:
#             W_sp, H_sp = self.split_size[0], self.split_size[1]
#         else:
#             print("ERROR MODE", idx)
#             exit(0)
#         self.H_sp = H_sp
#         self.W_sp = W_sp
#
#         if self.position_bias:
#             self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
#             position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
#             position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
#             biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
#             biases = biases.flatten(1).transpose(0, 1).contiguous().float()
#             self.register_buffer('rpe_biases', biases)
#
#             coords_h = torch.arange(self.H_sp)
#             coords_w = torch.arange(self.W_sp)
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
#             coords_flatten = torch.flatten(coords, 1)
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#             relative_coords[:, :, 0] += self.H_sp - 1
#             relative_coords[:, :, 1] += self.W_sp - 1
#             relative_coords[:, :, 0] *= 2 * self.W_sp - 1
#             relative_position_index = relative_coords.sum(-1)
#             self.register_buffer('relative_position_index', relative_position_index)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.attn = SWSA(dim, num_heads, window_list=[(H_sp, W_sp)], shift_list=[(0, 0)])
#
#     def forward(self, qkv, H, W, mask=None):
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         B, L, C = q.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#
#         q = q.reshape(B, C, H, W)
#         k = k.reshape(B, C, H, W)
#         v = v.reshape(B, C, H, W)
#
#         if self.position_bias:
#             pos = self.pos(self.rpe_biases)
#             rpb = pos[self.relative_position_index.view(-1)].view(
#                 self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1).permute(2, 0, 1).contiguous().unsqueeze(0)
#         else:
#             rpb = None
#
#         qkv = torch.stack((q, k, v), dim=2).view(B, C * 3, H, W)
#         x = self.attn(qkv)
#         x = x.reshape(B, C, H * W).transpose(1, 2).contiguous()
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
# class L_SA(nn.Module):
#     # The implementation builds on CAT code https://github.com/zhengchen1999/CAT/blob/main/basicsr/archs/cat_arch.py
#     def __init__(self, dim, num_heads,
#                  split_size=[2,4], shift_size=[1,2], qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., idx=0, reso=64, rs_id=0):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.split_size = split_size
#         self.shift_size = shift_size
#         self.idx = idx
#         self.rs_id = rs_id
#         self.patches_resolution = reso
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#
#         assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
#         assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"
#
#         self.branch_num = 2
#
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(drop)
#
#         self.attns = nn.ModuleList([
#                 WindowAttention(
#                     dim//2, idx = i,
#                     split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
#                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
#                 for i in range(self.branch_num)])
#
#         if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
#             attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
#
#             self.register_buffer("attn_mask_0", attn_mask[0])
#             self.register_buffer("attn_mask_1", attn_mask[1])
#         else:
#             attn_mask = None
#
#             self.register_buffer("attn_mask_0", None)
#             self.register_buffer("attn_mask_1", None)
#
#         self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv
#
#     def calculate_mask(self, H, W):
#         # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
#         # calculate attention mask for Rwin
#         img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
#         img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
#         h_slices_0 = (slice(0, -self.split_size[0]),
#                     slice(-self.split_size[0], -self.shift_size[0]),
#                     slice(-self.shift_size[0], None))
#         w_slices_0 = (slice(0, -self.split_size[1]),
#                     slice(-self.split_size[1], -self.shift_size[1]),
#                     slice(-self.shift_size[1], None))
#
#         h_slices_1 = (slice(0, -self.split_size[1]),
#                     slice(-self.split_size[1], -self.shift_size[1]),
#                     slice(-self.shift_size[1], None))
#         w_slices_1 = (slice(0, -self.split_size[0]),
#                     slice(-self.split_size[0], -self.shift_size[0]),
#                     slice(-self.shift_size[0], None))
#         cnt = 0
#         for h in h_slices_0:
#             for w in w_slices_0:
#                 img_mask_0[:, h, w, :] = cnt
#                 cnt += 1
#         cnt = 0
#         for h in h_slices_1:
#             for w in w_slices_1:
#                 img_mask_1[:, h, w, :] = cnt
#                 cnt += 1
#
#         # calculate mask for H-Shift
#         img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
#         img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1) # nW, sw[0], sw[1], 1
#         mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
#         attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
#         attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))
#
#         # calculate mask for V-Shift
#         img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
#         img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1) # nW, sw[1], sw[0], 1
#         mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
#         attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
#         attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))
#
#         return attn_mask_0, attn_mask_1
#
#     def forward(self, x, H, W):
#         """
#         Input: x: (B, H*W, C), x_size: (H, W)
#         Output: x: (B, H*W, C)
#         """
#
#         B, L, C = x.shape
#         assert L == H * W, "flatten img_tokens has wrong size"
#
#         qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
#         # v without partition
#         v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)
#
#
#         max_split_size = max(self.split_size[0], self.split_size[1])
#         pad_l = pad_t = 0
#         pad_r = (max_split_size - W % max_split_size) % max_split_size
#         pad_b = (max_split_size - H % max_split_size) % max_split_size
#
#         qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
#         qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
#         _H = pad_b + H
#         _W = pad_r + W
#         _L = _H * _W
#
#         if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (self.rs_id % 2 != 0 and self.idx % 4 == 0):
#             qkv = qkv.view(3, B, _H, _W, C)
#             # H-Shift
#             qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
#             qkv_0 = qkv_0.view(3, B, _L, C//2)
#             # V-Shift
#             qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
#             qkv_1 = qkv_1.view(3, B, _L, C//2)
#
#             if self.patches_resolution != _H or self.patches_resolution != _W:
#                 mask_tmp = self.calculate_mask(_H, _W)
#                 # H-Rwin
#                 x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
#                 # V-Rwin
#                 x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
#
#             else:
#                 # H-Rwin
#                 x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
#                 # V-Rwin
#                 x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)
#
#             x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
#             x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
#             x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
#             x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
#             # Concat
#             attened_x = torch.cat([x1,x2], dim=2)
#         else:
#             # V-Rwin
#             x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
#             # H-Rwin
#             x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
#             # Concat
#             attened_x = torch.cat([x1,x2], dim=2)
#
#         # mix
#         lcm = self.get_v(v)
#         lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)
#
#         x = attened_x + lcm
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
#
#
# class RG_SA(nn.Module):
#     """
#     Recursive-Generalization Self-Attention (RG-SA).
#     Args:
#         dim (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#         c_ratio (float): channel adjustment factor.
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5):
#         super(RG_SA, self).__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#
#         self.cr = int(dim * c_ratio) # scaled channel dimension
#
#         # self.scale = qk_scale or head_dim ** -0.5
#         self.scale = qk_scale or (head_dim * c_ratio) ** -0.5
#
#         # RGM
#         self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
#         self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
#         self.norm_act = nn.Sequential(
#             nn.LayerNorm(self.cr),
#             nn.GELU())
#         # CA
#         self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
#         self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
#         self.v = nn.Linear(self.cr, dim, bias=qkv_bias)
#
#         # CPE
#         self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
#
#         self.proj = nn.Linear(dim, dim)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#
#         _scale = 1
#
#         # reduction
#         _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
#
#         if self.training:
#             _time = max(int(math.log(H//4, 4)), int(math.log(W//4, 4)))
#         else:
#             _time = max(int(math.log(H//16, 4)), int(math.log(W//16, 4)))
#             if _time < 2: _time = 2 # testing _time must equal or larger than training _time (2)
#
#         _scale = 4 ** _time
#
#         # Recursion xT
#         for _ in range(_time):
#             _x = self.reduction1(_x)
#
#         _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C')
#         _x = self.norm_act(_x)
#
#         # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
#         q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
#         k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
#         v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
#
#         # corss-attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         # CPE
#         # v_shape=(B, H, N', C//H)
#         v = v + self.cpe(v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)).view(B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
# # class Transformer(nn.Module):
# #     def __init__(self, backbone, pretrained=None):
# #         super().__init__()
# #         self.encoder = getattr(pvt_v2, backbone)()
# #         if pretrained:
# #             checkpoint = torch.load('pvt_v2_b3.pth', map_location='cpu')
# #             if 'model' in checkpoint:
# #                 checkpoint_model = checkpoint['model']
# #             else:
# #                 checkpoint_model = checkpoint
# #             state_dict = self.encoder.state_dict()
# #             for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
# #                 if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
# #                     print(f"Removing key {k} from pretrained checkpoint")
# #                     del checkpoint_model[k]
# #             self.encoder.load_state_dict(checkpoint_model, strict=False)
# #
# #     def forward(self, x):
# #         # Define forward pass logic if needed
# #         return self.encoder(x)
# # def Encoder():
# #     model = Transformer('pvt_v2_b3', pretrained=True)
# #     return model.encoder
# # class RG_SA(nn.Module):
# #     """
# #     Recursive-Generalization Self-Attention (RG-SA).
# #     Args:
# #         dim (int): Number of input channels.
# #         num_heads (int): Number of attention heads.
# #         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
# #         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
# #         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
# #         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
# #         c_ratio (float): channel adjustment factor.
# #     """
# #
# #     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5, backbone='pvt_v2_b3', pretrained=True):
# #         super(RG_SA, self).__init__()
# #         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
# #         self.num_heads = num_heads
# #         head_dim = dim // num_heads
# #
# #         self.cr = int(dim * c_ratio)  # scaled channel dimension
# #
# #         # Initialize Transformer with PVT backbone
# #         self.transformer = Transformer(backbone, pretrained)
# #
# #         # self.scale = qk_scale or head_dim ** -0.5
# #         self.scale = qk_scale or (head_dim * c_ratio) ** -0.5
# #
# #         # RGM
# #         self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
# #         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
# #         self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
# #         self.norm_act = nn.Sequential(
# #             nn.LayerNorm(self.cr),
# #             nn.GELU())
# #         # CA
# #         self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
# #         self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
# #         self.v = nn.Linear(self.cr, dim, bias=qkv_bias)
# #
# #         # CPE
# #         self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
# #
# #         self.proj = nn.Linear(dim, dim)
# #         self.attn_drop = nn.Dropout(attn_drop)
# #         self.proj_drop = nn.Dropout(proj_drop)
# #
# #     def forward(self, x, H, W):
# #         B, N, C = x.shape
# #
# #         _scale = 1
# #
# #         # Use Transformer to encode input x
# #         _x = self.transformer(x)
# #
# #         _x = _x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
# #
# #         if self.training:
# #             _time = max(int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))
# #         else:
# #             _time = max(int(math.log(H // 16, 4)), int(math.log(W // 16, 4)))
# #             if _time < 2:
# #                 _time = 2  # testing _time must equal or larger than training _time (2)
# #
# #         _scale = 4 ** _time
# #
# #         # Recursion xT
# #         for _ in range(_time):
# #             _x = self.reduction1(_x)
# #
# #         _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C')
# #         _x = self.norm_act(_x)
# #
# #         # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
# #         q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
# #         k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
# #         v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
# #
# #         # corss-attention
# #         attn = (q @ k.transpose(-2, -1)) * self.scale
# #         attn = attn.softmax(dim=-1)
# #         attn = self.attn_drop(attn)
# #
# #         # CPE
# #         v = v + self.cpe(
# #             v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)).view(
# #             B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)
# #
# #         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
# #         x = self.proj(x)
# #         x = self.proj_drop(x)
# #
# #         return x
#
# class LSKblock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         self.conv1 = nn.Conv2d(dim, dim // 2, 1)
#         self.conv2 = nn.Conv2d(dim, dim // 2, 1)
#         self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
#         self.conv = nn.Conv2d(dim // 2, dim, 1)
#
#     def forward(self, x):
#         attn1 = self.conv0(x)
#         attn2 = self.conv_spatial(attn1)
#
#         attn1 = self.conv1(attn1)
#         attn2 = self.conv2(attn2)
#
#         attn = torch.cat([attn1, attn2], dim=1)
#         avg_attn = torch.mean(attn, dim=1, keepdim=True)
#         max_attn, _ = torch.max(attn, dim=1, keepdim=True)
#         agg = torch.cat([avg_attn, max_attn], dim=1)
#         sig = self.conv_squeeze(agg).sigmoid()
#         attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
#         attn = self.conv(attn)
#         return x * attn
#
# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
#                  attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, idx=0,
#                  rs_id=0, split_size=[2,4], shift_size=[1,2], reso=64, c_ratio=0.5, layerscale_value=1e-4):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         if idx % 2 == 0:
#             self.attn = L_SA(
#                 dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
#                 drop=drop, idx=idx, reso=reso, rs_id=rs_id
#             )
#         else:
#             self.attn = RG_SA(
#                 dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
#                 proj_drop=drop, c_ratio=c_ratio
#             )
#         # self.lsk_factor = LSKblock(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
#         self.norm2 = norm_layer(dim)
#
#         # HAI
#         self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
#
#     def forward(self, x, x_size):
#         #print('156133211111111111111111111111111516',x.size())
#         #print('156133211111111111111111111111111516', x_size)
#         z = x.size()
#         B , HW , C = z
#         # x :(B,H*W,C) --> (B,C,H*W) --> y:(B,C,H,W)
#         H , W = x_size
#
#         res = x
#         #x = x.view(0,2,1)
#
#         y = x.reshape(B,C,H*W)
#         y = x.reshape(B,C,H,W)
#         #print("before:sadassssdasdasd",y.size())
#         # x = self.lsk_factor(y)
#
#         x = x.reshape(B,H*W,C)
#
#         #print("after:sdddddddddddddddddddddddd",x.size())
#         x = x + self.drop_path(self.attn(self.norm1(x), H, W))
#
#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
#
#         # HAI
#         x = x + (res * self.gamma)
#
#         return x
# # class Block(nn.Module):
# #     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
# #                  attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, idx=0,
# #                  rs_id=0, split_size=[2,4], shift_size=[1,2], reso=64, c_ratio=0.5, layerscale_value=1e-4):
# #         super().__init__()
# #         self.norm1 = norm_layer(dim)
# #         if idx % 2 == 0:
# #             self.attn = L_SA(
# #                 dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
# #                 drop=drop, idx=idx, reso=reso, rs_id=rs_id
# #             )
# #         else:
# #             self.attn = RG_SA(
# #                 dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
# #                 proj_drop=drop, c_ratio=c_ratio
# #             )
# #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
# #
# #         mlp_hidden_dim = int(dim * mlp_ratio)
# #         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
# #         self.norm2 = norm_layer(dim)
# #
# #         self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
# #
# #     def forward(self, x, x_size):
# #         B, C, H, W = x.size()
# #         res = x
# #
# #         x = self.norm1(x)
# #         x = x + self.drop_path(self.attn(x, H, W))
# #         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
# #         x = x + (res * self.gamma)
# #
# #         return x
#
#
# class ResidualGroup(nn.Module):
#
#     def __init__(   self,
#                     dim,
#                     reso,
#                     num_heads,
#                     mlp_ratio=4.,
#                     qkv_bias=False,
#                     qk_scale=None,
#                     drop=0.,
#                     attn_drop=0.,
#                     drop_paths=None,
#                     act_layer=nn.GELU,
#                     norm_layer=nn.LayerNorm,
#                     depth=2,
#                     use_chk=False,
#                     resi_connection='1conv',
#                     rs_id=0,
#                     split_size=[8,8],
#                     c_ratio = 0.5):
#         super().__init__()
#         self.use_chk = use_chk
#         self.reso = reso
#
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_paths[i],
#                 act_layer=act_layer,
#                 norm_layer=norm_layer,
#                 idx = i,
#                 rs_id = rs_id,
#                 split_size = split_size,
#                 shift_size = [split_size[0]//2, split_size[1]//2],
#                 c_ratio = c_ratio,
#                 )for i in range(depth)])
#
#
#         if resi_connection == '1conv':
#             self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             self.conv = nn.Sequential(
#                 nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 nn.Conv2d(dim // 4, dim, 3, 1, 1))
#
#     def forward(self, x, x_size):
#         """
#         Input:
#
#         Output: x: (B, H*W, C)
#         """
#         H, W = x_size
#         res = x
#         for blk in self.blocks:
#             if self.use_chk:
#                 x = checkpoint.checkpoint(blk, x, x_size)
#             else:
#                 x = blk(x, x_size)
#         x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
#         x = self.conv(x)
#         x = rearrange(x, "b c h w -> b (h w) c")
#         x = res + x
#
#         return x
# # class ResidualGroup(nn.Module):
# #
# #     def __init__(self, dim, reso, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
# #                  drop=0., attn_drop=0., drop_paths=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
# #                  depth=2, use_chk=False, resi_connection='1conv', rs_id=0, split_size=[8,8],
# #                  c_ratio=0.5):
# #         super().__init__()
# #         self.use_chk = use_chk
# #         self.reso = reso
# #
# #         self.blocks = nn.ModuleList([
# #             Block(
# #                 dim=dim,
# #                 num_heads=num_heads,
# #                 mlp_ratio=mlp_ratio,
# #                 qkv_bias=qkv_bias,
# #                 qk_scale=qk_scale,
# #                 drop=drop,
# #                 attn_drop=attn_drop,
# #                 drop_path=drop_paths[i],
# #                 act_layer=act_layer,
# #                 norm_layer=norm_layer,
# #                 idx=i,
# #                 rs_id=rs_id,
# #                 split_size=split_size,
# #                 shift_size=[split_size[0] // 2, split_size[1] // 2],
# #                 c_ratio=c_ratio,
# #             ) for i in range(depth)])
# #
# #         if resi_connection == '1conv':
# #             self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
# #         elif resi_connection == '3conv':
# #             self.conv = nn.Sequential(
# #                 nn.Conv2d(dim, dim // 4, 3, 1, 1),
# #                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #                 nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
# #                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
# #                 nn.Conv2d(dim // 4, dim, 3, 1, 1))
# #
# #     def forward(self, x, x_size):
# #         """
# #         Input:
# #
# #         Output: x: (B, H*W, C)
# #         """
# #         H, W = x_size
# #         res = x
# #         block_outputs = []
# #
# #         for blk in self.blocks:
# #             if self.use_chk:
# #                 x = checkpoint.checkpoint(blk, x, x_size)
# #             else:
# #                 x = blk(x, x_size)
# #             block_outputs.append(x)
# #         x = torch.cat(block_outputs, dim=1)  # Concatenate all block outputs along the channel dimension
# #         x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
# #         x = self.conv(x)
# #         x = rearrange(x, "b c h w -> b (h w) c")
# #         x = res + x
# #
# #         ret
#
#
# class Upsample(nn.Sequential):
#     """Upsample module.
#     Args:
#         scale (int): Scale factor. Supported scales: 2^n and 3.
#         num_feat (int): Channel number of intermediate features.
#     """
#     def __init__(self, scale, num_feat):
#         m = []
#         if (scale & (scale - 1)) == 0:  # scale = 2^n
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
#                 m.append(nn.PixelShuffle(2))
#         elif scale == 3:
#             m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
#             m.append(nn.PixelShuffle(3))
#         else:
#             raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
#         super(Upsample, self).__init__(*m)
#
#
# ##鍓灊妯″潡
# class PruningModule:
#     def __init__(self, model, pruning_ratio=0.2):
#         self.model = model
#         self.pruning_ratio = pruning_ratio
#
#     def prune(self):
#         for name, module in self.model.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 self._prune_conv_layer(module)
#
#     def _prune_conv_layer(self, conv_layer):
#         # 鑾峰彇鍗风Н灞傛潈閲?
#         weights = conv_layer.weight.data.cpu().numpy()
#         # 璁＄畻鏉冮噸鐨凩1鑼冩暟
#         L1_norm = np.sum(np.abs(weights), axis=(1, 2, 3))
#         # 璁＄畻瑕佸壀鏋濈殑閫氶亾鏁伴噺
#         num_prune = int(self.pruning_ratio * weights.shape[0])
#         # 鑾峰彇瑕佸壀鏋濈殑閫氶亾绱㈠紩
#         prune_indices = np.argsort(L1_norm)[:num_prune]
#         # 鍓灊鏉冮噸
#         mask = np.ones(weights.shape[0], dtype=bool)
#         mask[prune_indices] = False
#         conv_layer.weight.data = torch.from_numpy(weights[mask, :, :, :]).to(conv_layer.weight.device)
#
#         # 濡傛灉鍗风Н灞傛湁鍋忕疆锛屼篃闇?瑕佸壀鏋?
#         if conv_layer.bias is not None:
#             biases = conv_layer.bias.data.cpu().numpy()
#             conv_layer.bias.data = torch.from_numpy(biases[mask]).to(conv_layer.bias.device)
#
#         # 鏇存柊鍗风Н灞傜殑杈撳嚭閫氶亾鏁?
#         conv_layer.out_channels = mask.sum()
#
# @ARCH_REGISTRY.register()
# class RGT(nn.Module):
#
#     def __init__(self,
#                 img_size=64,
#                 in_chans=3,
#                 embed_dim=180,
#                 depth=[2,2,2,2],
#                 num_heads=[2,2,2,2],
#                 mlp_ratio=4.,
#                 qkv_bias=True,
#                 qk_scale=None,
#                 drop_rate=0.,
#                 attn_drop_rate=0.,
#                 drop_path_rate=0.1,
#                 act_layer=nn.GELU,
#                 norm_layer=nn.LayerNorm,
#                 use_chk=False,
#                 upscale=2,
#                 img_range=1.,
#                 resi_connection='1conv',
#                 split_size=[8,8],
#                 c_ratio=0.5,
#                 **kwargs):
#         super().__init__()
#
#         num_in_ch = in_chans
#         num_out_ch = in_chans
#         num_feat = 64
#         self.img_range = img_range
#         if in_chans == 3:
#             rgb_mean = (0.4488, 0.4371, 0.4040)
#             self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
#         else:
#             self.mean = torch.zeros(1, 1, 1, 1)
#         self.upscale = upscale
#
#         # ------------------------- 1, Shallow Feature Extraction ------------------------- #
#         self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
#
#         # ------------------------- 2, Deep Feature Extraction ------------------------- #
#         self.num_layers = len(depth)
#         self.use_chk = use_chk
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         heads=num_heads
#
#         self.before_RG = nn.Sequential(
#             Rearrange('b c h w -> b (h w) c'),
#             nn.LayerNorm(embed_dim)
#         )
#
#         curr_dim = embed_dim
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
#
#         self.layers = nn.ModuleList()
#         for i in range(self.num_layers):
#             layer = ResidualGroup(
#                 dim=embed_dim,
#                 num_heads=heads[i],
#                 reso=img_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])],
#                 act_layer=act_layer,
#                 norm_layer=norm_layer,
#                 depth=depth[i],
#                 use_chk=use_chk,
#                 resi_connection=resi_connection,
#                 rs_id=i,
#                 split_size = split_size,
#                 c_ratio = c_ratio
#                 )
#             self.layers.append(layer)
#
#         self.norm = norm_layer(curr_dim)
#         # build the last conv layer in deep feature extraction
#         if resi_connection == '1conv':
#             self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         elif resi_connection == '3conv':
#             # to save parameters and memory
#             self.conv_after_body = nn.Sequential(
#                 nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
#
#         # ------------------------- 3, Reconstruction ------------------------- #
#         self.conv_before_upsample = nn.Sequential(
#                 nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
#         self.upsample = Upsample(upscale, num_feat)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward_features(self, x):
#         _, _, H, W = x.shape
#         x_size = [H, W]
#         x = self.before_RG(x)
#         for layer in self.layers:
#             x = layer(x, x_size)
#         x = self.norm(x)
#
#         x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
#
#         return x
#
#     def forward(self, x):
#         """
#         Input: x: (B, C, H, W)
#         """
#         self.mean = self.mean.type_as(x)
#         x = (x - self.mean) * self.img_range
#
#         x = self.conv_first(x)
#         x = self.conv_after_body(self.forward_features(x)) + x
#         x = self.conv_before_upsample(x)
#         x = self.conv_last(self.upsample(x))
#
#         x = x / self.img_range + self.mean
#         return x
#
#
# if __name__ == '__main__':
#     upscale = 1
#     height = 62
#     width = 66
#     model = RGT(
#         upscale=2,
#         in_chans=3,
#         img_size=64,
#         img_range=1.,
#         depth=[6,6,6,6,6,6],
#         embed_dim=180,
#         num_heads=[6,6,6,6,6,6],
#         mlp_ratio=2,
#         resi_connection='1conv',
#         split_size=[8, 8],
#         upsampler='pixelshuffle').cuda()
#     # print(model)
#     print(height, width)
#
#     x = torch.randn((1, 3, height, width)).cuda()
#     x = model(x)
#     print(x.shape)