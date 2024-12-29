import torch
from torch.nn import functional as F
import torch.nn as nn
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

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

@MODEL_REGISTRY.register()
class RGTModel(SRModel):

    def test(self):
        self.use_chop = self.opt['val']['use_chop'] if 'use_chop' in self.opt['val'] else False
        if not self.use_chop:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(self.lq)
                self.net_g.train()

        # test by partitioning
        else:
            _, C, h, w = self.lq.size()
            split_token_h = h // 200 + 1  # number of horizontal cut sections
            split_token_w = w // 200 + 1  # number of vertical cut sections

            patch_size_tmp_h = split_token_h
            patch_size_tmp_w = split_token_w
            
            # padding
            mod_pad_h, mod_pad_w = 0, 0
            if h % patch_size_tmp_h != 0:
                mod_pad_h = patch_size_tmp_h - h % patch_size_tmp_h
            if w % patch_size_tmp_w != 0:
                mod_pad_w = patch_size_tmp_w - w % patch_size_tmp_w
                
            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h+mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w+mod_pad_w]

            _, _, H, W = img.size()
            split_h = H // split_token_h  # height of each partition
            split_w = W // split_token_w  # width of each partition

            # overlapping
            shave_h = 16
            shave_w = 16
            scale = self.opt.get('scale', 1)
            ral = H // split_h
            row = W // split_w
            slices = []  # list of partition borders
            for i in range(ral):
                for j in range(row):
                    if i == 0 and i == ral - 1:
                        top = slice(i * split_h, (i + 1) * split_h)
                    elif i == 0:
                        top = slice(i*split_h, (i+1)*split_h+shave_h)
                    elif i == ral - 1:
                        top = slice(i*split_h-shave_h, (i+1)*split_h)
                    else:
                        top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                    if j == 0 and j == row - 1:
                        left = slice(j*split_w, (j+1)*split_w)
                    elif j == 0:
                        left = slice(j*split_w, (j+1)*split_w+shave_w)
                    elif j == row - 1:
                        left = slice(j*split_w-shave_w, (j+1)*split_w)
                    else:
                        left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                    temp = (top, left)
                    slices.append(temp)
            img_chops = []  # list of partitions
            for temp in slices:
                top, left = temp
                img_chops.append(img[..., top, left])
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    outputs = []
                    for chop in img_chops:
                        out = self.net_g_ema(chop)  # image processing of each partition
                        outputs.append(out)
                    _img = torch.zeros(1, C, H * scale, W * scale)
                    # merge
                    for i in range(ral):
                        for j in range(row):
                            top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                            left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                            if i == 0:
                                _top = slice(0, split_h * scale)
                            else:
                                _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                            if j == 0:
                                _left = slice(0, split_w*scale)
                            else:
                                _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                            _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                    self.output = _img
            else:
                self.net_g.eval()
                with torch.no_grad():
                    outputs = []
                    for chop in img_chops:
                        out = self.net_g(chop)  # image processing of each partition
                        outputs.append(out)
                    _img = torch.zeros(1, C, H * scale, W * scale)
                    # merge
                    for i in range(ral):
                            top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                            left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                            if i == 0:
                                _top = slice(0, split_h * scale)
                            else:
                                _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                            if j == 0:
                                _left = slice(0, split_w * scale)
                            else:
                                _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                            _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                    self.output = _img
                self.net_g.train()
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
