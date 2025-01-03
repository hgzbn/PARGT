# PARGT: Parallel Attention Recursive Generalization Transformer for Image Super-Resolution

> **Abstract:** Transformer architectures have demonstrated remarkable performance in image super-resolution (SR). However, existing Transformer-based models generally suffer from insufficient local feature modeling, weak feature representation capabilities, and unreasonable loss function design, especially when reconstructing high-resolution (HR) images, where the restoration of fine details is poor. To address these issues, we propose a novel SR model, Parallel Attention Recursive Generalization Transformer (PARGT) in this study, which can effectively capture the fine-grained interactions between local features of the
image and other regions, resulting in clearer and more coherent generated details. Specifically, we introduce the Parallel Local Self-attention (PL-SA) module, which enhances local features by parallelizing the Shift Window Pixel Attention Module (SWPAM) and Channel-Spatial Shuffle Attention Module(CSSAM). In addition, we introduce a new type of feed-forward network called Spatial Fusion Convolution Feed-forward Network (SFCFFN) for multi-scale information fusion. Finally, we optimize the reconstruction of high-frequency details by incorporating a Stationary Wavelet Transform (SWT). To the best of our knowledge, this is the first application of a parallel attention mechanism combined with a multi-scale feed-forward network for SR tasks. Experimental results on several challenging benchmark datasets demonstrate the superiority of our PARGT over state-of-the-art image SR models.The code will be available at https://github.com/hgzbn/PARGT.

---
## ⚙️ Dependencies

- Python 3.8
- PyTorch 1.9.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'PARGT'.
git clone [https://github.com/hgzbn/PARGT.git]
conda create -n PARGT python=3.8
conda activate PARGT
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
python setup.py develop
```


## <a name="datasets"></a>🖨️ Datasets

Used training and testing sets can be downloaded as follows:

| Training Set                                                 |                         Testing Set                          |                        Visual Results                        |
| :----------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images, 100 validation images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) [complete training dataset DF2K: [Google Drive](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) / [Baidu Disk](https://pan.baidu.com/s/1KIcPNz3qDsGSM0uDKl4DRw?pwd=74yc)] | Set5 + Set14 + BSD100 + Urban100 + Manga109 [complete testing dataset: [Google Drive](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1Tf8WT14vhlA49TO2lz3Y1Q?pwd=8xen)] | [Google Drive]() / [Baidu Disk](https://pan.baidu.com/s/10YeQAmkYI9lg2HnlzHBHxA?pwd=eb5i) |

Download training and testing datasets and put them into the corresponding folders of `datasets/`. See [datasets](datasets/README.md) for the detail of the directory structure.
## <a name="training"></a>🔧 Training

- Download [training](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) (DF2K, already processed) and [testing](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) (Set5, Set14, BSD100, Urban100, Manga109, already processed) datasets, place them in `datasets/`.

- Run the following scripts. The training configuration is in `options/train/`.

  ```shell
  # PARGT-S, input=64x64, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/train_RGT_S_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/train_RGT_S_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/train_RGT_S_x4.yml --launcher pytorch
  
  # PARGT, input=64x64, 4 GPUs
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/train_RGT_x2.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/train_RGT_x3.yml --launcher pytorch
  python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/train_RGT_x4.yml --launcher pytorch
  ```

- The training experiment is in `experiments/`.

## <a name="testing"></a>🔨 Testing

### 🌗 Test images with HR

- Download the pre-trained [models](https://drive.google.com/drive/folders/1UNn5LvnfQAi6eHAHz-mTYWu8vCJs5kwu?usp=sharing) and place them in `experiments/pretrained_models/`.

  We provide pre-trained models for image SR: PARGT-S and PARGT (x2, x3, x4).

- Download [testing](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) (Set5, Set14, BSD100, Urban100, Manga109) datasets, place them in `datasets/`.

- Run the following scripts. The testing configuration is in `options/test/` (e.g., [test_RGT_x2.yml](options/test/test_RGT_x2.yml)).

  Note 1:  You can set `use_chop: True` (default: False) in YML to chop the image for testing.

  ```shell
  # No self-ensemble
  # PARGT-S, reproduces results in Table 2 of the main paper
  python basicsr/test.py -opt options/test/test_RGT_S_x2.yml
  python basicsr/test.py -opt options/test/test_RGT_S_x3.yml
  python basicsr/test.py -opt options/test/test_RGT_S_x4.yml
  
  # PARGT, reproduces results in Table 2 of the main paper
  python basicsr/test.py -opt options/test/test_RGT_x2.yml
  python basicsr/test.py -opt options/test/test_RGT_x3.yml
  python basicsr/test.py -opt options/test/test_RGT_x4.yml
  ```

- The output is in `results/`.

### 🌓 Test images without HR

- Download the pre-trained [models](https://drive.google.com/drive/folders/1UNn5LvnfQAi6eHAHz-mTYWu8vCJs5kwu?usp=sharing) and place them in `experiments/pretrained_models/`.

  We provide pre-trained models for image SR: PARGT-S and PARGT (x2, x3, x4).

- Put your dataset (single LR images) in `datasets/single`. Some test images are in this folder.

- Run the following scripts. The testing configuration is in `options/test/` (e.g., [test_single_x2.yml](options/test/test_single_x2.yml)).

  Note 1: The default model is RGT. You can use other models like RGT-S by modifying the YML.

  Note 2:  You can set `use_chop: True` (default: False) in YML to chop the image for testing.

  ```shell
  # Test on your dataset
  python basicsr/test.py -opt options/test/test_single_x2.yml
  python basicsr/test.py -opt options/test/test_single_x3.yml
  python basicsr/test.py -opt options/test/test_single_x4.yml
  ```

- The output is in `results/`.

## <a name="results"></a>🔎 Results

We achieved state-of-the-art performance on synthetic and real-world blur dataset. Detailed results can be found in the paper.

## <a name="acknowledgements"></a>💡 Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).

