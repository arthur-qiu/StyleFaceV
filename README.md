# StyleFaceV - Official PyTorch Implementation

This repository provides the official PyTorch implementation for the following paper:

**StyleFaceV: Face Video Generation via Decomposing and Recomposing Pretrained StyleGAN3**</br>
[Haonan Qiu](http://haonanqiu.com/), [Yuming Jiang](https://yumingj.github.io/), [Hang Zhou](https://hangz-nju-cuhk.github.io/), [Wayne Wu](https://dblp.org/pid/50/8731.html), and [Ziwei Liu](https://liuziwei7.github.io/)</br>
Arxiv, 2022.

From [MMLab@NTU](https://www.mmlab-ntu.com/index.html) affliated with S-Lab, Nanyang Technological University and SenseTime Research.

<img src="docs/teaser.png" width="92%" height="92%"/>

[**[Project Page]**]() | [**[Paper]**]() | [**[Demo Video]**]()


### Generated Samples

## Updates

- [07/2022] Paper and demo video are released.
- [07/2022] Code is released.

## Installation
**Clone this repo:**
```bash
git clone https://github.com/arthur-qiu/StyleFaceV.git
cd StyleFaceV
```

**Dependencies:**

All dependencies for defining the environment are provided in `environment/text2human_env.yaml`.
We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/) to manage the python environment:

```bash
conda env create -f ./environment/stylefacev.yaml
conda activate stylefacev
```

## Datasets

Image Data: [Unaligned FFHQ](https://github.com/NVlabs/ffhq-dataset)

Video Data: [RAVDESS](https://zenodo.org/record/1188976)

Download the processed video data via this [Google Drive](https://drive.google.com/file/d/17tMHrpvTm08ixAwnzTI9dN0BhCjmCwgV/view?usp=sharing) or process the data via this [repo](https://github.com/AliaksandrSiarohin/video-preprocessing)

Put all the data at the path "../data".

transform the video data into .png form:

```bash
python scripts vid2img.py
```

## Sampling

### Pretrained Models

Pretrained models can be downloaded from this [Google Drive](https://drive.google.com/file/d/1VyI8_AbPwAUaZJPaPba8zxsFIWumlDen/view?usp=sharing). Unzip the file and put them under the dataset folder with the following structure:
```
pretrained_models
├── network-snapshot-005000.pkl  # styleGAN3 checkpoint finetuned on both RAVDNESS and unaligned FFHQ.
├── wing.ckpt                    # Face Alignment model from https://github.com/protossw512/AdaptiveWingLoss.
├── motion_net.pth               # trained motion sampler.
├── pre_net.pth
└── pre_pose_net.pth
checkpoints/stylefacev
├── latest_net_FE.pth            # appearance extractor + recompostion 
├── latest_net_FE_lm.pth         # first half of pose extractor
└── latest_net_FE_pose.pth       # second half of pose extractor
```

### Generating Videos

```bash
python test.py --dataroot ../data/actor_align_512_png --name stylefacev \\
    --network_pkl=pretrained_models/network-snapshot-005000.pkl --model sample \\
    --model_names FE,FE_pose,FE_lm --rnn_path pretrained_models/motion_net.pth \\
    --n_frames_G 60 --num_test=64 --results_dir './sample_results/'
```

## Training

### Pre Stage
This stage is purely trained on image data and will help the convergence.
```bash
python train.py --dataroot ../data/actor_align_512_png --name stylefacev_try \\
--network_pkl=pretrained_models/network-snapshot-005000.pkl --model stylepre
```

You can also use pre_net.pth and pre_pose_net.pth from the folder of pretrained_models.

### Decomposing and Recomposing Pipeline

```bash
python train.py --dataroot ../data/actor_align_512_png --name stylefacev \\
--network_pkl=pretrained_models/network-snapshot-005000.pkl --model stylefacevadv \\
--pose_path checkpoints/stylefacev_pre/latest_net_FE.pth \\
--pre_path checkpoints/stylefacev_pre/latest_net_FE.pth \\
--n_epochs 50 --n_epochs_decay 50 --lr 0.0002
```

### Motion Sampler

```bash
python train.py --dataroot ../data/actor_align_512_png --name motion \\
--network_pkl=pretrained_models/network-snapshot-005000.pkl --model stylernn \\
--pre_path checkpoints/stylefacev/latest_net_FE.pth \\
--pose_path checkpoints/stylefacev/latest_net_FE_pose.pth \\
--lm_path checkpoints/stylefacev/latest_net_FE_lm.pth \\
--n_frames_G 30 
```

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex

```