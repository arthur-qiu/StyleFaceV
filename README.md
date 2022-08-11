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

## Sampling

### Pretrained Models

Pretrained models can be downloaded from this [Google Drive](https://drive.google.com/file/d/1VyI8_AbPwAUaZJPaPba8zxsFIWumlDen/view?usp=sharing). Unzip the file and put them under the dataset folder with the following structure:
```
pretrain_models
├── .pth
├── .pth
├── .pth
├── .pth
├── .pth
└── .pth
```

### Generating Videos

```bash
python test.py --dataroot ../data/actor_align_512_png --name stylefacev --n_frames_G 60 --epoch 35 --network_pkl=pretrain_models/network-snapshot-005000.pkl --model sample --pose_path checkpoints/ffhq_stylevpose5/latest_net_FE.pth --model_names FE,FE_pose,FE_lm --rnn_path checkpoints/ffhq_stylep5ddrnnnewalign30/100_net_G.pth --num_test=1024 --results_dir './sample_results/'
```

## Training

### Stage 1

```bash
python train.py --dataroot ../data/actor_align_512_png --name stylefacev_pre --network_pkl=pretrain_models/network-snapshot-005000.pkl --model stylepre --pose_path checkpoints/ffhq_stylevideopose/latest_net_FE.pth
```

### Stage 2

```bash

```

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex

```