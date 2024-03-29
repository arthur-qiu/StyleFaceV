# StyleFaceV - Official PyTorch Implementation

This repository provides the official PyTorch implementation for the following paper:

**StyleFaceV: Face Video Generation via Decomposing and Recomposing Pretrained StyleGAN3**</br>
[Haonan Qiu](http://haonanqiu.com/), [Yuming Jiang](https://yumingj.github.io/), [Hang Zhou](https://hangz-nju-cuhk.github.io/), [Wayne Wu](https://wywu.github.io/), and [Ziwei Liu](https://liuziwei7.github.io/)</br>
Arxiv, 2022.

From [MMLab@NTU](https://www.mmlab-ntu.com/index.html) affiliated with S-Lab, Nanyang Technological University and SenseTime Research.

<img src="docs/teaser.png" width="92%" height="92%"/>

[**[Project Page]**](http://haonanqiu.com/projects/StyleFaceV.html) | [**[Paper]**](https://arxiv.org/abs/2208.07862) | [**[Demo Video]**](https://youtu.be/BZNLcD04-Fc)


### Generated Samples

<img src="docs/results1.gif" width="92%" height="92%"/>

<img src="docs/results2.gif" width="92%" height="92%"/>

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

All dependencies for defining the environment are provided in `environment/stylefacev.yaml`.
We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/) to manage the python environment:

```bash
conda env create -f ./environment/stylefacev.yaml
conda activate stylefacev
```

## Datasets

Image Data: [Unaligned FFHQ](https://github.com/NVlabs/ffhq-dataset)

Video Data: [RAVDESS](https://zenodo.org/record/1188976)

Download the processed video data via this [Google Drive](https://drive.google.com/file/d/17tMHrpvTm08ixAwnzTI9dN0BhCjmCwgV/view?usp=sharing) or process the data via this [repo](https://github.com/AliaksandrSiarohin/video-preprocessing)

Put all the data at the path `../data`.

Transform the video data into `.png` form:

```bash
python scripts/vid2img.py
```

## Sampling

### Pretrained Models

Pretrained models can be downloaded from this [Google Drive](https://drive.google.com/file/d/1c_JWfDjN44XpI8OG24p3FkdEufJGsv34/view?usp=sharing). Unzip the file and put them under the dataset folder with the following structure:
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
python test.py --dataroot ../data/actor_align_512_png --name stylefacev \
    --network_pkl=pretrained_models/network-snapshot-005000.pkl --model sample \
    --model_names FE,FE_pose,FE_lm --rnn_path pretrained_models/motion_net.pth \
    --n_frames_G 60 --num_test=64 --results_dir './sample_results/'
```

## Training

### Pre Stage

If you want to use new datasets, please finetune the StyleGAN3 model first.

This stage is purely trained on image data and will help the convergence.

```bash
python train.py --dataroot ../data/actor_align_512_png --name stylepose \
    --network_pkl=pretrained_models/network-snapshot-005000.pkl \
    --model stylevpose --n_epochs 5 --n_epochs_decay 5
python train.py --dataroot ../data/actor_align_512_png --name stylefacev_pre \
    --network_pkl=pretrained_models/network-snapshot-005000.pkl \
    --model stylepre --pose_path checkpoints/stylevpose/latest_net_FE.pth
```

You can also use `pre_net.pth` and `pre_pose_net.pth` from the folder of `pretrained_models`.

```bash
python train.py --dataroot ../data/actor_align_512_png --name stylefacev_pre \
    --network_pkl=pretrained_models/network-snapshot-005000.pkl --model stylepre \
    --pre_path pretrained_models/pre_net.pth --pose_path pretrained_models/pre_pose_net.pth
```

### Decomposing and Recomposing Pipeline

```bash
python train.py --dataroot ../data/actor_align_512_png --name stylefacev \
    --network_pkl=pretrained_models/network-snapshot-005000.pkl --model stylefacevadv \
    --pose_path pretrained_models/pre_pose_net.pth \
    --pre_path checkpoints/stylefacev_pre/latest_net_FE.pth \
    --n_epochs 50 --n_epochs_decay 50 --lr 0.0002
```

### Motion Sampler

```bash
python train.py --dataroot ../data/actor_align_512_png --name motion \
    --network_pkl=pretrained_models/network-snapshot-005000.pkl --model stylernn \
    --pre_path checkpoints/stylefacev/latest_net_FE.pth \
    --pose_path checkpoints/stylefacev/latest_net_FE_pose.pth \
    --lm_path checkpoints/stylefacev/latest_net_FE_lm.pth \
    --n_frames_G 30 
```

If you do not have a 32G GPU, reduce the `n_frames_G` (12 for 16G). Or only add supervision on pose representations:

```bash
python train.py --dataroot ../data/actor_align_512_png --name motion \
    --network_pkl=pretrained_models/network-snapshot-005000.pkl --model stylernns \
    --pose_path checkpoints/stylefacev/latest_net_FE_pose.pth \
    --lm_path checkpoints/stylefacev/latest_net_FE_lm.pth \
    --n_frames_G 30 
```

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2208.07862,
  doi = {10.48550/ARXIV.2208.07862},
  url = {https://arxiv.org/abs/2208.07862},
  author = {Qiu, Haonan and Jiang, Yuming and Zhou, Hang and Wu, Wayne and Liu, Ziwei},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {StyleFaceV: Face Video Generation via Decomposing and Recomposing Pretrained StyleGAN3},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
