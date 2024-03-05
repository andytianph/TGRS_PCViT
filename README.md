# PCViT: A Pyramid Convolutional Vision Transformer Detector for Object Detection in Remote-Sensing Imagery<a href="https://ieeexplore.ieee.org/document/10417056"><img src="https://img.shields.io/badge/TGRS-Paper-blue"></a>
### [Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Penghao Tian](https://github.com/andytianph), [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), Haitao Xu and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).

<p align="center">
  <a href="#updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#results-and-models">Results & Models</a> |
  <a href="#usage">Usage</a> |
  <a href="#Citation Details">Citation Details</a> |
  <a href="#Acknowledge">Acknowledge</a> 
</p >

This branch contains the official pytorch implementation for <a href="https://ieeexplore.ieee.org/document/10417056">PCViT: A Pyramid Convolutional Vision Transformer Detector for Object Detection in Remote-Sensing Imagery</a> [TGRS'24].

## Updates
### 2023.10.18
The codes of the PCViT has been released. The weights and logs will be uploaded soon. 

## Introduction
This repository contains codes, models and test results for the paper "[PCViT: A Pyramid Convolutional Vision Transformer Detector for Object Detection in Remote-Sensing Imagery](https://ieeexplore.ieee.org/document/10417056)".

<div align=center><img src="/Image/PCViT_baseline.png" width="80%" height="80%"></div>
Fig. 1: The structure of the baseline of the proposed PCViT. The proposed backbone constitutes a multiscale pyramid with three scale stages. The initial two stages consist of convolutional blocks, and the final stage consists of transformer blocks. Here, we refine the transformer block using the PCM  and LGKA module. Then, The multiscale features derived from the backbone are then fed into the subsequent FRPN neck to facilitate contextual information interaction before being directed to the detection head.

<div align=center><img src="/Image/Pretrain.png" width="80%" height="80%"></div>
Fig. 2: The pipeline of the proposed MPP. During pretraining, K masked perspectives of each image are randomly sampled in a mini-batch with MPM. Then, they will be fed to the encoder and the decoder for invisible reconstruction with targets.

<div align=center><img src="/Image/LGKA.png" width="80%" height="80%"></div>
Fig. 3: Local/Global k-NN Attention. In each group of transformer subblocks, we use local attention for the first two layers, that is, reduce computational complexity through 16x16 window attention. For propagation between windows, we use global attention in the third layer.


## Results and Models
#### MillionAID
The models are trained on 4 3090 machines with 2 images per gpu, which makes a batch size of 32 during training.
|Pretrain|Backbone | Input size | Params (M) | Pretrained model|
|-------|-------- | ----------  | ----- | ----- |
| MPP | PCViT | 224 × 224 | 112 | [Weights](--) |


#### Results from this repo on DIOR
The models are trained on 2 3090 machines with 2 images per gpu, which makes a batch size of 1 during training.

| Model | Pretrain | Machine | FrameWork | Box mAP@50 | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| PCViT | MPP | GPU | Faster RCNN | 80.25 | [config](-) | [log](-) | [Weights](--) |

## Usage

Environment:
- Python 3.8.5
- Pytorch 1.9.0+cu111
- torchvision 0.10.0+cu111
- timm 0.4.12
- mmcv-full 1.3.9


### Pretrain (4 × 3090 GPUs, 1 weeks)

1. Preparing the MillionAID: Download the [MillionAID](https://captain-whu.github.io/DiRS/). It is easy for users to record image names and revise corresponding codes `prtrain`.

2. To pretrain PCViT with **multi-node distributed training**, run the following on 1 node with 4 GPUs each (only mask 75% is supported): (batchsize: 128=4*32)

```bash
python -m torch.distributed.launch --nproc_per_node 4 main_pretrain.py \
--batch_size 32 --model fastconvmae_convvitae_base_patch16 \
--norm_pix_loss --mask_ratio 0.75 --epochs 100 \
--warmup_epochs 20 --blr 6.0e-4 --weight_decay 0.05
```
*Note: Padding the convolutional kernel of PCM in the pretrained PCViT with `convertK1toK3.py` for finetuning.*


### Finetune
We use PyTorch 1.9.0 or NGC docker 21.06, and mmcv 1.3.9 for the experiments.
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/andytianph/TGRS_PCViT.git
cd PCViT/finetune
pip install -v -e .
```

After install the two repos, install timm and einops, i.e.,
```bash
pip install timm==0.4.9 einops
```

Download the pretrained models from [MAE](https://github.com/facebookresearch/mae), [ViTAE](https://github.com/ViTAE-Transformer/ViTAE-Transformer) or [PCViT](weight), and then conduct the experiments by

```bash
# for single machine
bash tools/dist_train.sh <Config PATH> <NUM GPUs> --cfg-options model.pretrained=<Pretrained PATH>

# for multiple machines
python -m torch.distributed.launch --nnodes <Num Machines> --node_rank <Rank of Machine> --nproc_per_node <GPUs Per Machine> --master_addr <Master Addr> --master_port <Master Port> tools/train.py <Config PATH> --cfg-options model.pretrained=<Pretrained PATH> --launcher pytorch
```


## Citation Details
If you find this code helpful, please kindly cite:

```
@ARTICLE{10417056,
  author={Li, Jiaojiao and Tian, Penghao and Song, Rui and Xu, Haitao and Li, Yunsong and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={PCViT: A Pyramid Convolutional Vision Transformer Detector for Object Detection in Remote-Sensing Imagery}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  keywords={Transformers;Feature extraction;Task analysis;Object detection;Detectors;Nickel;Semantics;Convolutional neural network (CNN);feature pyramid network (FPN);multiscale object detection;remote-sensing images (RSIs);vision transformer (ViT)},
  doi={10.1109/TGRS.2024.3360456}}
```

## Acknowledge
We acknowledge the excellent implementation from [mmdetection](https://github.com/open-mmlab/mmdetection), [MAE](https://github.com/facebookresearch/mae), [Remote-Sensing-RVSA](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)
