# TGRS_PCViT
Official implementation for [TGRS'24] "PCViT: A Pyramid Convolutional Vision Transformer Detector for Object Detection in Remote-Sensing Imagery"

PCViT: A Pyramid Convolutional Vision Transformer Detector for Object Detection in Remote-Sensing Imagery, TGRS, 2024.
==
[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Penghao Tian](https://github.com/andytianph), [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), Haitao Xu and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).
***
Code for the paper: [PCViT: A Pyramid Convolutional Vision Transformer Detector for Object Detection in Remote-Sensing Imagery](https://ieeexplore.ieee.org/document/10417056).


<div align=center>< img src="/Image/PCViT_baseline.png" width="80%" height="80%"></div>
Fig. 1: The structure of the baseline of the proposed PCViT. The proposed backbone constitutes a multiscale pyramid with three scale stages. The initial two stages consist of convolutional blocks, and the final stage consists of transformer blocks. Here, we refine the transformer block using the PCM  and LGKA module. Then, The multiscale features derived from the backbone are then fed into the subsequent FRPN neck to facilitate contextual information interaction before being directed to the detection head.

<div align=center>< img src="/Image/Pretrain.png" width="80%" height="80%"></div>
Fig. 2: The pipeline of the proposed MPP. During pretraining, K masked perspectives of each image are randomly sampled in a mini-batch with MPM. Then, they will be fed to the encoder and the decoder for invisible reconstruction with targets.

<div align=center>< img src="/Image/Local or Global KNN Attention.png" width="80%" height="80%"></div>
Fig. 3: Local/Global k-NN Attention. In each group of transformer subblocks, we use local attention for the first two layers, that is, reduce computational complexity through 16x16 window attention. For propagation between windows, we use global attention in the third layer.

Training and Test Process
--
1) Please prepare the training and test data as operated in the paper. The datasets are Houston2013, Trento, MUUFL Gulfport. The data is placed under the 'data' folder. The file format is tif.
2) Run "demo.py" to to reproduce the Sal2RN results on Trento data set.

We have successfully tested it on Ubuntu 18.04 with PyTorch 1.12.0.

References
--
If you find this code helpful, please kindly cite:

[1]J. Li, Y. Liu, R. Song, Y. Li, K. Han and Q. Du, "Sal2RN: A Spatial-Spectral Salient Reinforcement Network for Hyperspectral and LiDAR Data Fusion Classification," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2022.3231930.

Citation Details
--
BibTeX entry:
```
@ARTICLE{9998520,
  author={Li, Jiaojiao and Liu, Yuzhe and Song, Rui and Li, Yunsong and Han, Kailiang and Du, Qian},
  journal={
