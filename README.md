# MQP
## A Novel Video Salient Object Detection Method via Semi-supervised Motion Quality Perception

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.1.0 with a single GeForce RTX 2080Ti GPU with 11GB Memory.
* Windows
* CUDA v10.1, cudnn v.7.5.0
* PyTorch 1.1.0
* torchvision

## Update
The training code has been uploaded
## Todo
Upload data preprocessing code
## Usage
1.Clone

git clone https://github.com/qduOliver/MQP.git

cd MQP/

2.Download the datasets

Download the following datasets and unzip them into your_data folder.
All datasets can be downloaded at this [data link](http://dpfan.net/news/).

* Davis
* Segtrack-v2
* Visal
* DAVSOD
* VOS

3.Download the pre-trained models

Because the Baidu Cloud link failed before, it has been updated now, please click the link below.
Download the following [pre-trained models](https://pan.baidu.com/s/1S8716W0CsXhCvscGgz_Fzg)(code:771o) into pretmodel folder. 

3.Train
run train.py

4.Test
run test.py

## Data
Our saliency detection results can be downloaded on [BaiduCloud](https://pan.baidu.com/s/1amXriy8kcrjF76iruk7cjA)(code:nkox). 


Thanks to [CPD](https://github.com/wuzhe71/CPD)  and [PWC-net](https://github.com/sniklaus/pytorch-pwc)


