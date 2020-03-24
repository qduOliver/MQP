# MQP
## A Universal Scheme to Boost Video Salient Object Detection Performance via Weakly Supervised Motion Quality Perception
## Prerequisites
The training and testing experiments are conducted using PyTorch 1.0.1 with a single GeForce RTX 2080Ti GPU with 11GB Memory.
* Windows
* CUDA v10.1, cudnn v.7.5.0
* PyTorch 1.1.0
* torchvision
## Update
## Todo
Upload our code
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
Here, we read the collection of data paths in a list file. You can edit the code in any way you like.

3.Download the pre-trained models

Download the following [pre-trained models](https://pan.baidu.com/s/1pf49N8nPCkMhO0RH01eR0Q)(code:bctu) into pretmodel folder. 

4.Test

run test.py

## Data
our saliency detection results can be downloaded on [BaiduCloud](https://pan.baidu.com/s/1685nRBX8BOx-tp53iiC4NQ)(code:3ron). 


Thanks to [CPD](https://github.com/wuzhe71/CPD)  and [PWC-net](https://github.com/sniklaus/pytorch-pwc)


