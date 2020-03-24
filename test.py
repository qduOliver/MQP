import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
from scipy import misc
import  cv2
import matplotlib.pyplot as plt
import time
import os
import torchvision
from data import get_loader_my

from model.MQP_models import MQP_VGG
from data import test_dataset_my
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path='./data/'
model = MQP_VGG() 

static_model=torch.load('./premodel/')
model.load_state_dict(static_model)  

model.cuda()
model.eval()

test_folders=os.listdir(dataset_path)

for i in range(0,len(test_folders)):
    dataset=test_folders[i]  
    save_path = r'./result/' + dataset + '/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset +'/'+'imgs/'
    test_loader = test_dataset_my(image_root, opt.testsize)
    names=[]
    for i in range(test_loader.size):
        image = image.cuda()
        star=time.time()
        _, ress = model(image)
        end=time.time()
        runtime=end-star
        print('runtime',runtime)
        h=shape[0]
        w=shape[1]
        res = F.upsample(ress, size=[h,w], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res*255)
           