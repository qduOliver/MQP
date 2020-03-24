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

from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data import test_dataset_my
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()
#dataset_path = 'G:/spyder_workpeace_song/total/dataset/images/rgb/480p/' 

#dataset_name='FBMS'
#dataset_path = 'G:/source/dataset/%s/'%dataset_name
#SegTrack-V2
#dataset_path='G:/spyder_workpeace_song/total/CPD-master/CPD-master/dataset/davis_test/'
#dataset_path='G:/source/dataset/flo_map/flojpg/'G:\source\dataset\flo_map\flo_train\anno_test

#davis_test   FBMS  Segtrack-v2 Visal VOS_test_png Easy-35
dataset_path='E:/dataset/dataset/Visal/'
#dataset_path=r'E:\dataset\dataset\flow\pwc_net_jpg\davis_train/'
#dataset_path='G:/source/dataset/flo_map/flo_label/flo_label_train/'

#dataset_name='davis'
#dataset_path='G:\spyder_workpeace_song\total\dataset\images\rgb\480p'
#Easy-35   

if opt.is_ResNet:
    model = CPD_ResNet()
    model.load_state_dict(torch.load('premodel/flo_fine_cpd_davsod866'))
else:
    model = CPD_VGG() 
    
    static_model=torch.load('test_train_weight/SSAV_replace(gt_xml_(0.6_01)_2028(davis_train)+17bad_456)w10_2890(lr4b10epoch30)/vgg.pth.19')
    model.load_state_dict(static_model)  

model.cuda()
model.eval()

#test_datasets = ['PASCAL', 'ECSSD', 'DUT-OMRON', 'DUTS-TEST', 'HKUIS']

#test_datasets=['blackswan/']
#test_folders=os.listdir(dataset_path+dataset_name)
test_folders=os.listdir(dataset_path)

for i in range(0,len(test_folders)):
    dataset=test_folders[i]  
    if opt.is_ResNet:
        save_path = './flo_result_yuan(inputbn_res)/davis_test/' + dataset + '/'
    else:
        save_path = r'./test_train_result_new/SSAV_replace(gt_xml_(0.6_01)_2028(davis_train)+17bad_456)w10_2890(lr4b10epoch30)_19/Visal/' + dataset + '/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset +'/'+'imgs/'
      
#    image_root = dataset_path + dataset +'/'
    
#    image_root = dataset_path +dataset_name+'/'+ dataset +'/'
#    image_root = dataset_path +'/'+ dataset +'/'
#    gt_root = 'G:/source/dataset/%s/'%dataset_name + dataset +'/'+'ground-truth/'
#    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    test_loader = test_dataset_my(image_root, opt.testsize)
    names=[]
    for i in range(test_loader.size):
        
        image, name ,index,shape= test_loader.load_data()
#        names.append(name)
#        print('image.shape',image.shape) [1,3,352,352]
#        print('rgbname',name)
#        gt = np.asarray(gt, np.float32)
#        gt /= (gt.max() + 1e-8)
        image = image.cuda()
#     
        star=time.time()
        _, ress = model(image)
        end=time.time()
        runtime=end-star
        print('runtime',runtime)
#            print('res',ress.shape)
#            res=torchvision.utils.make_grid(res).numpy()
        h=shape[0]
        w=shape[1]
        
        res = F.upsample(ress, size=[h,w], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
#                plt.imshow(res)
#                plt.show()
#                print('res',res.shape)
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#                print('res',res)
#                plt.imshow(res)
#                plt.show()
#                misc.imsave(save_path+names[i], res)
#                print(names[i+h-4])
        cv2.imwrite(save_path+name, res*255)
        end=time.time()
           
#    train_loader = get_loader_my(image_root, batchsize=5, trainsize=352)
#    for i, pack in enumerate(train_loader, start=1):
#       
#        images, gts = pack
#        #[5,3,352,352]
#        print('images',images.shape)
#        images = Variable(images)
#        images = images.cuda()
#        _, ress = model(images)
##        end=time.time()
##        runtime=end-star
##        print('runtime',runtime)
#        for j in range(5):
#            res = F.upsample(ress[j], size=[540,680], mode='bilinear', align_corners=False)
#            res = res.sigmoid().data.cpu().numpy().squeeze()
#            
#            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#            print('res',res)
##            misc.imsave(save_path+name, res)
#            cv2.imwrite(save_path+ int(i+j)+'.png', res*255)
#            end=time.time()