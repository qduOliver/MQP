import torch
from torch.autograd import Variable
import os, argparse
from datetime import datetime
from utils import clip_gradient, adjust_lr
from model.MQP_VGG import MQP_VGG
from data import get_loader_txt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=11, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

model_total= MQP_VGG()
rgb_pre_train=torch.load(r'premodel/CPD.pth',map_location='cuda:0')
model_dict = model_total.state_dict()
pretrained_dict = {k: v for k, v in rgb_pre_train.items() if k in model_dict}
model_dict.update(pretrained_dict)
model_total.load_state_dict(model_dict)
model_total.cuda()

params = model_total.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
CE = torch.nn.BCEWithLogitsLoss()

def train(train_loader, model_total,optimizer, epoch,dataset_name,video):
    model_total.train()

    for i, pack in enumerate(train_loader, start=0):
       
        optimizer.zero_grad()
        images,gts = pack
       
        images = Variable(images)
        gts = Variable(gts)
        
        images = images.cuda()
        gts = gts.cuda()
        att_s,d= model_total(images)
              
        loss1 = CE(att_s, gts)
        loss2 = CE(d, gts)
        
        loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} '.
              format(datetime.now(), epoch, opt.epoch, i, len(train_loader), loss1.data, loss2.data))
        
        save_path = r'../%s/%s/'%(dataset_name,video)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model_total.state_dict(), save_path + 'vgg_%d.pth' % epoch)
print("Let's go!")

######################################################################video
dataset_list = ['davis_test','Segtrack-v2','Visal','Easy-35','VOS_test_png']
#dataset_list = ['VOS_test_png']
for aa in range(0,len(dataset_list)):
    
    dataset_name = dataset_list[aa]
    path_txt_total = './txt/%s/'%(dataset_name)
    txt_list = os.listdir(path_txt_total)
    
    for sj in range(0,len(txt_list)):
        txt_name = txt_list[sj]
        print(txt_name)
        path_txt = path_txt_total+txt_name
        
        for epoch in range(1, opt.epoch):
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train_loader_txt=get_loader_txt(path_txt, batchsize=opt.batchsize, trainsize=opt.trainsize)
            train(train_loader_txt, model_total, optimizer, epoch,dataset_name,path_txt[:-4])