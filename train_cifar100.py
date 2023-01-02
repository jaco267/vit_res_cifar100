#%%
# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
import matplotlib.pyplot as plt
# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r',default=-1,type=int, help='resume from checkpoint')
parser.add_argument('--aug',default=False, action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='3')  #200
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)


args = parser.parse_args()


bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.aug
save_every = 2
print('use amp',use_amp)   #浮點數運算 減半  節省 memory 和訓練時間  accucy 不變  （因為neural net 對精度不敏感）
print('data augmentation',aug)
print('n_epochs',args.n_epochs,'batch size',bs)
print('save every',save_every,'epochs')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device',device)
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('\r==> Preparing data..')
if args.net=="vit_timm":
    size = 224
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True,)# num_workers=8)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,)# num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('\r==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    #python train_cifar10.py --net res18 --n_epochs 102 --lr 0.001 --bs 256
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    # python train_cifar10.py --net res50 --n_epochs 10 --lr 0.001 --bs 256   
    #*epoch 10 acc 87.9
    # python train_cifar10.py --net res50 --n_epochs 10 --lr 0.001 --bs 256 --aug 
    #*             77.34
    # python train_cifar10.py --net res50 --n_epochs 100 --lr 0.001 --bs 256
    #*epoch 100 acc 93.58
    net = ResNet50()
elif args.net=='res101':
    #python train_cifar10.py --net res101 --n_epochs 10 --lr 0.001 --bs 256 
    #*epoch 10 acc 87.9
    net = ResNet101()
elif args.net=="vit_timm":
    import timm                       #       384
    print(timm.list_models('vit*',pretrained=True))  #列出所有的選擇
    net = timm.create_model("vit_tiny_r_s16_p8_224", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 100)



net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)# use cosine scheduling

train_loss_list = [];  train_acc_list = [];   test_loss_list = [];   test_acc_list = []
lr_list = []

if args.resume>=0:
    # Load checkpoint.
    print(f'==> Resuming from checkpoint..{args.resume}')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.net}-patch_{args.patch}-epoch_{args.resume}-ckpt.t7')
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint['epoch']

    train_loss_list = checkpoint['train_loss']
    train_acc_list = checkpoint['train_acc']
    test_loss_list  = checkpoint['test_loss']
    test_acc_list   = checkpoint['test_acc']
    lr_list         = checkpoint['lr']
print('ss',start_epoch)
##### Training
def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0;    correct = 0;    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #inputs(512,3,32,32)  targets (512)
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)  #(batch_size,num_classes)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward();
        scaler.step(optimizer);   scaler.update();   optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0);   correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 
                   f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {(100.*correct/total):.3f}% ({correct}/{total})')
    return train_loss/(batch_idx+1), 100.*correct/total
##### Validation
def test(epoch):
    net.eval()
    test_loss = 0;    correct = 0;    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0);   correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 
                f'Loss: {test_loss/(batch_idx+1):.3f} | Acc: {(100.*correct/total):.3f}% ({correct}/{total})')
    
    # Save checkpoint.
    acc = 100.*correct/total

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss/(batch_idx+1), 100.*correct/total

net.cuda()
for epoch in range(start_epoch, start_epoch + args.n_epochs ):
    start = time.time()
    trainloss, train_acc = train(epoch)
    test_loss, test_acc  = test(epoch)

    scheduler.step() # step cosine scheduling
    
    train_loss_list.append(trainloss);   train_acc_list.append(train_acc)
    test_loss_list.append(test_loss);    test_acc_list.append(test_acc)
    lr_list.append(optimizer.param_groups[0]["lr"])    
    # Log training..
    if  (epoch+1)%save_every ==0:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict(),
              'epoch':  epoch + 1,
              'train_loss': train_loss_list,
              'train_acc':  train_acc_list,
              'test_loss':test_loss_list,
              'test_acc':test_acc_list,
              'lr': lr_list}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{args.net}-patch_{args.patch}-epoch_{epoch+1}-ckpt.t7')


    
fig, ax = plt.subplots(3,1,figsize=(8,8))
epoch_tot_num = np.arange(len(train_loss_list))
ax[0].plot(epoch_tot_num,train_loss_list,label='train loss')
ax[0].plot(epoch_tot_num,test_loss_list ,label='test loss')

ax[1].set_ylim(20,100)
ax[1].plot(epoch_tot_num,train_acc_list ,label='train acc')
ax[1].plot(epoch_tot_num,test_acc_list  ,label='test acc')
ax[2].plot(np.arange(len(lr_list)),lr_list  ,label='lr')
for i in range(3):    ax[i].legend()    
plt.tight_layout() 
plt.show()    
print(test_acc_list[-1])



# print("test and save prediction into csv")


# def test(epoch):
#     net.eval()
#     test_loss = 0;    correct = 0;    total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0);   correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 
#                 f'Loss: {test_loss/(batch_idx+1):.3f} | Acc: {(100.*correct/total):.3f}% ({correct}/{total})')
    
#     # Save checkpoint.
#     acc = 100.*correct/total

#     os.makedirs("log", exist_ok=True)
#     content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
#     print(content)
#     with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
#         appender.write(content + "\n")
#     return test_loss/(batch_idx+1), 100.*correct/total

# with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
#     writer = csv.writer(f, lineterminator='\n')
    # writer.writerow(list_loss) 
    # writer.writerow(list_acc) 


