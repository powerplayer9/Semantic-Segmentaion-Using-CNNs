import datetime
import os
import random

import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import pickle

import torchvision.transforms as standard_transforms
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as pyplot

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import cityscapes
from fcn8s import *
from duc_hdc import *
from unet import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d


args = {
    'train_batch_size': 2,
    'test_batch_size': 2,
    'epoch_num': 1,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'input_size': (256, 512),
    'momentum': 0.95,
    'lr_patience': 50,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes no snapshot
    'print_freq': 20,
    'val_batch_size': 2,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.05  # randomly sample some validation results to display
}

# Paths to trained models & epoch counts
'''
Comment the next few lines if training from scratch
'''
DUCHDC_epochCount = "EpochNumDUC.pkl"
FCN8_epochCount = "EpochNumFCN8.pkl"
Unet_epochCount = "EpochNumUnet.pkl"


DUCHDC_trainedModelPath = './ducModelFinal.pth'
FCN8_trainedModelPath = './fcnModelFinal.pth'
Unet_trainedModelPath = './unetModelFinal.pth'


# Transforms
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
short_size = int(min(args['input_size']) / 0.875)

joint_transform = joint_transforms.Compose([
    joint_transforms.Scale(short_size),
    joint_transforms.RandomCrop(args['input_size']),
    joint_transforms.RandomHorizontallyFlip()])

input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)])

target_transform = extended_transforms.MaskToTensor()
restore_transform = standard_transforms.Compose([
    extended_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()])

visualize = standard_transforms.ToTensor()

## Loading the datasets
train_set = cityscapes.CityScapes('fine', 'train', joint_transform=joint_transform,
                                  transform=input_transform, target_transform=target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], shuffle=True)

test_set = cityscapes.CityScapes('fine', 'test', joint_transform=joint_transform,
                                  transform=input_transform, target_transform=target_transform)
test_loader = DataLoader(train_set, batch_size=args['test_batch_size'], shuffle=False)

val_set = cityscapes.CityScapes('fine', 'val', joint_transform=joint_transform, transform=input_transform,
                                target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], shuffle=False)


def train(train_loader, net, criterion, optimizer, epoch, train_args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss = AverageMeter()
    curr_iter = len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs).to(2)
        labels = Variable(labels).to(2)

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == cityscapes.num_classes

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data[0], N)
                
        print ('Epoch:',epoch,' train_loss:', train_loss.avg,' Iter:', i ,'/', curr_iter)

def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        with torch.no_grad():
          inputs, gts = data
          N = inputs.size(0)
          inputs = Variable(inputs).to(2)
          gts = Variable(gts).to(2)
  
          outputs = net(inputs)
          predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
  
          val_loss.update(criterion(outputs, gts).data[0] / N, N)
  
          for i in inputs:
              if random.random() > train_args['val_img_sample_rate']:
                  inputs_all.append(None)
              else:
                  inputs_all.append(i.data.cpu())
          gts_all.append(gts.data.cpu().numpy())
          predictions_all.append(predictions)
          print ('Epoch: ', epoch, 'Val Iter: ',vi)
          
        
#        torch.cuda.empty_cache()

    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(predictions_all)
    
    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, cityscapes.num_classes)
  
    print('Epoch: ', epoch,'Val Loss: ',val_loss.avg, 'mean_IoU: ',mean_iu, 'Acc: ',acc, 'acc_cls', acc_cls, 'fwavacc: ', fwavacc)
    return val_loss.avg

# The NETWORK
# Get current Epoch Number from prev trained network
EpochNum = pickle.load( open( DUCHDC_epochCount, "rb" ) )				## CHANGE HERE WHEN YOU CHANGE NETWORK
print ('Total Epoch:' , EpochNum)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
If Training from scratch, choose your required network!!!
'''
#net = DUCHDC(num_classes=cityscapes.num_classes)

#Loading trained Network
net = torch.load(DUCHDC_trainedModelPath)								## CHANGE HERE WHEN YOU CHANGE NETWORK

net = net.to(device)
print(net)

criterion = CrossEntropyLoss2d(size_average=False, ignore_index=cityscapes.ignore_label)

optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}], momentum=args['momentum'])
         
optimizerAdam = optim.Adagrad([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}])

## Use optimizer based on the network
for epoch in range(args['epoch_num'] + 1):
    train(train_loader, net, criterion, optimizer, epoch, args)
    if epoch > 0:
      EpochNum = EpochNum + 1
    print ('Total Epoch:' , EpochNum)
        
    if (epoch%5) == 0:
      LastEpochNum = EpochNum
      pickle.dump( LastEpochNum, open( DUCHDC_epochCount, "wb" ) )          ## CHANGE HERE WHEN YOU CHANGE NETWORK
      
    torch.save(net, DUCHDC_trainedModelPath)								## CHANGE HERE WHEN YOU CHANGE NETWORK
    print('Netwrok Saved at :', epoch, 'epoch')
      
    


val_loss = validate(val_loader, net, criterion, optimizer, epoch, args, restore_transform, visualize)
scheduler.step(val_loss)

test_loss = validate(test_loader, net, criterion, optimizer, epoch, args, restore_transform, visualize)


