import datetime
import os
import random

import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as standard_transforms

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as pyplot

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import cityscapes

from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d


args = {
    'train_batch_size': 1,
    'test_batch_size': 1,
    'epoch_num': 10,
    'lr': 1e-10,
    'weight_decay': 5e-4,
    'input_size': (256, 512),
    'momentum': 0.95,
    'snapshot': '',  # empty string denotes no snapshot
    'print_freq': 20,
    'val_batch_size': 2,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.05  # randomly sample some validation results to display
}

# Paths to trained models & epoch counts

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

## Loading the test dataset
test_set = cityscapes.CityScapes('fine', 'test', joint_transform=joint_transform,
                                  transform=input_transform, target_transform=target_transform)
test_loader = DataLoader(train_set, batch_size=args['test_batch_size'], shuffle=False)


# The NETWORK
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = torch.load(DUCHDC_trainedModelPath)							## CHANGE HERE WHEN YOU CHANGE NETWORK

# For Multi GPU
#if torch.cuda.device_count() > 1:
#  print("Let's use", torch.cuda.device_count(), "GPUs!")
#net = torch.nn.DataParallel(net, device_ids=[0, 1])
net = net.to(device)
print(net)

criterion = CrossEntropyLoss2d(size_average=False, ignore_index=cityscapes.ignore_label)

optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}], momentum=args['momentum'])


# Output Images
for vi, data in enumerate(test_loader):
        with torch.no_grad():
          inputs, gts = data

          N = inputs.size(0)
			
		  # Sending Variables to gpu
          inputs = Variable(inputs).to(device)
          gts = Variable(gts).to(device)
          #print(np.shape(inputs))
		  
		  
          outputs = net(inputs)
          prediction = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
		  
          gts = gts.cpu().numpy()
		  
		  
#          inputs = inputs.cpu()
          #plt.imshow(cityscapes.colorize_mask(gts[0,:,:]))
		  
		  
          gts = cityscapes.colorize_mask(gts[0,:,:])
		  
		  # Save Location Path
          root = os.path.join(os.getcwd(),'outputs')
		  
		  # Saving the ground Truth image
          gts.save(os.path.join(root,'gtruth'+str(vi)+'.tif'))
#          gts.save('\outputs\gtruth.jpg')

		  # Saving the predicted image
          prediction = cityscapes.colorize_mask(prediction[0,:,:])
          #plt.imshow(prediction)
		  #plt.show()
          prediction.save(os.path.join(root,'predicted'+str(vi)+'.tif'))
#          prediction.save('predicted.jpg')

		# Status of saving
          if (vi%100) == 0:
              print('%d / %d' % (vi + 1, len(test_loader)))
          #break
