# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from PIL import Image
import itertools

from model.vpgnet_model import *
from model.dataloader import *
from model.metrics import *

# Image and annotation paths
img_path = '.'
train_annot_path = './data/vpgnet-sample.txt'
val_annot_path = './data/vpgnet-sample.txt'
classlist_path = './data/vpgnet_classlist.txt'
#test_annot_path = 'data/vpgnet_annot_test.txt'

# Train parameters
start_epoch = 0
max_epoch = 1 #100
save_freq = 1
eval_freq = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate dataset and dataloader
tfm = [tf.Resize((480,640), Image.BICUBIC),
       tf.ToTensor(),
       tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

trainloader = DataLoader(VPG_dataloader(imPath=img_path,
                                        annPath=train_annot_path,
                                        classList=classlist_path,
                                        mode='train',
                                        _transforms=tfm), 
                         batch_size=1, 
                         shuffle=True, 
                         #num_workers=1, 
                         drop_last=True)

valloader = DataLoader(VPG_dataloader(imPath=img_path,
                                      annPath=val_annot_path,
                                      classList=classlist_path,
                                      mode='val',
                                      _transforms=tfm), 
                         batch_size=2, 
                         shuffle=False, 
                         #num_workers=1, 
                         drop_last=True)

# print(f'len val: {len(valloader.dataset)}')
# raise ValueError

# Instantiate model
net = VPGNet()
net.to(device)

# Instantiate optimizer and LR scheduler
optim_fn = {'multiLabel': torch.optim.Adam(net.multilabel.parameters(), 
                                           lr=0.005, 
                                           betas=(0.5,0.999))}#,
            # 'objectMask': torch.optim.Adam(net.objectmask.parameters(), 
            #                                lr=0.005, 
            #                                betas=(0.5,0.999)),
            # 'gridBox': torch.optim.Adam(net.gridbox.parameters(), 
            #                             lr=0.005, 
            #                             betas=(0.5,0.999)),
            # 'vpp': torch.optim.Adam(itertools.chain(*[net.shared.parameters(), 
            #                                           net.vpp.parameters()]), 
            #                         lr=0.005, 
            #                         betas=(0.5,0.999))}

# Initialize loss
loss_fn = {'multiLabel': nn.CrossEntropyLoss()}#,
            # 'objectMask': nn.CrossEntropyLoss(),
            # 'gridBox': nn.L1Loss(),
            # 'vpp': nn.CrossEntropyLoss()}

# Extract classes
classes = valloader.dataset.classlist

### Train loop ###
for task in ['multiLabel']:
  # Initialize score dict
  f1_dict = {i+1: 0 for i in classes}
  recall_dict = {i+1: 0 for i in classes}
  precision_dict = {i+1: 0 for i in classes}
  best_avg_f1 = 0
  
  # Train task
  for ep in range(start_epoch, max_epoch):
    # Train epoch
    for i, batch in enumerate(trainloader):
      # Train iteration
      img, gt = batch['image'], batch[task]
      img.to(device), gt.to(device)
      # print(f'im shape: {img.shape}\ngt.shape: {gt.shape}')
      
      # Forward pass
      pred = net(img)[task]
      # print(f'pred shape: {pred.shape}')
      # Compute loss
      loss = loss_fn[task](pred, gt[:,0,:,:])
      
      # Zero optimizer
      optim_fn[task].zero_grad()
     
      # Backprop
      loss.backward()
      
      # Optimize weights
      optim_fn[task].step()
      if i%1==0:
        print(f'Epoch:[{ep+1}/{max_epoch}] Iter:[{i+1}/{len(trainloader)}] {task} loss:[{loss:.4f}]')
    
    # Save weigths every x epochs
    if (ep+1)%save_freq:
      print(f'Saving weights for task: {task} epoch: {ep+1}')
      torch.save(net.state_dict(), f'checkpoints/Epoch_{task}_{ep+1}.pth')
    
    # Validate every x epoch
    if (ep+1)%eval_freq:
      # Initialize score dict
      ep_f1_dict = {i+1: 0 for i in classes}
      ep_recall_dict = {i+1: 0 for i in classes}
      ep_precision_dict = {i+1: 0 for i in classes}
      
      for i, batch in enumerate(valloader):
        img, gt = batch['image'], batch[task]
        img.to(device)
        
        # Forward pass
        pred = net(img)[task]
        
        # Compute scores
        itr_f1, itr_recall, itr_precision = compute_class_scores(pred, gt, classes)
        
        # Accumulate scores
        ep_f1_dict = {c: ep_f1_dict[c]+itr_f1[c] for c in ep_f1_dict}
        ep_recall_dict = {c: ep_recall_dict[c]+itr_recall[c] for c in ep_recall_dict}
        ep_precision_dict = {c: ep_precision_dict[c]+itr_precision[c] for c in ep_precision_dict}
      
      # Compute avg class scores across val data 
      ep_f1_dict = {c: ep_f1_dict[c]/len(valloader.dataset) for c in ep_f1_dict}
      ep_recall_dict = {c: ep_recall_dict[c]/len(valloader.dataset) for c in ep_recall_dict}
      ep_precision_dict = {c: ep_precision_dict[c]/len(valloader.dataset) for c in ep_precision_dict}
      
      # Compute avg f1
      avg_f1 = sum(list(ep_f1_dict.values()))/valloader.dataset.numclass
      
      # Save best weigths if val results are best
      if avg_f1 > best_avg_f1:
        best_avg_f1 = avg_f1
        torch.save(net.state_dict(), 'checkpoints/Best_epoch.pth')
        print(f'Saved weights for best eval f1 score: {ep+1}')
        
  print(f'Completed Training {task} branch')


  
  