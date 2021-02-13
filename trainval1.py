# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.VPGNetV2 import *
from model.dataloader import *
from model.metrics import *

def init_optim_losses(task, **kwargs):
  # reset all param require_grad to True
  for param in net.parameters():
    param.requires_grad = True
  
  # based on task, select params for training and instantiate optim and loss
  if task=='vp': # only shared and vp layers trained
    for name, param in net.named_parameters():
      if 'ml' in name or 'om' in name:
        param.requires_grad = False
    optim_fn = torch.optim.Adam(net.parameters(), 
                                lr=lr_vp, 
                                betas=(0.5,0.999))
    loss_fn = nn.L1Loss() #nn.BCEWithLogitsLoss()
  
  elif task=='multilabel': # only multilabel layers trained
    for name, param in net.named_parameters():
      if not 'ml' in name:
        param.requires_grad = False
    optim_fn = torch.optim.Adam(net.parameters(), 
                                lr=lr_ml, 
                                betas=(0.5,0.999))
    loss_fn = nn.CrossEntropyLoss()
  
  elif task=='objectmask': # only object mask layers trained
    for name, param in net.named_parameters():
      if not 'om' in name:
        param.requires_grad = False
    optim_fn = torch.optim.Adam(net.parameters(), 
                                lr=lr_om, 
                                betas=(0.5,0.999))
    loss_fn = nn.CrossEntropyLoss()
  
  else:
    raise ValueError(f'Task {task} is not recognized, only accept "vp", "objectmask", or "multilabel"!')
  return optim_fn, loss_fn


def train_task_epoch(**kwargs):
  for i, batch in enumerate(trainloader):
    # Train iteration
    img, gt = batch['image'].to(device), batch[task].to(device)
    # print(f'im shape: {img.shape}\ngt.shape: {gt.shape}')
      
    # Forward pass
    pred = net(img)[task]
    ch = pred.shape[1]
    #if pred.view(batch_size,ch,-1).max(dim=2)[0].sum().item() < (batch_size*ch):
    #  print(f'\nraw values:\nmax: {pred.view(batch_size,ch,-1).max(dim=2)[0]}\nmin: {pred.view(batch_size,ch,-1).min(dim=2)[0]}]')
    print(f'\nraw values:\nmax: {pred.view(batch_size,ch,-1).max(dim=2)[0]}\nmin: {pred.view(batch_size,ch,-1).min(dim=2)[0]}]')
        
    # Compute loss
    loss = loss_fn(pred, gt[:,:,:,:])
      
    # Zero optimizer
    optim_fn.zero_grad()
     
    # Backprop
    loss.backward()
      
    # Optimize weights
    optim_fn.step()
    if i%10==0:
      print(f'Epoch:[{ep}/{max_epoch-1}] Iter:[{i}/{len(trainloader)-1}] {task} loss:[{loss:.4f}] ')
  
  # Save weigths every x epochs
  if (ep)%save_freq==0:
    print(f'\nSaving weights for task: {task} epoch: {ep}')
    torch.save(net.state_dict(), f'checkpoints/Epoch_{task}_{ep}.pth')
    

def val_task_epoch(**kwargs):
  if task=='vp':
    # Initialize accumulator for scores
    ep_f1, ep_recall, ep_precision = 0, 0, 0
      
  else:
    # Initialize score dict
    ep_f1_dict = {i+1: 0 for i in classes}
    ep_recall_dict = {i+1: 0 for i in classes}
    ep_precision_dict = {i+1: 0 for i in classes}
      
  # loop across valset
  for i1, batch in enumerate(tqdm(valloader)):
    img, gt = batch['image'].to(device), batch[task].to(device)
    img.to(device)
        
    # Forward pass
    pred = net(img)[task]
        
    # Compute VP point
    if task=='vp':
      pred_4sum = pred.sum(dim=1) # sum along 4 quadrants
      pred_max = pred_4sum.view(batch_size,-1).max(dim=1)[1]
      vp_x = pred_max//im_size[1]
      vp_y = pred_max%im_size[1]

      #if vp_x==0 or vp_y==0:    
      #  print(f'vp: ({vp_x}, {vp_y})')
            
      pred = torch.zeros(pred_4sum.shape)
      pred_viz = torch.zeros(pred_4sum.shape)
      for j in range(batch_size):
        pred[j, vp_x[j], vp_y[j]] = 1
        pred_viz[j, vp_x[j]-5:vp_x[j]+5, vp_y[j]-5:vp_y[j]+5] = 1
            
      ### paper vp computation ###
      # # compute p average
      # p_sum = pred[:,0,:,:].view(batch_size, -1).sum(dim=1).detach().cpu().numpy()
      # p_sz = np.prod(im_size)
      # p_avg_val = (1-(p_sum/p_sz))/4
      # p_avg = torch.FloatTensor(batch_size, 4, im_size[0], im_size[1])
      # for j in range(batch_size):
      #   p_avg[j,:,:,:].fill_(p_avg_val[j])  
          
      # # compute vp location
      # vp_loc = (torch.abs(p_avg - pred[:,1:,:,:])**2).sum(dim=1)
      # pred = torch.zeros(vp_loc.shape)
      # vp_loc = vp_loc.view(batch_size,-1).max(dim=1)[1]
      # vp_x, vp_y = vp_loc//im_size[1], vp_loc%im_size[1] # x and y coords of vp
      # for i in range(batch_size):
      #   pred[i, vp_x[i]-5:vp_x[i]+5, vp_y[i]-5:vp_y[i]+5] = 1
      #   pred[i, vp_x[i], vp_y[i]] = 1
          
      gt = 1 - gt[:,0,:,:] # 1-absence channel = vp channel
          
      # display samples
      if (i1%20)==0:
        f, axes = plt.subplots(1, 2)
        j1 = np.random.randint(batch_size)
        axes[0].imshow(img[j1,:,:,:].permute(1,2,0).cpu().numpy().astype(np.uint8))
        axes[0].set_title(f'img')
        axes[0].axis('off')          
        axes[1].imshow(pred_viz[j1,:,:].cpu().numpy())
        axes[1].set_title(f'vp')
        axes[1].axis('off')
        plt.savefig(f'samples/{task}_{ep}_val_sample.png')
        
      if pred.view(batch_size,-1).sum(dim=1).sum().item() < batch_size:
        print(f'num preds: {pred.view(batch_size,-1).sum(dim=1)}')
          
      # Compute scores
      itr_f1, itr_recall, itr_precision = compute_vp_scores(pred, gt)
      ep_f1 += itr_f1
      ep_recall += itr_recall
      ep_precision += itr_precision
      
    else:
      # Compute scores
      itr_f1, itr_recall, itr_precision = compute_class_scores(pred, gt, classes)
      
      # Accumulate scores
      ep_f1_dict = {c: ep_f1_dict[c]+itr_f1[c] for c in ep_f1_dict}
      ep_recall_dict = {c: ep_recall_dict[c]+itr_recall[c] for c in ep_recall_dict}
      ep_precision_dict = {c: ep_precision_dict[c]+itr_precision[c] for c in ep_precision_dict}
    
  # compute avg score and check if best score
  if task=='vp':
    ep_f1/=len(valloader.dataset)
    ep_recall/=len(valloader.dataset)
    ep_precision/=len(valloader.dataset)      
    # Compute avg f1
    avg_f1 = ep_f1
  
  else:
    # Compute avg class scores across val data 
    ep_f1_dict = {c: ep_f1_dict[c]/len(valloader.dataset) for c in ep_f1_dict}
    ep_recall_dict = {c: ep_recall_dict[c]/len(valloader.dataset) for c in ep_recall_dict}
    ep_precision_dict = {c: ep_precision_dict[c]/len(valloader.dataset) for c in ep_precision_dict}  
    # Compute avg f1
    avg_f1 = sum(list(ep_f1_dict.values()))/valloader.dataset.numclass    
  
  return avg_f1

if __name__ == '__main__':
  # Image and annotation paths
  train_path = './data/mat'
  val_path = './data/mat'
  classlist_path = './data/vpgnet_classlist.txt'
  #test_path = './data/mat'
  
  # Train parameters
  start_epoch = 0 # start epoch index
  max_epoch = 5 # max num of epochs for training in each task
  save_freq = 1 # freq of saving weights (epoch)
  eval_freq = 1 # freq of evaluation (epoch)
  batch_size = 2
  lr_vp = 0.0001
  lr_om = 0.0005
  lr_ml = 0.0005
  im_size = (480, 640) # row x col
  tasks = ['vp', 'objectmask', 'multilabel']
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Instantiate dataset and dataloader
  tfm = [tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
  
  trainloader = DataLoader(VPG_dataloader(dataPath=train_path,
                                          classList=classlist_path,
                                          mode='train',
                                          _transforms=tfm), 
                           batch_size=batch_size, 
                           shuffle=True, 
                           #num_workers=8, 
                           drop_last=True)
  
  valloader = DataLoader(VPG_dataloader(dataPath=val_path,
                                        classList=classlist_path,
                                        mode='val',
                                        _transforms=tfm), 
                           batch_size=batch_size, 
                           shuffle=False, 
                           #num_workers=8, 
                           drop_last=True)
  
  # Extract classes
  classes, numclass = trainloader.dataset.classlist, trainloader.dataset.numclass
  
  # Instantiate model
  net = VPGNet_v2(numclass=numclass).to(device)
  
  # init model weights
  init_weights(net, tasks[0], start_epoch)

  ### Train loop ###
  for task in tasks:
    print(f'Training "{task}" task')
    
    # Initialize score dict for objectmask and multilabel tasks only
    if task!='vp':
      f1_dict = {i+1: 0 for i in classes}
      recall_dict = {i+1: 0 for i in classes}
      precision_dict = {i+1: 0 for i in classes}
    
    # best f1 score -> trigger save best weights
    best_avg_f1 = 0
    
    
    # Initialize optim fn and loss fn
    optim_fn, loss_fn = init_optim_losses(task)
      
    # print(f'loss fn: {loss_fn}')
    # Train task
    for ep in range(start_epoch, max_epoch):
      # Train epoch
      train_task_epoch()
  
      # Validate every x epoch
      if (ep)%eval_freq==0:
        # print(f'Validating epoch {ep}')
        avg_f1 = val_task_epoch()
        
        # Save best weigths if val results are best
        print(f'Avg validation f1 score for epoch {ep}: {avg_f1}')
        if avg_f1 > best_avg_f1:
          best_avg_f1 = avg_f1
          torch.save(net.state_dict(), f'checkpoints/best_{task}_epoch.pth')
          
    print(f'Completed Training {task} branch')







