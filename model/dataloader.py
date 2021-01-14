# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

imPath = '../data/images'
trAnnPath = 'vpgnet_annot.txt'
vlAnnPath = 'vpgnet_annot.txt'
tsAnnPath = 'vpgnet_annot.txt'

class VPG_dataloader(Dataset):
  def __init__(self, 
               imPath, 
               trAnnPath, 
               vlAnnPath, 
               tsAnnPath,
               numClass=17,
               _transforms=None,            
               mode='train'):
    for i in [imPath, trAnnPath, vlAnnPath, tsAnnPath]:
      assert os.path.exists(i), f'Error: {i} does not exists!'
    self.mode = mode
    if _transforms is not None:
      self.transforms = tf.Compose(_transforms)
    else:
      self.transforms = tf.Compose([tf.ToTensor])
#    self.numClass = numClass
    self.imgpath = imPath
    self.trainlist = None
    self.vallist = None
    self.testlist = None
    if self.mode=='train':
      self.trainlist = self.extractAnnots(trAnnPath)
      self.num_imgs = len(self.trainlist)
    elif self.mode=='val':  
      self.vallist = self.extractAnnots(vlAnnPath)
      self.num_imgs = len(self.vallist)
    elif self.mode=='test':
      self.testlist = self.extractAnnots(tsAnnPath)
      self.num_imgs = len(self.testlist)
    else:
      raise ValueError(f'"{mode}" mode is not recognized!')
    print(f'Num "{self.mode}" imgs: {self.num_imgs}')
  
  def extractAnnots(self, annotfile):
    with open(annotfile, 'r') as f:
      annotlist = sorted(f.readlines())
      random.shuffle(annotlist)
    return annotlist
  
  def genLabels(self, annots):
    multiLabel = torch.FloatTensor(64,480,640).fill_(0.)
    objectMask = torch.FloatTensor(2,480,640).fill_(0.)
#    gridBox = torch.FloatTensor(self.numClass,480,640)
#    vpp = torch.FloatTensor(self.numClass,480,640)
    for coords in annots:
      x1,y1,x2,y2,lb = coords.split(' ')
      multiLabel[int(lb)-1,int(y1):int(y2),int(x1):int(x2)] = 1.
      objectMask[1,int(y1):int(y2),int(x1):int(x2)] = 1
    objectMask[0,:,:] = 1 - objectMask[1,:,:]
    multiLabel = tf.Resize((60,80), Image.NEAREST)(multiLabel)
    objectMask = tf.Resize((120,160), Image.NEAREST)(objectMask)
    return [multiLabel, objectMask]
  
#  def denorm(self)
  def __getitem__(self, i):
    if self.mode=='train':
      annots = self.trainlist[i].strip('\n').split('  ')
    elif self.mode=='val':
      annots = self.vallist[i].strip('\n').split('  ')
    elif self.mode=='test':
      annots = self.testlist[i].strip('\n').split('  ')
    img = self.transforms(Image.open(os.path.join(self.imgpath,annots[0])))
    self.num_labels = annots[1]
    multiLabel, objectMask = self.genLabels(annots[2:])
    return {'image': img, 'multiLabel': multiLabel, 'objectMask': objectMask}
  
  def __len__(self):
    return self.num_imgs

if __name__=='__main__':
  tfm = [tf.Resize((480,640), Image.BICUBIC),
         tf.ToTensor(),
         tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

  dataset = VPG_dataloader(imPath,
                           trAnnPath,
                           vlAnnPath,
                           tsAnnPath,
                           mode='test',
                           _transforms=tfm)
  
  dataloader = DataLoader(dataset, 
                          batch_size=1, 
                          shuffle=True, 
                          num_workers=1, 
                          drop_last=True)
  
  for i, batch in enumerate(dataloader):
    print(f"image: {batch['image'].shape}, multiLabel: {batch['multiLabel'].shape}, objectMask: {batch['objectMask'].shape}")
    op = np.array(batch['multiLabel'][0][12:15].mul(255.).clamp(0,255)).transpose(1,2,0)
#    op = np.array(batch['objectMask'][0,0,:,:].mul(255.).clamp(0,255))
#    plt.imshow(op)
#    print(f'max:{op.max():.4f}, min:{op.min():.4f}, avg:{op.mean():.4f}')
    plt.imshow(cv2.resize(op, (640,480), cv2.INTER_CUBIC).astype(np.uint8))
    plt.show()
    






