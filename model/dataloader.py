# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import numpy as np

class VPG_dataloader(Dataset):
  def __init__(self, 
               imPath, 
               annPath, 
               classList,
               _transforms=None,            
               mode='train'):
    
    assert os.path.exists(imPath), f'Error: {imPath} does not exists!'
    assert os.path.exists(annPath), f'Error: {annPath} does not exists!'
    
    self.mode = mode
    if _transforms is not None:
      self.transforms = tf.Compose(_transforms)
    else:
      self.transforms = tf.Compose([tf.ToTensor])
    
    self.classlist, self.numclass = self.get_classdict(classList)
    
    self.imgpath = imPath
    self.annotlist, self.num_imgs = self.extractAnnots(annPath)
    print(f'Num "{self.mode}" imgs: {self.num_imgs}')
  
  def extractAnnots(self, annotfile):
    with open(annotfile, 'r') as f:
      annotlist = sorted(f.readlines())
    return annotlist, len(annotlist)
  
  def get_classdict(self, classList):
    with open(classList, 'r') as f:
      classes = f.readlines()
      classes = {i+1: classes[i].strip('\n') for i in range(len(classes))}
    return classes, len(classes)
    
  def genLabels(self, annots):
    ### NOTE: gridBox and vpp NOT UP YET ###
    multiLabel = torch.LongTensor(1,480,640).fill_(0.) #64x480x640
    objectMask = torch.LongTensor(1,480,640).fill_(0.) #2x480x640
    gridBox = torch.LongTensor(4,480,640).fill_(0.) #4x480x640
    vpp = torch.LongTensor(5,480,640).fill_(0.) #5x480x640

    for coords in annots:
      x1,y1,x2,y2,lb = coords.split(' ')
      multiLabel[0,int(y1):int(y2),int(x1):int(x2)] = int(lb)
      objectMask[0,int(y1):int(y2),int(x1):int(x2)] = 1
    
    # objectMask[0,:,:] = 1 - objectMask[1,:,:]
    multiLabel = tf.Resize((60,80), Image.NEAREST)(multiLabel)
    objectMask = tf.Resize((120,160), Image.NEAREST)(objectMask)
    return [multiLabel, objectMask, gridBox, vpp]
  
  def __getitem__(self, i):
    annots = self.annotlist[i].strip('\n').split('  ')
    img = self.transforms(Image.open(os.path.join(self.imgpath,annots[0])))
    # index 0: image name, index 1: num labels, index 2 onwards: bbox annot
    self.num_labels = annots[1]
    multiLabel, objectMask, gridBox, vpp = self.genLabels(annots[2:])
    return {'image': img, 
            'multiLabel': multiLabel, 
            'objectMask': objectMask,
            'gridBox': gridBox,
            'vpp': vpp}
  
  def __len__(self):
    return self.num_imgs

if __name__=='__main__':
  import cv2

  imPath = '../data/images'
  annPath = '../data/vpgnet_annot.txt'
  classList = '../data/classlist_no-yellow-box.txt'
 
  tfm = [tf.Resize((480,640), Image.BICUBIC),
         tf.ToTensor(),
         tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

  dataset = VPG_dataloader(imPath,
                           annPath,
                           classList,
                           mode='test',
                           _transforms=tfm)
  
  dataloader = DataLoader(dataset, 
                          batch_size=1, 
                          shuffle=True, 
                          num_workers=1, 
                          drop_last=True)

  print(dataloader.dataset.get_classdict)
  
  for i, batch in enumerate(dataloader):
    print(f"image: {batch['image'].shape}, multiLabel: {batch['multiLabel'].shape}, objectMask: {batch['objectMask'].shape}")
    op = np.array(batch['multiLabel'][0][12:15].mul(255.).clamp(0,255)).transpose(1,2,0)
#    op = np.array(batch['objectMask'][0,0,:,:].mul(255.).clamp(0,255))
#    plt.imshow(op)
#    print(f'max:{op.max():.4f}, min:{op.min():.4f}, avg:{op.mean():.4f}')
    plt.imshow(cv2.resize(op, (640,480), cv2.INTER_CUBIC).astype(np.uint8))
    plt.show()
    
    
    






