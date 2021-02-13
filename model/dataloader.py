# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as tf
from scipy.io import loadmat
from PIL import Image
import glob
import os
import random
import numpy as np

class VPG_dataloader(Dataset):
  def __init__(self,
               dataPath, 
               classList,
               mode='train',
               _transforms=None):
    
    assert os.path.exists(dataPath), f'Error: dataPath "{dataPath}" does not exists!'
    
    if _transforms is not None:
      self.transforms = tf.Compose(_transforms)
    else:
      self.transforms = tf.Compose([tf.ToTensor])
    
    self.classlist, self.numclass = self.get_classdict(classList)
    
    self.data = sorted(glob.glob(f'{dataPath}/*/*.mat'))
    numdata = len(self.data)
    if mode == 'train':
      self.data = self.data[:int(.8*numdata)]
    elif mode == 'val':
      self.data = self.data[int(.8*numdata):]
    elif mode == 'test':
      self.data = self.data
    else:
      raise ValueError(f'Mode "{mode}" is not recognized. Only accept "train", "val", and "test".')
    self.lenData = len(self.data)
    
  def get_classdict(self, classList):
    with open(classList, 'r') as f:
      classes = f.readlines()
      classes = {i+1: classes[i].strip('\n') for i in range(len(classes))}
    return classes, len(classes)
    
  def loadData(self, matfile):
    data = loadmat(matfile)['rgb_seg_vp']
    im = self.tsrResize(torch.FloatTensor(data[:,:,:3].transpose(2,0,1)).unsqueeze(0), size=(480,640), dtype='float')[0] # ChxHxW
    fg = self.tsrResize(torch.FloatTensor(np.where(data[:,:,3]>0,1,0)).unsqueeze(0).unsqueeze(0))[0]
    objectmask = torch.cat((fg,1-fg), dim=0) # 2xHxW
    multilabel = self.tsrResize(torch.FloatTensor(data[:,:,3]).unsqueeze(0).unsqueeze(0))[0] # HxW
    vp = self.vp_label(data[:,:,4]) # HxW
    return im, multilabel, objectmask, vp
    
  def vp_label(self, vp_np):
    vp_abs = torch.FloatTensor(1-vp_np).unsqueeze(0) # only 1 vp element = 0, everything else = 1
    vp = torch.zeros(4,480,640).type(torch.FloatTensor)
    vp_y, vp_x = np.where(vp_np==1) # row, col
    try:
      vp_x, vp_y = vp_x[0], vp_y[0]
      vp[0, :vp_y+1, :vp_x+1] = 1 # Top left
      vp[1, :vp_y+1, vp_x:] = 1 # Top right
      vp[2, vp_y:, :vp_x+1] = 1 # Bottom left
      vp[3, vp_y:, vp_x:] = 1 # Bottom right
    except:
      vp[:,:,:] = 1 # set all to 1 if no VP 
    return torch.cat((vp_abs, vp), dim=0)
  
  def tsrResize(self, tsr, size=(96,128), dtype='long'):
    tsr_resz = F.interpolate(tsr, size, mode='nearest')
    if dtype=='float':
      out = tsr_resz.type(torch.FloatTensor)
    elif dtype=='long':
      out = tsr_resz.type(torch.LongTensor)
    else:
      raise ValueError(f'Only accept dtype "long" or "float", but got "{dtype}"')
    return out
  
  
  # def genLabels(self, annots):
  #   ### NOTE: gridBox and vpp NOT UP YET ###
  #   multiLabel = torch.LongTensor(1,480,640).fill_(0.) #64x480x640
  #   objectMask = torch.LongTensor(1,480,640).fill_(0.) #2x480x640
  #   gridBox = torch.LongTensor(4,480,640).fill_(0.) #4x480x640
  #   vpp = torch.LongTensor(5,480,640).fill_(0.) #5x480x640

  #   for coords in annots:
  #     x1,y1,x2,y2,lb = coords.split(' ')
  #     multiLabel[0,int(y1):int(y2),int(x1):int(x2)] = int(lb)
  #     objectMask[0,int(y1):int(y2),int(x1):int(x2)] = 1
    
  #   # objectMask[0,:,:] = 1 - objectMask[1,:,:]
  #   multiLabel = tf.Resize((96,128), Image.NEAREST)(multiLabel)
  #   objectMask = tf.Resize((96,128), Image.NEAREST)(objectMask)
  #   return [multiLabel, objectMask, gridBox, vpp]
  
  def __getitem__(self, i):
    # print(self.data[i].split('/')[-1])
    im, multilabel, objectmask, vp = self.loadData(self.data[i])
    im = self.transforms(im)
    return {'image': im, 
            'multilabel': multilabel, 
            'objectmask': objectmask,
            'vp': vp}
  
  def __len__(self):
    return self.lenData

if __name__=='__main__':
  import matplotlib.pyplot as plt

  dataPath = '../data/mat'
  classList = '../data/vpgnet_classlist.txt'
 
  tfm = [tf.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

  dataset = VPG_dataloader(dataPath,
                           classList,
                           _transforms=tfm)
  
  dataloader = DataLoader(dataset, 
                          batch_size=1, 
                          shuffle=True, 
                          num_workers=1, 
                          drop_last=True)

  #print(dataloader.dataset.classlist)
  
  for i, batch in enumerate(dataloader):
    print(f"image:\t{batch['image'].shape}\nmultilabel:\t{batch['multilabel'].shape}\nobjectmask:\t{batch['objectmask'].shape}\nvp:\t{batch['vp'].shape}")
    # display vp labels
    f, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3)
    
    # expand vp map only for better visualization
    m_y, m_x = np.where(batch['vp'][0,0,:,:].numpy() == 0)
    m_x, m_y = m_x.item(), m_y.item()
    print(f'vp loc: (x={m_x}, y={m_y})')
    vp_map = 1 - batch['vp'][0,0,:,:].numpy()
    vp_map[m_y-5:m_y+5, m_x-5:m_x+5] = 1
    
    # display vp
    ax1.imshow(batch['vp'][0,1,:,:])
    ax1.set_title('top left')
    ax1.axis('off')
    ax2.imshow(batch['vp'][0,2,:,:])
    ax2.set_title('top right')
    ax2.axis('off')
    ax3.imshow(1-vp_map)
    ax3.set_title('absence')
    ax3.axis('off')
    ax4.imshow(batch['vp'][0,3,:,:])
    ax4.set_title('bottom left')
    ax4.axis('off')
    ax5.imshow(batch['vp'][0,4,:,:])
    ax5.set_title('bottom right')
    ax5.axis('off')
    ax6.imshow(vp_map)
    ax6.set_title('vp')
    ax6.axis('off')
    
    # plt.imshow(batch['objectmask'][0,0,:,:].numpy())
    # plt.imshow(batch['objectmask'][0,1,:,:].numpy())
    # plt.imshow(batch['multilabel'][0,0,:,:].numpy())
    break
    
    # op = np.array(batch['multiLabel'][0][12:15].mul(255.).clamp(0,255)).transpose(1,2,0)
#    op = np.array(batch['objectMask'][0,0,:,:].mul(255.).clamp(0,255))
#    plt.imshow(op)
#    print(f'max:{op.max():.4f}, min:{op.min():.4f}, avg:{op.mean():.4f}')
    # plt.imshow(cv2.resize(op, (640,480), cv2.INTER_CUBIC).astype(np.uint8))
    # plt.show()
    
    
    
    
  
  
  



