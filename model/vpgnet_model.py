# -*- coding: utf-8 -*-
# VPGNet model updated 13/12/20
import torch
import torch.nn as nn

class VPGNet(nn.Module):
  def __init__(self):
    super(VPGNet, self).__init__()
    ### Shared Layers ###
    # Layer 1 (input) -> input shape: Bx3x480x640
    conv1 = [nn.Conv2d(3, 96, 11, 4, 0),
             nn.ReLU(),
             nn.LocalResponseNorm(5, alpha=0.0005, beta=0.75, k=2. ),#(size,alpha,beta,k)
             nn.MaxPool2d(3, 2),
             nn.Upsample((59, 79)),] # Upsize tensor from 58x78 to 59x79
    conv1 = nn.Sequential(*conv1)
    # Layer 2 -> input shape: Bx96x59x79
    conv2 = [nn.Conv2d(96, 256, 5, 1, 2),
             nn.ReLU(),
             nn.LocalResponseNorm(5, alpha=0.0005, beta=0.75, k=8. ),#(size,alpha,beta,k)
             nn.MaxPool2d(3, 2),]
    conv2 = nn.Sequential(*conv2)
    # Layer 3 -> input shape: Bx256x29x39
    conv3 = [nn.Conv2d(256, 384, 3, 1, 1),
             nn.ReLU(),]
    conv3 = nn.Sequential(*conv3)
    # Layer 4 -> input shape: Bx384x29x39
    conv4 = [nn.Conv2d(384, 384, 3, 1, 1),
             nn.ReLU(),]
    conv4 = nn.Sequential(*conv4)
    # Layer 5 -> input shape: Bx384x29x39
    conv5 = [nn.Conv2d(384, 384, 3, 1, 1),
             nn.ReLU(),
             nn.MaxPool2d(3, 2),]
    conv5 = nn.Sequential(*conv5)
    # Layer 6 -> input shape: Bx384x14x19
    conv6 = [nn.Conv2d(384, 4096, 6, 1, 3),
             nn.ReLU(),
             nn.Dropout2d(0.5),]
    conv6 = nn.Sequential(*conv6)
    self.shared = nn.Sequential(*[conv1,
                                  conv2,
                                  conv3,
                                  conv4,
                                  conv5,
                                  conv6])
    ### Branched Layers ###
    ## Grid Box ##
    # Layer 7a -> input shape: Bx4096x15x20
    conv7a = [nn.Conv2d(4096, 4096, 1, 1, 0),
              nn.ReLU(),
              nn.Dropout2d(0.5),]
    conv7a = nn.Sequential(*conv7a)
    # Layer 8a -> input shape: Bx4096x15x20 -> output: Bx256x15x20
    conv8a = [nn.Conv2d(4096, 256, 1, 1, 0),]
    conv8a = nn.Sequential(*conv8a)
    tile_a = tiling([1,256,15,20], 8)
    self.gridbox = nn.Sequential(*[conv7a,
                                   conv8a,
                                   tile_a])
    
    ## Object Mask ##
    # Layer 7b -> input shape: Bx4096x15x20
    conv7b = [nn.Conv2d(4096, 4096, 1, 1, 0),
              nn.ReLU(),
              nn.Dropout2d(0.5),]
    conv7b = nn.Sequential(*conv7b)
    # Layer 8b -> input shape: Bx4096x15x20 -> output: Bx256x15x20
    conv8b = [nn.Conv2d(4096, 128, 1, 1, 0),]
    conv8b = nn.Sequential(*conv8b)
    tile_b = tiling([1,128,15,20], 8)
    self.objectmask = nn.Sequential(*[conv7b,
                                      conv8b,
                                      tile_b])
    
    ## Multi-label ##
    # Layer 7c -> input shape: Bx4096x15x20
    conv7c = [nn.Conv2d(4096, 4096, 1, 1, 0),
              nn.ReLU(),
              nn.Dropout2d(0.5),]
    conv7c = nn.Sequential(*conv7c)# Layer 8c -> input shape: Bx4096x15x20 -> output: Bx256x15x20
    conv8c = [nn.Conv2d(4096, 1024, 1, 1, 0),]
    conv8c = nn.Sequential(*conv8c)
    tile_c = tiling([1,1024,15,20], 4)
    self.multilabel = nn.Sequential(*[conv7c,
                                      conv8c,
                                      tile_c])
    
    ## VPP ##
    # Layer 7d -> input shape: Bx4096x15x20
    conv7d = [nn.Conv2d(4096, 4096, 1, 1, 0),
              nn.ReLU(),
              nn.Dropout2d(0.5),]
    conv7d = nn.Sequential(*conv7d)
    # Layer 8d -> input shape: Bx4096x15x20 -> output: Bx256x15x20
    conv8d = [nn.Conv2d(4096, 320, 1, 1, 0),]
    conv8d = nn.Sequential(*conv8d)
    tile_d = tiling([1,320,15,20], 8)
    self.vpp = nn.Sequential(*[conv7d,
                               conv8d,
                               tile_d])
    
  def forward(self, x):
    x_shared = self.shared(x)
    
    # Grid box
    x_gridbox = self.gridbox(x_shared)
    
    # Object mask
    x_objectmask = self.objectmask(x_shared)

    # Multi-label
    x_multilabel = self.multilabel(x_shared)

    # VPP
    x_vpp = self.vpp(x_shared)
    return {'shared': x_shared, 
            'gridBox': x_gridbox, 
            'objectMask': x_objectmask, 
            'multiLabel': x_multilabel,
            'vpp': x_vpp}

class tiling(nn.Module):
# Tiling layer from orginal caffe-VPGNet 
  def __init__(self, x_dim, tile_dim):
    super(tiling, self).__init__()
    self.b,d,self.h,self.w = x_dim
    self.tile_dim = tile_dim
    self.out_d = int(d/(tile_dim**2)) # 4
    out_h = int(self.h*tile_dim) # 15
    out_w = int(self.w*tile_dim) # 20
    self.tiled_out = torch.FloatTensor(self.b, self.out_d, out_h, out_w).fill_(0.)  

  def forward(self, x):
    for ds in range(self.out_d):
      d_start = ds*(self.tile_dim**2)
      d_end = (ds+1)*(self.tile_dim**2)
      for hs in range(self.h):
        for ws in range(self.w):
          tile_select = x[:, d_start:d_end, hs, ws]
#          tile_select = x[:, ds*(self.tile_dim**2):(ds+1)*(self.tile_dim**2), hs, ws] 
          out_tile = tile_select.view(self.b, self.tile_dim, self.tile_dim)
          h_start = hs*self.tile_dim
          h_end = (1+hs)*self.tile_dim
          w_start = ws*self.tile_dim
          w_end = (1+ws)*self.tile_dim
          self.tiled_out[:, ds, h_start:h_end, w_start:w_end] = out_tile
#          self.tiled_out[:, ds, hs*self.tile_dim:(1+hs)*self.tile_dim, ws*self.tile_dim:(1+ws)*self.tile_dim] = out_tile
    return self.tiled_out

if __name__ == '__main__':
  model = VPGNet()
  print(model)

  y = model(torch.FloatTensor(1,3,480,640))

  print('shared layers output:',y['shared'].shape)
  print('grid box output:',y['gridBox'].shape)
  print('object mask output:',y['objectMask'].shape)
  print('multi label output:',y['multiLabel'].shape)
  print('vpp output:',y['vpp'].shape)
