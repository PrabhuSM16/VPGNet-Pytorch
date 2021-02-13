# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as sm


# inch, outch, kernel, stride, pad, dilation
class VPGNet_v2(nn.Module):
  def __init__(self, numclass, activ='relu', use_dropout=False, use_bias=False):
    super().__init__()
    assert numclass>0, f'Only accept numclass >0, but got {numclass}'
    ### shared layers -> encoder ###
    # s_conv1: in: 1x3x480x640, out: 1x128x96x128
    self.s_conv1 = nn.Sequential(*[nn.Conv2d(3,128,5,5,2,2,bias=use_bias),
                                 activation(activ),
                                 nn.LocalResponseNorm(5, alpha=0.0005, beta=0.75, k=2.),])
    # s_conv2: in: 1x128x96x128, out: 1x256x48x64
    self.s_conv2 = nn.Sequential(*[nn.MaxPool2d(2,2),
                                   nn.Conv2d(128,256,3,1,2,2,bias=use_bias),
                                   activation(activ),
                                   nn.LocalResponseNorm(5, alpha=0.0005, beta=0.75, k=2.),])
    # s_conv3: in: 1x256x48x64, out: 1x256x24x32
    self.s_conv3 = nn.Sequential(*[nn.MaxPool2d(2,2),
                                   nn.Conv2d(256,256,3,1,2,2,bias=use_bias),
                                   activation(activ),])
    # s_conv4: in: 1x256x24x32, out: 1x256x24x32
    # 5 x resblock layers
    self.s_conv4 = nn.Sequential(*[resLayer(256,256,numblocks=5,bias=use_bias)])
    # s_conv5: in: 1x256x24x32, out: 1x4096x24x32
    # Increase depth
    self.s_conv5 = nn.Sequential(*[nn.Conv2d(256,4096,3,1,1,bias=use_bias),
                                   activation(activ),])
    
    ### task layers -> decoder ###
    ## vanishing point task ##
    # vp_conv1: in: 1x4096x24x32, out: 1x256x48x64
    self.vp_conv1 = nn.Sequential(*[upConv(2,4096,256,3,1,1,bias=use_bias),
                                    activation(activ),])
    # ensure concat done to output!
    # vp_conv2: in: 1x512x48x64, out: 1x256x96x128
    self.vp_conv2 = nn.Sequential(*[upConv(2,512,128,3,1,1,bias=use_bias),
                                    activation(activ),])
    # ensure concat done to output!
    # vp_conv3: in: 1x256x96x128, out: 1x1x480x640
    self.vp_conv3 = nn.Sequential(*[upConv(5,256,5,3,1,1,bias=use_bias),
                                    activation(activ),])
    
    ## multilabel task ##
    # ml_conv1: in: 1x4096x24x32, out: 1x256x48x64
    self.ml_conv1 = nn.Sequential(*[upConv(2,4096,256,3,1,1,bias=use_bias),
                                    activation(activ),])
    # ensure concat done to output!
    # ml_conv2: in: 1x512x48x64, out: 1x256x96x128
    self.ml_conv2 = nn.Sequential(*[upConv(2,512,128,3,1,1,bias=use_bias),
                                    activation(activ),])
    # ensure concat done to output!
    # ml_conv3: in: 1x256x96x128, out: 1xnumclassx96x128
    self.ml_conv3 = nn.Sequential(*[nn.Conv2d(256,numclass,3,1,1,bias=use_bias),
                                    activation(activ),])
    
    ## object mask task ##
    # om_conv1: in: 1x4096x24x32, out: 1x256x48x64
    self.om_conv1 = nn.Sequential(*[upConv(2,4096,256,3,1,1,bias=use_bias),
                                    activation(activ),])
    # ensure concat done to output!
    # om_conv2: in: 1x512x48x64, out: 1x256x96x128
    self.om_conv2 = nn.Sequential(*[upConv(2,512,128,3,1,1,bias=use_bias),
                                    activation(activ),])
    # ensure concat done to output!
    # om_conv3: in: 1x256x96x128, out: 1x2x96x128
    self.om_conv3 = nn.Sequential(*[nn.Conv2d(256,2,3,1,1,bias=use_bias),
                                    activation(activ),])
      
  def forward(self, x):
    # shared layers #
    b = x.shape[0]
    x1 = self.s_conv1(x)
    # print(f'x1: max={x1.view(b,-1).max(dim=1)[0]}, min={x1.view(b,-1).min(dim=1)[0]}, median={x1.view(b,-1).median(dim=1)[0]}, mean={x1.view(b,-1).mean(dim=1)}')
    x2 = self.s_conv2(x1)
    # print(f'x2: max={x1.view(b,-1).max(dim=1)[0]}, min={x2.view(b,-1).min(dim=1)[0]}, median={x2.view(b,-1).median(dim=1)[0]}, mean={x2.view(b,-1).mean(dim=1)}')
    x3 = self.s_conv3(x2)
    # print(f'x3: max={x3.view(b,-1).max(dim=1)[0]}, min={x3.view(b,-1).min(dim=1)[0]}, median={x3.view(b,-1).median(dim=1)[0]}, mean={x3.view(b,-1).mean(dim=1)}')
    x4 = self.s_conv4(x3)
    # print(f'x4: max={x4.view(b,-1).max(dim=1)[0]}, min={x4.view(b,-1).min(dim=1)[0]}, median={x4.view(b,-1).median(dim=1)[0]}, mean={x4.view(b,-1).mean(dim=1)}')
    x_shared = self.s_conv5(x4)
    # print(f'x_shared: max={x_shared.view(b,-1).max(dim=1)[0]}, min={x_shared.view(b,-1).min(dim=1)[0]}, median={x_shared.view(b,-1).median(dim=1)[0]}, mean={x_shared.view(b,-1).mean(dim=1)}')
    
    # vp task layers #
    x_vp = self.vp_conv1(x_shared)
    # print(f'x_vp1: max={x_vp.view(b,-1).max(dim=1)[0]}, min={x_vp.view(b,-1).min(dim=1)[0]}, median={x_vp.view(b,-1).median(dim=1)[0]}, mean={x_vp.view(b,-1).mean(dim=1)}')
    x_vp = self.vp_conv2(torch.cat((x2,x_vp), dim=1))
    # print(f'x_vp2: max={x_vp.view(b,-1).max(dim=1)[0]}, min={x_vp.view(b,-1).min(dim=1)[0]}, median={x_vp.view(b,-1).median(dim=1)[0]}, mean={x_vp.view(b,-1).mean(dim=1)}')
    x_vp = self.vp_conv3(torch.cat((x1,x_vp), dim=1))
    # print(f'x_vp3: max={x_vp.view(b,-1).max(dim=1)[0]}, min={x_vp.view(b,-1).min(dim=1)[0]}, median={x_vp.view(b,-1).median(dim=1)[0]}, mean={x_vp.view(b,-1).mean(dim=1)}')
    
    # multilabel task layers #
    x_ml = self.ml_conv1(x_shared)
    x_ml = self.ml_conv2(torch.cat((x2,x_ml), dim=1))
    x_ml = self.ml_conv3(torch.cat((x1,x_ml), dim=1))
    
    # object mask task layers #
    x_om = self.om_conv1(x_shared)
    x_om = self.om_conv2(torch.cat((x2,x_om), dim=1))
    x_om = self.om_conv3(torch.cat((x1,x_om), dim=1))
    
    return {'shared': x_shared, 'vp': x_vp, 'multilabel': x_ml, 'objectmask': x_om}

class upConv(nn.Module):
  def __init__(self, scale, in_, out_, kernel, stride, padding, dilation=1, bias=False):
    super().__init__()
    assert scale>0, f'Ensure upshape size > 0, got {scale}'
    self.conv = nn.Conv2d(in_, out_, kernel, stride, padding, dilation, bias=bias)
    self.scale = scale
    
  def forward(self, x):
    x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
    return self.conv(x)
  
class resLayer(nn.Module):
  def __init__(self, in_, out_, numblocks, activ='relu', dilation=1, bias=False):
    super().__init__()
    blocks = []
    assert numblocks>0, f'Ensure number of basicResBlocks used is >0, got {numblocks}'
    for i in range(numblocks):
      blocks.append(basicResBlock(in_, out_, activ, dilation, bias))
    self.blocks = nn.Sequential(*blocks)
  
  def forward(self, x):
    return self.blocks(x)

class basicResBlock(nn.Module):
  def __init__(self, in_, out_, activ='relu', dilation=1, bias=False):
    super().__init__()
    resblock = [nn.Conv2d(in_, out_, 3, 1, dilation, dilation, bias=bias),
                nn.BatchNorm2d(out_),
                activation(activ),
                nn.Conv2d(out_, out_, 3, 1, dilation, dilation, bias=bias),
                nn.BatchNorm2d(out_),
                activation(activ),]
    self.resblock = nn.Sequential(*resblock)
    
  def forward(self, x):
   return x + self.resblock(x)
  
def activation(self, activ='relu', inplace=True):
  if activ == 'relu':
    return nn.ReLU(inplace)
  elif activ == 'prelu':
    return nn.PReLU(inplace)
  elif activ == 'sigmoid':
    return nn.Sigmoid(inplace)
  elif activ == 'tanh':
    return nn.Tanh(inplace)
  else:
    raise ValueError(f'{activ} activation is not implemented! Only accept "relu","prelu","sigmoid", and "tanh"')  

def weights_init_normal(m):
  if isinstance(m, nn.Conv2d):
    torch.nn.init.normal_(m.weight.data, 0.0, 0.1)  
    if m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 1.0)

def init_weights(net, task, epoch):
  tasks = ['vp', 'multilabel', 'objectmask']
  assert epoch>=0, f'Starting epoch must be > 0, got {epoch}.'
  assert task in tasks, f'Task must be "vp", "multilabel" or "objectmask", got {task}'
  
  if epoch==0:
    net.apply(weights_init_normal)
    print('Initialized weights for Epoch 0')
  elif epoch>0:
    net.load_state_dict(torch.load(f'checkpoints/Epoch_{task}_{epoch-1}.pth'))
    print(f'Loaded weights for task: "{task}" Epoch: {epoch-1}')

if __name__ == '__main__':
  x = torch.Tensor(1,3,480,640)
  model = VPGNet_v2(numclass=18)
  # print(model)
  out = model(x)
  print('shared:',out['shared'].shape)
  print('vp:',out['vp'].shape)
  print('multilabel:',out['multilabel'].shape)
  print('object mask:',out['objectmask'].shape)
  