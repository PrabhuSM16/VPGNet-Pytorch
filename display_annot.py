# -*- coding: utf-8 -*-
# Reads file name and annotations from listfile and displays
from scipy.io import loadmat
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ListFile',type=str,help='List file',default='vpgnet_db.txt')
parser.add_argument('--numClass',type=int,help='Number of classes',default=17)
args = parser.parse_args()

# random display colour for each class
colour = []
for i in range(1,args.numClass+1):
    colour.append((np.random.randint(0,255),
                   np.random.randint(0,255),
                   np.random.randint(0,255)))

# read file
with open(args.ListFile,'r') as f:
    labels = f.readlines()
    
# load image and seg file
for i, line in enumerate(labels):
    label = line.split('  ')
    arr = loadmat('.'+label[0])['rgb_seg_vp']
    im = arr[:,:,:3]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) #convert to BGR for opencv
    seg = arr[:,:,3]
    seg = (seg/seg.max())*255
    seg = np.repeat(np.expand_dims(seg, axis=2), 3, axis=2).astype(np.uint8)
    
    # generate 8x8 grid boxes and display
    for i, coords in enumerate(label[2:]):
        x1,y1,x2,y2,l = coords.split(' ')
        cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), colour[int(l)-1], -1)
    cv2.imshow(label[0].replace('.mat','.jpg'),np.hstack([im,seg]))
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key==ord('q'):
        break


