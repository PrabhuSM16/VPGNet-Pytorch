# -*- coding: utf-8 -*-
# generate jpeg images and annotations in list.txt from .mat files
from PIL import Image
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--DataPath',type=str,help='Full dataset path',default='F:/VPGNet-DB-5ch/tmp/*/*.mat')
parser.add_argument('--DstPath',type=str,help='Destination path',default='F:/VPGNet-DB-5ch/tmp')
args = parser.parse_args()

files = glob(args.DataPath)
print('Num files: {}'.format(len(files)))

os.makedirs(args.DstPath+'/JPEGImages', exist_ok=True)

with open(args.DstPath+'/vpgnet-db-5ch-list.txt','w') as f: 
    for i, file in enumerate(tqdm(files)):
#        fn = '{}/JPEGImages/{}'.format(args.DstPath, os.path.split(file)[-1].replace('.mat','.jpg'))
        fn = '{}/JPEGImages/{:05d}.jpg'.format(args.DstPath, i+1)
        arr = loadmat(file)['rgb_seg_vp']
#        print('fn:',fn)
#        break
        # extract RGB images and binary seg map
        img = arr[:,:,:3]
        seg = arr[:,:,3]
        num_class = seg.max() # max value = number of classes
        
        h,w = seg.shape
        num_h, num_w = h//8, w//8 # split image into 8x8 grids
        #print('height {}, width {}'.format(h, w))
        
        txt = ''
        num_objs = 0   
        # loop for all classes -> generate 8x8 box annotss
        for cl in range(1,num_class+1):
            seg2 = np.where(seg==cl, 1, 0) # if exist, set value=1 else value=0
            for y in range(num_h): # vertical parsing
                for x in range(num_w): #horizontal parsing
                    # record grid as long as there is a nonzero value present
                    if seg2[y*8:(y*8)+8, x*8:(x*8)+8].sum()>1:
                        txt+='{} {} {} {} {}  '.format(x*8,y*8,(x*8)+8,(y*8)+8,cl)
                        num_objs+=1
        if (i+1)<len(files):
            f.write('{}  {}  {}\n'.format(fn, num_objs, txt[:-2]))
        else:
            f.write('{}  {}  {}'.format(fn, num_objs, txt[:-2]))
        rgb_img = Image.fromarray(img)
        # img_1 = rgb_img.convert('RGB')
        rgb_img.save(fn)
        # plt.imsave(fn, img)

