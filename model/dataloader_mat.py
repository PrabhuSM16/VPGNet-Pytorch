# .mat file based dataloader
# @ZICHEN
#
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import loadmat

# Dataset Class for loading VPG .mat database
class VPG_dataset(Dataset):
    def __init__(self,
                 data_path,
                 class_list,
                 mode = 'train'):

        assert os.path.exists(data_path), f"Error: {data_path} does not exists!"

        self.mode = mode
        self.class_list, self.num_class = self.getClassdict(class_list)
        self.data_path = data_path
        self.data_list, self.num_imgs = self.getDataList(data_path)
        print(f"Mode: {self.mode}, Total data: {self.num_imgs}")

    def getClassdict(self, class_list):
        with open(class_list, 'r') as f:
            classes = f.readlines()
            classes = {i+1: classes[i].strip('\n') for i in range(len(classes))}
        return classes, len(classes)

    def getDataList(self, data_path):
        with open(data_path, 'r') as f:
            data_list = sorted(f.readlines())
        return data_list, len(data_list)

    def genImg(self, mat):
        img = mat[:, :, :3] # RGB image
        img = torch.from_numpy(img)
        return img

    def genVPMap(self, vp_mat):
        x, y = np.where(vp_mat == 1)
        map_stack = np.zeros((5, 480, 640))
        x = int(x)
        y = int(y)
        map_stack[0, x, y] = 1
        map_stack[1, :x+1, y:] = 1
        map_stack[2, :x+1, :y+1] = 1
        map_stack[3, x:, :y+1] = 1
        map_stack[4, x:, y:] = 1
        map_stack = torch.from_numpy(map_stack)
        return map_stack

    def genLabels(self, mat):
        # Initialize the grid box matrix
        grid_box = torch.LongTensor(4, 480, 640).fill_(0.)

        # Read mat file into matrix
        mat_data = loadmat(mat)
        multi_label[0, :, :] = torch.from_numpy(mat_data[:, :, 3]) # TBC, for type matching
        object_mask[0, :, :] = torch.from_numpy((mat_data[:, :, 3] != 0).astype(float))
        vp = genVPMap(mat_data[:, :, 4])

        multi_label = tf.Resize((60, 80), Image.NEAREST)(multi_label)
        object_mask = tf.Resize((120, 160), Image.NEAREST)(object_mask)
        vp = tf.Resize((120, 160), Image.NEAREST)(vp)
        return [multi_label, object_mask, grid_box, vp]


    def __getitem__(self, i):
        mat = self.data_list[i]
        img = self.genImg(mat)
        multi_label, object_mask, grid_box, vp = self.genLabels(mat)
        return {'image': img,
                'multi_label': multi_label,
                'object_mask': object_mask,
                'grid_box': grid_box,
                'vp': vp}


# Main function
if __name__=='__main__':

    # set path to all the .mat data
    # .mat database saves [r,g,b,class,vp] five channel info
    data_path = '../t_data/train_list.txt'
    class_list = '../data/classlist.txt'

    vpg_dataset = VPG_dataset(data_path,
                              class_list,
                              mode = 'test')
    
    dataloader = DataLoader(vpg_dataset, 
                            batch_size = 1, 
                            shuffle = True)
    
    for i, batch in enumerate(dataloader):
        print(f"image: {batch['image'].shape}, multiLabel: {batch['multiLabel'].shape}, objectMask: {batch['objectMask'].shape}")
        op = np.array(batch['multiLabel'][0][12:15].mul(255.).clamp(0,255)).transpose(1,2,0)
        
        
        






