# Ver 4.0.0
# @ZICHEN
#
# Generate annotated file list and masked images for all .mat file under current path iteratively (include subfolders).
#
# Usage:
#    python3 ./gen_label_v4.py [output]
#    Will search and generate file list for all .mat files in current path.
#    And 1. generate original *.png images into same direction with *.mat file
#        2. generate VPGNet format file list into './output_train', './output_val', './output_test'
#    NOTICE: if no [output] provided, will use './output' by default.

# Import Librarys
import sys
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import cv2
import random

# generate string for single image
def generate_string(label_list, file_name):
    temp_string = "{}  {}".format(file_name, len(label_list))
    for i in label_list:
        temp_string = temp_string + "  {} {} {} {} {}".format(i[1], i[0], i[1]+8, i[0]+8, i[2])
    return temp_string

# generate label_tuples for single image
# Input: rgb_label of size [height, width, 5], in which rgb_label[:,:,3] saves multi-class pixelwise class info.
# Output: label_list, a list of all grids like [[grid1], [grid2], ...] in which [grid1] is [x1, y1, class]
def generate_label_list(rgb_label):
    count = 0
    label_list = []
    for i in range(0, 480, 8):
        for j in range(0, 640, 8):
            current_cube = rgb_label[i:i+8,j:j+8,3:4]
            if not np.all(current_cube == 0):
                unique, counts = np.unique(current_cube, return_counts=True)
                if 0 in unique:
                    fill_value = unique[np.argmax(counts[1:])+1]
                else:
                    fill_value = unique[np.argmax(counts)]
                created_cube = [i,j,fill_value]
                label_list.append(created_cube)
                count += 1
            else:
                continue
    return label_list

# Generate 480*640*3 image with 
def generate_labeled_image(row_image, pixel_labels, color_mat, classes = 17):
    # Pixel_labels have to be 480*640*1 rather than 480*640*1
    labels = pixel_labels # WARNING: only designed for 480*640 for optimization for running
    labels = np.concatenate((labels, labels, labels), axis = 2).astype(np.uint8)
    temp_im = row_image
    for i in range(1, classes + 1): # from class 1 to max class
        temp_im = np.where(labels == np.array([i, i, i]), color_mat[i], temp_im)
    return temp_im.astype(np.uint8)
    
    

# Load all .mat file in current path iteratively
# Usage:
#     At begin of this file.
#
if __name__=="__main__":
    classess = 17 # Zichen: totally 17 classes in VPGNet DB
    if len(sys.argv) > 1:
        print("> Use {} as output file name.".format(sys.argv[1]))
        output_file = sys.argv[1]
    else:
        print("> No output file name provided. Use ./output as output")
        output_file = "output"
    train_file = output_file + "_train"
    test_file = output_file + "_test"
    val_file = output_file + "_val"
    print("> Indexing all .mat files iteratively...")
    all_mat = list(Path(".").rglob("*.mat"))
    print("> Found {} .mat files in dir {}, estimate {} sec...".format(len(all_mat), Path(".").absolute(), len(all_mat)/12))
    print("> Generate grid label file into {} ... \n(NOTICE: Please remove previous version of {} in case of overwritting.)".format(output_file,output_file))
    print("> Split test, validation and train as 1:1:5...")
    test_len = len(all_mat)//7
    val_len = test_len
    train_len = len(all_mat) - test_len - val_len
    test_list = random.sample(all_mat, test_len)
    remain_list = [x for x in all_mat if x not in test_list]
    val_list = random.sample(remain_list, val_len)
    train_list = [x for x in remain_list if x not in val_list]
    # Generate Train set
    print("> Generate train set into {}".format(train_file))
    for i in tqdm(train_list):
        png_output_path = i.with_suffix('.png')
        mat_file = loadmat(i)
        rgb_seg_vp_label = mat_file['rgb_seg_vp']
        rgb_image = rgb_seg_vp_label[:, :, :3]
        seg_label = rgb_seg_vp_label[:, :, 3].reshape(480,640,1)
        label_list = generate_label_list(rgb_seg_vp_label)
        image_string_temp = generate_string(label_list, "/{}".format(png_output_path))
        result_imwrite = cv2.imwrite(str(png_output_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)) # Generate original .png groundtruth
        with open(train_file, 'a') as f:
            f.write("{}\n".format(image_string_temp)) # write into file_list
    # Generate Test set
    print("> Generate train set into {}".format(train_file))
    for i in tqdm(test_list):
        png_output_path = i.with_suffix('.png')
        mat_file = loadmat(i)
        rgb_seg_vp_label = mat_file['rgb_seg_vp']
        rgb_image = rgb_seg_vp_label[:, :, :3]
        seg_label = rgb_seg_vp_label[:, :, 3].reshape(480,640,1)
        label_list = generate_label_list(rgb_seg_vp_label)
        image_string_temp = generate_string(label_list, "/{}".format(png_output_path))
        result_imwrite = cv2.imwrite(str(png_output_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)) # Generate original .png groundtruth
        with open(test_file, 'a') as f:
            f.write("{}\n".format(image_string_temp)) # write into file_list
    # Generate Val set
    print("> Generate train set into {}".format(train_file))
    for i in tqdm(val_list):
        png_output_path = i.with_suffix('.png')
        mat_file = loadmat(i)
        rgb_seg_vp_label = mat_file['rgb_seg_vp']
        rgb_image = rgb_seg_vp_label[:, :, :3]
        seg_label = rgb_seg_vp_label[:, :, 3].reshape(480,640,1)
        label_list = generate_label_list(rgb_seg_vp_label)
        image_string_temp = generate_string(label_list, "/{}".format(png_output_path))
        result_imwrite = cv2.imwrite(str(png_output_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)) # Generate original .png groundtruth
        with open(val_file, 'a') as f:
            f.write("{}\n".format(image_string_temp)) # write into file_list
