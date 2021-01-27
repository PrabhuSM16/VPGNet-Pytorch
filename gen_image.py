# Ver 2.0.0
# @ZICHEN
#
# Generate .png format original image and  annotated images for all .mat file under current path iteratively (include subfolders).
#
# Usage:
#    python3 ./gen_image.py [m|i]
#    Will search and index all .mat files in current path.
#    And according to your parameter [m|i], this script will:
#        1. i: generate original *.png images into same direction with *.mat file
#        2. m: generate masked image into same direction with *.mat file
#
# Example:
#    python3 ./gen_image.py im
#    Above command will generate both original images and annotated images.

# Import Librarys
import sys
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import cv2

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
# Usage & Example:
#     At the begining of this file
#
if __name__=="__main__":

    p_im = False
    p_mask = False

    param = sys.argv[1]
    if (('i' not in param) and ('m' not in param)):
        raise ValueError("wrong parameters! See gen_label_v3.py file for usage.")
    if 'i' in param:
        p_im = True
        print("> Will generate original image *.png")
    if 'm' in param:
        p_mask = True
        print("> will generate masked image *_mask.png")

    classess = 17 # Zichen: totally 17 classes in VPGNet DB

    # Traverse all .mat files in current directory
    print("> Indexing all .mat files iteratively...")
    all_mat = list(Path(".").rglob("*.mat"))
    print("> Found {} .mat files in dir {} ...".format(len(all_mat), Path(".").absolute()))

    # generate color matrix for different classes (TBC: can use fixed one.)
    color_mat = [np.full((480, 640, 3),np.array([0, 0, 0]))]
    for i in range(0, classess):
            temp_colour = np.random.choice(range(256), size=3)
            color_mat.append(np.full((480, 640, 3), temp_colour))

    # generate png original file and masked image for iteratively
    for i in tqdm(all_mat):
        mat_file = loadmat(i) # load mat file
        rgb_seg_vp_label = mat_file['rgb_seg_vp']
        rgb_image = rgb_seg_vp_label[:, :, :3]
        seg_label = rgb_seg_vp_label[:, :, 3].reshape(480,640,1)

        # generate original image
        if p_im:
            png_output_path = i.with_suffix('.png') # saving path for original image
            result_imwrite = cv2.imwrite(str(png_output_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)) # Generate original .png groundtruth

        # generate masked image
        if p_mask:
            masked_im_path = str(i).replace('.mat','_mask.png') # saving path for masked image
            alpha = 0.6 # set the transparency for mask
            mixed_im = generate_labeled_image(rgb_image, seg_label, color_mat, classess)
            mixed_im = cv2.addWeighted(mixed_im, alpha, rgb_image, 1 - alpha, 0)
            result_imwrite = cv2.imwrite(masked_im_path, cv2.cvtColor(mixed_im, cv2.COLOR_RGB2BGR))
