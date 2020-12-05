# CopyRight @TIAN ZICHEN
# Generate annotated file list and masked images for all .mat file under current path iteratively (include subfolders).
# Usage:
#    python3 ./gen_label.py [output_file.txt]
#    Will search and index all .mat files in current path.
#    And 1. generate VPGNet format file list into './output_file.txt'
#        2. generate masked images into same direction with .mat file
#    NOTICE: if no [output_file.txt] provided, will use './file_list.txt' by default.
import sys
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import cv2

# generate string for single image
def generate_string(label_list, file_name):
    temp_string = "{}  {}".format(file_name, len(label_list))
    for i in label_list:
        temp_string = temp_string + "  {} {} {} {} {}".format(i[1], i[0], i[1]+8, i[0]+8, i[2])
    return temp_string

# generate label_tuples for single image
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
#    python3 ./gen_label.py [output_file.txt]
#    Will search and index all .mat files in current path.
#    And 1. generate VPGNet format file list into './output_file.txt'
#        2. generate masked images into same direction with .mat file
#    NOTICE: if no [output_file.txt] provided, will use './file_list.txt' by default.
if __name__=="__main__":
    classess = 17
    if len(sys.argv) > 1:
        print("> Use {} as output file name.".format(sys.argv[1]))
        output_file = sys.argv[1]
    else:
        print("> No output file name provided. Use ./file_list.txt as output")
        output_file = "file_list.txt"
    print("> Indexing all .mat files iteratively...")
    all_mat = list(Path(".").rglob("*.mat"))
    print("> Found {} .mat files in dir {}, estimate {} sec...".format(len(all_mat), Path(".").absolute(), len(all_mat)/12))
    print("> Generate grid label file into {} ... \n(NOTICE: Please remove previous version of {} in case of overwritting.)".format(output_file,output_file))
    # generate color matrix (TBC: can use fixed one.)
    color_mat = [np.full((480, 640, 3),np.array([0, 0, 0]))]
    for i in range(0, classess):
            temp_colour = np.random.choice(range(256), size=3)
            color_mat.append(np.full((480, 640, 3), temp_colour))
    for i in tqdm(all_mat):
        mat_file = loadmat(i)
        rgb_seg_vp_label = mat_file['rgb_seg_vp']
        rgb_image = rgb_seg_vp_label[:, :, :3]
        seg_label = rgb_seg_vp_label[:, :, 3].reshape(480,640,1)
        label_list = generate_label_list(rgb_seg_vp_label)
        image_string_temp = generate_string(label_list, "/{}".format(i))
        with open(output_file, 'a') as f:
            f.write("{}\n".format(image_string_temp))
        # Generate labeled image under same folder as .mat file
        alpha = 0.6
        image_save_path = str(i).replace('mat', 'png')
        mixed_im = generate_labeled_image(rgb_image, seg_label, color_mat, classess)
        mixed_im = cv2.addWeighted(mixed_im, alpha, rgb_image, 1 - alpha, 0)
        result_imwrite = cv2.imwrite(image_save_path, cv2.cvtColor(mixed_im, cv2.COLOR_RGB2BGR))