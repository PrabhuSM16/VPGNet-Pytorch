from pathlib import Path
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import IPython.display
import PIL.Image

class_dict = ["background", "lane_solid_white", "lane_broken_white", 
              "lane_double_white", "lane_solid_yellow", "lane_broken_yellow", 
              "lane_double_yellow", "lane_broken_blue", "lane_slow", 
              "stop_line", "arrow_left", "arrow_right", "arrow_go_straight", 
              "arrow_u_turn", "speed_bump", "crossWalk", "safety_zone", 
              "other_road_markings"]


def getImageFromMat(mat):
    mat_file = loadmat(mat)
    rgb_seg_vp_label = mat_file['rgb_seg_vp']
    img = rgb_seg_vp_label[:, :, :3]
    return img

def genColor(classes):
    color_mat = [np.full((480, 640, 3),np.array([0, 0, 0]))]*classes
    color_map = [np.full((1,3), np.array([0,0,0]))]*classes
    for i in range(0, classes):
            temp_colour = np.random.choice(range(256), size=3)
            color_mat[i] = (np.full((480, 640, 3), temp_colour))
            color_map[i] = (np.full((1, 3), temp_colour))
    return color_mat, color_map

def genColorSample(color):
    example_mat = np.full((10, 10, 3), color)
    return example_mat.astype(np.uint8)
            
def genLabeledImage(row_image, pixel_labels, color_mat, classes = 18):
    # Pixel_labels have to be 480*640*1 rather than 480*640
    labels = pixel_labels # WARNING: only designed for 480*640 for optimization for running
    labels = np.concatenate((labels, labels, labels), axis = 2).astype(np.uint8)
    temp_im = row_image
    for i in range(1, classes): # from class 1 to max class #ZICHEN: to see the groundtruth, change to range(0, classes)
        temp_im = np.where(labels == np.array([i, i, i]), color_mat[i], temp_im)
    return temp_im.astype(np.uint8)
    
def getMaskFromMat(mat, classes = 18, color_mat = None):
    alpha = 0.6 # set the transparency for mask
    if color_mat == None:
        # generate color matrix for different classes (TBC: can use fixed one.)
        color_mat = [np.full((480, 640, 3),np.array([0, 0, 0]))]*classes
        for i in range(0, classes):
                temp_colour = np.random.choice(range(256), size=3)
                color_mat[i] = (np.full((480, 640, 3), temp_colour))
    mat_file = loadmat(mat)
    rgb_seg_vp_label = mat_file['rgb_seg_vp']
    img = rgb_seg_vp_label[:, :, :3]
    seg = rgb_seg_vp_label[:, :, 3].reshape((480,640,1))
    mixed_im = genLabeledImage(img, seg, color_mat, classes)
    mixed_im = cv2.addWeighted(mixed_im, alpha, img, 1 - alpha, 0)
    return mixed_im
    
def getHighestPath(search_path):
    all_mat = list(Path(search_path).rglob("*.mat"))
    h_value = [0]*18
    h_path = [""]*18
    for i in tqdm(all_mat):
        mat_file = loadmat(i)
        rgb_seg_vp_label = mat_file['rgb_seg_vp']
        seg_label = rgb_seg_vp_label[:, :, 3]
        for class_num in range(0, 18):
            count = np.count_nonzero(seg_label == class_num)
            if count >= h_value[class_num]:
                h_value[class_num] = count
                h_path[class_num] = i
    return h_path


if __name__ = "__main__":
    # Recommend: below code run in Jupyter Notebook
    classes = 18
    root_path = "./"
    h_path = getHighestPath(root_path)
    color_mat, color_map = genColor(classes)
    for i in range(1, classes):
        print(class_dict[i], h_path[i])
        color_ex = PIL.Image.fromarray(genColorSample(color_map[i]))
        disp_im = PIL.Image.fromarray(getMaskFromMat(h_path[i], classes, color_mat))
        IPython.display.display(color_ex)
        IPython.display.display(disp_im)