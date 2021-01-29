from pathlib import Path
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import IPython.display
import PIL.Image

class_dict = ["background", "lane_solid_white", "lane_broken_white", "lane_double_white", "lane_solid_yellow", "lane_broken_yellow", "lane_double_yellow", "lane_broken_blue", "lane_slow", "stop_line", "arrow_left", "arrow_right", "arrow_go_straight", "arrow_u_turn", "speed_bump", "crossWalk", "safety_zone", "other_road_markings"]

def getImageFromMat(mat):
    mat_file = loadmat(mat)
    rgb_seg_vp_label = mat_file['rgb_seg_vp']
    img = rgb_seg_vp_label[:, :, :3]
    return img

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
    h_path = getHighestPath("./")
    for i in range(0, 18):
        print(class_dict[i], h_path[i])
        ori_im = PIL.Image.fromarray(getImageFromMat(h_path[i]))
        IPython.display.display(ori_im)