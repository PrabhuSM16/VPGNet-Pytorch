import numpy as np
from sklearn.metrics import f1_score, recall_score

# Lane Detection:
# Calculate the F1 score between two 480*640*1 label map
# 1. extend the centered points to circles and get the circle labeled map
# 2. compare predicted pixels with circle labeled map
# 3. TP: Predict pos, label pos
# 4. FP: Predict pos, label neg
# 5. FN: Predict neg, label pos
# (TP: predict in circles; FN: remain pixels not in circles; FP: predict no in circle)

def create_circular_mask(h, w, center=None, radius=None):
    # Create a circular mask from center on an (h,w) map with euclidean distance radius
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def vpg_lane_f1(map1, map2, mask_R = 4):
    # For single class metric, calculate two map1,2 whose label are binary (0 for no, and !0 for label)
    # Input:
    #     map1: (h,w,1) size label predicted map
    #     map2: (h,w,1) size label groundtruth map, every 8*8 pixels an grid with same label
    #     mask_R: euclidean distance of radius R, default 4
    # Return:
    #     single_class_f1: f1 score for map1 and map2 described in VPGNet Sec5.3

    # First extend the grid-labeled map2 to circle-labeled map extend_mask with boundary R
    
    map1_mask = map1 > 0
    map2_mask = map2 > 0 # Assume map1,2 only have one class
    extend_mask = numpy.ones((480, 640), dtype=bool) # extended groundtruth (from 8*8 square grid to radius R circle)
    for i in range(0, 480, 8):
        for j in range(0, 640, 8):
            if map2_mask(i,j) == True: # if this pixel have label, this 8*8 grid should have same label
                area_mask = create_circular_mask(480, 640, center = (i,j), radius = mask_R)
                extend_mask = extend_mask + area_mask # add the area_mask to blank mask
                
    # Compare map1 and the extended mask for f1 score
    single_class_f1 = f1_score(extend_mask.flatten(), map1_mask.flatten())
    return single_class_f1


# Road Markings:
# Calculate the Recall Score between two 480*640*1 label map
# 1. use precise groundtruth labeled map
# 2. compare predicted pixels with labeled map
# 3. TP: Predict pos, label pos
# 4. FN: Predict neg, label pos
# (TP: predict in circles; FN: remain pixels not in label map pos area)
def vpg_rm_recall(map1, map2):
    # For single class metric, calculate two map1,2 whose label are binary (0 for no, and !0 for label)
    # Input:
    #     map1: (h,w,1) size label predicted map
    #     map2: (h,w,1) size label groundtruth map, every 8*8 pixels an grid with same label
    # Return:
    #     single_class_recall: recall score for map1 and map2
    map1_mask = map1 > 0
    map2_mask = map2 > 0 # Assume map1,2 only have one class
    single_class_recall = recall_score(map2_mask.flatten(), map1_mask.flatten())
    return single_class_recall


# Vanishing Point:
# Calculate the Recall Score between two 480*640*1 label map
def vpg_VP_recall(map1, map2, mask_R = 4):
    # For single class metric, calculate two map1,2 whose label are binary (0 for no, and !0 for label)
    # Input:
    #     map1: (h,w,1) size label predicted map
    #     map2: (h,w,1) size label groundtruth map, every 8*8 pixels an grid with same label, with only 1 pixel as vanishing point
    #     mask_R: euclidean distance of radius R, default 4
    # Return:
    #     single_class_recall: recall score for map1 and map2
    map1_mask = map1 > 0
    map2_mask = map2 > 0 # Assume map1,2 only have one class, and map2 only have 1 positive value
    extend_mask = numpy.ones((480, 640), dtype=bool) # extended groundtruth (from 8*8 square grid to radius R circle)
    vp_point_pos = np. where(map2_mask)
    VP_circle_mask = create_circular_mask(480, 640, center = vp_point_pos, radius = mask_R)
    VP_recall = recall_score(VP_circle_mask.flatten(), map1_mask.flatten())
    return VP_recall