## Most functions here are referenced from the repository: https://github.com/ArayCHN/VPGNet_for_lane
## Functions are mostly unedited, can be optimised in the future

import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import cv2
import math
import scipy
from scipy import cluster
import os
import time # time the execution time
#import IPM 	# imports IPM.py, which in turn imports IPM.cpp. Haven't been able to do this yet.

DOWNSCALE = 1.0 # 0.3 default
UPSCALE = 1.0 / DOWNSCALE
IMAGE_SIZE_RESCALE = 3
DEST = "./output_log" # where the output pix are stored, can be changed by arg

## Functions needed
## Functions below are referenced from 'lane_extension_polyline_for_VPG.py' file in the repository.

def preprocess(file, line_type, suppress_output): # 0.006s
    '''pre-process the image of connected lines, final result masked_img'''

    # NOTICE! picture has to be float when passed into IPM cpp, return value is also float!
    # the image for adjust() is always 640 * 480
    # file is a gray image!

    tmp = file
    # tmp = tmp.astype(dtype = np.float32, copy = False)
    resize_x, resize_y = tmp.shape[1] / 640.0, tmp.shape[0] / 480.0

    ret, tmp = cv2.threshold(tmp, 200, 255, cv2.THRESH_BINARY)
    if not suppress_output:
        cv2.imwrite('%s/%s'%(DEST, 'threshold_original.png'), tmp)

    thresh_img = cv2.resize(tmp, (int(DOWNSCALE * 640), int(DOWNSCALE * 480))) # image downsample, but will be converted back to gray image again!!!
    ret, thresh_img = cv2.threshold(thresh_img, 100, 255, cv2.THRESH_BINARY)
    thresh_img = thresh_img.astype(np.uint8)
    
    #time3 = time.time()
    #print time1 - time0, time2 - time1, time3 - time2
    # mask the original graph
    APPLY_MASK = 0 # 1: apply, 0: not apply
    x_size = thresh_img.shape[1]
    y_size = thresh_img.shape[0]
    
    # below: a mask of shape Chinese Character "ao" (1st)
    # pt1 = (0, 0) # specify 8 vertices of the (U-shaped, concave) mask
    # pt2 = (int(0.01 * x_size), 0)
    # pt3 = (pt2[0], int(0.1 * y_size)) # default 0.57
    # pt4 = (int(0.99 * x_size), pt3[1])
    # pt5 = (pt4[0], 0)
    # pt6 = (x_size - 1, 0)
    # pt7 = (x_size - 1, int(y_size * 1))
    # pt8 = (0, int(y_size * 1))

    # below: a mask of shape Chinese Character "tu" (1st)
    if APPLY_MASK:
        pt1 = (0, int(0.2 * y_size)) # specify 8 vertices of the (U-shaped, concave) mask
        pt2 = (int(0.45 * x_size), int(0.2 * y_size))
        pt3 = (pt2[0], int(0.01 * y_size)) # default 0.57
        pt4 = (int(0.55 * x_size), pt3[1])
        pt5 = (pt4[0], int(0.2 * y_size))
        pt6 = (x_size - 1, int(0.2 * y_size))
        pt7 = (x_size - 1, int(y_size * 1))
        pt8 = (0, y_size - 1)

        mask = np.zeros_like(thresh_img)
        vertices = np.array([[pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]])
        mask = cv2.fillPoly(mask, vertices, 255)
        masked_img = cv2.bitwise_and(thresh_img, mask) # apply mask!
    else:
        masked_img = thresh_img # don't wanna use mask!
    if not suppress_output:
        cv2.imwrite('%s/%s'%(DEST, 'thresh_img.png'), thresh_img)
        cv2.imwrite('%s/%s'%(DEST, 'masked_img.png'), masked_img)

    return masked_img, resize_x, resize_y


def houghlines(masked_img_connected, suppress_output):
    """Performs houghlines algorithm"""
    scale = 0.5 # 0.175 # scale the pic to perform houghlinesP, for speed!
    rho = 2
    theta = np.pi / 180 # / 2 # resolution: 0.5 degree
    threshold = int(150 * scale * DOWNSCALE) # the number of votes (voted by random points on the picture)
    min_line_length = int(80 * scale * DOWNSCALE) # line length
    max_line_gap = 20 * scale # the gap between points on the line, no limit here
    masked_img_connected = cv2.resize(masked_img_connected, (0, 0), fx = scale, fy = scale)
    lines = cv2.HoughLinesP(masked_img_connected, rho, theta, threshold, np.array([]), min_line_length, max_line_gap) # find the lines

    # adjust all lines' end points to middle!
    if lines is None:
        return None
    for i in range(lines.shape[0]):
        for y1, x1, y2, x2 in lines[i]:
            #x1, y1 = find_mid(x1, y1, x2, y2, masked_img_connected)
            #x2, y2 = find_mid(x2, y2, x1, y1, masked_img_connected)
            lines[i] = [[int(y1 / scale), int(x1 / scale), int(y2 / scale), int(x2 / scale)]]

    # plot the original hughlinesP result!
    if not suppress_output:
        hough_img = masked_img_connected.copy()
        cv2.imwrite('%s/%s'%(DEST, 'houghlines_raw.png'), hough_img)
        hough_img = cv2.imread('%s/%s'%(DEST, 'houghlines_raw.png')) # convert gray scale to BGR
        if lines is None: return []
        for i in range(lines.shape[0]):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(hough_img, (int(x1*scale), int(y1*scale)), (int(x2*scale), int(y2*scale)), (0, 255, 0), 1) # paint lines in green

        # hough_img = cv2.resize(hough_img, (0, 0), fx = UPSCALE, fy = UPSCALE)
        cv2.imwrite('%s/%s'%(DEST, 'houghlines_raw.png'), hough_img)
    return lines # lines in 640 * 480 * DOWNSCALE pic

def check(z, i, n, clustered, checked, clusters, cluster_id):
    """
    Figures out which member belongs to which cluster after the scipy lib carries out the clustering algorithm
    and stores the process of the clustering in z
    See detailed usage in the example below in cluster_lines(), where this function is called
    """

    if z[i, 0] < n and z[i, 1] < n:
        checked[i] = 1
        clustered[int(z[i, 0])] = 1
        clustered[int(z[i, 1])] = 1
        clusters[cluster_id].append(int(z[i, 0]))
        clusters[cluster_id].append(int(z[i, 1]))
    elif z[i, 0] >= n and z[i, 1] < n:
        checked[i] = 1
        clustered[int(z[i, 1])] = 1
        clusters[cluster_id].append(int(z[i, 1]))
        check(z, int(z[i, 0]) - n, n, clustered, checked, clusters, cluster_id)
    elif z[i, 0] < n and z[i, 1] >= n:
        checked[i] = 1
        clustered[int(z[i, 0])] = 1
        clusters[cluster_id].append(int(z[i, 0]))
        check(z, int(z[i, 1]) - n, n, clustered, checked, clusters, cluster_id)
    else: # z[i, 0] >= n and z[i, 1] >= n
        checked[i] = 1
        check(z, int(z[i, 0]) - n, n, clustered, checked, clusters, cluster_id)
        check(z, int(z[i, 1]) - n, n, clustered, checked, clusters, cluster_id)
    return

def cluster_lines(masked_img_connected, lines, suppress_output):
    """
    Clusters the results from houghlines(), lines too close will be clustered into one line.
    "centroid", distances from cluster is determined by the distance of centroids.
    """
    # filter the results, lines too close will be taken as one line!
    # 1. convert the lines to angle-intercept space - NOTE: intercept is on x-axis on the bottom of the image!
    cluster_threshold = int(30 * DOWNSCALE)
    y0 = int(0.5 * masked_img_connected.shape[0]) # the intercept here needs to be carefully chosen!
    n = lines.shape[0]

    if n == 0: # n = 0 or 1, can't do cluster!
        return []
    if n == 1:
        # print lines[0]
        [[x1, y1, x2, y2]] = lines[0]
        eps = 0.1
        theta = ( math.atan2(abs(y2 - y1), (x2 - x1) * abs(y2 - y1) / (y2 - y1 + eps)) / np.pi * 180.0)
        intercept = ((x1 - x2) * (y0 - y1) / (y1 - y2 + eps)) + x1
        k = math.tan(theta / 180.0 * np.pi)
        b = - k * intercept + y0
        return [(k, b)]

    y = np.zeros((n, 2), dtype = float) # stores all lines' data
    for i in range(lines.shape[0]):
        for x1,y1,x2,y2 in lines[i]:
            eps = 0.1 # prevent division by 0
            theta = ( math.atan2(abs(y2 - y1), (x2 - x1) * abs(y2 - y1) / (y2 - y1 + eps)) / np.pi * 180.0)
            intercept = ((x1 - x2) * (y0 - y1) / (y1 - y2 + eps)) + x1
            y[i, :] = [theta * DOWNSCALE, intercept] # intercept: x value at y = y0

    # 2. perform clustering
    z = cluster.hierarchy.centroid(y) # finish clustering
    ending = 0
    while ending < z.shape[0] and z[ending, 2] < cluster_threshold: # cluster distance < cluster_threshold, continue clustering!
        ending += 1
    ending -= 1 # the last cluster where distance < 10

    # below: figure out which point belongs to which cluster
    clustered = np.zeros((n), dtype = int) # each line, whether clustered
    cluster_id = -1
    clusters = []
    checked = np.zeros((n), dtype = int) # each element in z, whether checked
    for i in range(ending, -1, -1):
        if not checked[i]:
            clusters.append([])
            cluster_id += 1
            check(z, i, n, clustered, checked, clusters, cluster_id) # recursively obtain all members in a cluster

    for i in range(n):
        if not clustered[i]:
            clusters.append([i]) # points not clusterd will be a single cluster

    if not suppress_output:
        print "cluster representatives: format (intercept, theta)"

    ave_lines = []
    for each_cluster in clusters:
        if not suppress_output:
            print each_cluster
        sum_intercept = 0
        sum_theta = 0
        tot = len(each_cluster)
        for each_line in each_cluster:
            sum_theta += y[each_line, 0] / DOWNSCALE
            sum_intercept += y[each_line, 1]
        ave_intercept = sum_intercept / tot
        ave_theta = sum_theta / tot
        if ave_theta == 0:
            ave_theta = 0.001 # to prevent runtime error
        if not suppress_output:
            print ave_intercept, ave_theta
        y1 = y0
        x1 = ave_intercept
        y2 = masked_img_connected.shape[0] / 2
        x2 = x1 + int((y2 - y1) / math.tan(ave_theta / 180.0 * np.pi))
        y3 = masked_img_connected.shape[0]
        x3 = x1 + int((y3 - y1) / math.tan(ave_theta / 180.0 * np.pi))

        # cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1) # paint lines in green
        # cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 255), 1)
        k = math.tan(ave_theta / 180.0 * np.pi)
        b = - k * ave_intercept + y0
        ave_lines.append((k, b))

    if not suppress_output:
        print "cluster representatives all printed!"

    return ave_lines

## Commented functions below reference IPM, which I have issues importing

# def clean_up(orig_img, lines, suppress_output): # orig_image: 640 * 480
#     # plot the result on original picture
#     threshold_img = cv2.imread('%s/%s'%(DEST, "thresh_img.png"))

#     # final_lines = []
#     draw_lines_x = []
#     draw_lines_y = []

#     # make polyline sparser
#     npx, npy, lines = make_sparse(lines)

#     # print polyline
#     for line in lines:
#         print 'The next line is:', line

#     for line in lines: # lines format: lines = [line1, line2, line3, ...], linei = np.array([[x, y], [x, y], ...])
#         for i in range(len(line) - 1):
#             if not suppress_output:
#                 cv2.line(threshold_img, (int(line[i][0]), int(line[i][1])), (int(line[i + 1][0]), int(line[i + 1][1])), (0, 0, 255), 1)
#             x1, y1 = line[i]
#             x2, y2 = line[i + 1]
#             # k = (y2 - y1)/(x2 - x1 + 0.0001)
#             # b = y1 - x1*(y2 - y1)/(x2-x1+0.0001) # y = kx + b
#             # final_lines.append((k, b)) # collect all lines for evaluation
#             draw_lines_x.append(x1 * UPSCALE / IMAGE_SIZE_RESCALE) # collect all lines for IPM
#             draw_lines_y.append(y1 * UPSCALE / IMAGE_SIZE_RESCALE)
#             if i == len(line) - 2:
#                 draw_lines_x.append(x2 * UPSCALE / IMAGE_SIZE_RESCALE)
#                 draw_lines_y.append(y2 * UPSCALE / IMAGE_SIZE_RESCALE)

#     # lines[] here is within image of 640 * 480
#     npx = np.array(draw_lines_x, dtype = np.float32) # in order to pass into c++
#     npy = np.array(draw_lines_y, dtype = np.float32)

#     #step = IPM.points_ipm2image(npx, npy) # convert npx, npy to picture coordinates

#     tot = 0
#     for line in lines: # lines format: lines = [line1, line2, line3, ...], linei = [(x, y), (x, y), ...]
#         for i in range(len(line) - 1):
#             cv2.line(orig_img, (int(npx[tot]*IMAGE_SIZE_RESCALE), int(npy[tot]*IMAGE_SIZE_RESCALE)), \
#                 (int(npx[tot+1]*IMAGE_SIZE_RESCALE), int(npy[tot+1]*IMAGE_SIZE_RESCALE)), (0, 0, 255), 1)
#             tot += 1
#         tot += 1

#     cv2.imwrite('%s/%s'%(DEST, 'threshold.png'), threshold_img)
#     cv2.imwrite('%s/%s'%(DEST, "labeled.png"), orig_img)

#     return lines, npx, npy

# def scale_back(lines, npx, npy, resize_x, resize_y):
#     tot = 0
#     lines_in_img = []
#     for i in range(len(lines)): # lines format: lines = [line1, line2, line3, ...], linei = [(x, y), (x, y), ...]
#         lines_in_img.append([])
#         for j in range(len(lines[i])):
#             if (npx[tot] >= 0) and (npx[tot] < 640) and (npy[tot] >= 0) and (npy[tot] < 480):
#                 lines_in_img[i].append((int(npx[tot] * resize_x), int(npy[tot] * resize_y)))
#             tot += 1
#     return lines_in_img

# def convert_img2gnd(npx, npy, lines):
#     IPM.points_image2ground(npx, npy)
#     tot = 0
#     lines_in_gnd = []
#     for i in range(len(lines)): # lines format: lines = [line1, line2, line3, ...], linei = [(x, y), (x, y), ...]
#         lines_in_gnd.append([])
#         for j in range(len(lines[i])):
#             if (npy[tot] >= 0):
#                 lines_in_gnd[i].append((npx[tot], npy[tot]))
#             tot += 1
#     return lines_in_gnd

# def make_sparse(lines):
#     """
#     Makes polyline sparse by a scale
#     """
#     sparse_scale = 8
#     sparse_lines = []
#     npx = np.array([])
#     npy = np.array([])
#     for line in lines:
#         tmp = []
#         for i in range(len(line)):
#             if line[i, 1] != 0 or line[i, 0] != 0:
#                 last = i
#             if line[i, 1] == 0 and line[i, 0] == 0: # get rid of the additional 0s in line[]
#                 break
#             if i % sparse_scale == 0:
#                 tmp.append(line[i])
#                 npx = np.append(npx, line[i, 0])
#                 npy = np.append(npy, line[i, 1])
#         if tmp == []:
#             continue
#         if len(tmp) == 1: # only one point in line, invalid! add the last point also
#             tmp.append(line[last])
#             npx = np.append(npx, line[last, 0])
#             npy = np.append(npy, line[last, 1])
#         sparse_lines.append(tmp)
#     npx = npx.astype(dtype=np.float32, copy=False)
#     npy = npy.astype(dtype=np.float32, copy=False)
#     return npx, npy, sparse_lines


def work(file, do_adjust, suppress_output, time1, time2): # the public API
    # threshold + resize
    masked_img_connected, resize_x, resize_y = preprocess(file, 'connected', suppress_output) # img: ipm'ed image
    # masked_img_connected: (640 * 480)*DOWNSCALE
    if not suppress_output:
        orig_img = cv2.resize(file, (640, 480))
        cv2.imwrite('%s/%s'%(DEST, "o.png"), orig_img)
        orig_img = cv2.imread('%s/%s'%(DEST, "o.png"))

    # initial line extraction: with opencv HoughLines algorithm
    time25 = time.time()
    lines = houghlines(masked_img_connected, suppress_output) # (640 * 480)*DOWNSCALE
    time3 = time.time()

    if lines is not None:

        ave_lines = cluster_lines(masked_img_connected, lines, suppress_output) # cluster results from houghlines
        # further cluster and filter out the lines that are noises! only retain the largest cluster whose inclinations are close enough
        # ave_lines = cluster_directions(ave_lines, suppress_output)
        time4 = time.time()
        
        lines_in_gnd = []
        lines = []

        for (k, b) in ave_lines:
            if not suppress_output:
                # print k, b
                pass

            if do_adjust:
                # do adjustment (refinement), further adjust all lines to the middle
                # line = adjust(k, b, masked_img_connected.shape[0], masked_img_connected.shape[1], masked_img_connected, DOWNSCALE) # python adjust
                line = np.zeros((200, 2), dtype = np.int32) # C++ adjust
                masked_img_connected = np.array(masked_img_connected, dtype = np.int32)
                ## Adjust function below uses author's C++ script, which I haven't been able to import
                # adjust_line_for_VPG.adjust(line, k, b, DOWNSCALE, masked_img_connected) 

            else:
                # only use straight line, don't do refinement
                y = masked_img_connected.shape[0] - 1
                line = np.array([[int((y - b)/k), y]])
                while y >= 10 * DOWNSCALE:
                    y -= int(10 * DOWNSCALE)
                    line = np.append(line, [[int((y - b)/k), y]], axis = 0) # lines are in the image of the size (640 * 480)*DOWNSCALE

            if line != [] and (line[0, 0] != 0 or line[0, 1] != 0): # this line exists
                lines.append(line)

        time5 = time.time() # lines are in the image of the size (640 * 480)* DOWNSCALE

        ## Below uses functions from commented functions above

        # if lines != []:

        #     # filter through lines, make polyline control points sparser, and convert them to image coordinates
        #     if not suppress_output:
        #         lines, npx, npy = clean_up(orig_img, lines, suppress_output)
        #         # rescale npx, npy back to original image (not 640*480!) and store in the same shape as lines
        #         lines_in_img = scale_back(lines, npx, npy, resize_x, resize_y)
        #         # further convert to ground coordinates: (real-world)
        #         lines_in_gnd = convert_img2gnd(npx, npy, lines)
        #         print lines_in_gnd

        #     else:
        #         npx, npy, lines = make_sparse(lines) # lines are in image of (640 * 480) * DOWNSCALE; real ipm is in (640 * 480) / IMAGE_SIZE_RESCALE
        #         lines_in_gnd = convert_img2gnd(npx, npy, lines) # Note: this function also modifies lines[]!
        #         # lines format: a list of numpy ndarrays [line1, line2, ...], line1 = np.array( [[x1, y1], [x2, y2], ...] )
        # else:
        #     lines_in_gnd = []
    
    else:
        lines_in_gnd = []
        lines = []

    time6 = time.time()
    #print "readfile time: ", time2 - time1, "preprocess:", time25 - time2, "houghlines: ", time3 - time25, "clustering: ", time4 - time3
    #print "adjust in C++: ", time5 - time4, "clean up: ", time6 - time5, "total time: ", time6 - time1
    #print "total time not counting file reading: ", time6 - time2

    return time6 - time2, lines_in_gnd, lines


# Class referenced from 'lane_detection_workflow.py'

class LaneDetector:
    """
    The class which includes the entire pipeline of lane detection
    """

    def __init__(self, workspace_root='.'):
        if not os.path.exists(os.path.join(os.getcwd(), workspace_root)):
            os.mkdir(workspace_root)

        self.model = './deploy.prototxt'
        self.pretrained = 'snapshots/split_iter_10000.caffemodel'
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.net = caffe.Net(self.model, self.pretrained, caffe.TEST)
        print ("successfully loaded classifier")

    def load_image(self, filename):
        """ load image from filename and store it in VPGNet """
        self.filename = filename
        self.img = caffe.io.load_image(filename)
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('data', (2, 1, 0))
        transformed_img = transformer.preprocess('data', self.img) # swap R, B channel, the final input to the network should be RGB
        self.net.blobs['data'].data[...] = transformed_img

    def forward(self):
        """ forward-propagation """
        self.net.forward()

    def extract_mask(self):
        """
        get the result from VPGNet
        the result is a binary mask, the mask is smaller than original picture because it is shrinked by grid_size
        """
        obj_mask = self.net.blobs['binary-mask'].data
        x_offset_mask = 4 # offset to align output with original pic: due to padding
        y_offset_mask = 4
        masked_img = self.img.copy()
        mask_grid_size = self.img.shape[0] / obj_mask.shape[2]
        small_mask = obj_mask[0, 1, ...] * 255
        self.resized_mask = cv2.resize(small_mask, (640, 480))
        translationM = np.float32([[1, 0, x_offset_mask*mask_grid_size], [0, 1, y_offset_mask*mask_grid_size]])
        self.resized_mask = cv2.warpAffine(self.resized_mask, translationM, (640, 480)) # translate (shift) the image
        # cv2.imwrite(workspace_root + 'mask.png', resized_mask)
        # return self.resized_mask

    def post_process(self, t1):
        """
        do post-processing of the mask and extract actual polylines out of it

        t: time spent on post-processing
        lines: polyline, format see lane_extension_polyline_for_VPG.py
        """
        self.t, self.lines_in_gnd, self.lines_in_img = work(self.resized_mask, do_adjust=True, 
                                                               suppress_output=True, time1=t1, time2=t1)
        # work(file, do_adjust, suppress_output, time1, time2)

        return self.t

    def visualize(self, num):
        """
        visualize the result of post-processing on the original picture

        num: the index of the picture being processed
        the output is stored in VPG_log/
        """
        original_img = cv2.imread(self.filename)
        original_img = cv2.resize(original_img, (640, 480))
        for line in self.lines_in_img:
            for i in range(len(line) - 1):
                cv2.line(original_img, (line[i][0], line[i][1]), (line[i+1][0], line[i+1][1]), (0, 0, 255), 5)
        cv2.imwrite('VPG_log/labeled/%d_labeled.png'%num, original_img)
        cv2.imwrite('VPG_log/labeled/%d_mask.png'%num, self.resized_mask)


workspace_root = 'VPG_log/'
detector = LaneDetector(workspace_root)
t_sum = 0
t_pp = 0
t_net = 0
for i in range(50): # Original range is 245
    detector.load_image('/media/herman/WD_BLACK/Ubuntu/FYP/VPGNet-master/caltech-lanes/cordova2/f'+str(i).zfill(5)+'.png')
    t0 = time.time()
    detector.forward()
    mask = detector.extract_mask()
    t1 = time.time()
    t = detector.post_process(t1)
    t_pp += t
    t_sum += time.time() - t0
    t_net += t1 - t0
    detector.visualize(i)
    # os.system('mv output_log/threshold.png VPG_log/log/%d.png'%i)
    # os.system('mv output_log/o.png VPG_log/log/%d_raw.png'%i)
    
# print 'total time ', t_sum / 245.0
# print 'VPGNet time ', t_net / 245.0
# print 'post-processing time ', t_pp / 245.0

