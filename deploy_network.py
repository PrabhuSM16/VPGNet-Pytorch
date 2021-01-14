import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score
from scipy.io import loadmat
from scipy import ndimage
import sys
import os
import time # time the execution time

import caffe
import cv2

import shelve # store workspace

class LaneDetector:

	def __init__(self, workspace_root='.'):
	    if not os.path.exists(os.path.join(os.getcwd(), workspace_root)):
	        os.mkdir(workspace_root)

	    self.model = './deploy.prototxt'

	    ## Caltech dataset trained
	    # self.pretrained = 'snapshots/split_iter_100000.caffemodel'

	    ## VPG dataset trained
	    self.pretrained = 'snapshots/VPG_trained/split_iter_100000.caffemodel'
	    caffe.set_mode_gpu()
	    caffe.set_device(0)
	    self.net = caffe.Net(self.model, self.pretrained, caffe.TEST)
	    print ("successfully loaded classifier")

	# visualize net shape:
	# for name, blob in net.blobs.iteritems():
	#    print("{:<5}: {}".format(name, blob.data.shape))

	def load_image(self, filename):
	    self.filename = filename
	    self.img = caffe.io.load_image(filename)
	    print self.img.shape
	    transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
	    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
	    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	    transformer.set_channel_swap('data', (2, 1, 0))
	    self.transformed_img = transformer.preprocess('data', self.img) # swap R, B channel, the final input to the network should be RGB
	    print self.transformed_img.shape
	    self.net.blobs['data'].data[...] = self.transformed_img

	def forward(self):
	    """ forward-propagation """
	    self.net.forward()

	def extract_mask(self, num):
		# Visualize the test result:

		for i in range(3):
		    for j in range(self.transformed_img.shape[1]):
		        for k in range(self.transformed_img.shape[2]):
		            self.img[j, k, i] = self.transformed_img[i, j, k]
		# cv2.imwrite(workspace_root + "example.png", self.img)

		obj_mask = self.net.blobs['binary-mask'].data
		mlabel = self.net.blobs['multi-label'].data # mlabel: saves 18 feature maps for different classes
		bbox = self.net.blobs['bb-output-tiled'].data # bbox: not sure

		x_offset_mask = 4 # offset to align output with original pic: due to padding
		y_offset_mask = 4

		masked_img = self.img.copy()
		mask_grid_size = self.img.shape[0] / obj_mask.shape[2]
		tot = 0
		for i in range(120):
		    for j in range(160):
		        mapped_value =  int(obj_mask[0, 0, i, j] * 255)
		        obj_mask[0, 0, i, j] = mapped_value

		        mapped_value =  int(obj_mask[0, 1, i, j] * 255)
		        obj_mask[0, 1, i, j] = mapped_value

		        if mapped_value > 100:
		            masked_img[(i+y_offset_mask)*mask_grid_size : (i+1+y_offset_mask)*mask_grid_size + 1, (j+x_offset_mask)*mask_grid_size : (j+x_offset_mask+1)*mask_grid_size + 1]\
		             = (mapped_value, mapped_value, mapped_value) # mask with white block

		small_mask = obj_mask[0, 1, ...]
		resized_mask = cv2.resize(small_mask, (640, 480))
		translationM = np.float32([[1, 0, x_offset_mask*mask_grid_size], [0, 1, y_offset_mask*mask_grid_size]])
		resized_mask = cv2.warpAffine(resized_mask, translationM, (640, 480)) # translate (shift) the image
		# cv2.imwrite('VPG_log/labeled/%d_mask.png'%num, resized_mask)
		# print resized_mask.shape
		# cv2.imwrite('VPG_log/labeled/%d_masked.png'%num, masked_img)

	def visualize(self, num):
		# visualize classification
		original_img = cv2.imread(self.filename)
		original_img = cv2.resize(original_img, (640, 480))
		classification = self.net.blobs['multi-label'].data
		classes = []
		y_offset_class = 1 # offset for classification error
		x_offset_class = 1
		grid_size = self.img.shape[0]/60

		# create color for visualizing classification
		def color_options(x):
		    return {
		        1: (0, 255, 0), # green color 					# lane_solid_white (?)
		        2: (255, 0, 0), # blue							# Disconnected White Lines
		        3: (0, 0, 255), # red 							# Connected White Lines
		        4: (0, 0, 0),	# black							# Yellow Lines
		        5: (204, 204, 0), # dark yellow					# lane_broken_yellow
		        6: (102, 102, 0), # darker yellow				# lane_double_yellow
		        7: (51, 204, 255), #light blue					# lane_broken_blue
		        8: (255, 100, 0), #orange						# lane_slow
		        9: (128, 0, 0),	#maroon							# stop_line
		        10: (230, 230, 0), #yellow						# arrow_left
		        11: (230, 230, 0),								# arrow_right
		        12: (230, 230, 0),								# arrow_go_straight
		        13: (230, 230, 0),								# arrow_u_turn
		        14: (230, 230, 0),								# speed_bump
		        15: (208, 208, 225), # grey						# crossWalk
		        16: (208, 208, 225), #grey						# safety_zone
		        17: (255, 100, 208), #pink						# other_road_markings
		        18: (102, 0, 102) #purple						# ???
		    }[x]

		for i in range(60):
		    classes.append([])
		    for j in range(80):
		        max_value = 0
		        maxi = 0
		        # Finding max value 
		        for k in range(64):
		            if classification[0, k, i, j] > max_value:
		                max_value = classification[0, k, i, j]
		                maxi = k
		        classes[i].append(maxi)
		        if maxi != 0:
		            pt1 = ((j + y_offset_class)*grid_size, (i+x_offset_class)*grid_size)
		            pt2 = ((j + y_offset_class)*grid_size+grid_size, (i+x_offset_class)*grid_size+grid_size)
		            # print maxi
		            # print(maxi)
		            cv2.rectangle(original_img, pt1, pt2, color_options(maxi), 2)
		            # if maxi not in [1, 2, 3, 4]:
		            #     print "ERROR OCCURRED: an unknown class detected!"

		cv2.imwrite('VPG_log/labeled/%d_labeled.png'%num, original_img)

		# bounding box visualization
		# bb = net.blobs['bb-output-tiled'].data
		# print bb.shape
		# bb_visualize0 = bb[0, 0, ...]*255
		# bb_visualize1 = bb[0, 1, ...]*255
		# bb_visualize2 = bb[0, 2, ...]*255
		# bb_visualize3 = bb[0, 3, ...]*255
		# cv2.imwrite('bb_visualize0.png', bb_visualize0)
		# cv2.imwrite('bb_visualize1.png', bb_visualize1)
		# cv2.imwrite('bb_visualize2.png', bb_visualize2)
		# cv2.imwrite('bb_visualize3.png', bb_visualize3)

		# keys = ['classification', 'obj_mask', 'x_offset_class', 'y_offset_class', 
		# 'mask_grid_size', 'img', 'max_value', 'x_offset_mask', 'y_offset_mask', 'grid_size', 'transformed_img', 
		# 'classes', 'masked_img', 'resized_mask', 'small_mask']

		# shelf_file_handle = shelve.open(workspace_root + 'shelve.out', 'n')

		# for key in keys:
		#     print 'saving variable: ', key
		#     shelf_file_handle[key] = globals()[key]
		# shelf_file_handle.close()

	def f1_score_list(self, num, mat_path):
		ground_truth = loadmat(mat_path)['rgb_seg_vp']
		mlabel = self.net.blobs['multi-label'].data # mlabel: saves 18 feature maps for different classes

		# Obtain classes in ground truth into a list
		ground_transform = ground_truth[:,:, 3] # Get the different classes in the ground truth
		ground_classes = set()
		for x in range(1,18):
			if (x in ground_transform):
				# print(x)
				ground_classes.add(x)
		ground_classes = sorted(ground_classes)
		print("Ground Truth has classes: " + str(ground_classes))

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

		def class_names(class_num):
			return {
		        1: 'lane_solid_white',					
		        2: 'lane_broken_white',						
		        3: 'lane_double_white',						
		        4: 'lane_solid_yellow',				
		        5: 'lane_broken_yellow',				
		        6: 'lane_double_yellow',		
		        7: 'lane_broken_blue',					
		        8: 'lane_slow',					
		        9: 'stop_line',					
		        10: 'arrow_left',					
		        11: 'arrow_right',							
		        12: 'arrow_go_straight',				
		        13: 'arrow_u_turn',					
		        14: 'speed_bump',			
		        15: 'crossWalk',	
		        16: 'safety_zone',				
		        17: 'other_road_markings',		
		        18: 'unknown'			
		    }[class_num]

		## Saving the 18 different class masks
		# for i in range(0,18): # 18 classes in total, corresponding to github/VPGNet/vpgnet-labels.txt
		#     small_mask = mlabel[0, i, ...] * 255
		#     resized_mask = cv2.resize(small_mask, (640, 480))
		#     class_image = resized_mask.astype('uint8')
		#     print(i) # show which class it is
		#     cv2.imwrite('VPG_log/labeled/multi_class_%d.png'%i, class_image)

		# ground_mask = ground_transform > 0 # Assume ground truth only have one class. Returns Boolean.
		# extend_mask = np.ones((480, 640), dtype=bool) # extended groundtruth (from 8*8 square grid to radius R circle)
		# for i in range(0, 480, 8):
		#     for j in range(0, 640, 8):
		#         if ground_mask[i,j].any() == True: # if this pixel have label, this 8*8 grid should have same label
		#             area_mask = create_circular_mask(480, 640, center = (i,j), radius = 4)
		#             extend_mask = extend_mask + area_mask # add the area_mask to blank mask
		            
		# Compare map1 and the extended mask for f1 score
		f1_dict = {}
		for x in ground_classes:
			# Obtaining detected classes from network
			class_type = class_names(x)
			small_mask = mlabel[0, x, ...] * 255
			resized_mask = cv2.resize(small_mask, (640, 480))
			class_image = resized_mask.astype('uint8')
			class_mask = class_image > 0 # Returns Boolean.

			# # Trying to get individual class mask from ground truth
			# ground_type = ground_transform # ground_transform is the ground truth
			# for i in range(0, 480):
			#     for j in range(0, 640):
			#         if ground_type[i,j] != x: # x is the class, i.e. range(1,18)
			#             ground_type[i,j] = 0

			# ground_mask = ground_type > 0 # Returns Boolean.

			temp_ground_map = [None] * 18	# Create empty list for 18 classes
			temp_ground_map[x] = (ground_transform == x)
			temp_ground_map[x] = temp_ground_map[x].astype(np.uint8) # Translate True/False to 1-0

			extend_mask = np.ones((480, 640), dtype=bool) # extended groundtruth (from 8*8 square grid to radius R circle)
			for i in range(0, 480, 8):
			    for j in range(0, 640, 8):
			        if temp_ground_map[x][i,j] == True: # if this pixel have label, this 8*8 grid should have same label
			            area_mask = create_circular_mask(480, 640, center = (i,j), radius = 4)
			            extend_mask = extend_mask + area_mask # add the area_mask to blank mask
			single_class_f1 = f1_score(extend_mask.flatten(), class_mask.flatten())
			f1_dict[class_type] = single_class_f1
		return f1_dict


workspace_root = 'VPG_log/'
detector = LaneDetector(workspace_root)

## Deploy on VPG dataset
# vpg_img = '000300'
# detector.load_image('/media/herman/WD_BLACK/Ubuntu/FYP/VPGNet_dataset/scene_1/20160512_1329_00/'+vpg_img+'.png')
# detector.forward()
# mask = detector.extract_mask(int(vpg_img))
# detector.visualize(int(vpg_img))

## Deploy on caltech dataset
# for i in range(337): # cordova1:245, c2:406, washington1:337, w2:232
#     detector.load_image('/media/herman/WD_BLACK/Ubuntu/FYP/VPGNet-master/caltech-lanes/washington1/f'+str(i).zfill(5)+'.png')
#     detector.forward()
#     mask = detector.extract_mask(i)
#     detector.visualize(i)

## Testing with F1 score
detector.load_image('/media/herman/WD_BLACK/Ubuntu/FYP/VPGNet_dataset/scene_1/20160512_1329_00/Original/000181.png')
path_to_mat = '/media/herman/WD_BLACK/Ubuntu/FYP/VPGNet_dataset/scene_1/20160512_1329_00/000181.mat'
detector.forward()
# mask = detector.extract_mask(181)
detector.visualize(181)
f1_dict = detector.f1_score_list(181, path_to_mat)
print f1_dict
