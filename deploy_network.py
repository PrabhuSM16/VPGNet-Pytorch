import numpy as np
import matplotlib.pyplot as plt
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
	    self.pretrained = 'snapshots/split_iter_10000.caffemodel'
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
	    transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
	    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
	    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	    transformer.set_channel_swap('data', (2, 1, 0))
	    self.transformed_img = transformer.preprocess('data', self.img) # swap R, B channel, the final input to the network should be RGB
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
		# print obj_mask.shape
		# print transformed_img.shape

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
		cv2.imwrite('VPG_log/labeled/%d_mask.png'%num, resized_mask)
		cv2.imwrite('VPG_log/labeled/%d_masked.png'%num, masked_img)

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
		        1: (0, 255, 0), # green color
		        2: (255, 0, 0), # blue			# White Lines (?)
		        3: (0, 0, 255), # red 			# Only appears on boundary of road
		        4: (0, 0, 0)	# black			# Yellow Lines (?)
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
		            cv2.rectangle(original_img, pt1, pt2, color_options(maxi), 2)
		            if maxi not in [1, 2, 3, 4]:
		                print "ERROR OCCURRED: an unknown class detected!"

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


workspace_root = 'VPG_log/'
detector = LaneDetector(workspace_root)

for i in range(80): # Original range is 245
    detector.load_image('/media/herman/WD_BLACK/Ubuntu/FYP/VPGNet-master/caltech-lanes/cordova2/f'+str(i).zfill(5)+'.png')
    detector.forward()
    mask = detector.extract_mask(i)
    detector.visualize(i)
