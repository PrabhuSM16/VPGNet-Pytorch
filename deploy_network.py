import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import cv2

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = './deploy.prototxt'
PRETRAINED = 'snapshots/split_iter_10000.caffemodel'

# load the model
caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
print ("successfully loaded classifier")

# Test on Image
image_path = '/media/herman/WD_BLACK/Ubuntu/FYP/VPGNet-master/caltech-lanes/cordova2/f00000.png'
image = cv2.imread(image_path)
net.blobs['data'].reshape(1, image.shape[2], image.shape[0], image.shape[1])

# Transform Image for pre-processing
# Input in Caffe data layer is (C, H, W)
transformer = caffe.io.Transformer({'data': (1, image.shape[2], image.shape[0], image.shape[1])})
transformer.set_transpose('data', (2, 0, 1)) # To reshape from (H, W, C) to (C, H, W) ...
transformer.set_raw_scale('data', 1/255.) # To scale to [0, 1] ...
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# Forward Pass for prediction
net.forward()
score_bb = net.blobs['bb-output-tiled'].data	#blobs['blob name']
score_multi = net.blobs['multi-label'].data
score_binary = net.blobs['binary-mask'].data

# print("bb has shape ", net.blobs['bb-output-tiled'].data.shape) #(1, 4, 120, 160)
# print("multi has shape ", net.blobs['multi-label'].data.shape)  #(1, 64, 60, 80)
# print("binary has shape ", net.blobs['binary-mask'].data.shape)	#(1, 2, 120, 160)

# Printing the output from each task
# print('bb score: ', score_bb)
# print('multi label score: ', score_multi)
# print('binary score: ', score_binary)


# Splitting channels from bb 
# bb_ch0 = score_bb[:, 0]	#(1, 4, 120, 160) 
#[:,0] means first column, using empty slice
# print("Channel 1 is: ", bb_ch0)
# bb_ch1 = score_bb[:, 1]
# print("Channel 2 is: ", bb_ch1)
# bb_ch2 = score_bb[:, 2]
# print("Channel 3 is: ", bb_ch2)
# bb_ch3 = score_bb[:, 3]
# print("Channel 4 is: ", bb_ch3)


# # Attempting to Check for None values
# for elem in bb_ch0.flat:
#         if elem is None:
#         	print("NONE DETECTED")
        # else:
        # 	print("No NONEs")


# # Trying to imshow the bb output (without splitting channels)
# print(score_bb.shape)
# score_bb = np.transpose(score_bb, (0, 3, 2, 1))
# print(score_bb.shape)
# cv2.imshow('image', score_bb)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()


# # Trying to imshow the output image of bb channel 1
# print(bb_ch0.shape) #(1, 120, 160) 
# print("Channel is: ", bb_ch0)
# bb_ch0 = np.transpose(bb_ch0, (2, 1, 0))
# print(bb_ch0.shape)
# cv2.imshow('image', bb_ch0)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()


# # Trying to imshow the output image of bb channel 2
# print(bb_ch1.shape) #(1, 120, 160) 
# print("Channel is: ", bb_ch1)
# bb_ch1 = np.transpose(bb_ch1, (2, 1, 0))
# print(bb_ch1.shape)
# cv2.imshow('image', bb_ch1)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()


# # Trying to imshow the output image of bb channel 3
# print(bb_ch2.shape) #(1, 120, 160) 
# print("Channel is: ", bb_ch2)
# bb_ch2 = np.transpose(bb_ch2, (2, 1, 0))
# print(bb_ch2.shape)
# cv2.imshow('image', bb_ch2)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()


# # Trying to imshow the output image of bb channel 4
# print(bb_ch3.shape) #(1, 120, 160) 
# print("Channel is: ", bb_ch3)
# bb_ch3 = np.transpose(bb_ch3, (2, 1, 0))
# print(bb_ch3.shape)
# cv2.imshow('image', bb_ch3)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()


# # Checking the different channels from multi-label
# multi_ch0 = score_multi[:, 3]
# print(multi_ch0)
# print(multi_ch0.shape) #(1, 120, 160) 
# print("Channel is: ", multi_ch0)
# multi_ch0 = np.transpose(multi_ch0, (1, 2, 0))
# print(multi_ch0.shape)
# cv2.imshow('image', multi_ch0)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()


# # Obtaining the different layer names
# def get_layers(net):
#     """
#     Get the layer names of the network.
    
#     :param net: caffe network
#     :type net: caffe.Net
#     :return: layer names
#     :rtype: [string]
#     """
    
#     return [layer for layer in net.params.keys()]

# layer_name = get_layers(net)
# print(layer_name)


# Attempting to Resize
# print(score_bb)
# # dsize = (640, 480)
# print("bb shape is: ", score_bb.shape)
# bb_resize = cv2.resize(score_bb, dsize, interpolation = cv2.INTER_AREA)
# print("New size of bb is: ", bb_resize.shape)


# Showing Original Image
# image_show = cv2.imread(image_path)
# print(image_show)
# imgplot = cv2.imshow('image', image_show)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()


# Visulisation method from another website, not edited 
# def visualize_kernels(net, layer, zoom = 5):
#     """
#     Visualize kernels in the given convolutional layer.
    
#     :param net: caffe network
#     :type net: caffe.Net
#     :param layer: layer name
#     :type layer: string
#     :param zoom: the number of pixels (in width and height) per kernel weight
#     :type zoom: int
#     :return: image visualizing the kernels in a grid
#     :rtype: numpy.ndarray
#     """
    
#     num_kernels = net.params[layer][0].data.shape[0]
#     num_channels = net.params[layer][0].data.shape[1]
#     kernel_height = net.params[layer][0].data.shape[2]
#     kernel_width = net.params[layer][0].data.shape[3]

#     print(num_kernels)
#     print(num_channels)
#     print(kernel_height)
#     print(kernel_width)
    
#     image = np.zeros((num_kernels*zoom*kernel_height, num_channels*zoom*kernel_width))
#     for k in range(num_kernels):
#         for c in range(num_channels):
#             kernel = net.params[layer][0].data[k, c, :, :]
#             kernel = cv2.resize(kernel, (zoom*kernel_height, zoom*kernel_width), kernel, 0, 0, cv2.INTER_NEAREST)
#             kernel = (kernel - np.min(kernel))/(np.max(kernel) - np.min(kernel))
#             image[k*zoom*kernel_height:(k + 1)*zoom*kernel_height, c*zoom*kernel_width:(c + 1)*zoom*kernel_width] = kernel
    
#     return image

# visualize_kernels(net, "bb-output", zoom=5)

