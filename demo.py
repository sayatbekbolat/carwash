import numpy as np
import torchvision
import glob
from collections import Counter
import argparse
import sys
import cv2
import os
import matplotlib.image as mpimg
# from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
  UnNormalizer, Normalizer
# from car import *
# import distance 
# import time
# import nomer_detector 'coroutine' object is not subscriptable
# 

# assert torch.__version__.split('.')[0] == '1'
import tensorflow as tf
# from tensorflow.python.client import device_lib
# tf.test.is_built_with_gpu_support()
# ab = tf.test.is_gpu_available(
#     cuda_only=True,
#     min_cuda_compute_capability=None
# )

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
ess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
# tf.debugging.set_log_device_placement(True)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# change this property
NOMEROFF_NET_DIR = os.path.dirname(os.path.realpath(__file__))
# print(NOMEROFF_NET_DIR)

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')
# print(MASK_RCNN_DIR)
sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, textPostprocessingAsync

print("LOADING MODELS...")
print('CUDA available: {}'.format(torch.cuda.is_available()))
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel("latest")
rectDetector = RectDetector()

# Initialize text detector.
rootDir = 'images/new_conf_22_07/'
already = os.listdir(rootDir)
print(already)

# import operator
# optionsDetector = OptionsDetector()
# optionsDetector.load("latest")
# def fix_rect(arrPoints):
#     newArrPoints = arrPoints[0].tolist()
#     (minXidx, minPoint) = min(enumerate(newArrPoints), key=operator.itemgetter(1))
#     res = newArrPoints[minXidx:4]+newArrPoints[0:minXidx]
#     dx1 = res[1][0]-res[0][0]
#     dx3 = res[3][0]-res[0][0]
#     if (dx1>dx3):
#         return np.array([[res[3]]+res[0:3]])
#     else:
#         return np.array([res])
max_img_w = 1600
i = 0
import tqdm
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in tqdm.tqdm_notebook(fileList):
        # print(1)
        try:
            baseName = os.path.splitext(os.path.basename(fname))[0]
            img_path = os.path.join(dirName, fname)
            print(i, img_path)
            
            img = mpimg.imread(img_path)
            #plt.axis("off")
            #plt.imshow(img)
            #plt.show()
             
            # corect size for better speed
            img_w = img.shape[1]
            img_h = img.shape[0]
            img_w_r = 1
            img_h_r = 1
            if img_w > max_img_w:
                resized_img = cv2.resize(img, (max_img_w, int(max_img_w/img_w*img_h)))
                img_w_r = img_w/max_img_w
                img_h_r = img_h/(max_img_w/img_w*img_h)
            else:
                resized_img = img
            
            # print('NP')
            NP = nnet.detect([resized_img]) 
            # print('after NP')
            # Generate image mask.
            cv_img_masks = filters.cv_img_mask(NP)

            # Detect points.
            arrPoints = rectDetector.detect(cv_img_masks, outboundHeightOffset=0, fixGeometry=True, fixRectangleAngle=10)
            # print(arrPoints)
            # arrPoints = fix_rect(arrPoints)
            arrPoints[..., 1:2] = arrPoints[..., 1:2]*img_h_r
            arrPoints[..., 0:1] = arrPoints[..., 0:1]*img_w_r
            # cut zones
            zones = rectDetector.get_cv_zonesBGR(img, arrPoints)
            toShowZones = rectDetector.get_cv_zonesRGB(img, arrPoints)

            # find standart
            # regionIds, stateIds, countLines = optionsDetector.predict(zones)
            # if countLines:
            #     if countLines[0]>1:
            #         i += 1
                #regionNames = optionsDetector.getRegionLabels(regionIds)
            foundNumber=0
            for zone, points in zip(toShowZones, arrPoints):
                #plt.axis("off")
                mpimg.imsave("new_nums/img_22_07/{}_{}.png".format(baseName, foundNumber), zone)
                foundNumber += 1
                    #plt.imshow(zone)
                #plt.show()


            #print(regionNames)'coroutine' object is not subscriptable


            # find text with postprocessing by standart  
            #textArr = textDetector.predict(zones, regionNames, countLines)
            #textArr = await textPostprocessingAsync(textArr, regionNames)
            #print(textArr)
        except Exception as e:
            print("Error")
            print(e)
print(i)