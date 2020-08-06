# Import all necessary libraries.
import os
import cv2
import numpy as np
import sys
import json
import matplotlib.image as mpimg
import cv2
# import tensorflow as tf
from tensorflow.python.client import device_lib
import uuid
# tf.test.is_built_with_gpu_support()
# ab = tf.test.is_gpu_available(
#     cuda_only=True,
#     min_cuda_compute_capability=None
# )
# import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
ess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
# tf.debugging.set_log_device_placement(True)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# # change this property
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
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel("latest")

rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# Initialize text detector.
textDetector = TextDetector({
    
    "kz": {
        "for_regions": ["kz"],
        "model_path": "../nomeroff-net/models/anpr_ocr_kz1_002-gpu.h5"
    }
    
}, mode='gpu')

# textDetector = TextDetector({
    
#     "kz": {
#         "for_regions": ["kz"],
#         "model_path": "latest"
#     }
    
# })

# Walking through the ./examples/images/ directory and checking each of the images for license plates.
print("START RECOGNIZING")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
rootDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../images/')

max_img_w = 1600
# Initialize npdetector with default configuration file.
def num_predict(img):
    
    # for dirName, subdirList, fileList in os.walk(rootDir):
    #     for fname in fileList:
    

        # img_path = os.path.join(dirName, fname)
        # print(img_path)t
        # img = mpimg.imread(img_path)
        
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
        # resized_img = cv2.resize(img, (img_w*2, img_h*2))
        resized_img = img

    NP = nnet.detect([resized_img]) 
    
    # Generate image mask.
    cv_img_masks = filters.cv_img_mask(NP)
    # Detect points.
    arrPoints = rectDetector.detect(cv_img_masks, outboundHeightOffset=0, fixGeometry=True, fixRectangleAngle=10)
    arrPoints[..., 1:2] = arrPoints[..., 1:2]*img_h_r
    arrPoints[..., 0:1] = arrPoints[..., 0:1]*img_w_r

    # print(arrPoints)
    zones = rectDetector.get_cv_zonesBGR(img, arrPoints)
    toShowZones = rectDetector.get_cv_zonesRGB(img, arrPoints)
    foundNumber=0
    for zone, points in zip(toShowZones, arrPoints):
        #plt.axis("off")
        mpimg.imsave('images/zones_error/'+str(uuid.uuid4())+'.png', zone)
        foundNumber += 1
    # cv2.imwrite('images/zones_error/zone.png', zones)
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)
    # print('zones.shape')
    # for zone in zones:
    # print(zones.shap
    # print('aaa')
    textArr = textDetector.predict(zones, ['kz'], [1])
    # print('aaaa')
    textArr = textPostprocessing(textArr, regionNames)

    return textArr