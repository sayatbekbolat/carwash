#!/usr/bin/python3.7
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import sys
import cv2
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from retinanet import model
import traceback
from retinanet.dataloader import  CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer
from car import *
import distance 
import pika
import json
import base64
import nomer_detector 
import uuid
from PIL import Image
from io import BytesIO
import logging
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf
import vector
retinanet = model.resnet50(num_classes=1,)
use_gpu = True
transform=transforms.Compose([Normalizer(), Resizer()])

if use_gpu:
    retinanet = retinanet.cuda()
    print('its gpu running')
resizer = Resizer()

retinanet = torch.load('retina/csv_retinanet_56.pt')
# retinanet.eval()
credentials = pika.credentials.PlainCredentials('queue', 'ugazmHU5R', erase_on_connect=False)
connection = pika.BlockingConnection(pika.ConnectionParameters('europharma.lean-solutions.kz', '5672', '/', credentials, heartbeat=20000))
channel = connection.channel()
channel.basic_qos(prefetch_count=1)
# logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                    # format="%(asctime)-15s %(levelname)-8s %(message)s")
# logging.info("New start!")
# channel.queue_declare(queue='carwash_camera_ml')

cars = {'1': None,
        '2': None,
        '3': None,
        '4': None,
        '6': None}

def compare_num(num, num1):
    print("numbers fo compare")
    print(str(num)+'  '+str(num1))
    print((distance.levenshtein(num, num1)/max([len(num), len(num1)])))
    return (distance.levenshtein(num, num1)/max([len(num), len(num1)]))

def calc_center(p1, p2):
    center = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
    return center

def check_center(center):
    x1, x2, y1, y2 = 100, 400, 100, 400
    if (center[0] > x1 and center[0] < x2 and 
        center[1] > y1 and center[1] < y2) : 
        return True
    else : 
        return False

def send(car):
    credentials = pika.credentials.PlainCredentials('queue', 'ugazmHU5R', erase_on_connect=False)
    conn = pika.BlockingConnection(pika.ConnectionParameters('europharma.lean-solutions.kz', '5672', '/', credentials, heartbeat=600))
    chan = conn.channel()
    now = datetime.now()
    out_time = datetime.timestamp(now)
    in_time, res_num, out_time = car.out(out_time)
    # logging.info(str(in_time), str(car.get_id()[0]))
    res = {
        'num_id':str(res_num),
        'in_time':str(in_time),
        'out_time':str(out_time),
        'camera_id':str(car.get_id()[0]),
        'image':str(car.get_image())[2:-1]
    }
    print('car sended', res['camera_id'])
    chan.queue_declare(queue='carwash_ml_back')
    chan.basic_publish(exchange='',
                    routing_key='carwash_ml_back',
                    body=json.dumps(res))
    conn.close()

def img_prepare(data):
    img_str = base64.decodebytes(data['img_base64'].encode())
    nparr = np.frombuffer(img_str, dtype=np.uint8)
    # print(nparr.shape)
    img = np.array(nparr).reshape(int(data['height']), int(data['width']), int(data['channels']))
    return img

def callback(ch, method, properties, body):
    try:
        data = json.loads(body)
        car = cars[data['cam_id']]
        # in_time = data['time']
        # print(data['img_base64'])
        frame = img_prepare(data)
        h, w, c = frame.shape
        x_s = w / 512
        y_s = h / 512
        # print(h,w,c)
        mean = np.array([[[0.485, 0.456, 0.406]]])
        std = np.array([[[0.229, 0.224, 0.225]]])    
        collector = []    
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)    
        dim = (width, height)
        img = cv2.resize(img, (512, 512)) #dim, interpolation = cv2.INTER_AREA)
        img_dis = img.copy()
        img = img.astype(np.float32)/255.0            
        img = ((img.astype(np.float32)-mean)/std)            
        img_tensor = torch.from_numpy(img)
        
        scores, classification, transformed_anchors = retinanet(img_tensor.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
        idxs = np.where(scores.cpu()>0.5)

        if idxs[0].shape[0] > 0:
            bbox = transformed_anchors[idxs[0][0], :]
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            center = calc_center(p1, p2)
            if check_center(center):
                print("There is a car")
                if car is None:

                    
                    img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)
                    # print(img_dis.shape)
                    pil_img = Image.fromarray(img_dis)
                    buffered = BytesIO()
                    pil_img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue())
                    now = datetime.now()
                    in_time = datetime.timestamp(now)
                    car = Car(inTime = str(in_time), img = img_str, ID = data['cam_id'])

                bbox[0] = int(np.round((int(bbox[0]) * x_s)))
                bbox[1] = int(np.round((int(bbox[1]) * y_s)))
                bbox[2] = int(np.round((int(bbox[2]) * x_s)))
                bbox[3] = int(np.round((int(bbox[3]) * y_s)))
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[2]), int(bbox[3]))
                im = frame[p1[1]:p2[1], p1[0]:p2[0]]
                # print('im shape')
                # print(im.shape)
                cv2.imwrite('images/new_conf_01_08/'+str(uuid.uuid4())+'.jpeg', im)

                num = nomer_detector.num_predict(im)
                print(num)          
                num = ''.join(num).split()
                
                if len(num)>0:
                    if len(num[0])>5:
                        num = num[0]
                        # print(num)
                        last_num = car.get_last_num()
                        if last_num!='empty':
                            if compare_num(last_num, num)>0.4:
                                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                                print('car changed')
                                send(car)
                                now = datetime.now()
                                in_time = datetime.timestamp(now)

                                pil_img = Image.fromarray(img_dis)
                                buffered = BytesIO()
                                pil_img.save(buffered, format="JPEG")
                                img_str = base64.b64encode(buffered.getvalue())

                                car = Car(inTime = str(in_time), img = img_str, ID = data['cam_id'])
                                
                        else:
                            if car.get_last_num()!='empty':
                                car.add_num(car.get_last_num())
                        print('ADDED NUM', num)
                        car.add_num(num)
            else:
                if car:
                    send(car)
                car = None

        cars[data['cam_id']]=car

        ch.basic_ack(delivery_tag = method.delivery_tag) 
    except Exception as e:
        print(e)
channel.basic_consume(queue='carwash_camera_ml', on_message_callback=callback)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
