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
import videostream
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from retinanet import model
import traceback
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
from car import *
import distance 
from twisted.internet import task, reactor

import nomer_detector 

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


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

def main():
    is_available = False
    camera = cv2.VideoCapture('images/cars/15_05_R_200124130800_1.mkv')
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])    
    collector = []    
    print('start')
    c = 0
    retinanet = model.resnet50(num_classes=80,)
    use_gpu = True
    transform=transforms.Compose([Normalizer(), Resizer()])

    if use_gpu:
        retinanet = retinanet.cuda()
    resizer = Resizer()
    retinanet = torch.load('retina/csv_retinanet_56.pt')
    retinanet.eval()
    print('asd')
    car = None
    while True:
        ret, frame = camera.read()
        if frame is not None:
            if c<4500:        
                if c%150==0:
                    c+=1
                    try:
                        time.sleep(0.5)
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
                        # print(classification)
                        
                        idxs = np.where(scores.cpu()>0.5)

                        if idxs[0].shape[0] > 0:
                            bbox = transformed_anchors[idxs[0][0], :]
                            # print(bbox)
                            p1 = (int(bbox[0]), int(bbox[1]))
                            p2 = (int(bbox[2]), int(bbox[3]))
                            center = calc_center(p1, p2)
                            if check_center(center):
                                print("There is a car")
                                if car is None:
                                    now = datetime.now()
                                    in_time = datetime.timestamp(now)
                                    car = Car(inTime = str(in_time))
                                im = img_dis[p1[1]:p2[1], p1[0]:p2[0]]

                                num = nomer_detector.num_predict(frame)
                              
                                num = ''.join(num).split()
                                
                                print(num)
                                if len(num)>0:
                                    num = num[0]
                                    last_num = car.get_last_num()
                                    print('LAST NUM')
                                    print(last_num) 
                                    if last_num!='empty':
                                        # print(compare_num(last_num, num[0]))
                                        if compare_num(last_num, num)>0.4:
                                        
                                            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                                            print('car changed')
                                            now = datetime.now()
                                            out_time = datetime.timestamp(now)
                                            in_time, res_num, out_time = car.res_num(out_time)
                                            print(in_time)
                                            res = {
                                                'num_id':res_num,
                                                'in_time':in_time,
                                                'out_time':out_time
                                            }
                                            print(res)
                                            car = None
                                    car.add_num(num)
                                    print('ADDED '+str(num))

                                else:
                                    print('No number detected!!!')
                                    last_num = car.get_last_num() 
                                    if last_num!='empty': 
                                        num = car.get_last_num()
                                        car.add_num(num)
                            else:
                                if car is None:
                                    now = datetime.now()
                                    out_time = datetime.timestamp(now)
                                    in_time, res_num, out_time = car.out(out_time)
                                    print(in_time)
                                    res = {
                                        'num_id':res_num,
                                        'in_time':in_time,
                                        'out_time':out_time
                                    }
                                    print(res)
                                    car = None


                                print(num, len(num))

                        

                            # ollector = add_box(collector, calc_center(p1, p2))
                            # print('added to collector')
                            # print(collector)
                            # is_car_there = check_collecor(collector)
                            # print(is_car_there, is_available)                
                            # if is_car_there:
                            #     if is_available:
                            #         print('checking car')
                            #         x = p1[0]
                            #         y = p1[1]
                            #         im = img_dis[x:x+p2[0], y:y+p2[1]]
                            #         # new_car.update(collector)
                            #         num = nomer_detector.num_predict(im)
                            #         # print(num)
                            #         # new_car.add_num(num)
                            #     else:
                            #         ct = datetime.now()
                            #         in_time = (ct.year, ct.month, ct.day, ct.hour, ct.minute)
                            #         new_car = Car(TRESH, collector, in_time)
                            #         # car_process(new_car)
                            #         is_available = True
                            #         print('CAR entered')
                            # else:
                            #     if is_available:
                            #         print('OUT CAR')
                            #         is_available = False
                            #         res_num = new_car.res_num()
                            #         res_json = {'num': res_num, 
                            #                     'intime': new_car.inTime, 
                            #                     'outtime':(ct.year, ct.month, ct.day, ct.hour, ct.minute)}
                            #         print(res_json)
                                    # car out 
                            cv2.rectangle(img_dis, p1, p2, color=(0, 0, 255), thickness=2)
                            cv2.circle(img_dis, calc_center(p1, p2), 5, color=(0, 0, 255), thickness=2 )

                        if cv2.waitKey(int(1000/10)) & 0xFF == ord('q'):
                            break
                        cv2.imshow('img', cv2.cvtColor(img_dis, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        raise e
                else:
                    c+=1
                    continue
            else:
                c = 0
        else:
            break
    camera.release()
 
# Closes all the frames
    cv2.destroyAllWindows()

        

            

if __name__ == '__main__':
    
    TRESH = 35
    main()
