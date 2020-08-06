import cv2
import base64
import argparse
import json
import pika
import glob
import time
from datetime import datetime

# print('asd')
parser = argparse.ArgumentParser()
parser.add_argument("--cam_id", help="id of cam")
# images/cars/15_05_R_200124130800_1.mkv
args = parser.parse_args()
rtsp = args.cam_id

base = f"rtsp://192.168.1.51:554/user=admin&password=8996&channel={rtsp}&stream=0.tcp"

def main():
    try:
        cap = cv2.VideoCapture(base)
        k = 0
        data = dict()
        ret, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # time.sleep(20)
        # print("OK")
    # img = cv2.resize(img, (256,256))
        h, w, c = img.shape
        str_img = base64.encodebytes(img.tobytes()).decode("utf-8")
        print(len(str_img))
        data['cam_id'] = rtsp
        now = datetime.now()
        data['time'] = datetime.timestamp(now)
        data['img_base64'] = str_img
        data['width'] = w
        data['height'] = h
        data['channels'] = c
        # k = 0
        # print(1)
        
        credentials = pika.credentials.PlainCredentials('queue', 'ugazmHU5R', erase_on_connect=False)
        # print(2)
        connection = pika.BlockingConnection(pika.ConnectionParameters('europharma.lean-solutions.kz', '5672', '/', credentials, heartbeat = 600))
        # connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        # print(3)
        channel = connection.channel()
        # print(4)
        channel.queue_declare(queue='carwash_camera_ml')
        channel.basic_publish(exchange='', routing_key='carwash_camera_ml', body=json.dumps(data))
        print(" [x] Sent message")
        connection.close()
        # cv2.imshow('img', img)
        # if cv2.waitKey(int(1000/10)) & 0xFF == ord('q'):
        #                 break
        
        # cap.release()
        # cv2.destroyAllWindows()
    except Exception as e:
        # print(ctime(time()))
        print(e)
        time.sleep(20)
while True:
    main()
    time.sleep(2)
