 
#!/usr/bin/env python
import pika
import time
import json
import base64
import numpy as np
import cv2
credentials = pika.credentials.PlainCredentials('queue', 'ugazmHU5R', erase_on_connect=False)
connection = pika.BlockingConnection(pika.ConnectionParameters('europharma.lean-solutions.kz', '5672', '/', credentials))
channel = connection.channel()


print(' [*] Waiting for messages. To exit press CTRL+C')


def img_prepare(data):
    img_str = base64.decodebytes(data['img_base64'].encode())
    nparr = np.frombuffer(img_str, dtype=np.uint8)
    print(nparr.shape)
    img = np.array(nparr).reshape(int(data['height']), int(data['width']), int(data['channels']))
    return img
def callback(ch, method, properties, body):
    print(" [x] Received %r" % len(body))

    data = json.loads(body)
    
    print(len(data['img_base64']))
    img = img_prepare(data)
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='carwash_camera_ml', on_message_callback=callback)

channel.start_consuming()
''' pil_img = Image.fromarray(orig)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())'''