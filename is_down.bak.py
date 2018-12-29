# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import redis
import pickle
import cmath
import time
import os
import numpy as np
from sys import platform


redis = redis.Redis(host='localhost', port=6379, db=0)


def pross():
    
    start = time.time()
    ret = redis.lrange("keysList", 0, -1)
    print(">>>" + str(len(ret)))
    count = 1
    for key in ret:
       ret_data = redis.get(key)
       if ret_data is None:
        continue
       ret = pickle.loads(ret_data)
       for keypoint in ret["keypoints"]:
           x1 = keypoint[2][0]
           y1 = keypoint[2][1]
           x2 = keypoint[9][0]
           y2 = keypoint[9][1]
           x3 = keypoint[10][0]
           y3 = keypoint[10][1]
           if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0 or x3 == 0 or y3 == 0:
               x1 = keypoint[5][0]
               y1 = keypoint[5][1]
               x2 = keypoint[12][0]
               y2 = keypoint[12][1]
               x3 = keypoint[13][0]
               y3 = keypoint[13][1]
               if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0 or x3 == 0 or y3 == 0:
                   continue
           a2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
           b2 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)
           c2 = (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)
           a = cmath.sqrt(a2)
           b = cmath.sqrt(b2)
           c = cmath.sqrt(c2)
           pos = (a2+b2-c2)/(2*a*b)
           angle = cmath.acos(pos)
           realangle = angle*180/cmath.pi
           #print("realangle:" + str(realangle.real) + ",keypoint:" + str(keypoint))

           if (realangle.real >= 80 and realangle.real <= 100) :
               image = np.asarray(bytearray(ret["image"]), dtype="uint8")
               image = cv2.imdecode(image, cv2.IMREAD_COLOR)
               save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(key))
               cv2.imwrite(save_path, image)
               print("/home/woody/tmp/openpose" + key)
               #return False
           else:
               image = np.asarray(bytearray(ret["image"]), dtype="uint8")
               image = cv2.imdecode(image, cv2.IMREAD_COLOR)
               save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/wu", str(key))
               cv2.imwrite(save_path, image)
               #return True
       count = count + 1
    print(str(count) + ">>>" + str(time.time() - start))

if __name__ == '__main__':
    pross()
