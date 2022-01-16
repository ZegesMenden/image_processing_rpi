# image processing library
from re import T
import cv2

# math and array functions
import numpy as np
from light_tracker import light_tracker

vid = cv2.VideoCapture(0)

# sets the exposure of the camera
vid.set(10,0.1)

light_tracker_1: light_tracker = light_tracker()

while True:
    
    inp = input()
    
    
    light_tracker_1.update(vid)
    cv2.imshow('frame', light_tracker_1.video_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()