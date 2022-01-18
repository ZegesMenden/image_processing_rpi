# image processing library
from re import T
import cv2

# math and array functions
import numpy as np
from light_tracker import camera_controller

import os

clear = lambda: os.system('cls')

vid_1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
vid_2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)

# sets the exposure of the camera
vid_1.set(10,0.5)
vid_2.set(10,0.5)

cam_1: camera_controller = camera_controller()
cam_2: camera_controller = camera_controller()

while True:
    _1, frame_raw_1 = vid_1.read()
    _2, frame_raw_2 = vid_2.read()
    cam_1.update(frame_raw_1)
    cam_2.update(frame_raw_2)
    
    cv2.imshow('frame1', cam_1.video_out)
    cv2.imshow('frame2', cam_2.video_out)
    
    
    
#     clear()
    
#     print(f'''
# locations x:
# {cam_1.positions_x}
# locations y:
# {cam_1.positions_y}

# ''')
    
#     print(f'''
# locations x:
# {cam_2.positions_x}
# locations y:
# {cam_2.positions_y}

# ''')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # cv2.imwrite('frame1_3.jpg', frame_raw_1)
        # cv2.imwrite('frame2_3.jpg', frame_raw_2)
        break
vid_1.release()
vid_2.release()

cv2.destroyAllWindows()