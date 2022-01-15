import time
import cv2
import numpy as np
from numpy.linalg import norm
import argparse

vid = cv2.VideoCapture(0)
vid.set(10,0.1)



while(True):
    start_time = time.time()
    ret, frame = vid.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )

    lower = np.array([90, 90, 90])
    upper = np.array([255,255,255])
    
    mask = cv2.inRange(rgb, lower, upper)
    
    res = cv2.bitwise_and(frame,frame,mask=mask)

    ny = int(frame.shape[1]/10)
    nx = int(frame.shape[0]/10)

    rough_location_x = 0
    rough_location_y = 0
    
    loc_count = 0

    for y_large_sweep in range(0,ny):
        
        for x_large_sweep in range(0,nx):
            
            if res[x_large_sweep*10][y_large_sweep*10][0] >= 1:
                res[x_large_sweep*10][y_large_sweep*10][0] = 0
                res[x_large_sweep*10][y_large_sweep*10][1] = 0
                res[x_large_sweep*10][y_large_sweep*10][2] = 0
                
                rough_location_x += x_large_sweep*10
                rough_location_y += y_large_sweep*10
                
                loc_count += 1
                
    if loc_count > 0:
        
        rough_location_x /= loc_count
        rough_location_y /= loc_count
        
        res = cv2.circle(res, [int(rough_location_y), int(rough_location_x)], 100, [255, 255, 255], 5)

    res = cv2.putText(res, "FPS: %f" % (1.0 / (time.time() - start_time)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('image.png', res)
    # cv2.imshow('thresh.png', frame)
    # cv2.imshow('orig',frame)
    # cv2.imshow('fff',res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()

