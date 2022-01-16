# image processing library
import cv2
import glob
import cv2 as cv

# math and array functions
import numpy as np

class light_tracker:
    
    def __init__(self):
        
        # current position at index 0 and previous 9 positions
        self.positions_x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.positions_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.lower_light_threshold = np.array([90, 90, 90])
        self.upper_light_threshold = np.array([255,255,255])

        self.smoothed_location_x = 0.0
        self.smoothed_location_y = 0.0
        
        self.r = 0.0
        
    def calibrate_camera(self, images_folder):
        # stolen from https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
        
        images_names = sorted(glob.glob(images_folder))
        images = []
        for imname in images_names:
            im = cv.imread(imname, 1)
            images.append(im)
    
        #criteria used by checkerboard pattern detector.
        #Change this if the code can't find the checkerboard
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
        rows = 5 #number of checkerboard rows.
        columns = 8 #number of checkerboard columns.
        world_scaling = 1. #change this to the real world square size. Or not.
    
        #coordinates of squares in the checkerboard world space
        objp = np.zeros((rows*columns,3), np.float32)
        objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
        objp = world_scaling* objp
    
        #frame dimensions. Frames should be the same size.
        width = images[0].shape[1]
        height = images[0].shape[0]
    
        #Pixel coordinates of checkerboards
        imgpoints = [] # 2d points in image plane.
    
        #coordinates of the checkerboard in checkerboard world space.
        objpoints = [] # 3d point in real world space
    
    
        for frame in images:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
            #find the checkerboard
            ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
    
            if ret == True:
    
                #Convolution size used to improve corner detection. Don't make this too large.
                conv_size = (11, 11)
    
                #opencv can attempt to improve the checkerboard coordinates
                corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
                cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
                cv.imshow('img', frame)
                k = cv.waitKey(500)
    
                objpoints.append(objp)
                imgpoints.append(corners)
 
 
 
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('distortion coeffs:', dist)
        print('Rs:\n', rvecs)
        print('Ts:\n', tvecs)
    
        self.mtx = mtx
        self.dist = dist

    
    def update(self, video):
        
        ret, frame = video.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mask = cv2.inRange(rgb, self.lower_light_threshold, self.upper_light_threshold)
        
        res = cv2.bitwise_and(frame,frame,mask=mask)

        ny = int(res.shape[1]/10)
        nx = int(res.shape[0]/10)

        rough_location_x = 0
        rough_location_y = 0
        
        loc_count = 0

        for y_large_sweep in range(0,ny):
            
            for x_large_sweep in range(0,nx):
                
                xls_10 = x_large_sweep * 10
                yls_10 = y_large_sweep * 10
                
                if res[xls_10][yls_10][0] >= 1:
                    
                    # show detection grid
                    
                    # res[xls_10][yls_10][0] = 0
                    # res[xls_10][yls_10][1] = 0
                    # res[xls_10][yls_10][2] = 0
                    
                    rough_location_x += xls_10
                    rough_location_y += yls_10
                    
                    loc_count += 1
                    
                    # if the light that we are seeing is close to the current position estimate, chances are that its the light we are looking for, so we increase this measurement's weight
                    if xls_10 >= self.smoothed_location_x - 100 and xls_10 <= self.smoothed_location_x + 100 and yls_10 >= self.smoothed_location_y - 100 and yls_10 <= self.smoothed_location_y + 100:
                        rough_location_x += xls_10*2
                        rough_location_y += yls_10*2
                        
                        loc_count += 2
            
        # if there are bright pixels that we detected      
        if loc_count > 0:
            
            # divide the total number of pixels by the number of times we detected pixels to get a rough location
            # this should be decent enough to get a en estimate of the location
            rough_location_x /= loc_count
            rough_location_y /= loc_count
            
            # create a running average of position measurements
            for num in range(0,9):
                if num < 9:
                    self.positions_y[9-num] = self.positions_y[8-num]
                    self.positions_x[9-num] = self.positions_x[8-num]
            
            self.positions_x[0] = rough_location_x
            self.positions_y[0] = rough_location_y
            
            smoothed_location_x = sum(self.positions_x) / len(self.positions_x)
            smoothed_location_y = sum(self.positions_y) / len(self.positions_y)
            
            # bias the raw position with our smoothed position estimate
            self.positions_x[0] = rough_location_x*0.5 + self.smoothed_location_x*0.5
            self.positions_y[0] = rough_location_y*0.5 + self.smoothed_location_y*0.5
            
            self.r = (((max(self.positions_x) - min(self.positions_x)) ** 2 + (max(self.positions_y) - min(self.positions_y)) ** 2) ** 0.5)*0.5 + self.r*0.5
                
            mask = cv2.circle(res, [int(smoothed_location_y), int(smoothed_location_x)], 20 + int(self.r), [255, 255, 255], 5)
        
        # FPS calculation (does not work well)
        
        # for x in range(0,8):
        #     FPS[9-x]=FPS[9-x-1]
        # FPS[0] = (1/(time.time()-start_time))
        # rlFPS = sum(FPS)/len(FPS)
        # res = cv2.putText(res, f"FPS: {rlFPS}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # print(rlFPS)
        
        # shows number of light points detected
        res = cv2.putText(res, f"light zones: {loc_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        self.video_out = res
    
    