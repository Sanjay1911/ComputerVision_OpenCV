#!/usr/bin/env python3

#reference: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2 as cv
import glob
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


#FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS 
chessboardSize = (8,6)
frameSize = (640,480)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
prev_shape=None
size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

images = glob.glob('PATH_ROSBAG_IMAGES')

for image in images:

    img = cv.imread(image)
    img=cv.resize(img,(2048,1536)) # result.txt file specification
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray_img, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        # refining pixel co-ordinates
        corners2 = cv.cornerSubPix(gray_img, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        savepath = "PATH_CALIBRATION_OUTPUT"
        cv.imwrite(os.path.join(savepath , 'calibresult'+str(image[-5])+'.png'), img)
        cv.imshow('img', img)
        cv.waitKey(1000)


cv.destroyAllWindows()






ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)
#save both the camera matrix and distortion coefficients in a single txt file  without using numpy
with open('PATH_OUTPUT_TEXT_FOLDER', 'w') as f:
    f.write('Camera Matrix:\n')
    f.write(np.array2string(cameraMatrix, separator=', '))
    f.write('\n')
    f.write('Distortion Coefficients:\n')
    f.write(np.array2string(dist, separator=', '))
print("Camera Calibrated: ",ret)
print("Camera Matrix:\n",cameraMatrix)
print("\nDistortion Parameters:\n",dist)
print("Rotation Vector:\n",rvecs)
print("\nTranslation Vectors:\n",tvecs)

#TODO Extrinsic parameters


