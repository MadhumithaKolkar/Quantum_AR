#Created by : Madhumitha Kolkar 2024

import cv2
import numpy as np

#Getting the Image ready for feature detection
input_image = cv2.imread('The_kid_who_came_from_space.jpg')
input_image = cv2.resize(input_image, (400,550),interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# Initiate ORB object
orb = cv2.ORB_create(nfeatures=1000)

# find the keypoints with ORB
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# draw only the location of the keypoints without size
final_keypoints = cv2.drawKeypoints(gray_image, keypoints,input_image,(0,255,0))

cv2.imshow('ORB keypoints', final_keypoints)
cv2.waitKey()