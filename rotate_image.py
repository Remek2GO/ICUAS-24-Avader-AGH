import cv2
import numpy as np


# Read the image
image = cv2.imread('/home/vision/Documents/Repositorium/icuas24_avader/images/91_manual_color.png')

image = cv2.imread("/home/vision/Documents/Repositorium/icuas24_avader/test.png")

# Store height and width of the image
height, width = image.shape[:2]

center = (width/2, height/2) 
  
# using cv2.getRotationMatrix2D()  
# to get the rotation matrix 
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=5, scale=1) 
  
# rotate the image using cv2.warpAffine  
# 90 degree anticlockwise 
rotated_image = cv2.warpAffine( 
    src=image, M=rotate_matrix, dsize=(width, height)) 

# Display the original image
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
