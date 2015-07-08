#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import numpy
 
parser = argparse.ArgumentParser()
parser.add_argument("imgpath")
args = parser.parse_args()
 
target_shape = (256, 256)
imgpath=args.imgpath 
print imgpath
img = cv2.imread(imgpath)
height, width, depth = img.shape
output_side_length=256
new_height = output_side_length
new_width = output_side_length
if height > width:
 new_height = output_side_length * height / width
else:
 new_width = output_side_length * width / height
resized_img = cv2.resize(img, (new_width, new_height))
height_offset = (new_height - output_side_length) / 2
width_offset = (new_width - output_side_length) / 2
cropped_img = resized_img[height_offset:height_offset + output_side_length,
width_offset:width_offset + output_side_length]
#cv2.imwrite(output_file, cropped_img)

cv2.imshow("hhoge",cropped_img)
cv2.waitKey(0)

cv2.imwrite("_"+imgpath, cropped_img) 
