#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import numpy
 
parser = argparse.ArgumentParser()
parser.add_argument("source_dir")
parser.add_argument("target_dir")
args = parser.parse_args()
 
target_shape = (256, 256)
output_side_length=256

for source_imgpath in os.listdir(args.source_dir):
  print source_imgpath
  img = cv2.imread(args.source_dir+"/"+source_imgpath)
  height, width, depth = img.shape
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
  cv2.imwrite(args.target_dir+"/"+source_imgpath, cropped_img) 
