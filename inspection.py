#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time

import numpy as np
from PIL import Image


import six
#import six.moves.cPickle as pickle
import cPickle as pickle
from six.moves import queue

import chainer
import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from matplotlib.ticker import * 
from chainer import serializers



parser = argparse.ArgumentParser(
    description='Image inspection using chainer')
parser.add_argument('image', help='Path to inspection image file')
parser.add_argument('--model','-m',default='model', help='Path to model file')
parser.add_argument('--mean', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
args = parser.parse_args()


def read_image(path, center=False, flip=False):
  image = np.asarray(Image.open(path)).transpose(2, 0, 1)
  if center:
    top = left = cropwidth / 2
  else:
    top = random.randint(0, cropwidth - 1)
    left = random.randint(0, cropwidth - 1)
  bottom = model.insize + top
  right = model.insize + left
  image = image[:, top:bottom, left:right].astype(np.float32)
  image -= mean_image[:, top:bottom, left:right]
  image /= 255
  if flip and random.randint(0, 1) == 0:
    return image[:, :, ::-1]
  else:
    return image

import nin

mean_image = pickle.load(open(args.mean, 'rb'))


model = nin.NIN()
serializers.load_hdf5("gpu1out.h5", model)
cropwidth = 256 - model.insize
model.to_cpu()


def predict(net, x):
    h = F.max_pooling_2d(F.relu(net.mlpconv1(x)), 3, stride=2)
    h = F.max_pooling_2d(F.relu(net.mlpconv2(h)), 3, stride=2)
    h = F.max_pooling_2d(F.relu(net.mlpconv3(h)), 3, stride=2)
    h = net.mlpconv4(F.dropout(h, train=net.train))
    h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))
    return F.softmax(h)

#setattr(model, 'predict', predict)

img = read_image(args.image)
x = np.ndarray(
        (1, 3, model.insize, model.insize), dtype=np.float32)
x[0]=img
x = chainer.Variable(np.asarray(x), volatile='on')

score = predict(model,x)
#score=cuda.to_cpu(score.data)

categories = np.loadtxt("labels.txt", str, delimiter="\t")

top_k = 20
prediction = zip(score.data[0].tolist(), categories)
prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
for rank, (score, name) in enumerate(prediction[:top_k], start=1):
    print('#%d | %s | %4.1f%%' % (rank, name, score * 100))

