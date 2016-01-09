import chainer
import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
from chainer.links import caffe
from matplotlib.ticker import * 
from chainer import serializers
import nin

float32=0

model = nin.NIN()
serializers.load_hdf5("gpu0out", model)


def plotD(dim,data):
	size = int(math.ceil(math.sqrt(dim[0])))
	if(len(dim)==4):
		for i,channel in enumerate(data):
			ax = plt.subplot(size,size, i+1)
			ax.xaxis.set_major_locator(NullLocator())
			ax.yaxis.set_major_locator(NullLocator())
			accum = channel[0]
			for ch in channel:
				accum += ch
			accum /= len(channel)
			ax.imshow(accum, interpolation='nearest')
	else:
		plt.imshow(W.data, interpolation='nearest')


def plot(W):
	dim = eval('('+W.label+')')[0]
	size = int(math.ceil(math.sqrt(dim[0])))
	if(len(dim)==4):
		for i,channel in enumerate(W.data):
			ax = plt.subplot(size,size, i+1)
			ax.xaxis.set_major_locator(NullLocator())
			ax.yaxis.set_major_locator(NullLocator())
			accum = channel[0]
			for ch in channel:
				accum += ch
			accum /= len(channel)
			ax.imshow(accum, interpolation='nearest')
	else:
		plt.imshow(W.data, interpolation='nearest')


def showPlot(layer):
	plt.clf()
	W = layer.params().next()
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	fig.suptitle(W.label, fontweight="bold",color="white")
	plot(W)
	plt.show()

def showW(W):
	plt.clf()
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	fig.suptitle(W.label, fontweight="bold",color="white")
	plot(W)
	plt.show()


def getW(layer):
	return layer.params().next()

def savePlot2(layer):
	plt.clf()
	W = layer.params().next()
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	fig.suptitle(W.label, fontweight="bold",color="white")
	plot(W)
	plt.draw()
	plt.savefig(W.label+".png")


def savePlot(W,name):
	plt.clf()
	fig = plt.figure()
	fig.suptitle(name+" "+W.label, fontweight="bold")
	plot(W)
	plt.draw()
	plt.savefig(name+".png")

def layers(model):
	for layer in model.namedparams():
		if layer[0].find("W") > -1:
			print layer[0],layer[1].label
			savePlot(layer[1],layer[0].replace("/","_"))

def layersName(model):
	for layer in model.namedparams():
		print layer[0],layer[1].label

def combine(m1,m2):
	l1i = m1.namedparams()
	for l2 in m2.namedparams():
		l1 = l1i.next()
		l1[1].data = (l1[1].data + l2[1].data ) * 0.5
	return m1





def look(i):
	for o in i:
		print o
		dir(o)

def lookName(i):
	for o in i:
		print o[1].name


plt.gray()
