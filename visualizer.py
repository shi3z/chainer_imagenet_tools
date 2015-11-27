import chainer
import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
from chainer.links import caffe
from matplotlib.ticker import * 
import matplotlib.image as mpimg

float32=0


def plot(layer):
	dim = eval('('+layer.W.label+')')[0]
	size = int(math.ceil(math.sqrt(dim[0])))
	if(len(dim)==4):
		for i,channel in enumerate(layer.W.data):
			ax = plt.subplot(size,size, i)
			ax.xaxis.set_major_locator(NullLocator())
			ax.yaxis.set_major_locator(NullLocator())
			accum = channel[0]
			for ch in channel:
				accum += ch
			accum /= len(channel)
			ax.imshow(accum, interpolation='nearest')
	else:
		plt.imshow(layer.W.data, interpolation='nearest')


def showPlot(layer):
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	fig.suptitle(layer.W.label, fontweight="bold",color="white")
	plot(layer)
	plt.show()


def savePlot(layer,name):
	fig = plt.figure()
	fig.suptitle(name+" "+layer.W.label, fontweight="bold")
	plot(layer)
	plt.draw()
	plt.savefig(name+".png")
#	mpimg.imsave(name+".png",img)

def save(func):
	for candidate in func.layers:
		if(candidate[0]) in dir(func):
			name=candidate[0]
			savePlot(func[name],name)



