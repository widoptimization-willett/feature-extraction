import os.path

import numpy as np
from . import Measurement

import skimage
import caffe

class Caffenet(Measurement):
	default_options = {
		'caffe_root': os.path.expanduser('~/caffe/'),
		'caffe_mode': 'cpu',
		'layer': 'fc7',
	}

	def __init__(self, options=None):
		super(Caffenet, self).__init__(options)

		if self.options.caffe_mode == 'cpu':
			caffe.set_mode_cpu()
		elif self.options.caffe_mode == 'gpu':
			caffe.set_mode_gpu()

		caffe_root = self.options.caffe_root = os.path.expanduser(self.options.caffe_root)

		# caffenet definition files
		model_definition = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
		model_weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

		# set up the caffe network
		self.net = caffe.Net(model_definition,
						model_weights,
						caffe.TEST)

		mu = np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
		mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

		self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
		self.transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
		self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
		self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

	def compute(self, image):
		image = skimage.exposure.rescale_intensity(image.astype('float64'), image.dtype.name, (0, 1))
		im_rgb = skimage.color.gray2rgb(image)

		# load data
		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im_rgb)
		# run the network forward
		self.net.forward()

		return self.net.blobs[self.options.layer].data[0].copy().ravel()
