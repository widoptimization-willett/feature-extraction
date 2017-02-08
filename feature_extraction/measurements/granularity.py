# coding=utf-8

import numpy as np
import skimage
import skimage.morphology as morph

from . import Measurement
from ..util.cleanup import cell_aoi_and_clip

class Granularity(Measurement):
	default_options = {
		'subsampling': 2, # 2x downsampling

		'element_size': 20,
		'spectrum_length': 16,

		# standard stuff
		'clip_cell_borders': True,
		'cell_border_erosion': None,
	}

	def compute(self, image):
		# -- get mask, potentially clip image
		image, aoi_mask = cell_aoi_and_clip(image, clip=self.options.clip_cell_borders,
										erosion=self.options.cell_border_erosion)

		# -- subsample image/mask
		image = skimage.measure.block_reduce(image,
			(self.options.subsampling, self.options.subsampling))
		aoi_mask = skimage.measure.block_reduce(aoi_mask,
			(self.options.subsampling, self.options.subsampling)) > 0.9 # reconvert to bool after averaging during subsampling

		# -- remove background pixels using a black tophat transform.
		# that is, heavily open the (subsampled) input image with a disk,
		# and then subtract that from the original image.
		bg_image = morph.opening(image, morph.disk(self.options.element_size))
		image -= bg_image
		image[image < 0] = 0 # make sure we don't have negative intensities

		# -- perform "accelerated granulometry"
		# adapted from lines [271-334] of cellprofiler/src/modules/measuregranularity.py

		# structuring/neighborhood element
		elem = np.array([[False, True, False],
                         [True,  True,  True],
                         [False, True, False]])
		# set up an array for recording means at each spectrum step and populate with
		# the initial mean
		means = [max(np.mean(image[aoi_mask]), np.finfo(float).eps)]
		# set up an array for recording {G_k(I)}, k∈{1,spectrum_length}
		G = []

		im_ero = image.copy() # im_ero holds the result of the erosion, which is eroded at each step with ``elem``.
		for i in range(1, self.options.spectrum_length):
			im_ero = morph.erosion(im_ero, elem)
			im_rec = morph.reconstruction(im_ero, image, selem=elem)
			means.append(np.mean(im_rec[aoi_mask]))
			G_k = (means[i-1] - means[i]) * 100 / means[0]
			G.append(G_k)

		return G
