# encoding: utf8

import numpy as np
from . import Measurement
from ..util.cleanup import cell_boundary_mask
import skimage.morphology as morph

import matplotlib.pyplot as plt

class EdgeIntensityRatio(Measurement):
	default_options = {
		'border_width': 10 # pixels
	}

	def compute(self, image):
		measurements = []
		for width in np.hstack([self.options.border_width]):
			# -- find the outer boundary of the cell
			cellmask = cell_boundary_mask(image)

			# -- erode the boundary in by ``width``
			inner_mask = morph.binary_erosion(cellmask, morph.disk(width))
			
			# -- compute a mask of the border strip between the inner part and outer boundary of the cell
			border_mask = cellmask & ~inner_mask

			# if the inner mask is empty
			if np.sum(inner_mask) == 0:
				measurements.append(0)
				continue

			# -- find the ratio of the average intensities between the border and interior of the cell
			intensity_ratio = np.mean(image[border_mask])/np.mean(image[inner_mask])

			measurements.append(intensity_ratio)

		return measurements
