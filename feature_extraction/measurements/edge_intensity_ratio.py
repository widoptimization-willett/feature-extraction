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

			# -- find the ratio of the average intensities between the border and interior of the cell
			intensity_ratio = np.mean(image[border_mask])/np.mean(image[inner_mask])

			intensity_ratio = 0 if intensity_ratio==np.nan else intensity_ratio

			measurements.append(intensity_ratio)

		return measurements
