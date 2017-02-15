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
		# -- find the outer boundary of the cell
		cellmask = cell_boundary_mask(image)

		# -- erode the boundary in by `border_width`
		inner_mask = morph.binary_erosion(cellmask, morph.disk(self.options.border_width))
		
		# -- compute a mask of the border strip between the inner part and outer boundary of the cell
		border_mask = cellmask & ~inner_mask

		# -- find the ratio of the average intensities between the inner part and border of the cell
		intensity_ratio = np.mean(image[inner_mask])/np.mean([border_mask])

		return [intensity_ratio]
