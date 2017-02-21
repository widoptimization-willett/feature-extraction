import numpy as np
from . import Measurement
from ..util.cleanup import cell_boundary_mask
import skimage.measure

import matplotlib.pyplot as plt

class EulerNumber(Measurement):
	default_options = {
		'threshold_values': np.linspace(0.1, 0.9, 10)*65535
	}

	def compute(self, image):
		measurements = []
		cellmask = cell_boundary_mask(image)

		for thresh in np.hstack([self.options.threshold_values]):
			im_bin = image > thresh
			regionprops = skimage.measure.regionprops(im_bin*1)
			measurements.append(regionprops[0].euler_number)

		return measurements
