import numpy as np
import scipy as sp
from . import Measurement
import feature_extraction.util.cleanup as cleanup
from skimage.morphology import binary_erosion, disk
from centrosome .haralick import Haralick

class HaralickTexture(Measurement):
	default_options = {
		'haralick_scale': 10,
		'haralick_angle': 'average',

		'clip_cell_borders': True,
		'erode_cell': False,
		'erode_cell_amount': False,
	}
	def __init__(self, options=None):
		super(HaralickTexture, self).__init__(options)

	def compute(self, image):
		# -- preprocessing
		# by default, the AoI mask will include the whole image
		mask = np.ones_like(image).astype(bool)
		if self.options.clip_cell_borders:
			# get the cell boundary mask
			mask = cleanup.cell_boundary_mask(image)

			# if we're told to, erode the mask with a disk by some amount
			if self.options.erode_cell:
				mask = binary_erosion(cleanup.cell_boundary_mask(), disk(self.options.erode_cell_amount))

			# mask the image
			# TODO(liam): we can probably use scipy.MaskedArray to get a speedup here
			image = image.copy()
			image[~mask] = 0 # set everything *outside* the cell to 0

		# -- haralick setup and run
		# we're looking at the entire cell's haralick texture parameters,
		# so we'll generate a single label covering the entire AoI
		labels = mask*1 # convert the boolean mask into a zeros/ones label array

		return []
