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

	def _get_haralick_scales(self, scale, angle):
		return {'vertical': (scale, 0),
				'horizontal': (0, scale),
				'diagonal': (scale, scale),
				'antidiagonal': (scale, -scale)}[angle]

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

		if self.options.haralick_angle == 'average':
			fvecs = []
			for angle in ['vertical', 'horizontal', 'diagonal', 'antidiagonal']:
				scale_i, scale_j = self._get_haralick_scales(self.options.haralick_scale,
					angle)

				# hstack since .all() outputs an array of 1-arrays
				fvecs.append(np.hstack(Haralick(image, labels, scale_i, scale_j).all()))
			fvec = np.mean(fvecs, axis=0)
		else:
			scale_i, scale_j = self._get_haralick_scales(self.options.haralick_scale,
				self.options.haralick_angle)

			# hstack since .all() outputs an array of 1-arrays
			fvec = np.hstack(Haralick(image, labels, scale_i, scale_j).all())

		return fvec
