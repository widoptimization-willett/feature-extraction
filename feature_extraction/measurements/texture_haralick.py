import numpy as np
import scipy as sp
from . import Measurement
from ..util.cleanup import cell_aoi_and_clip
from skimage.morphology import binary_erosion, disk
from centrosome .haralick import Haralick

class HaralickTexture(Measurement):
	"""
	This Measurement measures the texture of a image or cell by the Haralick method.
	Code is derived from that in CellProfiler; CellProfiler's image library, Centrosome,
	is used to perform the Haralick computations.

	Notes
	-----
	This Measurement accepts the following parameters:

	scale : int
		The scale at which to measure texture correlations.
		This is the distance at which intensities are correlated.
		
		Default: 10 pixels.
	angle : {'average', 'horizontal', 'vertical', 'diagonal', 'antidiagonal'}
		The direction to measure intensity correlation in (that is, the direction
		of the offset to chose pixels).
		If set to ``'average'``, the Haralick texture parameters are calculated for all directions
		and then averaged, which produces a kind of rotational invariance.

		Default: ``'average'``.
	clip_cell_borders : bool
		When ``True``, a RoI mask will be identified of the cell only.
		The Haralick routines will then operate only on this region.

		Default: ``True``
	cell_border_erosion : int or None
		If `clip_cell_borders` is ``True``, the RoI mask will be eroded with a disk of size `cell_border_erosion`.
		May be useful if the cells have edge artifacting.

		Default: `None`
	"""
	default_options = {
		'scale': 10,
		'angle': 'average',

		'clip_cell_borders': True,
		'cell_border_erosion': None,
	}

	def _get_haralick_scales(self, scale, angle):
		return {'vertical': (scale, 0),
				'horizontal': (0, scale),
				'diagonal': (scale, scale),
				'antidiagonal': (scale, -scale)}[angle]

	def compute(self, image):
		# -- preprocessing
		image, aoi_mask = cell_aoi_and_clip(image, clip=self.options.clip_cell_borders,
										erosion=self.options.cell_border_erosion)
		# -- haralick setup and run
		# we're looking at the entire cell's haralick texture parameters,
		# so we'll generate a single label covering the entire AoI
		labels = aoi_mask*1 # convert the boolean mask into a zeros/ones label array

		if self.options.angle == 'average':
			fvecs = []
			for angle in ['vertical', 'horizontal', 'diagonal', 'antidiagonal']:
				scale_i, scale_j = self._get_haralick_scales(self.options.scale,
					angle)

				# hstack since .all() outputs an array of 1-arrays
				fvecs.append(np.hstack(Haralick(image, labels, scale_i, scale_j).all()))
			fvec = np.mean(fvecs, axis=0)
		else:
			scale_i, scale_j = self._get_haralick_scales(self.options.scale,
				self.options.angle)

			# hstack since .all() outputs an array of 1-arrays
			fvec = np.hstack(Haralick(image, labels, scale_i, scale_j).all())

		return fvec
