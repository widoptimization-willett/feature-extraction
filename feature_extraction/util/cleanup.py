import numpy as np
import scipy as sp
import skimage.morphology as morphology

def cell_boundary_mask(image):
	"""
	Identifies the clipping boundary of a cell image.
	Returns this as a mask, where True corresponds to "inside the cell".

	This is done by finding a mask that is True where image != 0
	(as the clipped area will be perfectly zero).
	Imperfections in imaging or low gain may cause internal zeros;
	these are removed by using scipy.ndimage.binary_fill_holes().
	"""

	cellmask = (image != 0)
	return sp.ndimage.binary_fill_holes(cellmask)

def cell_aoi_and_clip(image, clip=False, erosion=None):
	# by default, the AoI mask will include the whole image
	if not clip:
		mask = np.ones_like(image).astype(bool)
		return image, mask

	mask = cell_boundary_mask(image)
	if erosion:
		# if we're told to, erode the mask with a disk of given size
		mask = morphology.binary_erosion(mask, morphology.disk(erosion))

	# mask the image
	# TODO(liam): we can probably use scipy.MaskedArray to get a speedup here
	image = image.copy()
	image[~mask] = 0 # set everything *outside* the cell to 0

	return image, mask
