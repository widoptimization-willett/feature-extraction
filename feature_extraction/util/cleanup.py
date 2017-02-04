import numpy as np
import scipy as sp

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
