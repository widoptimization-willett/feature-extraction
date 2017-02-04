import numpy as np
from . import Measurement

class PixelAverage(Measurement):
	"""
	This is an incredibly basic example of a feature-extraction measurement.

	Returns
	-------
	np.array
		Contains a single element equal to the average of all pixel values in the image.
	"""

	def compute(self, image):
		return [np.average(image)]
