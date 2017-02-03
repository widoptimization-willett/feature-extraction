import numpy as np

"""
This is an incredibly basic example of a feature-extraction measurement
(simply returning the average of all pixel values in the image).
"""
class PixelAverage(object):
	def compute(self, image):
		return [np.average(image)]
