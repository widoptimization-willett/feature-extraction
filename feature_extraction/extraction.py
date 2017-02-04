import numpy as np

def extract_features(image, measurements):
	"""
	Given an image as a Numpy array and a set of measurement objects
	implementing a compute method returning a feature vector, return a combined
	feature vector.
	"""

	# TODO(liam): parallelize multiple measurements on an image by using Celery
	
	return np.hstack([m.compute(image) for m in measurements])
