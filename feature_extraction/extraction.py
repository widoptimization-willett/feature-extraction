import numpy as np

"""
Given an image as a Numpy array and a set of measurement objects
implementing a compute method returning a feature vector, return a combined
feature vector.
"""
def extract_features(image, measurements):
	# TODO(liam): parallelize multiple measurements on an image by using Celery
	return np.ravel([m.compute(image) for m in measurements])

