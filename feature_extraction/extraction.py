import numpy as np
import skimage.exposure as exposure
from .util import AttributeDict

def extract_features(image, measurements):
	"""
	Given an image as a Numpy array and a set of measurement objects
	implementing a compute method returning a feature vector, return a combined
	feature vector.
	"""

	# TODO(liam): parallelize multiple measurements on an image by using Celery
	
	return np.hstack([m.compute(image) for m in measurements])

def normalize_features(X):
	# recenter features and normalize over the dataset
	X -= np.mean(X, axis=0)
	X /= np.linalg.norm(X, axis=0)

	# normalize for each record
	X /= np.vstack(np.linalg.norm(X, axis=1))

	return X

def feature_postprocessing(X, options):
	_options = AttributeDict({'normalize': True, 'fill_nans': False})
	_options.update(options or {}); options = _options

	if options.fill_nans:
		X = np.nan_to_num(X)

	if options.normalize:
		X = normalize_features(X)

	return X

def image_preprocessing(im, options):
	_options = AttributeDict({'normalize': True, 'equalize': None})
	_options.update(options or {}); options = _options

	if options.normalize:
		im = exposure.rescale_intensity(im)

	print options

	if options.equalize:
		if options.equalize['method'] == "histogram":
			im = exposure.equalize_hist(im)
		elif options.equalize['method'] == "stretch":
			pmin, pmax = np.percentile(im,
				(options.equalize['saturation'], 100-options.equalize['saturation']))
			im = exposure.rescale_intensity(im, in_range=(pmin, pmax))

	return im
