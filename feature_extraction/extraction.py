import numpy as np
import skimage.exposure as exposure
from .util import AttributeDict

from sklearn.decomposition import PCA

def extract_features(image, measurements, debug=False):
	"""
	Given an image as a Numpy array and a set of measurement objects
	implementing a compute method returning a feature vector, return a combined
	feature vector.
	"""

	# TODO(liam): parallelize multiple measurements on an image by using Celery
	
	def trace(m, x):
		"""if the debug flag is set to true, log parameters of each measurement"""
		if debug:
			print("{}: len(x) = {}".format(type(m).__name__, len(x)))

		return x
	return np.hstack([trace(m, m.compute(image)) for m in measurements])

def normalize_features(X):
	# -- recenter features and normalize over the dataset
	X -= np.mean(X, axis=0)

	# normalize where nonzero
	nz = X != 0
	X[nz] /= np.linalg.norm(X[nz], axis=0)

	# normalize for each record
	X /= np.vstack(np.linalg.norm(X, axis=1))

	return X

def feature_postprocessing(X, options):
	_options = AttributeDict({'normalize': True, 'fill_nans': False, 'pca': None})
	_options.update(options or {}); options = _options

	# make sure everything is a float64
	X = X.astype('float64')

	if options.fill_nans:
		X = np.nan_to_num(X)

	if options.normalize:
		X = normalize_features(X)

	if options.pca:
		pca = PCA(n_components=options.pca['components'])
		X = pca.fit_transform(X)

	return X

def image_preprocessing(im, options):
	_options = AttributeDict({'normalize': True, 'equalize': None})
	_options.update(options or {}); options = _options

	if options.normalize:
		im = exposure.rescale_intensity(im)

	if options.equalize:
		if options.equalize['method'] == "histogram":
			im = exposure.equalize_hist(im)
		elif options.equalize['method'] == "stretch":
			pmin, pmax = np.percentile(im,
				(options.equalize['saturation'], 100-options.equalize['saturation']))
			im = exposure.rescale_intensity(im, in_range=(pmin, pmax))

	return im
