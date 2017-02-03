import json

import click
import numpy as np
from PIL import Image

from . import extraction
from .measurements.pixelaverage import PixelAverage

# TODO(liam): replace with CLI arguments or a manifest
ENABLED_MEASURMENTS = [PixelAverage()]

@click.command()
@click.argument('files', nargs=-1, required=True) # unlimited number of args can be passed (eg. globbing)
@click.option('-o', '--output', default='features.json')
def extract_features(files, output):
	# TODO(liam): replace this with sending to celery and joining
	processed_files = [] # array of maps containing metadata and feature vectors mapped to files
	for filename in files:
		# -- load data
		_im = Image.open(filename)
		im = np.array(_im) # convert to a numpy array
		assert im.ndim == 2 # rank should be 2 if we're only considering grayscale images

		# -- extract features
		feature_vector = extraction.extract_features(im, ENABLED_MEASURMENTS)

		# -- add the features + metadata to the output list
		processed_files.append({'filename': filename,
			'feature_vector': feature_vector.tolist()}) # do .tolist() since json can't serialize `np.array`s

	json.dump(processed_files, open(output, 'w'), indent=4)
