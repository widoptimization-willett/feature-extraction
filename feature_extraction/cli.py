import json

import click
import numpy as np
from PIL import Image

from . import extraction, pipeline
from .measurements import PixelAverage, HaralickTexture

@click.command()
@click.argument('pipeline_manifest', required=True)
@click.argument('files', nargs=-1, required=True) # unlimited number of args can be passed (eg. globbing)
@click.option('-o', '--output', default='features.json')
def extract_features(pipeline_manifest, files, output):
	pipe = pipeline.construct_from_manifest(open(pipeline_manifest))
	# TODO(liam): replace this with sending to celery and joining
	processed_files = [] # array of maps containing metadata and feature vectors mapped to files
	for filename in files:
		# -- load data
		im = np.array(Image.open(filename)) # load with Pillow, convert to a numpy array
		assert im.ndim == 2 # rank should be 2 if we're only considering grayscale images

		# -- extract features
		feature_vector = extraction.extract_features(im, pipe)

		# -- add the features + metadata to the output list
		processed_files.append({'filename': filename,
			'feature_vector': feature_vector.tolist()}) # do .tolist() since json can't serialize `np.array`s

	json.dump(processed_files, open(output, 'w'), indent=4)
