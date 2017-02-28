import json

import click
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity

from . import extraction, pipeline


@click.command()
@click.argument('pipeline_manifest', required=True)
@click.argument('files', nargs=-1, required=True) # unlimited number of args can be passed (eg. globbing)
@click.option('-o', '--output', default='features.json')
def extract_features(pipeline_manifest, files, output):
	preprocess_options, pipe, postprocess_options = pipeline.construct_from_manifest(open(pipeline_manifest))
	# TODO(liam): replace this with sending to celery and joining
	X_raw = [] # raw feature matrix
	for filename in tqdm(files):
		# -- load data
		# load with Pillow, convert to a numpy array, rescale to 16 bits of depth
		im = rescale_intensity(np.array(Image.open(filename)), 'dtype', 'uint16')
		assert im.ndim == 2 # rank should be 2 if we're only considering grayscale images

		# -- preprocess
		im = extraction.image_preprocessing(im, preprocess_options)

		# -- extract features
		x = extraction.extract_features(im, pipe)

		# -- add to the feature vector
		X_raw.append(x)

	X = extraction.feature_postprocessing(np.array(X_raw), postprocess_options)

	feature_map = dict(zip(files, 
		map(lambda x: x.tolist(), X)))

	json.dump(feature_map, open(output, 'w'), indent=4)
