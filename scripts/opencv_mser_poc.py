import click
import numpy as np
import cv2
from skimage.exposure import rescale_intensity
from PIL import Image
from feature_extraction import extraction, pipeline


@click.command()
@click.argument('files', nargs=-1, required=True) # unlimited number of args can be passed (eg. globbing)
def run(files):
	preprocess_options, pipe, postprocess_options = pipeline.construct_from_manifest(open('pipelines/mser_test.yml'))
	for filename in files[0:2]:
		print(filename)
		# im = rescale_intensity(np.array(Image.open(files[0])), 'dtype', 'uint8')
		# assert im.ndim == 2 # rank should be 2 if we're only considering grayscale images
		im = cv2.imread(filename)
		im = extraction.image_preprocessing(im, preprocess_options)
		# im_ = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
		print(im.shape, im.dtype, im.min(), im.max())

		vis = im.copy()

		mser = cv2.MSER_create(_delta=5, _min_area=5)

		regions, _ = mser.detectRegions(im)

		hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
		cv2.polylines(vis, hulls, 1, (0, 255, 0))

		cv2.imshow('im', vis)

		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == '__main__':
	run()
