class Measurement(object):
	"""
	A generic feature measurement.
	"""

	def __init__(self, options=None):
		"""
		When initializing this measurement, options can be passed.
		These are exposed to internal algorithms as `self.options`.
		"""
		self.options = options

from .pixelaverage import PixelAverage
from .texture_haralick import HaralickTexture
