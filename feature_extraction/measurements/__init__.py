from collections import defaultdict
from ..util import AttributeDict

class Measurement(object):
	"""
	A generic feature measurement.

	Attributes
	----------
	default_options
		Can be set by subclasses to set default option values
	"""

	default_options = {}

	def __init__(self, options=None):
		"""
		When initializing this measurement, options can be passed.
		These are exposed to internal algorithms as `self.options`.

		Parameters
		----------
		options : dict
			A dict of options for this measurement.
		"""
		self.options = AttributeDict()
		self.options.update(self.default_options or {})
		self.options.update(options or {})

from .pixelaverage import PixelAverage
from .texture_haralick import HaralickTexture
from .granularity import Granularity
from .edge_intensity_ratio import EdgeIntensityRatio
from .euler_number import EulerNumber
