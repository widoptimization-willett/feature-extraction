import yaml
from . import measurements

"""
A 'pipeline' is just a python list containing a number of instantiated Measurement objects.

This module can load them from a YAML manifest.
"""

def construct_from_manifest(file):
	"""
	Construct a pipeline from a YAML manifest

	Parameters
	----------
	file : file
		An open file handle to a YAML process manifest

	Returns
	-------
	pipeline : list
		A list of instantiated Measurement objects.
	postprocess_options : dict
		Options for the postprocessor
	"""

	data = yaml.safe_load(file)

	pipeline = []
	for m in data['measurements']:
		class_name = m['module']
		options = m; options.pop('module')
		_class = getattr(measurements, class_name)

		pipeline.append(_class(options)) # instantiate w/ options, append to pipeline


	preprocess_options = data.get('preprocessing', {})
	postprocess_options = data.get('postprocessing', {})

	return preprocess_options, pipeline, postprocess_options
