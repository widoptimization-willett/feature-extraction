import os
from setuptools import setup

if os.environ.get('READTHEDOCS'):
	INSTALL_REQUIRES = []
else:
	INSTALL_REQUIRES = ['PyYAML', 'numpy', 'Pillow', 'Click', 'scikit-image', 'centrosome']

setup(
	name='feature-extraction',
	author='Liam Marshall',
	author_email='limarshall@wisc.edu',
	version='0.1',
	license='Apache',

	packages=['feature_extraction'],
	install_requires=INSTALL_REQUIRES,

	entry_points='''
	[console_scripts]
	extract_features=feature_extraction.cli:extract_features
	''',
)
