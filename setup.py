from setuptools import setup

setup(
	name='feature-extraction',
	author='Liam Marshall',
	author_email='limarshall@wisc.edu',
	version='0.1',
	license='Apache',

	packages=['feature_extraction'],
	install_requires=['numpy', 'Pillow', 'Click'],

	entry_points='''
	[console_scripts]
	extract_features=feature_extraction.cli:extract_features
	''',
)
