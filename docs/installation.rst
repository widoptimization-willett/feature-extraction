.. _installation:

Installation
============

Unfortunately, due to some of the libraries it uses, `feature_extraction` *does not support Python 3*.
Ideally, you should use Python 2.7.

Dependencies
------------
To install `feature_extraction`'s core, in addition to Python 2.x, you need a C compiler. You should probably install Cython from your system package manager.
**If you want to use the Caffe feature measurements, you need to install that first: :ref:`installing-caffe`.**

Installation
------------
Clone the repository:

.. code-block:: sh

	git clone https://github.com/widoptimization-willett/feature-extraction.git
	cd feature-extraction

Create a virtualenv, and activate it:

.. code-block:: sh

	virtualenv -p python2 --system-site-packages venv
	. venv/bin/activate

Install `feature_extraction` in the virtualenv:

.. code-block:: sh

	pip install -e .

Test that `feature_extraction` was successfully installed:

.. code-block:: sh

	# should print help for the extract_features command
	extract_features --help
