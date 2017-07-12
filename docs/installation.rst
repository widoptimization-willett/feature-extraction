.. _installation:

Installation
============

Unfortunately, due to some of the libraries it uses, `feature_extraction` *does not support Python 3*.
Ideally, you should use Python 2.7.

Dependencies
------------
To install `feature_extraction`'s core, in addition to Python 2.x, you need a C compiler.
You should also probably install Cython from your system package manager.


**If you want to use the Caffe feature measurements, you need to install that first:** :ref:`installing_caffe`.

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


.. _installing_caffe:

Installing Caffe
----------------

Caffe is a pretty difficult package to install and use. This document
contains some of the pitfalls I’ve run into while using Caffe, and
solutions or workarounds I’ve found.

While you *can* install Caffe from your package manager on Debian and
derived Linux distributions, it’s far easier to debug and correct issues
if you build Caffe from source.

Caffe provides installation guides for various operating systems:

* `Linux (Ubuntu) <http://caffe.berkeleyvision.org/install_apt.html>`_
* `Linux (Debian) <http://caffe.berkeleyvision.org/install_apt_debian.html>`_
* `Linux (Fedora/RHEL/CentOS) <http://caffe.berkeleyvision.org/install_yum.html>`_
* `Linux (macOS) <http://caffe.berkeleyvision.org/install_osx.html>`_
* `Windows <https://github.com/BVLC/caffe/tree/windows>`_ (**unsupported!**)

*Note:* If you're on Windows, it's probably easier to just find a macOS or Linux machine.

One common trip-up is ``Makefile.config``. The options to pay attention
to are:

-  ``CPU_ONLY``: uncomment this option to force Caffe to build without
   GPU support. Useful if you have an unsupported GPU or don’t have CUDA
   installed.

-  ``USE_CUDNN``: uncomment this option if you have both CUDA and cuDNN
   installed from Nvidia’s developer site.

-  ``CUDA_DIR``: point this where you installed CUDA. If you have
   ``USE_CUDNN`` uncommented or ``CPU_ONLY`` commented, make sure this
   is set properly.

-  | ``BLAS``: set this to ``atlas`` if your BLAS library is ATLAS,
     ``mkl`` if your BLAS is MKL, and ``open`` if your BLAS is OpenBLAS.
     If you don’t have a BLAS implementation installed, I recommend
     OpenBLAS.
   | If you installed OpenBLAS with Homebrew, make sure to follow the
     instructions in ``Makefile.configure`` wrt ``BLAS_LIB``.

-  ``PYTHON_INCLUDE``: point this at your Python include directory. It’s
   prefilled with ``/usr/include/python``; if you want to build for only
   Python 2.7, change all references to that to
   ``/usr/include/python2.7``.
