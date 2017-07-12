Usage
=====

The ``extract_features`` tool
-----------------------------

After installation, you will find an ``extract_features`` script installed into your virtualenv. This script is used to batch-extract features from image datasets, using a "pipeline manifest" describing what operations will be performed to extract the features.

Basic Usage
^^^^^^^^^^^

The `feature-extraction` repository contains several example pipelines in ``pipelines/``.

As an example, given a folder ``~/Dataset`` containing images (in this case, TIFFs), we can extract Caffenet features for the dataset by running

.. code-block:: sh

    $ extract_features -o features.json pipelines/caffenet.yml ~/Dataset/*.tif

Pipeline Manifests
^^^^^^^^^^^^^^^^^^

Pipeline manifests are YAML files describing the sequence of operations to be applied to the dataset to generate features.
A pipline consists of:

* Preprocessing steps, such as image normalization or cropping
* Measurement steps, which generate features in the feature vector
* Postprocessing steps, such as feature vector centering and normalization or PCA

This is an example of a pipline used for extracting Caffenet features:

.. code-block:: yaml

    # The measurements block is a list of measurements to be performed.
    # The measurements are concatenated in the order listed here to
    # generate the feature vector.
    measurements:
      - module: Caffenet
        # options for the measurement go here
        caffe_root: ~/build/caffe-rc4

    preprocessing:
      equalize:
        method: 'stretch'
        saturation: 0.2

    postprocessing:
      normalize: yes
      fill_nans: yes
      pca:
        components: 80

Output
^^^^^^

The output is a JSON file containing an object which maps filenames to feature vectors.

If you are exporting to, for instance, ``.npy`` files from this format, it is recommended to impose a consistent filename ordering. *Note:* A script to do this is planned and will be added to the distribution (soon).
