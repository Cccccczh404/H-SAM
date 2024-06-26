# Data preprocessing
## Synapse dataset
The following desription provides the operation of data preprocessing for Synapse dataset, corresponding to the operations in [preprocess_data.py](preprocess_data.py)
1. Please download the Synapse dataset from the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/). Convert them to numpy format, clip the value range to \[-125,275\], normalize the 3D image to \[0-1\], extract the 2D slices from the 3D image for training and store the 3D images as h5 files for inference.
2. According to the [data description](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) of the Synapse dataset, the map of the semantic labels between the original data and the processed data is shown below.

Organ | Label of the original data | Label of the processed data
------------ | -------------|----
spleen | 1 | 1
right kidney | 2 | 2
left kidney | 3 | 3
gallbladder | 4 | 4
liver | 6 | 5
stomach | 7 | 6
aorta | 8 | 7
pancreas | 11 | 8

## LA dataset
The following desription provides the operation of data preprocessing for LA dataset:
1. Please download the LA dataset and related list files from https://github.com/yulequan/UA-MT/tree/master/data.
2. Run [preprocess_LA.py](preprocess_LA.py).

## PROMISE12 dataset
The following desription provides the operation of data preprocessing for PROMISE12 dataset:
1. Please download the preprocessed PROMISE12 dataset from .
2. Run [preprocess_PROMISE12.py](preprocess_PROMISE12.py).
