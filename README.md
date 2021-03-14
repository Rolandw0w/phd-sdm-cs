# PhD research

## Introduction
This project is a part of my PhD research.
It includes several hybrid neural memory models based on [Kanerva's Sparse Distributed Memory (SDM)](https://en.wikipedia.org/wiki/Sparse_distributed_memory).
Core is written in C++ and CUDA.

## Structure
### CUDA-related code
All the low-level operations and GPU-related routines are implemented in C++ under **REPO/core/** path and can be built as a CMake application.
Required CUDA version is 10+.

### Data, analysis, plots.
All the high-level operations, like restructuring input data, Google Cloud Vision API requests, analysing experiments' reports, and plotting are implemented in Python 3.
Required Python version is 3.7+.

## How to
### Set up the environment
1. Clone this repository, to some path (**REPO**)
2. Download  [CIFAR-10 Python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
3. Extract tar.gz/cifar-10-batches-py/* to some **CIFAR_10_path**, ideally to **REPO/data**.
4. Install Python3 libraries
> pip3 install -r **REPO**/py/requirements.txt
   
### Get features from Google Cloud Vision (JSON, readable)
Run
> cd **REPO**/py/features
> python3 get_features.py GCP_CREDENTIALS_PATH --input INPUT_PATH --N N --output OUTPUT

where:
**INPUT_PATH** - absolute path with CIFAR-10 binaries; **CIFAR_10_path** from previous chapter (default is **REPO/data**).

**N** - number of images to pass to Google Vision API (default is **12000**).

**OUTPUT** - absolute path to output JSON file; Google Vision features will be stored there (default is **REPO/data/features/json**).

### Get features from Google Cloud Vision (binary, for further SDM/CUDA processing)
Run
> cd **REPO**/py/features
> python3 process_features.py --input INPUT --score_threshold S --non_zero_images NZI --output O

**INPUT** - path to OUTPUT from the previous step (default is **REPO/data/features.json**).

**S** - lower threshold for a feature (default is **0.7**).

**NZI** - required number of images with at least one feature (default is 9000).

**O** - absolute path for the output binary file with features (default is **REPO/data/features.bin**).
