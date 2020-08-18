# ML

Efficient C++ implementations of some ML algorithms with Python bindings.

## Contents

- clustering (EM algorithm)
- decision trees for classification and regression
- linear regression

More is coming!

Precompiled Python bindings for Windows are available on [PyPi](https://pypi.org/project/cppyml/).


## Requirements

### Windows

Tested with Visual Studio C++ 2019 and Python 3.7.7.

### Linux

Tested on Ubuntu LTS.

- SCons
- g++
- Python 3.x

### C++ dependencies

Avoid using old Ubuntu packages, prefer getting the source from GitHub.

- pybind11
- googletest
- benchmark (Google C++ benchmarking library)
- Python 3.x header files
- Eigen++
- (Linux only) pthread


### Python dependencies

For build:
- setuptools

For demos and tests:
- numpy
- pandas
- sklearn
- scipy
- unittest

## Installation

### Windows

Python extension 

### Linux

You have to build from source for now, using SCons.

## Licence

Available under GNU GENERAL PUBLIC LICENSE Version 3.
