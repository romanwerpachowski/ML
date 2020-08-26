# ML++ and cppyml

ML++ is a set of efficient C++ implementations of some ML algorithms.

cppyml is a Python extension module built on top of M++. Precompiled cppyml binaries for Windows are available on [PyPi](https://pypi.org/project/cppyml/).

## Contents

- clustering (EM algorithm)
- decision trees for classification and regression
- linear regression (incl. recursive)

More is coming!

## Requirements

### Windows

Tested with Visual Studio C++ 2019 and Python 3.7.7.

### Linux

Tested on Ubuntu 18.04.5 LTS.

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

## Building

### Windows binaries

Pre-built Python extension can be installed via `pip install cppyml` (requires Python 3.7.x).

If you want to build from source, build the Visual Studio solution in Release mode.

### Linux binaries

You have to build from source for now, using SCons. Call `scons` in the main project directory
to build in Debug mode, or `scons mode=release` for a Release (optimised) build.

### Doxygen documentation

C++ code has Doxygen-compatible comments. To generate HTML documentation from them, run (from the main project directory):

```bash
cd ML
doxygen
```

(requires `doxygen` and `graphviz` to be installed). The documentation will be written to `ML/docs/html` subdirectory.

## Python extension

Documentation for Python extension project `cppyml` is [here](../cppyml/README.md).