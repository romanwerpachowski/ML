# ML++ and cppyml

ML++ is a set of efficient C++ implementations of some ML algorithms.

cppyml is a Python extension module built on top of M++. Precompiled cppyml binaries for Windows are available on [PyPi](https://pypi.org/project/cppyml/).

Current version number for both projects is 0.3.0

## Contents

- clustering (EM algorithm)
- decision trees for classification and regression
- linear regression (incl. recursive and ridge)

More is coming! See Doxygen documentation for details.

## Requirements

### Windows

Tested with Visual Studio C++ 2019 and Python 3.7.7.

### Linux

Tested with on Ubuntu 20.04.1 LTS (Focal Fossa).
A list of required packages for is in the `ubuntu_required_packages.txt` file.

- SCons >= 4.0.1
- g++ >= 7.5.0
- Python >= 3.6

### C++ dependencies

- pybind11
- benchmark (Google C++ benchmarking library)
- Python 3.x header files
- Eigen++
- (Linux only) pthread


### Python dependencies

For the build:
- setuptools

For demos and tests:
- nose
- numpy
- pandas
- sklearn
- scipy
- unittest

## Building

### Windows binaries

Use the Visual Studio build process.

Pre-built Python extension `cppyml` can be installed via `pip install cppyml` (requires Python 3.7.x).
If you want to build it from source, see [cppyml documentation](cppyml.md).

### Linux binaries

Call `scons` in the main repository directory to build in Debug mode, or `scons mode=release` 
for a Release (optimised) build.

### Doxygen documentation

C++ code has Doxygen-compatible comments. To generate HTML documentation from them, run 
(from the main repository directory):

```bash
cd ML
doxygen
```

(requires `doxygen` and `graphviz` to be installed). The documentation will be written to
`ML/docs/html` subdirectory. A pre-built version of this documentation for the current release
is [here](html/index.html).

## Python extension

Documentation for the Python extension project `cppyml` is [here](cppyml.md).
