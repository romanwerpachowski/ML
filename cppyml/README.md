# cppyml

Optimised Python extension for machine learning based on the [ML++](https://github.com/romanwerpachowski/ML) C++ library.

© 2020-21 Roman Werpachowski.

## Licence

Available under GNU GENERAL PUBLIC LICENSE Version 3.

## Documentation

See [here](https://romanwerpachowski.github.io/ML/cppyml.html).

## How to build and install

### Linux

See `scripts/ci_debug_build.sh` for Debug build and `scripts/ci_release_build.sh` for Release build.

### Windows

1. Add a path to the `libs` directory of a Python distribution to the Visual Studio properties file `LocalDependencies.props` in the "Linker -> General -> Additional Library Directories" section. For example, I use `C:\Users\Roman\anaconda3\libs`.

2. Build all projects in either DebugStatic or ReleaseStatic mode.

3. From the `cppyml` directory, run

   1. `python setup.py install --debug` to install the Python module linked to DebugStatic build of the ML++ library.
   2. `python setup.py install` to install the Python module linked to ReleaseStatic build of the ML++ library.

   Remember to install the `cppyml` package in a Python environment with the same Python version as the one used to build it!