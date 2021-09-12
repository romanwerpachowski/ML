# cppyml

Optimised Python extension for machine learning based on the [ML++](https://github.com/romanwerpachowski/ML) C++ library.

© 2020 Roman Werpachowski.

## Contents

- ordinary least squares (OLS) regression
- recursive OLS regression
- classification and regression decision trees with pruning
- Gaussian E-M clustering

## Documentation

Automatically generated documentation is [here](cppyml/html/index.html)

## Installation

### Linux

Build from source using `SCons` and install using `sudo python setup.py install`:

```bash
scons mode=release
cd cppyml
sudo python setup.py install
```

### Windows


1. Add a path to the `libs` directory of a Python distribution to the Visual Studio properties file `LocalDependencies.props` in the "Linker -> General -> Additional Library Directories" section.
   For example, I use `C:\Users\Roman\anaconda3\libs`.

2. Build the solution in ReleaseStatic mode.

3. From the `cppyml` directory, run `python setup.py install` to install the Python module linked to ReleaseStatic build of the ML++ library.
   Remember to install the `cppyml` package in a Python environment with the same Python version as the one used to build it!


## Example

```python
import numpy as np
from cppyml import linear_regression

n = 25
x = np.random.randn(n)
y = 0.1 * x - 0.9 + 0.2 * np.random.randn(n)
result = linear_regression.univariate(x, y)
```

See also unit tests in `tests`.