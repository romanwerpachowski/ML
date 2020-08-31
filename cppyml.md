# cppyml

Optimised Python extension for machine learning based on the [ML++](https://github.com/romanwerpachowski/ML) C++ library.

(C) 2020 Roman Werpachowski.

## Contents

- ordinary least squares (OLS) regression
- recursive OLS regression
- classification and regression decision trees with pruning
- Gaussian E-M clustering

## Installation

### Linux

Build from source using `SCons` and install using `python setup.py install`:

```bash
scons mode=release
cd cppyml
sudo python setup.py install
```

### Windows

Precompiled binaries are available on [PyPi](https://pypi.org/project/cppyml/).
Install them using `pip install cppyml`.

To build from source under Windows, build the solution in Release mode and then do:
```bash
cd cppyml
python setup.py install
```

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