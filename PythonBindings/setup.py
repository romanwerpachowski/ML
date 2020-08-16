"""Setup script for PyML package.

To install the package, run "python setup.py install" from PythonBindings
directory.
"""
from setuptools import setup, find_packages


# Whether to install a Debug version of the binary files.
DEBUG_BINARIES = True

# Package name.
NAME = "PyML"

# Binary files.
BINARY_FILES = ["PyML.pyd", "ML.dll"]

setup(
    name=NAME,
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    author="Roman Werpachowski",
    url="https://github.com/romanwerpachowski/ML",
    author_email="roman.werpachowski@gmail.com",
    description="Efficient implementations of selected ML algorithms for Python.",
    package_data={NAME: BINARY_FILES}
)