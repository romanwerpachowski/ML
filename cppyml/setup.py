"""Setup script for cppyml package.

To install the package, run "python setup.py install" from cppyml
directory.

To build a distributable version, run "python setup.py sdist bdist_wheel".
"""
import glob
import os
from setuptools import setup, find_packages
from setuptools.dist import Distribution
import shutil


# Whether to install a Debug version of the binary files.
DEBUG_BINARIES = False

# Name of the C++ build mode.
CPP_BUILD_MODE = "Debug" if DEBUG_BINARIES else "Release"

# Package name.
NAME = "cppyml"

# Directory where setup.py is located.
SETUP_DIRNAME = os.path.abspath(os.path.dirname(__file__))

# Base directory of the whole project.
BASE_DIRECTORY = os.path.abspath(os.path.join(SETUP_DIRNAME, ".."))

# Filenames and paths of binary files needed.
if os.name == "posix":
    PYML_FILENAME = "cppyml.so"
    PYML_PATH = os.path.join(BASE_DIRECTORY, "cppyml", "build", CPP_BUILD_MODE, PYML_FILENAME)
    BINARY_FILENAMES = [PYML_FILENAME]
    SRC_PATHS = [PYML_PATH]
else:
    ML_FILENAME = "ML.dll"
    PYML_FILENAME = "cppyml.pyd"
    BINARY_DIRECTORY = os.path.join(BASE_DIRECTORY, "x64", CPP_BUILD_MODE)
    ML_PATH = os.path.join(BINARY_DIRECTORY, ML_FILENAME)
    PYML_PATH = os.path.join(BINARY_DIRECTORY, PYML_FILENAME)
    BINARY_FILENAMES = [ML_FILENAME, PYML_FILENAME]
    SRC_PATHS = [ML_PATH, PYML_PATH]


def get_binary_files():
    package_dirname = os.path.join(SETUP_DIRNAME, NAME)
    # Remove old files.
    for extension in ["dll", "pyd", "so"]:
        paths = glob.glob(os.path.join(package_dirname, "*.%s" % extension))
        for path in paths:
            os.remove(path)
            print("Removed %s" % path)
    for src_path, dst_filename in zip(SRC_PATHS, BINARY_FILENAMES):
        dst_path = os.path.join(package_dirname, dst_filename)
        shutil.copyfile(src_path, dst_path)
        print("Copied %s to %s" % (src_path, dst_path))


def load_readme():
    with open(os.path.join(SETUP_DIRNAME, "README.md"), "r") as f:
        return f.read()


get_binary_files()

PACKAGE_DATA = BINARY_FILENAMES


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

setup(
    name=NAME,
    author="Roman Werpachowski",
    url="https://github.com/romanwerpachowski/ML",
    author_email="roman.werpachowski@gmail.com",    
    license="GPL-3.0",
    keywords="machine-learning ML extension algorithms numerical optimised",   
    version="0.1.2",
    description="Efficient implementations of selected ML algorithms for Python.",
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    package_data={NAME: PACKAGE_DATA},
    distclass=BinaryDistribution,
)
