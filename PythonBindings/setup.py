"""Setup script for PyML package.

To install the package, run "python setup.py install" from PythonBindings
directory.
"""
import os
from setuptools import setup, find_packages
import shutil


# Whether to install a Debug version of the binary files.
DEBUG_BINARIES = False

# Name of the C++ build mode.
CPP_BUILD_MODE = "Debug" if DEBUG_BINARIES else "Release"

# Package name.
NAME = "PyML"

# Directory where setup.py is located.
SETUP_DIRNAME = os.path.dirname(__file__)

# Base directory of the whole project.
BASE_DIRECTORY = os.path.abspath(os.path.join(SETUP_DIRNAME, ".."))

# Filenames and paths of binary files needed.
if os.name == "posix":
    ML_FILENAME = "libML.so"
    PYML_FILENAME = "PyML.so"
    ML_PATH = os.path.join(BASE_DIRECTORY, "ML", "build", CPP_BUILD_MODE, ML_FILENAME)
    PYML_PATH = os.path.join(BASE_DIRECTORY, "PythonBindings", "build", CPP_BUILD_MODE, PYML_FILENAME)
else:
    ML_FILENAME = "ML.dll"
    PYML_FILENAME = "PyML.pyd"
    BINARY_DIRECTORY = os.path.join(BASE_DIRECTORY, "x64", CPP_BUILD_MODE)
    ML_PATH = os.path.join(BINARY_DIRECTORY, ML_FILENAME)
    PYML_PATH = os.path.join(BINARY_DIRECTORY, PYML_FILENAME)
BINARY_FILENAMES = [ML_FILENAME, PYML_FILENAME]

def setup_binary_files():
    package_dirname = os.path.join(SETUP_DIRNAME, NAME)
    src_paths = [ML_PATH, PYML_PATH]
    dst_filenames = BINARY_FILENAMES 
    for src_path, dst_filename in zip(src_paths, dst_filenames):
        dst_path = os.path.join(package_dirname, dst_filename)
        shutil.copyfile(src_path, dst_path)
        print("Copied %s to %s" % (src_path, dst_path))

setup_binary_files()

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
    package_data={NAME: BINARY_FILENAMES}
)