"""Setup script for cppyml package.

(C) 2020 Roman Werpachowski.

To install the package, run "python setup.py install" from cppyml
directory.

To build a distributable version, run "python setup.py sdist bdist_wheel".
"""
from setuptools.dist import Distribution
from setuptools import setup, find_packages
import argparse
import json
import glob
import os
import sys
import shutil

argparser = argparse.ArgumentParser(add_help=False)
argparser.add_argument("--debug", dest="debug", default=False, action="store_true",
                       help="Use C++ debug build")
args, unknown = argparser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

# Whether to install a Debug version of the binary files.
DEBUG_BINARIES = args.debug

# Name of the C++ build mode.
CPP_BUILD_MODE = "Debug" if DEBUG_BINARIES else "Release"

# Package version.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.json")) as f:
    VERSION = json.load(f)

# Package name.
NAME = "cppyml"

# Directory where setup.py is located.
SETUP_DIRNAME = os.path.abspath(os.path.dirname(__file__))

# Base directory of the whole project.
BASE_DIRECTORY = os.path.abspath(os.path.join(SETUP_DIRNAME, ".."))

# Architecture is 64-bit by default.

# Filenames and paths of binary files needed.
if os.name == "posix":
    PYML_FILENAME = "cppyml.so"   
    PYML_PATH = os.path.join(BASE_DIRECTORY, "cppyml",
                             "build", CPP_BUILD_MODE, PYML_FILENAME)
else:
    PYML_FILENAME = "cppyml.pyd"
    BINARY_DIRECTORY = os.path.join(BASE_DIRECTORY, "x64", CPP_BUILD_MODE + "Static")
    PYML_PATH = os.path.join(BINARY_DIRECTORY, PYML_FILENAME)

    
BINARY_FILENAMES = [PYML_FILENAME]
SRC_PATHS = [PYML_PATH]


def get_binary_files():
    package_dirname = os.path.join(SETUP_DIRNAME, NAME)
    # Remove old files.
    for extension in ["dll", "pyd", "so"]:
        paths = glob.glob(os.path.join(package_dirname, "*.%s" % extension))
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                print("File %s does not exist", path)
            else:
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
    url="https://romanwerpachowski.github.io/ML/",
    author_email="roman.werpachowski@gmail.com",
    license="GPL-3.0",
    keywords="machine-learning ML extension algorithms numerical optimised",
    version=VERSION,
    description="Efficient implementations of selected ML algorithms for Python.",
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    package_data={NAME: PACKAGE_DATA},
    distclass=BinaryDistribution,
)
