"""Setup script for PyML package.

To install the package, run "python setup.py install" from PythonBindings
directory.
"""
import os
from setuptools import setup, find_packages
import shutil


# Whether to install a Debug version of the binary files.
DEBUG_BINARIES = False

# Package name.
NAME = "PyML"

# Filename templates for required binary files.
BINARY_FILES_TEMPLATES = ["PyML%s.pyd", "ML%s.dll"]

# Names of required binary files.
BINARY_FILES = [tmpl % "" for tmpl in BINARY_FILES_TEMPLATES]


def setup_binary_files():
    setup_dirname = os.path.dirname(__file__)
    package_dirname = os.path.join(setup_dirname, NAME)
    suffix = "-Debug" if DEBUG_BINARIES else "-Release"
    for tmpl, dst_name in zip(BINARY_FILES_TEMPLATES, BINARY_FILES):
        src = os.path.join(package_dirname, tmpl % suffix)
        dst = os.path.join(package_dirname, dst_name)
        shutil.copyfile(src, dst)
        print("Copied %s to %s" % (src, dst))

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
    package_data={NAME: BINARY_FILES}
)