import glob
import os
import sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.7"]

module = Extension(
    "PyML", sources = glob.glob(".cpp"),
    include_dirs=["pybind11/include"],
    language="c++",
    extra_compile_args = cpp_args,
    )

setup(
    name = "PyML",
    version = "0.1",
    description = "Python package with machine learning models implemented in C++ (PyBind11)",
    ext_modules = [module],
)