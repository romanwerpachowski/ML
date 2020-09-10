#!/bin/bash
set -ev

#*** Generate documentation for the C++ code using Doxygen. ***
doxygen Doxyfile

#*** Generate documentation for Python extension using Sphinx. ***
cd docs/sphinx
make clean
make html
cd ..
mv -f sphinx/_build cppyml
cd ..
rm -rf sphinx
