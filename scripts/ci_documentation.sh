#!/bin/bash
set -ev

CWD=$(pwd)

#*** Generate documentation for the C++ code using Doxygen. ***
doxygen Doxyfile

#*** Generate documentation for Python extension using Sphinx. ***
cd "${CWD}"
cd docs/sphinx
make clean
make html
cd ..
rm -rf cppyml
mv -f sphinx/_build cppyml
#rm -rf sphinx
cd cppyml/html
# Work around Jekyll's unwillingness to serve directories with names
# beginning in underscores.
mv _static static
mv _sources sources
sed -i -e 's/_static\//static\//g' *.html
sed -i -e 's/_sources\//sources\//g' *.html
# Go back to main directory.
cd "${CWD}"
git add docs/cppyml
