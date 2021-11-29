#!/bin/bash
set -ev

#*** Release build ***
export TERM="xterm"
scons mode=release -j 2

#*** Deploy Python module locally ***
CWD="${PWD}"
cd cppyml
sudo python3 setup.py install bdist_wheel
# Copy the single generated wheel file to known name.
cp dist/*.whl ${CWD}/cppyml.whl
cd ${CWD}

#*** Python unit tests using the C++ release build ***
nosetests3 --exe -w cppyml/tests

#*** Python demos ***
for demo in Demo/*.py; do python3 $demo; done
