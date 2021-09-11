#!/bin/bash
set -ev

#*** Debug build ***
export TERM="xterm"
scons -j 2

#*** C++ unit tests ***
./Tests/build/Debug/Tests

#*** Deploy Python module locally (the debug version) ***
CWD="${PWD}"
cd cppyml
sudo python3 setup.py install --debug
cd ${CWD}

#*** Python unit tests using the C++ debug build ***
nosetests3 --exe -w cppyml/tests
