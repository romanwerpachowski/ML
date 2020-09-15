#!/bin/bash
set -ev

# Target architecture.
ARCH="$1"

#*** Debug build ***
scons -j 2 arch=${ARCH}

#*** C++ unit tests ***
./Tests/build/Debug/${ARCH}/Tests

#*** Deploy Python module locally (the debug version) ***
CWD="${PWD}"
cd cppyml
sudo python3 setup.py install --debug
cd ${CWD}

#*** Python unit tests using the C++ debug build ***
nosetests3 --exe -w cppyml/tests
