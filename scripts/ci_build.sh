#!/bin/bash
set -ev

#*** Debug build ***
scons -j 2

#*** Release build ***
scons mode=release -j 2

#*** C++ unit tests ***
./Tests/build/Debug/Tests

#*** C++ demos ***
./Demo/build/Release/Demo

#*** Deploy Python module locally ***
CWD="${PWD}"
cd cppyml
sudo python3 setup.py install
cd ${CWD}

#*** Python unit tests ***
nosetests3 --exe -w cppyml/tests

#*** Benchmarks ***
./Benchmarks/build/Release/Benchmarks
