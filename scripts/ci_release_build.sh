#!/bin/bash
set -ev

#*** Release build ***
scons mode=release -j 2

#*** C++ demos ***
./Demo/build/Release/Demo

#*** Deploy Python module locally ***
CWD="${PWD}"
cd cppyml
sudo python3 setup.py install
cd ${CWD}

#*** Python unit tests using the C++ release build ***
nosetests3 --exe -w cppyml/tests

#*** Python demos ***
for demo in Demo/*.py; do python3 $demo; done

#*** Benchmarks ***
./Benchmarks/build/Release/Benchmarks
