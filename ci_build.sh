#!/bin/bash

set -ev

#*** Debug build ***
scons -j 2

#*** Release build ***
scons mode=release -j 2

#*** Unit tests ***
./Tests/build/Debug/Tests

#*** Demos ***
./Demo/build/Release/Demo

#*** Benchmarks ***
./Benchmarks/build/Release/Benchmarks