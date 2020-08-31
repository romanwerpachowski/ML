#!/bin/bash
set -ev

#*** Debug build ***
scons -j 2

#*** C++ unit tests ***
./Tests/build/Debug/Tests