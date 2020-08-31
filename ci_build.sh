#!/bin/bash

set -ev

# Build.
scons -j 2
scons mode=release -j 2

# Test.
./Tests/build/Debug/Tests

# Benchmark.
./Benchmarks/build/Release/Benchmarks