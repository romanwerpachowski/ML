#!/bin/bash

scons ML Tests -j 2
scons ML Benchmarks mode=release -j 2
