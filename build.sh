#!/bin/bash

set -ev

scons -j 2
scons mode=release -j 2
