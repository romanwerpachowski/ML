#!/bin/bash

sudo apt-get update

for package in `cat ubuntu_required_packages.txt`; do
    sudo apt-get -y install "${package}"
done
