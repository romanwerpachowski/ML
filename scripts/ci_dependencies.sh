#!/bin/bash
set -ev

PACKAGES=$(cat ubuntu_required_packages.txt | tr -s '\n' ' ')

sudo apt-get update
sudo apt-get -y install ${PACKAGES}
