#!/bin/bash
set -ev

PACKAGES=$(cat "$1" | tr -s '\n' ' ')

sudo apt-get update
sudo apt-get -y install ${PACKAGES}
