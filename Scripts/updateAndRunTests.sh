#!/bin/bash

# To be run in build directory
# Paths to other repos need to be updated

cp ../Tests/BOSSTests.cpp ./BOSS-prefix/src/BOSS/Tests/

cmake --build . -j4

export HAZARD_ADAPTIVE_CONSTANTS=/home/jcp122/repos/BOSSHazardAdaptiveEngine/Source/constants/machineConstantValues.json

./deps/bin/Tests --library ./libBOSSHazardAdaptiveEngine.so [hazard-adaptive-engine]
