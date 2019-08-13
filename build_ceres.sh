#!/bin/bash

mkdir -p ceres
cd ceres
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
sed -i 's/\(^option(BUILD_SHARED_LIBS.*\)OFF/\1ON/' CMakeLists.txt
rm -rf build
mkdir build
cd build
cmake ..
make -j8

