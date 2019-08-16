#!/bin/bash

mkdir -p ceres
cd ceres
wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz
tar xvzf ceres-solver-1.14.0.tar.gz
cd ceres-solver-1.14.0
sed -i 's/\(^option(BUILD_SHARED_LIBS.*\)OFF/\1ON/' CMakeLists.txt
rm -rf build
mkdir build
cd build
cmake ..
make -j8
mv ceres/ceres-solver-1.14.0/build/lib/libceres* lib/utils/extend_utils/lib

