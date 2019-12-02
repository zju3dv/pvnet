#!/bin/bash
export ROOT_PATH=$1
mkdir -p $ROOT_PATH/ceres
cd $ROOT_PATH/ceres
wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz
tar xvzf ceres-solver-1.14.0.tar.gz
cd ceres-solver-1.14.0
sed -i 's/\(^option(BUILD_SHARED_LIBS.*\)OFF/\1ON/' CMakeLists.txt
rm -rf build
mkdir build
cd build
cmake ..
make -j8
sudo make install
