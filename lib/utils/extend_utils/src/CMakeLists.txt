cmake_minimum_required(VERSION 3.5)
project(pnp_solver)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH /home/liuyuan/software/lib/cmake/Ceres)
set(Ceres_DIR /home/liuyuan/software/lib/cmake/Ceres)
find_package(Ceres REQUIRED)
if(NOT Ceres_FOUND)
    message(ERROR "ceres not found")
else(NOT Ceres_FOUND)
    message(STATUS ${CERES_VERSION})
    message(STATUS ${CERES_LIBRARIES})
endif(NOT Ceres_FOUND)

find_package(Eigen)

message(STATUS ${EIGEN_VERSION})
message(STATUS ${EIGEN_INCLUDE_DIRS})
message(STATUS ${EIGEN_LIBRARIES})

include_directories(/home/liuyuan/software/include ${EIGEN_INCLUDE_DIRS})

add_executable(pnp_solver main.cpp)
target_link_libraries(pnp_solver
        /home/liuyuan/software/lib/libceres.so
        /usr/lib/x86_64-linux-gnu/libglog.so)