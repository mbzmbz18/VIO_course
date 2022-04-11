cmake_minimum_required(VERSION 2.8)
project(curve_fitting)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++14 -O0")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

add_executable(gaussNewton main_gaussnewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})