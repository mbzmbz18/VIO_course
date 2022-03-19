cmake_minimum_required( VERSION 2.8 )
project( eigen_rotation )

set(CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_FLAGS "-std=c++11" )

include_directories("/usr/include/eigen3")

#set(Sophus_DIR "/home/bozhang/slam_projects/vio_course/VIO_course/Sophus-master/build")
#include_directories("/home/bozhang/slam_projects/vio_course/VIO_course/Sophus-master" )
#link_libraries("/home/bozhang/slam_projects/vio_course/VIO_course/Sophus-master/build/libSophus.so")

#list(APPEND CMAKE_PREFIX_PATH "/home/bozhang/Desktop/SLAM_course/Sophus-master")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})


#IF( NOT Sophus_LIBRARIES )
#    SET( Sophus_LIBRARIES "libSophus.so" )
#ENDIF()
#message( STATUS "Sophus library status:" )
message( STATUS "includes: ${Sophus_INCLUDE_DIRS}" )
#message( STATUS "libraries: ${Sophus_LIBRARIES}" )

add_executable(eigen_rotation eigen_rotation.cpp)
target_link_libraries(eigen_rotation ${Sophus_LIBRARIES})