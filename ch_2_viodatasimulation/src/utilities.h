#ifndef IMUSIMWITHPOINTLINE_UTILITIES_H
#define IMUSIMWITHPOINTLINE_UTILITIES_H

#include "imu.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <vector>
#include <fstream>

// 存储空间中的所有三维点的坐标到文件
void save_points(std::string filename, 
                 std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points);

// 存储当前帧相机可以观测到的空间中所有三维点的坐标以及其对应的二维图像坐标到文件
void save_features(std::string filename,
                   std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points,
                   std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> features);

// 存储当前帧相机可以观测到的空间中所有线段(两个端点的坐标)到文件
void save_lines(std::string filename,
                std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> features);

// 从文件中加载IMU数据
void LoadPose(std::string filename, std::vector<MotionData>& pose);

// 存储IMU数据到文件
void save_Pose(std::string filename, std::vector<MotionData> pose);

// 存储IMU数据到文件(TUM格式)
void save_Pose_asTUM(std::string filename, std::vector<MotionData> pose);

#endif //IMUSIMWITHPOINTLINE_UTILITIES_H