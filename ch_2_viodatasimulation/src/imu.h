#ifndef IMUSIMWITHPOINTLINE_IMU_H
#define IMUSIMWITHPOINTLINE_IMU_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <vector>

#include "param.h"

// 运动数据类（针对每一帧）
struct MotionData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp;               // 当前的时间帧

    Eigen::Matrix3d Rwb;            // 当前的旋转
    Eigen::Vector3d twb;            // 当前的平移
    Eigen::Vector3d imu_velocity;   // 当前的速度

    Eigen::Vector3d imu_acc;        // 当前的acc测量值，即body的加速度
    Eigen::Vector3d imu_gyro;       // 当前的gyro测量值，即body系下的角速度
    Eigen::Vector3d imu_acc_bias;   // 当前的acc的bias
    Eigen::Vector3d imu_gyro_bias;  // 当前的gyro的bias
};

// 根据欧拉角得到旋转矩阵，用于转换body下的观察的点到inertial下的点
Eigen::Matrix3d euler2Rotation(Eigen::Vector3d eulerAngles);

// 根据欧拉角得到一个3x3矩阵，用于转换inertial下的欧拉角速度到body下的角速度
Eigen::Matrix3d eulerRates2bodyRates(Eigen::Vector3d eulerAngles);

// IMU类
class IMU
{
public:
    // 构造函数
    IMU(Param p);

    // 生成当前帧IMU运动数据
    MotionData MotionModel(double t);
    // 将当前帧的IMU数据加入相关噪声
    void addIMUnoise(MotionData& data);
    // 测试IMU，对相应的测量值进行积分以得到运动轨迹
    void testImu(std::string src, std::string dist);

    Param param_;   // IMU相关参数

    Eigen::Vector3d gyro_bias_;         // IMU状态量：gyro的bias，会随时间变化
    Eigen::Vector3d acc_bias_;          // IMU状态量：acc的bias，会随时间变化
    Eigen::Vector3d init_twb_;          // 初始平移P
    Eigen::Vector3d init_velocity_;     // 初始速度V
    Eigen::Matrix3d init_Rwb_;          // 初始旋转Q
};

#endif //IMUSIMWITHPOINTLINE_IMU_H
