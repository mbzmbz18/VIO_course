#ifndef IMUSIMWITHPOINTLINE_IMU_H
#define IMUSIMWITHPOINTLINE_IMU_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <vector>

#include "param.h"

// 运动数据类，针对每一帧
struct MotionData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp;

    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;

    Eigen::Vector3d imu_acc;
    Eigen::Vector3d imu_gyro;
    Eigen::Vector3d imu_gyro_bias;
    Eigen::Vector3d imu_acc_bias;

    Eigen::Vector3d imu_velocity;
};

// euler2Rotation:   body frame to interitail frame
Eigen::Matrix3d euler2Rotation(Eigen::Vector3d eulerAngles);
Eigen::Matrix3d eulerRates2bodyRates(Eigen::Vector3d eulerAngles);

// IMU类
class IMU
{
public:
    // 构造函数
    IMU(Param p);

    // IMU参数
    Param param_;

    // IMUBias
    Eigen::Vector3d gyro_bias_;
    Eigen::Vector3d acc_bias_;

    // 初始状态
    Eigen::Vector3d init_velocity_;
    Eigen::Vector3d init_twb_;
    Eigen::Matrix3d init_Rwb_;

    // 函数：运动模型
    MotionData MotionModel(double t);
    // 函数：加入噪声
    void addIMUnoise(MotionData& data);
    // 函数：测试IMU
    void testImu(std::string src, std::string dist); // imu数据进行积分，用来看imu轨迹

};

#endif //IMUSIMWITHPOINTLINE_IMU_H
