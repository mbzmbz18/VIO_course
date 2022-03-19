#ifndef IMUSIM_PARAM_H
#define IMUSIM_PARAM_H

#include <eigen3/Eigen/Core>

// 参数类，生成所有相关的参数
class Param
{
public:
    // 构造函数
    Param();

    // 时间参数
    int imu_frequency = 200; // IMU频率
    int cam_frequency = 30; // 相机频率
    double imu_timestep = 1./imu_frequency;
    double cam_timestep = 1./cam_frequency;
    double t_start = 0.; // 模拟开始时间
    double t_end = 20; // 模拟结束时间

    // IMUBias
    double gyro_bias_sigma = 1.0e-5;
    double acc_bias_sigma = 0.0001;
    // IMU噪声
    double gyro_noise_sigma = 0.015;    // rad/s * 1/sqrt(hz)
    double acc_noise_sigma = 0.019;     //　m/(s^2) * 1/sqrt(hz)
    // 像素噪声
    double pixel_noise = 1;             // 1 pixel noise

    // 相机参数
    double fx = 460;
    double fy = 460;
    double cx = 255;
    double cy = 255;
    double image_w = 640;
    double image_h = 640;

    // IMU相机外参数（cam相对body)
    Eigen::Matrix3d R_bc;
    Eigen::Vector3d t_bc;
};


#endif //IMUSIM_PARAM_H
