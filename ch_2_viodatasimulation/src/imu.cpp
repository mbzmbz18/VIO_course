#include <random>
#include "imu.h"
#include "utilities.h"

// 根据欧拉角得到旋转矩阵，用于转换body下的观察的点到inertial下的点
Eigen::Matrix3d euler2Rotation(Eigen::Vector3d eulerAngles)
{
    // 相关参数
    double roll = eulerAngles(0);   // roll: r
    double pitch = eulerAngles(1);  // pitch: p
    double yaw = eulerAngles(2);    // yaw: y
    double cr = cos(roll);  double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);
    double cy = cos(yaw);   double sy = sin(yaw);
    // 计算旋转矩阵body->inertial，公式见课件L2-s43
    Eigen::Matrix3d Rib;
    Rib << cy*cp ,   cy*sp*sr - sy*cr,   sy*sr + cy* cr*sp,
           sy*cp,    cy *cr + sy*sr*sp,  sp*sy*cr - cy*sr,
           -sp,      cp*sr,              cp*cr;
    return Rib;
}

// 根据欧拉角得到一个3x3矩阵，用于转换inertial下的欧拉角速度到body下的角速度
Eigen::Matrix3d eulerRates2bodyRates(Eigen::Vector3d eulerAngles)
{
    // 相关参数
    double roll = eulerAngles(0);
    double pitch = eulerAngles(1);
    double cr = cos(roll);  double sr = sin(roll);
    double cp = cos(pitch); double sp = sin(pitch);
    // 计算3x3变换矩阵，用于转换inertial下的欧拉角速度到body下的角速度
    // 公式见课件L2-s44-(31)
    Eigen::Matrix3d transform_matrix;
    transform_matrix << 1,  0,    -sp,
                        0,  cr,   sr*cp,
                        0,  -sr,  cr*cp;
    return transform_matrix;
}

// 构造函数
IMU::IMU(Param p): param_(p)
{
    // 将acc和gyro的Bias均初始化为0
    gyro_bias_ = Eigen::Vector3d::Zero();
    acc_bias_ = Eigen::Vector3d::Zero();
}

// 生成当前帧IMU运动数据
MotionData IMU::MotionModel(double t)
{
    // 初始化当前帧运动数据
    MotionData data;
    // 设置运动方程的一些重要参数
    float ellipse_x = 15;  // 椭圆的x轴
    float ellipse_y = 20;  // 椭圆的y轴，x,y轴做椭圆运动
    float z = 1;           // z轴方向上做正弦运动
    float K1 = 10;         // z轴方向上的正弦频率是x，y的K1倍
    float K = M_PI/10;     // 20*K = 2pi，由于我们采取的模拟时间是20s, 系数K控制yaw正好旋转一圈，运动一周
    double k_roll = 0.1;
    double k_pitch = 0.2;

    // 当前时刻t下，body运动的平移部分（相对世界坐标系）
    // 位置
    Eigen::Vector3d position(ellipse_x * cos(K*t) + 5, ellipse_y * sin(K*t) + 5,  z * sin(K1*K*t) + 5);
    // 位置的导数，即速度
    Eigen::Vector3d dp(-K * ellipse_x * sin(K*t), K * ellipse_y * cos(K*t), z*K1*K * cos(K1*K*t));
    // 位置的导数的导数，即加速度
    double K2 = K*K;
    Eigen::Vector3d ddp(-K2 * ellipse_x * cos(K*t), -K2 * ellipse_y * sin(K*t), -z*K1*K1*K2 * sin(K1*K*t));

    // 当前时刻t下，body运动的旋转部分（相对世界坐标系）
    // 欧拉角，其中roll~[-0.1, 0.1], pitch~[-0.2, 0.2], yaw~[0, 2pi]
    // 欧拉角是从inertial经过给定的三个转角转到body，其具体定义见课件L2-s41,42
    Eigen::Vector3d eulerAngles(k_roll * cos(t), k_pitch * sin(t), K * t);   
    // 欧拉角的导数，即欧拉角速度
    Eigen::Vector3d eulerAnglesRates(-k_roll * sin(t) , k_pitch * cos(t) , K);
    // 首先计算一个3x3矩阵，用于转换inertial下的欧拉角速度到body下的角速度
    // 基于计算的3x3矩阵，从欧拉角速度得到body角速度，即gyro的理论测量值
    Eigen::Vector3d imu_gyro = eulerRates2bodyRates(eulerAngles) * eulerAnglesRates;
    // 导航坐标系东北天ENU中的重力向量
    Eigen::Vector3d gn(0, 0, -9.81); // ENU:(0,0,-9.81) / NED:(0,0,9,81)
    // 从欧拉角得到旋转矩阵，使完成body->world的转换
    Eigen::Matrix3d Rwb = euler2Rotation(eulerAngles);
    // acc的理论测量值，其公式见课件L2-s32-(16)
    Eigen::Vector3d imu_acc = Rwb.transpose() * (ddp - gn);

    // 将计算的运动数据赋值到当前帧
    data.imu_gyro = imu_gyro;   // gyro的测量值(也是理论值，因为没有误差)
    data.imu_acc = imu_acc;     // acc的测量值(也是理论值，因为没有误差)
    data.Rwb = Rwb;             // 旋转
    data.twb = position;        // 平移
    data.imu_velocity = dp;     // 速度
    data.timestamp = t;         // 时间戳
    return data;
}

// 将当前帧的IMU数据加入相关噪声
void IMU::addIMUnoise(MotionData& data)
{
    // 随机数生成器
    std::random_device rd;
    std::default_random_engine generator_(rd());
    std::normal_distribution<double> noise(0.0, 1.0);   // 高斯分布，均值为0.0，方差为1.0

    // 当前时刻t下，加入相关的噪声项
    // 创建随机的gyro高斯白噪声
    Eigen::Vector3d noise_gyro(noise(generator_), noise(generator_), noise(generator_));
    // 创建gyro的协方差矩阵
    Eigen::Matrix3d gyro_sqrt_cov = param_.gyro_noise_sigma * Eigen::Matrix3d::Identity();
    // 将gyro的理论测量值加入相关的噪声，这里包括，gyro的离散高斯白噪声以及gyro的bias
    // 其中，gyro的离散高斯白噪声计算，其公式见课件L2-s26-(11)
    //      gyro的bias默认为上一时刻的bias
    data.imu_gyro = data.imu_gyro + gyro_sqrt_cov * noise_gyro / sqrt(param_.imu_timestep) + gyro_bias_;
    // 创建随机的acc高斯白噪声
    Eigen::Vector3d noise_acc(noise(generator_), noise(generator_), noise(generator_));
    // 创建acc的协方差矩阵
    Eigen::Matrix3d acc_sqrt_cov = param_.acc_noise_sigma * Eigen::Matrix3d::Identity();
    // 将acc的理论测量值加入相关的噪声，这里包括，acc的离散高斯白噪声以及acc的bias
    // 其中，acc的离散高斯白噪声计算，其公式见课件L2-s26-(11)
    //      acc的bias默认为上一时刻的bias
    data.imu_acc = data.imu_acc + acc_sqrt_cov * noise_acc / sqrt( param_.imu_timestep ) + acc_bias_;

    // 更新当前gyro的bias，以便在下一时刻使用
    // 创建随机的gyro的bias的白噪声，即随即游走
    Eigen::Vector3d noise_gyro_bias(noise(generator_), noise(generator_), noise(generator_));
    // 将白噪声加入gyro的bias，其公式见课件L2-s28-(15)
    gyro_bias_ = gyro_bias_ + param_.gyro_bias_sigma * sqrt(param_.imu_timestep ) * noise_gyro_bias;
    // 更新
    data.imu_gyro_bias = gyro_bias_;
    // 更新当前acc的bias，以便在下一时刻使用
    // 创建随机的acc的bias的白噪声，即随即游走
    Eigen::Vector3d noise_acc_bias(noise(generator_), noise(generator_), noise(generator_));
    // 将白噪声加入gyro的bias，其公式见课件L2-s28-(15)
    acc_bias_ = acc_bias_ + param_.acc_bias_sigma * sqrt(param_.imu_timestep ) * noise_acc_bias;
    // 更新
    data.imu_acc_bias = acc_bias_;
}

// 读取通过模拟生成的IMU数据并用IMU动力学模型计算运动轨迹，用来验证数据以及模型的有效性
void IMU::testImu(std::string src, std::string dist)
{
    std::cout << "test by integrating IMU data ..." << std::endl;
    // 加载IMU的所有运动数据
    std::vector<MotionData> all_imudata;
    LoadPose(src, all_imudata);
    // 存储输出的目标位置
    std::ofstream save_points;
    save_points.open(dist);

    // 初始化系统的初始状态，这些系统的状态量在后面会更新
    // 注意：由于没有状态估计系统，acc和gyro的bias是未知的，不列入系统的状态量也不进行估计
    double dt = param_.imu_timestep;        // 时间戳
    Eigen::Vector3d Pwb = init_twb_;        // 平移：系统状态P
    Eigen::Vector3d Vw = init_velocity_;    // 速度：系统状态V
    Eigen::Quaterniond Qwb(init_Rwb_);      // 旋转：系统状态Q
    // 初始化相关参数
    Eigen::Vector3d gw(0, 0, -9.81);        // 重力向量(ENU)
    int cnt_int_pose = 0;                   // 计数
    bool use_euler = false;         // 使用欧拉法/中值法进行数值积分
    std::cout << "number of IMU data: " << all_imudata.size() << std::endl; // 4001，包含初始状态
    
    // 从第1帧开始，遍历所有帧
    for (int i = 1; i < all_imudata.size(); ++i) {

        // 得到当前帧(curr)的时间戳
        double curr_timestamp = all_imudata[i].timestamp;
        // 得到当前帧(curr)的IMU运动数据
        MotionData curr_imudata = all_imudata[i];
        // 得到上一帧(last)的IMU运动数据
        MotionData last_imudata = all_imudata[i-1];

        if (use_euler) {
            // IMU动力学模型：欧拉积分
            // 欧拉积分的公式见课件L2-s37-(24) 
            // 注意：这里只是简单将IMU的测量数据进行积分
            //      由于没有状态估计系统，acc和gyro的bias是未知的，因此可以将它们设置为0，在积分的时候被忽略
            // 首先计算旋转部分的更新量
            Eigen::Vector3d w = last_imudata.imu_gyro; // 需要gyro的测量值，另外gyro的bias设置为0
            Eigen::Vector3d dqi = 0.5 * w * dt;
            Eigen::Quaterniond dq;
            dq.w() = 1;
            dq.x() = dqi.x();
            dq.y() = dqi.y();
            dq.z() = dqi.z();
            dq.normalize(); // 注意：将四元数归一化，只有单位四元数才能代表旋转
            // 其次计算平移部分的更新量
            Eigen::Vector3d a = Qwb * (last_imudata.imu_acc) + gw; // 需要acc的测量值，另外acc的bias设置为0
            // 执行系统的状态量PVQ的更新
            Vw = Vw + a * dt;                           // 更新当前帧(curr)的速度
            Pwb = Pwb + Vw * dt + 0.5 * dt * dt * a;    // 更新当前帧(curr)的平移
            Qwb = Qwb * dq;                             // 更新当前帧(curr)的旋转
        } 
        else {
            // IMU动力学模型：中值积分
            // 中值积分的公式见课件L2-s38-(26) 
            // 注意：这里只是简单将IMU的测量数据进行积分
            //      由于没有状态估计系统，acc和gyro的bias是未知的，因此可以将它们设置为0，在积分的时候被忽略
            // 首先计算旋转部分的更新量
            Eigen::Vector3d w = 0.5 * (curr_imudata.imu_gyro + last_imudata.imu_gyro); // 需要gyro的测量值，
                                                                                       // 另外gyro的bias设置为0
            Eigen::Vector3d dqi = 0.5 * w * dt;
            Eigen::Quaterniond dq;
            dq.w() = 1;
            dq.x() = dqi.x();
            dq.y() = dqi.y();
            dq.z() = dqi.z();
            dq.normalize(); // 注意：将四元数归一化，只有单位四元数才能代表旋转
            // 提前更新旋转Q，以用于中值公式
            Eigen::Quaterniond curr_Qwb = Qwb * dq;
            // 其次计算平移部分的更新量
            Eigen::Vector3d a_curr = curr_Qwb * curr_imudata.imu_acc + gw;  // 需要acc的测量值，另外acc的bias设置为0
            Eigen::Vector3d a_last = Qwb * last_imudata.imu_acc + gw;       // 需要acc的测量值，另外acc的bias设置为0
            Eigen::Vector3d a = 0.5 * (a_curr + a_last);
            // 执行系统的状态量PVQ的更新
            Vw = Vw + a * dt;                           // 更新当前帧(curr)的速度
            Pwb = Pwb + Vw * dt + 0.5 * dt * dt * a;    // 更新当前帧(curr)的平移  
            Qwb = curr_Qwb;                             // 更新当前帧(curr)的旋转
        }

        // 按着IMUquaternion，IMUpostion，Camquaternion, Campostion的格式存储
        // 由于没有Cam，所以IMU存了两次
        save_points << curr_imudata.timestamp << " "
                    << Qwb.w() << " "
                    << Qwb.x() << " "
                    << Qwb.y() << " "
                    << Qwb.z() << " "
                    << Pwb(0) << " "
                    << Pwb(1) << " "
                    << Pwb(2) << " "
                    << Qwb.w() << " "
                    << Qwb.x() << " "
                    << Qwb.y() << " "
                    << Qwb.z() << " "
                    << Pwb(0) << " "
                    << Pwb(1) << " "
                    << Pwb(2) << " "
                    << std::endl;
        // 计数
        cnt_int_pose++;
    }
    std::cout << "number of integrated IMU pose: " << cnt_int_pose << std::endl;
    std::cout << "test is end" << std::endl;
}