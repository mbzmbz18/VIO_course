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
    data.imu_gyro = imu_gyro;   // 当前gyro的测量值(理想，因为没有噪声)
    data.imu_acc = imu_acc;     // 当前acc的测量值(理想，因为没有噪声)
    data.Rwb = Rwb;             // 当前旋转
    data.twb = position;        // 当前平移
    data.imu_velocity = dp;     // 当前速度
    data.timestamp = t;         // 当前时间戳
    return data;
}

// 将当前帧的IMU数据加入相关噪声
void IMU::addIMUnoise(MotionData& data)
{
    // 随机数生成器
    std::random_device rd;
    std::default_random_engine generator_(rd());
    std::normal_distribution<double> noise(0.0, 1.0);   // 高斯分布，均值为0.0，方差为1.0
    // 当前时刻t下，下面将IMU的数据加入相关的噪声项

    // Step 1: 当前gyro的测量值(理想)加入当前的高斯白噪声以及上一时刻gyro的bias，得到当前时刻gyro的测量值(真实)
    // 创建N(0,1)的随机高斯白噪声(3x1)
    Eigen::Vector3d white_noise1(noise(generator_), noise(generator_), noise(generator_));
    // 创建gyro高斯白噪声的协方差矩阵(3x3)
    Eigen::Matrix3d gyro_sqrt_cov = param_.gyro_noise_sigma * Eigen::Matrix3d::Identity();
    // 计算离散情况下gyro的高斯白噪声，其公式见课件L2-26-(10)
    Eigen::Vector3d noise_gyro = gyro_sqrt_cov / sqrt(param_.imu_timestep) * white_noise1;
    // 将gyro的理论测量值加入相关的噪声，这里包括，gyro的离散高斯白噪声以及gyro的bias
    //      gyro的bias默认为上一时刻的bias，即此时该bias还未更新
    data.imu_gyro = data.imu_gyro + noise_gyro + gyro_bias_;

    // Step 2: 当前acc的测量值(理想)加入当前的高斯白噪声以及上一时刻acc的bias，得到当前时刻acc的测量值(真实)
    // 创建N(0,1)的随机高斯白噪声(3x1)
    Eigen::Vector3d white_noise2(noise(generator_), noise(generator_), noise(generator_));
    // 创建acc高斯白噪声的协方差矩阵(3x3)
    Eigen::Matrix3d acc_sqrt_cov = param_.acc_noise_sigma * Eigen::Matrix3d::Identity();
    // 计算离散情况下acc的高斯白噪声，其公式见课件L2-26-(10)
    Eigen::Vector3d noise_acc = acc_sqrt_cov / sqrt(param_.imu_timestep) * white_noise2;
    // 将acc的理论测量值加入相关的噪声，这里包括，acc的离散高斯白噪声以及acc的bias
    //      acc的bias默认为上一时刻的bias，即此时该bias还未更新
    data.imu_acc = data.imu_acc + noise_acc + acc_bias_;

    // Step 3: 上一时刻gyro的bias加入bias的随机游走，得到更新当前时刻gyro的bias
    // 创建N(0,1)的随机高斯白噪声(3x1)
    Eigen::Vector3d white_noise3(noise(generator_), noise(generator_), noise(generator_));
    // 计算gyro的bias的随机游走，其公式见课件L2-28-(14)
    Eigen::Vector3d bias_change_gyro = param_.gyro_bias_sigma * sqrt(param_.imu_timestep) * white_noise3;
    // 计算并更新gyro的bias，其公式见课件L2-s28-(14)
    gyro_bias_ = gyro_bias_ + bias_change_gyro;
    data.imu_gyro_bias = gyro_bias_;

    // Step 4: 上一时刻acc的bias加入bias的随机游走，得到更新当前时刻acc的bias
    // 创建N(0,1)的随机高斯白噪声(3x1)
    Eigen::Vector3d white_noise4(noise(generator_), noise(generator_), noise(generator_));
    // 计算acc的bias的随机游走，其公式见课件L2-28-(14)
    Eigen::Vector3d bias_change_acc = param_.acc_bias_sigma * sqrt(param_.imu_timestep) * white_noise4;
    // 计算并更新acc的bias，其公式见课件L2-s28-(14)
    acc_bias_ = acc_bias_ + bias_change_acc;
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

        // Step 0: 提取相关IMU数据
        // 得到当前帧(curr)的时间戳
        double curr_timestamp = all_imudata[i].timestamp;
        // 得到当前帧(curr)的IMU运动数据
        MotionData curr_imudata = all_imudata[i];
        // 得到上一帧(last)的IMU运动数据
        MotionData last_imudata = all_imudata[i-1];

        if (use_euler) {
            // Step 1: IMU动力学模型：欧拉积分
            // 欧拉积分的公式见课件L2-s37-(24) 
            // 注意：这里只是简单将IMU的测量数据进行积分
            //      由于没有状态估计系统，acc和gyro的bias是未知的，因此可以将它们设置为0，在积分的时候被忽略
            // 首先计算旋转部分的更新量
            Eigen::Vector3d w = last_imudata.imu_gyro - Eigen::Vector3d(0,0,0); // 需要上一时刻gyro的测量值，
                                                                                // 另外gyro的bias设置为0
            Eigen::Vector3d dqi = 0.5 * w * dt;
            Eigen::Quaterniond dq;
            dq.w() = 1;
            dq.x() = dqi.x();
            dq.y() = dqi.y();
            dq.z() = dqi.z();
            dq.normalize(); // 注意：将四元数归一化，只有单位四元数才能代表旋转
            // 其次计算平移部分的更新量
            Eigen::Vector3d a = Qwb * (last_imudata.imu_acc - Eigen::Vector3d(0,0,0)) + gw; // 需要上一时刻acc的测量值，
                                                                                            // 另外acc的bias设置为0
            // 执行系统的状态量PVQ的更新
            Pwb = Pwb + Vw * dt + 0.5 * dt * dt * a;    // 更新当前帧(curr)的平移
            Vw = Vw + a * dt;                           // 更新当前帧(curr)的速度
            Qwb = Qwb * dq;                             // 更新当前帧(curr)的旋转
        } 
        else {
            // Step 2: IMU动力学模型：中值积分
            // 中值积分的公式见课件L2-s38-(26) 
            // 注意：这里只是简单将IMU的测量数据进行积分
            //      由于没有状态估计系统，acc和gyro的bias是未知的，因此可以将它们设置为0，在积分的时候被忽略
            // 首先计算旋转部分的更新量
            Eigen::Vector3d w = 0.5 * (curr_imudata.imu_gyro - Eigen::Vector3d(0,0,0) + 
                                       last_imudata.imu_gyro - Eigen::Vector3d(0,0,0)); // 需要gyro的测量值，
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
            Eigen::Vector3d a_curr = curr_Qwb * (curr_imudata.imu_acc - Eigen::Vector3d(0,0,0)) + gw;
            Eigen::Vector3d a_last = Qwb * (last_imudata.imu_acc - Eigen::Vector3d(0,0,0)) + gw;    // 需要acc的测量值，
                                                                                                    // 另外acc的bias设置为0
            Eigen::Vector3d a = 0.5 * (a_curr + a_last); // 中值法
            // 执行系统的状态量PVQ的更新
            Pwb = Pwb + Vw * dt + 0.5 * dt * dt * a;    // 更新当前帧(curr)的平移  
            Vw = Vw + a * dt;                           // 更新当前帧(curr)的速度
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