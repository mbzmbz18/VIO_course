#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

struct Pose
{
    // 构造函数
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t): Rwc(R), qwc(R), twc(t) {};

    Eigen::Matrix3d Rwc;    // 旋转
    Eigen::Quaterniond qwc; // 旋转
    Eigen::Vector3d twc;    // 平移
};


int main()
{
    // 相关参数
    int featureNums = 20;                       // 特征点数量
    int poseNums = 10;                          // 位姿数量
    int dims = poseNums * 6 + featureNums * 3;  // 待优化变量总维度 = 10*6 + 20*3 = 120
    double fx = 1.;     // 相机内参
    double fy = 1.;     // 相机内参

    // 初始化H矩阵
    Eigen::MatrixXd H(dims, dims);  // 尺寸为120x120
    H.setZero();                    // 将H矩阵全部初始化为0，实际应用中，H是稀疏的，即大多的元素都为0

    // 制作相机的位姿
    std::vector<Pose> camera_pose;
    double radius = 8;
    for (int n = 0; n < poseNums; ++n ) {
        // 当前的角度
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4圆弧有10个pose
        // 绕z轴旋转当前的角度
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        // 相应的平移部分
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        // 加入到相机位姿
        camera_pose.push_back(Pose(R, t));
    }

    // 通过随机数制作三维特征点
    std::default_random_engine generator;
    std::vector<Eigen::Vector3d> points;
    for (int j = 0; j < featureNums; ++j) {
        // 当前的特征点坐标，其id为j
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(8., 10.);
        double tx = xy_rand(generator);
        double ty = xy_rand(generator);
        double tz = z_rand(generator);
        Eigen::Vector3d Pw(tx, ty, tz);
        // 加入到特征点坐标
        points.push_back(Pw);
        // 遍历所有的相机位姿
        for (int i = 0; i < poseNums; ++i) {
            // 当前的相机位姿，其id为i
            // 计算当前特征点在当前相机中的坐标，构成一次当前的观测
            Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
            Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);
            double x = Pc.x();
            double y = Pc.y();
            double z = Pc.z();
            double z_2 = z * z;
            // 初始化当前观测的雅可比：de/dPc，其公式见视觉SLAM-s186-(7.43)
            Eigen::Matrix<double, 2, 3> jacobian_uv_Pc;
            jacobian_uv_Pc << fx/z, 0 , -x*fx/z_2,
                              0, fy/z, -y*fy/z_2;
            // 初始化当前观测的雅可比：de/dP，应用于优化特征点的坐标，其公式见视觉SLAM-s.187-(7.48)
            Eigen::Matrix<double, 2, 3> jacobian_Pj = jacobian_uv_Pc * Rcw;
            // 初始化当前观测的雅可比：de/dT，应用于优化相机的位姿，其公式见视觉SLAM-s187-(7.46)
            // 另外注意：与公式相比，这里定义旋转在前，平移在后
            Eigen::Matrix<double, 2, 6> jacobian_Ti;
            jacobian_Ti << -x*y*fx/z_2, (1+x*x/z_2)*fx, -y/z*fx, fx/z, 0, -x*fx/z_2,
                           -(1+y*y/z_2)*fy, x*y/z_2*fy, x/z*fy, 0, fy/z, -y*fy/z_2;

            // 根据当前观测的雅可比，填入H矩阵的相应位置，注意，这里有两种雅可比：
            // 一种是de/dP，即应用于优化特征点的坐标：jacobian_Pj (2x3)
            // 一种是de/dT，即应用于优化相机的位姿：jacobian_Ti (2x6)
            // H矩阵的维度： (60+60) x (60+60)
            // A.block(r, c, rows, cols) == A(r:r+rows, c:c+cols)
            // Step 1: 对应H矩阵的左上角，对应纯相机位姿部分
            H.block(i*6, i*6, 6, 6) += jacobian_Ti.transpose() * jacobian_Ti;   // 6x6
            // Step 2: 对应H矩阵的右下角，对应纯特征点坐标部分
            H.block(6*poseNums + j*3, 6*poseNums + j*3, 3, 3) += jacobian_Pj.transpose() * jacobian_Pj; // 3x3
            // Step 3: 对应H矩阵的右上角
            H.block(i*6, 6*poseNums + j*3, 6, 3) += jacobian_Ti.transpose() * jacobian_Pj;  // 6x3
            // Step 4: 对应H矩阵的左下角
            H.block(6*poseNums + j*3, i*6, 3, 6) += jacobian_Pj.transpose() * jacobian_Ti;  // 3x6
        }
    }

    // std::cout << H << std::endl;
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H);
    // std::cout << saes.eigenvalues() <<std::endl;

    // 将H矩阵进行SVD分解
    // 正确结果为：矩阵奇异值的最后7个维度接近于0，说明零空间的维度为7
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << svd.singularValues() << std::endl;
  
    return 0;
}
