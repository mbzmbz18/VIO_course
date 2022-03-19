#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;
using namespace std;


// 旋转矩阵更新方法一：通过旋转向量计算exp(w^)
Eigen::Matrix3d computeMatrixByAngleAxis(Eigen::Vector3d w) 
{
    // 得到角度，即旋转向量的模长
    double angle = sqrt(w(0) * w(0) + w(1) * w(1) + w(2) * w(2));
    // 得到轴，即将旋转向量归一化
    Eigen::Vector3d axis = w / angle;
    // 定义旋转向量
    Eigen::AngleAxisd r_angleaxis = Eigen::AngleAxisd(angle, axis);
    // 通过旋转向量计算exp(w^)
    Eigen::Matrix3d expw = r_angleaxis.toRotationMatrix();
    return expw;
}

// 旋转矩阵更新方法二：通过Sophus提供的函数计算exp(w^)
Eigen::Matrix3d computeMatrixBySophus(Eigen::Vector3d w) 
{
    // 通过Sophus提供的函数直接计算exp(w^)
    Sophus::SO3d expw_sophus = Sophus::SO3d::exp(w);
    Eigen::Matrix3d expw = expw_sophus.matrix();
    return expw;
}

// 旋转矩阵更新方法三：通过罗德里格斯公式计算exp(w^)
Eigen::Matrix3d computeMatrixByRodrigues(Eigen::Vector3d w) 
{
    // 单位矩阵
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    // 得到角度，即旋转向量的模长
    double alpha = sqrt(w(0) * w(0) + w(1) * w(1) + w(2) * w(2));
    // 得到轴，即将旋转向量归一化
    Eigen::Vector3d n = w / alpha;
    // 矩阵n^
    Eigen::Matrix3d skew_n;
    skew_n << 0, -n(2), n(1),
              n(2), 0, -n(0),
              -n(1), n(0), 0;
    // 使用罗德里格斯公式计算对应的旋转矩阵exp(w^)
    // R = cos(alpha)I + (1-cos(alpha))nn^T + sin(alpha)n^
    Eigen::Matrix3d expw;
    expw = cos(alpha) * I + (1 - cos(alpha)) * n * n.transpose() + sin(alpha) * skew_n;
    return expw;
}

// 主函数
int main()
{
    // 定义一个旋转向量
    Eigen::AngleAxisd r_angleaxis = Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d(0, 0, 1));
    // 从旋转向量生成旋转矩阵R
    Eigen::Matrix3d R = r_angleaxis.toRotationMatrix();
    // 从旋转向量生成四元数q
    Eigen::Quaterniond q = Eigen::Quaterniond(r_angleaxis);

    // 定义一个旋转的更新量w
    Eigen::Vector3d w(0.01, 0.02, 0.03);

    // 计算旋转的更新量对应的旋转矩阵，即exp(w^)
    Eigen::Matrix3d expw1 = computeMatrixByAngleAxis(w);
    Eigen::Matrix3d expw2 = computeMatrixBySophus(w);
    Eigen::Matrix3d expw3 = computeMatrixByRodrigues(w);
    // 更新旋转矩阵: R <- R * exp(w^)
    Eigen::Matrix3d R_new1 = R * expw1;
    Eigen::Matrix3d R_new2 = R * expw2;
    Eigen::Matrix3d R_new3 = R * expw3;

    // 计算旋转的更新量对应的四元数，即[1, 1/2w]^T
    Eigen::Quaterniond q_update(1, 0.5*w(0), 0.5*w(1), 0.5*w(2));
    // 更新四元数：q <- q * [1, 1/2w]^T
    Eigen::Quaterniond q_new = q * q_update;
    Eigen::Matrix3d R_new_q = q_new.toRotationMatrix();

    // 显示结果的对比
    cout << "Update using rotation matrix:" << endl;
    cout << "-------------------------------------" << endl;
    cout << "use AngleAxis: " << endl << R_new1 << endl;
    cout << "-------------------------------------" << endl;
    cout << "use Sophus: " << endl << R_new2 << endl;
    cout << "-------------------------------------" << endl;
    cout << "use Rodrigues: " << endl << R_new3 << endl;
    cout << "-------------------------------------" << endl;
    cout << endl;
    cout << "Update using quaternion:" << endl;
    cout << "-------------------------------------" << endl;
    cout << "use quaternion: " << endl << R_new_q << endl;

    return 0;
}