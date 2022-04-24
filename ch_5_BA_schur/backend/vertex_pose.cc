#include "backend/vertex_pose.h"
#include "sophus/so3.hpp"
//#include <iostream>

namespace myslam {
namespace backend {

// 该Vertex对应待估计状态量的更新：加法，可重定义，默认是向量加
void VertexPose::Plus(const VecX& delta) {

    // 得到当前该Vertex的待估计参数(引用)，以便进行状态的更新
    VecX& parameters = Parameters();
    // 平移部分，直接向量加
    parameters.head<3>() += delta.head<3>();
    // 旋转部分，利用四元数更新
    Qd q(parameters[6], parameters[3], parameters[4], parameters[5]);
    q = q * Sophus::SO3d::exp(Vec3(delta[3], delta[4], delta[5])).unit_quaternion();  // right multiplication with so3
    q.normalized();
    parameters[3] = q.x();
    parameters[4] = q.y();
    parameters[5] = q.z();
    parameters[6] = q.w();
    // Qd test = Sophus::SO3d::exp(Vec3(0.2, 0.1, 0.1)).unit_quaternion() * Sophus::SO3d::exp(-Vec3(0.2, 0.1, 0.1)).unit_quaternion();
    // std::cout << test.x()<<" "<< test.y()<<" "<<test.z()<<" "<<test.w() <<std::endl;
}

}
}
