#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include "backend/vertex_pose.h"
#include "backend/vertex_point_xyz.h"
#include "backend/edge_reprojection.h"
#include "backend/eigen_types.h"

#include <iostream>

namespace myslam {
namespace backend {

/*    
std::vector<std::shared_ptr<Vertex>> verticies_; // 该边对应的顶点
VecX residual_;                 // 残差
std::vector<MatXX> jacobians_;  // 雅可比，每个雅可比维度是 residual x vertex[i]
MatXX information_;             // 信息矩阵
VecX observation_;              // 观测信息
*/

// 基于当前迭代的待估计状态量，计算Edge的残差
void EdgeReprojection::ComputeResidual() {

    // 得到当前迭代中，待估计的特征点的参数(即逆深度)
    double inv_dep_i = verticies_[0]->Parameters()[0];      // vertices_[0]: 特征点Vertex对象
    // 得到当前迭代中，初始化帧的参数(即旋转和平移)
    VecX param_i = verticies_[1]->Parameters();             // vertices_[1]: 相应特征点初始化帧Vertex对象
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();
    // 得到当前迭代中，观测帧的参数(即旋转和平移)
    VecX param_j = verticies_[2]->Parameters();             // vertices_[2]: 相应特征点的观测帧Vertex对象
    Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Vec3 Pj = param_j.head<3>();
    // 计算相关量，相关公式见L3-s61-(60)
    // 注意：pts_i_为在初始化帧i中的观测(归一化坐标)，pts_j_为在观测帧j中的观测(归一化坐标)
    // 首先计算预测的特征点在j帧中的相机坐标(预测)
    Vec3 pts_camera_i = pts_i_ / inv_dep_i;                 // 特征点在i帧中的相机坐标
    Vec3 pts_imu_i = qic * pts_camera_i + tic;              // 特征点在i帧中的Body坐标
    Vec3 pts_w = Qi * pts_imu_i + Pi;                       // 根据i帧的位姿，计算特征点的世界坐标
    Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);           // 根据j帧的位姿，计算特征点在j帧中的Body坐标
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);  // 特征点在j帧中的相机坐标
    double dep_j = pts_camera_j.z();                        // 特征点在j帧中的深度
    // 计算重投影的残差(预测-观测)
    residual_ = (pts_camera_j/dep_j).head<2>() - pts_j_.head<2>();
    // residual_ = information_ * residual_;   // remove information here, we multi information matrix in problem solver
}

// 设置IMU相机的外参
void EdgeReprojection::SetTranslationImuFromCamera(Eigen::Quaterniond& qic_, Vec3& tic_) {
    qic = qic_;
    tic = tic_;
}

// 基于当前迭代的待估计状态量，计算该Edge的小雅可比
void EdgeReprojection::ComputeJacobians() {

    // Step 1：计算相关的变量，这里和计算残差类似
    // 得到当前迭代中，待估计的特征点的参数(即逆深度)
    double inv_dep_i = verticies_[0]->Parameters()[0];      // vertices_[0]: 特征点Vertex对象
    // 得到当前迭代中，初始化帧的参数(即旋转和平移)
    VecX param_i = verticies_[1]->Parameters();             // vertices_[1]: 相应特征点初始化帧Vertex对象
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();
    // 得到当前迭代中，观测帧的参数(即旋转和平移)
    VecX param_j = verticies_[2]->Parameters();             // vertices_[2]: 相应特征点的观测帧Vertex对象
    Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
    Vec3 Pj = param_j.head<3>();
    // 得到相关旋转矩阵用于后续计算
    Mat33 Ri = Qi.toRotationMatrix();   // R_world_i
    Mat33 Rj = Qj.toRotationMatrix();   // R_world_j
    Mat33 ric = qic.toRotationMatrix(); // r_i_c
    // 计算相关量，相关公式见L3-s61-(60)
    // 注意：pts_i_为在初始化帧i中的观测(归一化坐标)，pts_j_为在观测帧j中的观测(归一化坐标)
    // 首先计算预测的特征点在j帧中的相机坐标(预测)
    Vec3 pts_camera_i = pts_i_ / inv_dep_i;                 // 特征点在i帧中的相机坐标
    Vec3 pts_imu_i = qic * pts_camera_i + tic;              // 特征点在i帧中的Body坐标
    Vec3 pts_w = Qi * pts_imu_i + Pi;                       // 根据i帧的位姿，计算特征点的世界坐标
    Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);           // 根据j帧的位姿，计算特征点在j帧中的Body坐标
    Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);  // 特征点在j帧中的相机坐标
    double dep_j = pts_camera_j.z();                        // 特征点在j帧中的深度

    // Step 2：计算小雅可比
    // 小雅可比为该Edge视觉重投影误差对对应状态量Vertex的求导，包括初始化i帧，观测帧j帧，(IMU相机外参），特征点
    // 根据链式法则，小雅可比的计算可以分成两步：
    // 1.Step：残差对特征点在j帧相机坐标的求导
    // 2.Step：特征点在j帧相机坐标对相关状态量的求导
    
    // Step 2.1：残差对特征点在j帧相机坐标的求导
    // 计算残差对特征点在j帧相机坐标的求导，其公式见课件L3-s63-(63)
    Mat23 reduce(2, 3);
    reduce << 1./dep_j, 0, -pts_camera_j(0)/(dep_j*dep_j),
              0, 1./dep_j, -pts_camera_j(1)/(dep_j * dep_j);
    // reduce = information_ * reduce;
    // Step 2.2.1
    // 计算特征点在j帧相机坐标对i帧状态量的求导
    Eigen::Matrix<double, 3, 6> jaco_i; // 特征点在j帧相机坐标为3维，i帧状态量为6维(平移(3)和旋转(！3！)）
    // 平移部分，其公式见课件L3-s63-(64)
    jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
    // 旋转部分，其公式见课件L3-s63-(67)
    jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Sophus::SO3d::hat(pts_imu_i);
    // 结合1.Step和2.Step，得到视觉重投影误差对对应初始化i帧状态量的求导
    Eigen::Matrix<double, 2, 6> jacobian_pose_i;    // 重投影误差为2维，i帧状态量为6维(平移(3)和旋转(！3！)）
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
    // Step 2.2.2
    // 计算特征点在j帧相机坐标对j帧状态量的求导
    Eigen::Matrix<double, 3, 6> jaco_j; // 特征点在j帧相机坐标为3维，j帧状态量为6维(平移(3)和旋转(！3！)）
    // 平移部分，其公式见课件L3-s65-(68)
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    // 旋转部分，其公式见课件L3-s65-(70)
    jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
    // 结合1.Step和2.Step，得到视觉重投影误差对对应初始化i帧状态量的求导
    Eigen::Matrix<double, 2, 6> jacobian_pose_j;    // 重投影误差为2维，i帧状态量为6维(平移(3)和旋转(！3！)）
    jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
    // Step 2.2.3
    // 计算特征点在j帧相机坐标对特征点状态量的求导(状态量仅包括特征点的逆深度)
    // 其公式见L3-s68-(76)
    Eigen::Vector3d jacobian_feat;  // 特征点在j帧相机坐标为3维，特征点状态量为1维(逆深度)
    jacobian_feat = ric.transpose() * Rj.transpose() * Ri * ric * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);
    // 结合1.Step和2.Step，得到视觉重投影误差对对应特征点在j帧相机坐标的求导
    Eigen::Vector2d jacobian_feature;   // 重投影误差为2维，特征点状态量为1维(逆深度)
    jacobian_feature = reduce * jacobian_feat;

    // Step 3：对计算出的雅可比进行赋值
    // 赋值当前计算的雅可比，注意需要按照指定的顺序
    jacobians_[0] = jacobian_feature;       // 对特征点Vertex
    jacobians_[1] = jacobian_pose_i;        // 对第0帧相机pose，即逆深度初始化帧
    jacobians_[2] = jacobian_pose_j;        // 对当前观测帧相机pose

    // ------------- check jacobians -----------------
    // {
    //     std::cout << jacobians_[0] <<std::endl;
    //     const double eps = 1e-6;
    //     inv_dep_i += eps;
    //     Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
    //     Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    //     Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    //     Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    //     Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    //     Eigen::Vector2d tmp_residual;
    //     double dep_j = pts_camera_j.z();
    //     tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
    //     tmp_residual = information_ * tmp_residual;
    //     std::cout <<"num jacobian: "<<  (tmp_residual - residual_) / eps <<std::endl;
    // }
}

// 基于当前迭代的待估计状态量，计算Edge的残差
void EdgeReprojectionXYZ::ComputeResidual() {
    Vec3 pts_w = verticies_[0]->Parameters();

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

    double dep_i = pts_camera_i.z();
    residual_ = (pts_camera_i / dep_i).head<2>() - obs_.head<2>();
}

// 设置IMU相机的外参
void EdgeReprojectionXYZ::SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_) {
    qic = qic_;
    tic = tic_;
}

// 基于当前迭代的待估计状态量，计算Edge的小雅可比
void EdgeReprojectionXYZ::ComputeJacobians() {

    Vec3 pts_w = verticies_[0]->Parameters();

    VecX param_i = verticies_[1]->Parameters();
    Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
    Vec3 Pi = param_i.head<3>();

    Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
    Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

    double dep_i = pts_camera_i.z();

    Mat33 Ri = Qi.toRotationMatrix();
    Mat33 ric = qic.toRotationMatrix();
    Mat23 reduce(2, 3);
    reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i),
        0, 1. / dep_i, -pts_camera_i(1) / (dep_i * dep_i);

    Eigen::Matrix<double, 2, 6> jacobian_pose_i;
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * -Ri.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

    Eigen::Matrix<double, 2, 3> jacobian_feature;
    jacobian_feature = reduce * ric.transpose() * Ri.transpose();

    jacobians_[0] = jacobian_feature;
    jacobians_[1] = jacobian_pose_i;

}

// 基于当前迭代的待估计状态量，计算Edge的残差
void EdgeReprojectionPoseOnly::ComputeResidual() {
    VecX pose_params = verticies_[0]->Parameters();
    Sophus::SE3d pose(
        Qd(pose_params[6], pose_params[3], pose_params[4], pose_params[5]),
        pose_params.head<3>()
    );

    Vec3 pc = pose * landmark_world_;
    pc = pc / pc[2];
    Vec2 pixel = (K_ * pc).head<2>() - observation_;
    // TODO:: residual_ = ????
    residual_ = pixel;
}

// 基于当前迭代的待估计状态量，计算Edge的小雅可比
void EdgeReprojectionPoseOnly::ComputeJacobians() {
    // TODO implement jacobian here
}

}
}