#ifndef MYSLAM_BACKEND_VISUALEDGE_H
#define MYSLAM_BACKEND_VISUALEDGE_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "backend/eigen_types.h"
#include "backend/edge.h"

namespace myslam {
namespace backend {

/**
 * @brief 该Edge是视觉重投影误差，此Edge为三元边，即与之相连的三个顶点有：
 * - 特征点Vertex，通过逆深度参数化
 * - 逆深度初始化该特征点的相机Vertex：T_World_From_Body1
 * - 观测到该特征点的相机Vertex，构造了重投影误差：T_World_From_Body2
 * 注意：该Edge的相关verticies_顶点顺序必须为 InveseDepth、T_World_From_Body1、T_World_From_Body2
 */
class EdgeReprojection : public Edge 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 构造函数，调用基类Edge的构造函数，残差的维度为2，对应的Vertex数量为3
    EdgeReprojection(const Vec3& pts_i, const Vec3& pts_j)
        : Edge(2, 3, std::vector<std::string>{"VertexInverseDepth", "VertexPose", "VertexPose"}) {
        // 设置在相应两帧中的观测值，i为特征点初始化帧，j为特征点在当前观测帧中的观测
        pts_i_ = pts_i; // 用于预测特征点在j帧的观测
        pts_j_ = pts_j; // 作为特征点在j帧的真实观测值
    }

    // 返回边的类型信息，具体在子类中定义
    virtual std::string TypeInfo() const override { return "EdgeReprojection"; }

    // 基于当前迭代的待估计状态量，计算Edge的残差，具体在子类中定义
    virtual void ComputeResidual() override;

    // 基于当前迭代的待估计状态量，计算该Edge的小雅可比，具体在子类中定义
    virtual void ComputeJacobians() override;

    // 设置IMU相机的外参
    void SetTranslationImuFromCamera(Eigen::Quaterniond& qic_, Vec3& tic_);

private:
    Qd qic;                 // IMU相机外参
    Vec3 tic;               // IMU相机外参
    Vec3 pts_i_, pts_j_;    // 特征点在相应两帧中的观测值，i为特征点初始化帧，j为特征点在当前观测帧中的观测
};

/**
* @brief 此边是视觉重投影误差，此边为二元边，与之相连的两个顶点有：
* - 特征点Vertex，通过世界XYZ参数化
* - 观测到该路标点的相机Vertex，构造重投影误差：T_World_From_Body1
* 注意：该Edge的相关verticies_顶点顺序必须为 XYZ、T_World_From_Body1
*/
class EdgeReprojectionXYZ : public Edge 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 构造函数，调用基类Edge的构造函数，残差的维度为2，对应的Vertex数量为2
    EdgeReprojectionXYZ(const Vec3& pts_i)
        : Edge(2, 2, std::vector<std::string>{"VertexXYZ", "VertexPose"}) {
        obs_ = pts_i;   // 设置特征点相应的观测
    }

    /// 返回边的类型信息
    virtual std::string TypeInfo() const override { return "EdgeReprojectionXYZ"; }

    /// 计算残差
    virtual void ComputeResidual() override;

    /// 计算雅可比
    virtual void ComputeJacobians() override;

    void SetTranslationImuFromCamera(Eigen::Quaterniond &qic_, Vec3 &tic_);

private:
    Qd qic;     // IMU相机外参
    Vec3 tic;   // IMU相机外参
    Vec3 obs_;  // 在相应帧中的观测
};

/**
 * @brief 此边是视觉重投影误差，此边为二元边，但仅计算重投影的相机pose
 */
class EdgeReprojectionPoseOnly : public Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 构造函数，调用基类Edge的构造函数，残差的维度为2，对应的Vertex数量为1
    EdgeReprojectionPoseOnly(const Vec3& landmark_world, const Mat33& K) :
        Edge(2, 1, std::vector<std::string>{"VertexPose"}),
        landmark_world_(landmark_world), K_(K) {}

    // 返回边的类型信息
    virtual std::string TypeInfo() const override { return "EdgeReprojectionPoseOnly"; }

    // 计算残差
    virtual void ComputeResidual() override;

    // 计算雅可比
    virtual void ComputeJacobians() override;

private:
    Vec3 landmark_world_;   // 特征点的世界坐标
    Mat33 K_;               // 相机内参
};

}
}

#endif
