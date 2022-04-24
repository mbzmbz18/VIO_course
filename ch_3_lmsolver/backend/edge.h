#ifndef MYSLAM_BACKEND_EDGE_H
#define MYSLAM_BACKEND_EDGE_H

#include <memory>
#include <string>
#include "eigen_types.h"

namespace myslam {
namespace backend {

class Vertex;

/**
 * @brief 边负责计算残差，残差是 预测-观测，其维度在构造函数中定义
 * 代价函数是 残差*信息*残差，是一个标量数值，由后端求和后最小化
 */
class Edge 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief 构造函数，会自动化分配雅可比的空间
     * @param residual_dimension 残差的维度
     * @param num_verticies 相关Vertex数量
     * @param verticies_types 顶点类型名称，可以不给，不给的话check中不会检查
     */
    explicit Edge(int residual_dimension, int num_verticies,
                  const std::vector<std::string>& verticies_types = std::vector<std::string>());

    // 析构函数
    virtual ~Edge();

    // 返回id
    unsigned long Id() const { return id_; }

    // 设置一个相关的Vertex
    bool AddVertex(std::shared_ptr<Vertex> vertex) {
        verticies_.emplace_back(vertex);
        return true;
    }

    // 设置一些相关的Vertex
    bool SetVertex(const std::vector<std::shared_ptr<Vertex>>& vertices) {
        verticies_ = vertices;
        return true;
    }

    // 返回第i个相关的Vertex
    std::shared_ptr<Vertex> GetVertex(int i) {
        return verticies_[i];
    }

    // 返回所有相关的Vertex
    std::vector<std::shared_ptr<Vertex>> Verticies() const {
        return verticies_;
    }

    // 返回相关Vertex的个数
    size_t NumVertices() const { return verticies_.size(); }

    // 返回边的类型信息，以便于debug，具体在子类中实现
    virtual std::string TypeInfo() const = 0;

    // 计算残差，具体由子类实现
    virtual void ComputeResidual() = 0;

    // 计算所有的小雅可比，具体由子类实现，且本后端不支持自动求导，需要实现每个子类的雅可比计算方法
    virtual void ComputeJacobians() = 0;

    // // 计算该edge对Hession矩阵的影响，由子类实现
    // virtual void ComputeHessionFactor() = 0;

    // 计算平方误差，会乘以信息矩阵
    double Chi2();

    // 返回残差
    VecX Residual() const { return residual_; }

    // 返回所有的小雅可比
    std::vector<MatXX> Jacobians() const { return jacobians_; }

    // 设置信息矩阵, information_ = sqrt_Omega = w
    void SetInformation(const MatXX& information) {
        information_ = information;
    }

    // 返回信息矩阵
    MatXX Information() const {
        return information_;
    }

    // 设置观测信息
    void SetObservation(const VecX &observation) {
        observation_ = observation;
    }

    // 返回观测信息
    VecX Observation() const { return observation_; }

    // 检查边的信息是否全部设置
    bool CheckValid();

    int OrderingId() const { return ordering_id_; }

    void SetOrderingId(int id) { ordering_id_ = id; };

protected:
    unsigned long id_;  // edge id
    int ordering_id_;   // edge id in problem
    std::vector<std::string> verticies_types_;          // 对应所有顶点类型信息，用于debug
    std::vector<std::shared_ptr<Vertex>> verticies_;    // 对应的所有顶点
    VecX residual_;                 // 残差
    std::vector<MatXX> jacobians_;  // 所有的雅可比(d残差/dVertex)，每个雅可比维度是residual * vertex[i]
    MatXX information_;             // 信息矩阵
    VecX observation_;              // 观测信息
};

}
}

#endif