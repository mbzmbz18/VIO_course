#ifndef MYSLAM_BACKEND_EDGE_H
#define MYSLAM_BACKEND_EDGE_H

#include <memory>
#include <string>
#include "backend/eigen_types.h"

namespace myslam {
namespace backend {

class Vertex;

/**
 * @brief Edge负责计算残差，残差 = 预测 - 观测，维度在构造函数中定义
 * 代价函数是 残差*信息*残差，是一个数值(标量)，由后端求和后最小化
 */
class Edge 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief 构造函数，会自动化配雅可比的空间
     * @param residual_dimension 残差的维度
     * @param num_verticies 相关Vertex的数量
     * @param verticies_types 相关Vertex类型名称，可以不给，不给的话check中不会检查
     */
    explicit Edge(int residual_dimension, int num_verticies,
                  const std::vector<std::string>& verticies_types = std::vector<std::string>());

    virtual ~Edge();

    // 返回id
    unsigned long Id() const { return id_; }

    /**
     * @brief 设置一个相关的Vertex
     * @param vertex 对应的vertex对象
     */
    bool AddVertex(std::shared_ptr<Vertex> vertex) {
        verticies_.emplace_back(vertex);
        return true;
    }

    /**
     * @brief 设置一些相关的Vertex
     * @param vertices 一些Vertex对象，在vector中按指定的顺序进行排列
     * @return
     */
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

    // 返回关联Vertex的个数
    size_t NumVertices() const { return verticies_.size(); }

    // 返回边的类型信息，在子类中实现
    virtual std::string TypeInfo() const = 0;

    // 计算残差，由子类实现
    virtual void ComputeResidual() = 0;

    // 计算雅可比，由子类实现，本后端不支持自动求导，需要实现每个子类的雅可比计算方法
    virtual void ComputeJacobians() = 0;

    // // 计算该edge对Hession矩阵的影响，由子类实现
    // virtual void ComputeHessionFactor() = 0;

    // 计算平方误差，会乘以信息矩阵
    double Chi2();

    // 返回残差
    VecX Residual() const { return residual_; }

    // 返回雅可比
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

    // 检查edge的信息是否全部设置
    bool CheckValid();

    int OrderingId() const { return ordering_id_; }

    void SetOrderingId(int id) { ordering_id_ = id; };

protected:
    unsigned long id_;  // edge id
    int ordering_id_;   // edge id in problem
    std::vector<std::string> verticies_types_;          // 各顶点类型信息，用于debug
    std::vector<std::shared_ptr<Vertex>> verticies_;    // 该边对应的顶点，注意：一个Edge可能关联多个Vertex
    VecX residual_;                 // 残差
    std::vector<MatXX> jacobians_;  // 一些相关的小雅可比，每个小雅可比维度是残差维度*当前待估计状态量的维度，
                                    // 注意：一个Edge可能关联多个Vertex，因此注意小雅可比要按指定的顺序存储
    MatXX information_;             // 信息矩阵
    VecX observation_;              // 观测信息
};

}
}

#endif
