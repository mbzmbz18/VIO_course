#ifndef MYSLAM_BACKEND_VERTEX_H
#define MYSLAM_BACKEND_VERTEX_H

#include "eigen_types.h"

namespace myslam {
namespace backend {

/**
 * @brief 顶点，对应一个parameter block
 * 变量值以VecX存储，需要在构造时指定维度。在构造实际问题时，需要自定义专门的子类赋写该基类
 */
class Vertex 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * 构造函数
     * @param num_dimension 顶点自身维度
     * @param local_dimension 本地参数化维度，默认为-1时认为与本身维度一样
     */
    explicit Vertex(int num_dimension, int local_dimension = -1);

    // 析构函数
    virtual ~Vertex();

    // 得到待优化变量的维度
    int Dimension() const;

    // 得到待优化变量的局部参数化维度
    int LocalDimension() const;

    // 得到该顶点的id
    unsigned long Id() const { return id_; }

    // 得到待优化变量的参数值
    VecX Parameters() const { return parameters_; }

    // 得到待优化变量的参数值的引用
    VecX& Parameters() { return parameters_; }

    // 设置待优化变量的参数值(初始参数值)
    void SetParameters(const VecX& params) { parameters_ = params; }

    // 加法的更新，可重定义，默认是向量的加法
    virtual void Plus(const VecX& delta);

    // 返回顶点的名称，具体在子类中实现
    virtual std::string TypeInfo() const = 0;

    int OrderingId() const { return ordering_id_; }

    void SetOrderingId(unsigned long id) { ordering_id_ = id; };

    // 固定该点的估计值
    void SetFixed(bool fixed = true) {
        fixed_ = fixed;
    }

    // 查看该点是否被固定
    bool IsFixed() const { return fixed_; }

protected:
    VecX parameters_;       // 该Vertex实际存储的变量值，拥有相应的变量维度
    int local_dimension_;   // 该Vertex局部参数化的维度
    unsigned long id_;      // 该Vertex的id，自动生成
    bool fixed_ = false;    // 该Vertex是否固定

    // ordering_id_是Vertex在problem中排序后的id，用于寻找雅可比对应的块
    // ordering_id_带有维度信息，例如ordering_id=6则对应Hessian中的第6列
    // ordering_id_从0开始
    unsigned long ordering_id_ = 0;
};

}
}

#endif