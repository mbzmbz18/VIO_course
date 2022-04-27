#ifndef MYSLAM_BACKEND_VERTEX_H
#define MYSLAM_BACKEND_VERTEX_H

#include <backend/eigen_types.h>

namespace myslam {
namespace backend {

/**
 * @brief Vertex的基类，对应一个parameter block
 * 每个Vertex对应的待估计状态以VecX存储，需要在构造时指定维度
 * 可以衍生出不同的子类Vertex
 */
class Vertex 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief 构造函数
     * @param num_dimension 顶点自身的参数总维度
     * @param local_dimension 局部参数化维度，为-1时认为与自身参数的总维度一样
     */
    explicit Vertex(int num_dimension, int local_dimension = -1);

    virtual ~Vertex();

    // 返回该Vertex的参数总维度
    int Dimension() const;

    // 返回该Vertex的局部参数化维度，更新状态量时使用
    int LocalDimension() const;

    // 该Vertex的id
    unsigned long Id() const { return id_; }

    // 返回该Vertex当前的状态量，对应参数总维度
    VecX Parameters() const { return parameters_; }

    // 返回参数值的引用，以便修改该Vertex当前的状态量
    VecX& Parameters() { return parameters_; }

    // 设置该Vertex当前的状态量
    void SetParameters(const VecX& params) { parameters_ = params; }

    // 对应该Vertex待估计状态量的更新：加法，可在子类中重定义，默认是向量加
    virtual void Plus(const VecX& delta);

    // 返回该Vertex的名称，在子类中实现
    virtual std::string TypeInfo() const = 0;

    // 得到该Vertex的orderingId
    int OrderingId() const { return ordering_id_; }

    // 设置该Vertex的orderingId
    void SetOrderingId(unsigned long id) { ordering_id_ = id; };

    // 固定该Vertex的估计值
    void SetFixed(bool fixed = true) {
        fixed_ = fixed;
    }

    // 测试该Vertex是否被固定
    bool IsFixed() const { return fixed_; }

protected:
    VecX parameters_;           // 该Vertex当前实际存储的状态量，对应参数总维度
    int local_dimension_;       // 该Vertex的局部参数化维度
    unsigned long id_;          // 该Vertex的id，自动生成(在初始化Vertex时唯一指定)

    // ordering_id_是将该Vertex加入到problem中后的排序id，从零开始，用于在H中寻找小雅可比的对应块
    // ordering_id_需要在problem构建完毕后再单独设置，比如对于slam问题，需要将Pose与Landmark做区分
    // ordering_id_带有维度信息，
    // 例如 ordering_id_ = 6 对应Hessian中第6列开始的部分，具体列数与该Vertex的local_dimension_相同
    unsigned long ordering_id_ = 0;

    bool fixed_ = false;        // 该Vertex是否被固定
};

}
}

#endif
