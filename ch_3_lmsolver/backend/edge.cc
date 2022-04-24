#include "vertex.h"
#include "edge.h"
// #include <glog/logging.h>
#include <iostream>

using namespace std;

namespace myslam {
namespace backend {

// 全部变量，用于计数Edge的id
unsigned long global_edge_id = 0;

// 构造函数，会自动化配雅可比的空间
Edge::Edge(int residual_dimension, int num_verticies,
           const std::vector<std::string>& verticies_types) 
{
    // 定义残差的维度
    residual_.resize(residual_dimension, 1);
    // TODO: 这里可能会存在问题，比如这里resize了3个，后续调用edge->addVertex，使得vertex前面会存在空元素
    // verticies_.resize(num_verticies); 
    
    // 定义相关Vertex的类型
    if (!verticies_types.empty()) {
        verticies_types_ = verticies_types;
    }
    // 定义其雅可比的数量，即该Edge与几个Vertex有关
    // 注意：这里的雅可比是“小雅可比“，它只与当前Edge关联的Vertex有关
    //      实际问题中，雅可比的维度由残差维度和系统状态量总维度共同决定，
    //      但由于系统可能包含的状态量过多(多个Vertex)，且并不是每个状态量都与当前的Edge有关
    //      为了防止构建的雅可比过于稀疏，这里只计算并存储一些所谓的“小雅可比“
    //      每个“小雅可比”都仅仅代表当前残差与其某个相关的Vertex的雅可比
    //      由于当前Edge可能对应多个Vertex，因此也会有多个“小雅可比”
    //      在最终构建Hessian矩阵的过程中，只需要关注与这些“小雅可比“的值就好，其他位置均为0
    jacobians_.resize(num_verticies);
    // id
    id_ = global_edge_id++;
    // 创建信息矩阵
    Eigen::MatrixXd information(residual_dimension, residual_dimension);
    information.setIdentity();
    information_ = information;

    // cout << "Edge construct residual_dimension=" << residual_dimension
    //      << ", num_verticies=" << num_verticies << ", id_=" << id_ << endl;
}

// 析构函数
Edge::~Edge() {}

// 计算平方误差，是一个标量。另外这里会考虑信息矩阵
double Edge::Chi2() 
{
    // TODO:: we should not Multiply information here, because we have computed Jacobian = sqrt_info * Jacobian
    return residual_.transpose() * information_ * residual_;
    // return residual_.transpose() * residual_;   // 当计算 residual 的时候已经乘以了 sqrt_info, 这里不要再乘
}

// 检查边的信息是否全部设置
bool Edge::CheckValid() 
{
    if (!verticies_types_.empty()) {
        // check type info
        for (size_t i = 0; i < verticies_.size(); ++i) {
            if (verticies_types_[i] != verticies_[i]->TypeInfo()) {
                cout << "Vertex type does not match, should be " << verticies_types_[i] <<
                     ", but set to " << verticies_[i]->TypeInfo() << endl;
                return false;
            }
        }
    }
    /*
    CHECK_EQ(information_.rows(), information_.cols());
    CHECK_EQ(residual_.rows(), information_.rows());
    CHECK_EQ(residual_.rows(), observation_.rows());
    // check jacobians
    for (size_t i = 0; i < jacobians_.size(); ++i) {
        CHECK_EQ(jacobians_[i].rows(), residual_.rows());
        CHECK_EQ(jacobians_[i].cols(), verticies_[i]->LocalDimension());
    }
    */
    return true;
}

}
}