#include "backend/vertex.h"
#include <iostream>

namespace myslam {
namespace backend {

// 全局变量，用于记录Vertex的id
unsigned long global_vertex_id = 0;

// 构造函数
Vertex::Vertex(int num_dimension, int local_dimension) {
    // 该Vertex的总维度
    parameters_.resize(num_dimension, 1);
    // 该Vertex的本地维度
    local_dimension_ = local_dimension > 0 ? local_dimension : num_dimension;
    // 该Vertex的id(在初始化Vertex时唯一指定)
    id_ = global_vertex_id++;
    // std::cout << "Vertex construct num_dimension: " << num_dimension
    //           << " local_dimension: " << local_dimension << " id_: " << id_ << std::endl;
}

Vertex::~Vertex() {}

int Vertex::Dimension() const {
    return parameters_.rows();
}

int Vertex::LocalDimension() const {
    return local_dimension_;
}

// 对应待估计状态的更新：加法，可在子类中重定义，默认是向量加
void Vertex::Plus(const VecX& delta) {
    parameters_ += delta;
}

}
}