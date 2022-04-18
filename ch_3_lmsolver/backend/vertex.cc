#include "vertex.h"
#include <iostream>

namespace myslam {
namespace backend {

// 全局变量，用于记录Vertex的数量
unsigned long global_vertex_id = 0;

// 构造函数，local_dimension默认与num_dimension相等
Vertex::Vertex(int num_dimension, int local_dimension) 
{
    // 设置待优化变量的维度
    parameters_.resize(num_dimension, 1);
    // 设置待优化变量局部参数化的维度
    local_dimension_ = local_dimension > 0 ? local_dimension : num_dimension;
    // 更新Vertex的数量
    id_ = global_vertex_id++;
    // std::cout << "Vertex construct num_dimension: " << num_dimension
    //           << " local_dimension: " << local_dimension << " id_: " << id_ << std::endl;
}

// 析构函数
Vertex::~Vertex() {}

// 得到待优化变量的维度
int Vertex::Dimension() const 
{
    return parameters_.rows();
}

// 得到待优化变量的局部参数化维度
int Vertex::LocalDimension() const 
{
    return local_dimension_;
}

// 加法的更新，可重定义，默认是向量的加法
void Vertex::Plus(const VecX& delta) 
{
    parameters_ += delta;
}

}
}