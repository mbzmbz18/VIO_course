#ifndef MYSLAM_BACKEND_INVERSE_DEPTH_H
#define MYSLAM_BACKEND_INVERSE_DEPTH_H

#include "backend/vertex.h"

namespace myslam {
namespace backend {

/**
 * @brief Vertex：以逆深度形式存储的特征点Vertex
 * parameters: inverse_depth (1 DoF)
 */
class VertexInverseDepth : public Vertex 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 构造函数，调用基类Vertex的构造函数，自身维度为1，本地参数化维度也为1(默认)
    VertexInverseDepth() : Vertex(1) {}

    virtual std::string TypeInfo() const { return "VertexInverseDepth"; }
};

}
}

#endif
