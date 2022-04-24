#ifndef MYSLAM_BACKEND_POINTVERTEX_H
#define MYSLAM_BACKEND_POINTVERTEX_H

#include "backend/vertex.h"

namespace myslam {
namespace backend {

/**
 * @brief 以xyz形式存储的特征点Vertex
 */
class VertexPointXYZ : public Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 构造函数，调用基类Vertex的构造函数
    VertexPointXYZ() : Vertex(3) {}

    std::string TypeInfo() const { return "VertexPointXYZ"; }
};

}
}

#endif
