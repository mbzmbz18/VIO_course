#ifndef MYSLAM_BACKEND_MOTIONVERTEX_H
#define MYSLAM_BACKEND_MOTIONVERTEX_H

#include <memory>
#include "backend/vertex.h"

namespace myslam {
namespace backend {

/**
 * 运动数据的Vertex
 * parameters: v, ba, bg 9 DoF
 */
class VertexMotion : public Vertex 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 构造函数，调用基类Vertex的构造函数
    VertexMotion() : Vertex(9) {}

    std::string TypeInfo() const { return "VertexMotion"; }

};

}
}

#endif
