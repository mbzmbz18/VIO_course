#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>
#include <fstream>

#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"

typedef unsigned long ulong;

namespace myslam {
namespace backend {

class Problem 
{
public:
    /**
     * 构建问题的类型：SLAM问题还是通用的问题
     * 如果是SLAM问题，那么pose和landmark是区分开的，H以稀疏方式存储，且SLAM问题只接受一些特定的Vertex和Edge
     * 如果是通用问题，那么H是稠密的，除非用户设定某些vertex为marginalized
     */
    enum class ProblemType { SLAM_PROBLEM, GENERIC_PROBLEM };

    typedef unsigned long ulong;
    // typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
    typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
    typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
    typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 构造函数
    Problem(ProblemType problemType);
    // 析构函数
    ~Problem();

    // 添加，删除Vertex
    bool AddVertex(std::shared_ptr<Vertex> vertex);
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);

    // 添加，删除Edge
    bool AddEdge(std::shared_ptr<Edge> edge);
    bool RemoveEdge(std::shared_ptr<Edge> edge);

    // 取得在优化中被判断为outlier部分的边，方便前端去除outlier
    void GetOutlierEdges(std::vector<std::shared_ptr<Edge>>& outlier_edges);

    // 求解构建的最小二乘问题，即求解的主函数
    bool Solve(int iterations);

    // 边缘化一个frame和以它为host的landmark
    bool Marginalize(std::shared_ptr<Vertex> frameVertex,
                     const std::vector<std::shared_ptr<Vertex>>& landmarkVerticies);
    // 边缘化
    bool Marginalize(const std::shared_ptr<Vertex> frameVertex);

    // test compute prior
    void TestComputePrior();

private:

    // Solve的实现，解通用问题
    bool SolveGenericProblem(int iterations);
    // Solve的实现，解SLAM问题
    bool SolveSLAMProblem(int iterations);

    // 设置各顶点的ordering_index
    void SetOrdering();

    // set ordering for new vertex in slam problem
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    // 构造大H矩阵
    void MakeHessian();

    // schur求解SBA
    void SchurSBA();

    // 解线性方程
    void SolveLinearSystem();

    // 依据当前迭代的更新量，对系统待估计状态进行更新
    void UpdateStates();

    // 有时候 update 后残差会变大，需要退回去，重来
    void RollbackStates(); 

    // 计算并更新Prior部分
    void ComputePrior();

    // 判断一个顶点是否为Pose顶点
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    // 判断一个顶点是否为landmark顶点
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    // 在新增顶点后，需要调整几个hessian的大小
    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    // 检查ordering是否正确
    bool CheckOrdering();

    void LogoutVectorSize();

    // 获取某个顶点连接到的边
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    // 计算LM算法的初始阻尼系数Lambda
    void ComputeLambdaInitLM();

    // Hessian对角线加上阻尼系数Lambda
    void AddLambdatoHessianLM();
    // Hessian对角线减去之前添加的阻尼系数Lambda
    void RemoveLambdaHessianLM();

    // 判断阻尼引子Lambda在上次迭代求解过程中是否可以，以及下一步Lambda应该怎么缩放
    bool IsGoodStepInLM();

    // PCG迭代线性求解器
    VecX PCGSolver(const MatXX& A, const VecX& b, int maxIter);

    double currentLambda_;      // 当前迭代的阻尼系数Lambda
    double currentChi_;         // 当前迭代的残差值
    double stopThresholdLM_;    // LM迭代退出的阈值条件，即一个特定的残差值
    double ni_;                 // 用于控制阻尼系数Lambda缩放大小

    ProblemType problemType_;   // 最小二乘问题的类型

    // 整个信息矩阵
    MatXX Hessian_;
    VecX b_;
    VecX delta_x_;

    // 先验部分信息
    MatXX H_prior_;
    VecX b_prior_;
    MatXX Jt_prior_inv_;
    VecX err_prior_;

    // SBA的Pose部分
    MatXX H_pp_schur_;
    VecX b_pp_schur_;

    // Heesian 的 Landmark 和 pose 部分
    MatXX H_pp_;
    VecX b_pp_;
    MatXX H_ll_;
    VecX b_ll_;

    // 所有的Vertex
    HashVertex verticies_;

    // 所有的Edges
    HashEdge edges_;

    // 由vertex id查询edge
    HashVertexIdToEdge vertexToEdge_;

    // Ordering相关的
    ulong ordering_poses_ = 0;
    ulong ordering_landmarks_ = 0;
    ulong ordering_generic_ = 0;
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;        // 以ordering排序的pose顶点
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;    // 以ordering排序的landmark顶点

    // 需要被marg的一些Vertex
    HashVertex verticies_marg_;

    bool bDebug = false;
    double t_hessian_cost_ = 0.0;
    double t_PCGsovle_cost_ = 0.0;

    std::ofstream output_file_; // 输出文件路径
};

}
}

#endif