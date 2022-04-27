#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>

#include "backend/eigen_types.h"
#include "backend/edge.h"
#include "backend/vertex.h"

typedef unsigned long ulong;

namespace myslam {
namespace backend {

class Problem 
{
public:
    /**
     * @brief 问题的类型：SLAM问题还是通用的问题
     * 如果是SLAM问题，那么pose和landmark是区分开的，Hessian以稀疏方式存储
     * SLAM问题只接受一些特定的Vertex和Edge
     * 如果是通用问题，那么hessian是稠密的，除非用户设定某些vertex为marginalized
     */
    enum class ProblemType {
        SLAM_PROBLEM,
        GENERIC_PROBLEM
    };

    typedef unsigned long ulong;
    // typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
    typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
    typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
    typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Problem(ProblemType problemType);

    ~Problem();

    // 向问题中添加Vertex
    bool AddVertex(std::shared_ptr<Vertex> vertex);
    // 向问题中删除Vertex
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);

    // 向问题中添加Edge
    bool AddEdge(std::shared_ptr<Edge> edge);
    // 向问题中删除Edge
    bool RemoveEdge(std::shared_ptr<Edge> edge);

    /**
     * @brief 取得在优化中被判断为outlier部分的边，方便前端去除outlier
     * @param outlier_edges
     */
    void GetOutlierEdges(std::vector<std::shared_ptr<Edge>>& outlier_edges);

    /**
     * @brief 求解此问题
     * @param iterations 最大迭代次数
     */
    bool Solve(int iterations);

    // 边缘化一个frame和以它为host的landmark
    bool Marginalize(std::shared_ptr<Vertex> frameVertex,
                     const std::vector<std::shared_ptr<Vertex>>& landmarkVerticies);
    bool Marginalize(const std::shared_ptr<Vertex> frameVertex);

    // test compute prior
    void TestMarginalize();

private:

    // Solve的实现，解通用问题
    bool SolveGenericProblem(int iterations);

    // Solve的实现，解SLAM问题
    bool SolveSLAMProblem(int iterations);

    // 设置各顶点的ordering_index
    void SetOrdering();

    // 如果是SLAM问题，设置该Vertex对应的OrderingId
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    // 构造当前迭代的H矩阵
    void MakeHessian();

    // schur求解SBA
    void SchurSBA();

    // 解线性方程
    void SolveLinearSystem();

    // 更新状态变量
    void UpdateStates();

    void RollbackStates(); // 有时候 update 后残差会变大，需要退回去，重来

    // 计算并更新Prior部分
    void ComputePrior();

    // 判断一个顶点是否为Pose顶点
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    // 判断一个顶点是否为landmark顶点
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    // 在新增顶点后，需要调整几个hessian的大小
    // 这里仅应用于滑动窗口等类似情形，在添加新的Vertex之后，旧的先验信息依旧想保留下来，不至于重构problem
    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    // 检查ordering是否正确
    bool CheckOrdering();

    // 显示输出
    void LogoutVectorSize();

    // 获取某个顶点连接到的边
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    // 计算LM算法的初始Lambda
    void ComputeLambdaInitLM();

    // Hessian对角线加上或者减去Lambda
    void AddLambdatoHessianLM();
    void RemoveLambdaHessianLM();

    // LM算法中用于判断Lambda在上次迭代中是否可以，以及Lambda怎么缩放
    bool IsGoodStepInLM();

    // PCG迭代线性求解器
    VecX PCGSolver(const MatXX& A, const VecX& b, int maxIter);

    // LM算法相关参数
    double currentLambda_;      // 当前迭代中的阻尼因子Lambda
    double currentChi_;         // 当前迭代所有Edge的残差之和
    double stopThresholdLM_;    // 残差阈值，代表LM迭代退出的条件
    double ni_;                 // 控制Lambda缩放大小

    // 最小二乘的问题类型
    ProblemType problemType_;   

    // 整个Hessian和b向量，在每次迭代后会更新
    MatXX Hessian_;
    VecX b_;
    // 每次迭代中求解的状态量更新
    VecX delta_x_;

    // 滑动窗口相关
    // 先验部分信息
    MatXX H_prior_;
    VecX b_prior_;
    MatXX Jt_prior_inv_;
    VecX err_prior_;

    // 舒尔补相关
    // SBA的Pose部分
    MatXX H_pp_schur_;
    VecX b_pp_schur_;
    // Hessian的Landmark和pose部分
    MatXX H_pp_;
    VecX b_pp_;
    MatXX H_ll_;
    VecX b_ll_;

    // 所有的Vertex，这里不区分Vertex的类型，以Id排序
    HashVertex verticies_;
    // 所有的Edge，这里不区分Edge的类型，以Id排序
    HashEdge edges_;
    // 由Vertex的Id查询其关联的edge
    HashVertexIdToEdge vertexToEdge_;

    // OrderingId相关，在设置OrderingId时用于计数
    ulong ordering_poses_ = 0;
    ulong ordering_landmarks_ = 0;
    ulong ordering_generic_ = 0;
    // SLAM问题，用于记录所有加入problem的Pose类型的Vertex，以Id排序
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;
    // SLAM问题，用于记录所有加入problem的Landmark类型的Vertex，以Id排序
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;

    // verticies need to marg. <Ordering_id_, Vertex>
    HashVertex verticies_marg_;

    bool bDebug = false;
    double t_hessian_cost_ = 0.0;
    double t_PCGsovle_cost_ = 0.0;
};

}
}

#endif
