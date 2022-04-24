#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>
#include "problem.h"
#include "../utils/tic_toc.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

namespace myslam {
namespace backend {

void Problem::LogoutVectorSize() 
{
    // LOG(INFO) <<
    //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
    //           " edges:" << edges_.size();
}

// 构造函数
Problem::Problem(ProblemType problemType): problemType_(problemType) 
{
    LogoutVectorSize();
    verticies_marg_.clear();
}

// 析构函数
Problem::~Problem() {}

// 向问题中添加Vertex
bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) 
{
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }
    return true;
}

// 向问题中添加Edge
bool Problem::AddEdge(shared_ptr<Edge> edge) 
{
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }
    // 遍历该Edge所有相关的Vertex
    for (auto& vertex : edge->Verticies()) {
        // 添加关联：<Vertex, Edge>
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }
    return true;
}

// 求解构建的最小二乘问题
bool Problem::Solve(int iterations) 
{
    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }
    TicToc t_solve;

    // Step 1: 相关初始化操作，确定迭代优化的初始条件
    // 统计优化变量的维数(可能存在不同类型的Vertex，都需要被构建到优化问题中)，为构建H矩阵做准备
    SetOrdering();
    // 遍历所有Edges, 构建初始迭代时的H = J^T * J矩阵和b向量
    MakeHessian();
    // 计算LM算法的初始迭代时的阻尼系数，记作Lambda
    ComputeLambdaInitLM();

    // 迭代相关参数
    bool stop = false;
    int iter = 0;

    // 打开输出文件，写入迭代初始的阻尼系数Lambda
    output_file_.open("data_mu.txt", ios::out);
    
    // 进入迭代优化求解
    while (!stop && (iter < iterations)) {

        // 打印当前迭代中的信息：当前的残差和阻尼因子
        std::cout << "iter: " << iter << " , chi = " << currentChi_ 
                  << " , Lambda = " << currentLambda_ << std::endl;
        // 写入当前迭代中的阻尼因子Lambda
        output_file_ << iter << " " << currentLambda_ << std::endl;

        // 当前迭代的参数
        bool oneStepSuccess = false;
        int false_cnt = 0;

        // 进入当前的迭代求解
        // LM：不断尝试有效的阻尼系数Lambda, 直到当前的迭代求解过程可以被判断为是成功的
        while (!oneStepSuccess) {  
        
            // 当前迭代的Hessian对角线加上当前的阻尼系数Lambda，对应LM的求解
            AddLambdatoHessianLM();
            // 解线性方程 H * dx = b，即求解更新量delta_x_的值
            SolveLinearSystem();
            // Hessian对角线减去之前添加的阻尼系数Lambda
            RemoveLambdaHessianLM();

            // 优化退出条件1： 当前迭代过程的更新量delta_x_很小，则退出
            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
                stop = true;   
                break;           // 设置flag并break用于退出整个求解过程
            }

            // 更新当前迭代待估计的状态量 X = X + delta_x
            UpdateStates();
            // 更新当前迭代待估计的状态量后，
            // 可以根据更新的状态量计算新的残差，并判断刚刚完成的LM迭代是否可行

            // 判断当前当前迭代步是否可行以及LM的Lambda怎么更新
            // oneStepSuccess为True则说明成功
            oneStepSuccess = IsGoodStepInLM();

            // 后续处理，如果当前的迭代求解被认为是成功的
            if (oneStepSuccess) {
                // 在新的线性化点(即当前更新后的待估计状态量)处构建Hessian，为下一次的迭代做准备
                MakeHessian();
                // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
                // double b_max = 0.0;
                // for (int i = 0; i < b_.size(); ++i) {
                //     b_max = max(fabs(b_(i)), b_max);
                // }
                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
                // stop = (b_max <= 1e-12);
                false_cnt = 0;
            } 
            // 如果当前的迭代求解被认为是失败的
            else {
                // 误差没下降，回滚到之前的待估计状态量，在当前迭代中重新尝试新的阻尼因子
                false_cnt++;
                RollbackStates();   
            }
        }
        // 当前的迭代的求解过程是成功的，退出当前的迭代
        iter++;

        // 优化退出条件3： 如果当前迭代完成后的残差currentChi_跟迭代初始时的残差相比下降了1e6倍，则退出
        if (sqrt(currentChi_) <= stopThresholdLM_) {
            stop = true;
        }
    } 
    // 迭代优化求解完成
    // 关闭文件流
    output_file_.close();
    // 输出相关信息
    std::cout << " problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << " makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
    return true;
}

// 设置各Vertex的ordering_index
void Problem::SetOrdering() 
{
    // 每次调用该函数，重新计数
    ordering_poses_ = 0;
    ordering_landmarks_ = 0;
    ordering_generic_ = 0;
    // Note: verticies_是map类型的, 顺序是按照id号排序的
    // 统计待估计的所有变量的总维度
    for (auto vertex : verticies_) {
        ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量局部参数化的总维数
    }
}

// 构造当前迭代过程中的大 H 矩阵和 b 向量
void Problem::MakeHessian() 
{
    TicToc t_h;

    // 初始化当前迭代过程中的大 H 矩阵和 b 向量，注意维度
    ulong size = ordering_generic_;     // 得到待优化变量的总维度(使用其局部参数化的维度)
    MatXX H(MatXX::Zero(size, size));   // 根据维度进行初始化大 Hessian 为全0矩阵
    VecX b(VecX::Zero(size));           // 根据维度进行初始化 b 为全0向量

    // TODO:: accelate, accelate, accelate
    // #ifdef USE_OPENMP
    // #pragma omp parallel for
    // #endif

    // 遍历问题中的每一个Edge，并计算他们当前迭代中的残差和小雅克比，以得到最后的大 H = J^T * J 和 b 
    for (auto& edge : edges_) {
        // 基于当前的待估状态量，计算并更新当前Edge在当前迭代中的残差和雅可比
        // 更新后的残差和雅可比会存储到Edge的成员变量中
        edge.second->ComputeResidual();     // 残差
        edge.second->ComputeJacobians();    // 所有小雅可比
        // 得到当前Edge所有相关的Vertex，由于取决于Edge的类型，一个Edge可能对应多个Vertex
        auto verticies = edge.second->Verticies();  // 所有的相关Vertex
        // 得到当前Edge对所有相关Vertex的雅可比
        auto jacobians = edge.second->Jacobians();      // 所有相关Vertex的雅可比
        assert(jacobians.size() == verticies.size());
        // 遍历当前Edge所有相关的Vertex，由于取决于Edge的类型，一个Edge可能对应多个Vertex
        for (size_t i = 0; i < verticies.size(); ++i) {
            // 当前Vertex
            auto v_i = verticies[i];
            if (v_i->IsFixed()) {
                continue;    // Hessian里不需要添加它的信息，也就是它的雅克比为0
            }
            // 得到对应当前Edge，当前Vertex的小雅可比
            auto jacobian_i = jacobians[i];         // 小雅可比
            ulong index_i = v_i->OrderingId();      // id
            ulong dim_i = v_i->LocalDimension();    // dim
            // 计算小 J^T * W
            MatXX JtW = jacobian_i.transpose() * edge.second->Information();
            // 由于Edge可能是多元边，即一个Edge会关联多个Vertex
            // 这些被当前Edge关联的Vertex之间都会产生相应的联系，其表现是在计算雅可比或Hessian时
            // 遍历当前Edge所有其他相关的Vertex
            for (size_t j = i; j < verticies.size(); ++j) {
                // 当前Vertex
                auto v_j = verticies[j];
                if (v_j->IsFixed()) {
                    continue;   // Hessian里不需要添加它的信息，也就是它的雅克比为0
                } 
                // 得到对应当前Edge，当前Vertex的小雅可比
                auto jacobian_j = jacobians[j];         // 小雅可比
                ulong index_j = v_j->OrderingId();      // id
                ulong dim_j = v_j->LocalDimension();    // dim
                assert(v_j->OrderingId() != -1);
                // 计算小Hessian: H = J^T * W * J
                MatXX hessian = JtW * jacobian_j;
                // 利用当前计算的小hessian更新总的H矩阵
                // H的计算是一个从0矩阵开始逐渐在对应位置累加小hessian的过程
                // 更新上三角部分(右上)
                // A.block(r, c, n_rows, n_cols) == A(r:r+n_rows, c:c+n_cols)
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角部分(左下)
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                }
            }
            // 利用当前计算的小雅可比和残差更新总的b
            // b的计算是一个从0向量开始逐渐在对应位置累加小雅可比的过程
            b.segment(index_i, dim_i).noalias() -= JtW * edge.second->Residual();
        }
    }
    // 根据上面的计算结果，更新当前迭代时刻的H和b
    Hessian_ = H;
    b_ = b;
    // 累计计算时间
    t_hessian_cost_ += t_h.toc();
    // reset当前迭代中的更新量
    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

// 解线性方程Hx = b
void Problem::SolveLinearSystem() 
{
    // 简单取逆计算
    delta_x_ = Hessian_.inverse() * b_;
    // we can use PCG iterative method or use sparse Cholesky
    //delta_x_ = H.ldlt().solve(b_);
}

// 依据当前迭代的更新量，对系统待估计状态进行更新
void Problem::UpdateStates() 
{
    // 遍历所有的Vertex，依次进行相应的状态更新
    for (auto vertex: verticies_) {
        // 得到当前Vertex
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        // 得到当前Vertex对应的更新量
        VecX delta = delta_x_.segment(idx, dim);
        // 所有的参数x叠加一个增量 x_{k+1} = x_{k} + delta_x
        vertex.second->Plus(delta);
    }
}

// 有时在完成当前迭代的状态量更新后，残差会增加，这时就需要回滚到原来的状态
void Problem::RollbackStates() 
{
    // 遍历所有的Vertex，依次进行相应的状态回滚
    for (auto vertex: verticies_) {
        // 得到当前的Vertex
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        // 得到当前Vertex对应的更新量
        VecX delta = delta_x_.segment(idx, dim);
        // 之前更新了待估计的状态量后导致损失函数增加了，
        // 不要这次迭代结果，需要把之前更新的量减去
        vertex.second->Plus(-delta);
    }
}

// 计算LM算法的初始阻尼系数Lambda
void Problem::ComputeLambdaInitLM() 
{
    // 初始化相关参数
    ni_ = 2.;               // 初始的阻尼因子缩放系数
    currentLambda_ = -1.;   // 初始的阻尼因子
    currentChi_ = 0.0;      // 初始的系统残差和

    // TODO: robust cost chi2
    // 遍历problem当前的所有Edge，计算残差的和(考虑信息矩阵)
    for (auto edge: edges_) {
        currentChi_ += edge.second->Chi2();
    }
    // 遍历所有的prior_error，计算残差的和
    if (err_prior_.rows() > 0) {
        currentChi_ += err_prior_.norm();
    }
    // 以当前初始的残差和为标准，设置迭代完成时要求的残差和
    stopThresholdLM_ = 1e-6 * currentChi_;  // 迭代停止条件为：残差下降为初始的1e-6倍
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // 得到当前Hessian对角线上的最大元素
    double maxDiagonal = 0;
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
    }
    double tau = 1e-5;
    // 选择初始的阻尼系数Lambda，其公式见L3-s16-(9)
    currentLambda_ = tau * maxDiagonal;
    // 在之后的每次迭代过程中，Lambda值还会不断变化
}

// Hessian对角线加上阻尼系数Lambda
void Problem::AddLambdatoHessianLM() 
{
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) += currentLambda_;
    }
}

// Hessian对角线减去之前添加的阻尼系数Lambda
void Problem::RemoveLambdaHessianLM() 
{
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) -= currentLambda_;
    }
}

// 判断阻尼引子Lambda在上次迭代求解过程中是否可以，以及下一步Lambda应该怎么缩放
bool Problem::IsGoodStepInLM() 
{
    // 计算比例因子的分母部分，其公式见L3-s17-(11)
    double scale = 0;
    scale = delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale += 1e-3;    // 保证分母不为0
    // recompute residuals after update state
    // 在当前迭代完成并更新状态量后，再次统计所有的残差
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->Chi2();
    }
    // 将新旧残差进行对比，
    // 计算比例因子rho，其公式见L3-s17-(10)
    double rho = (currentChi_ - tempChi) / scale;

    // 根据计算出的比例因子判断当前迭代
    // 这里采用Nielsen策略，其公式见课件L3-s20-(13)
    // 如果 rho > 0，则说明状态量的更新导致了残差的下降，当前的迭代是有效的
    if (rho > 0 && isfinite(tempChi)) {
        double alpha = 1. - pow((2*rho-1), 3);
        alpha = std::min(alpha, 2./3.);
        double scaleFactor = (std::max)(1./3., alpha);
        currentLambda_ *= scaleFactor;      // 更新阻尼因子，用于下一次迭代
        ni_ = 2;
        currentChi_ = tempChi;              // 更新残差值
        return true;
    } 
    // 如果 rho < 0，则说明状态量的更新导致了残差的增加，则需要阻止当前迭代，重新执行一遍当前的迭代
    // 重新执行当前迭代考虑增大阻尼，减小步长
    else {
        currentLambda_ *= ni_;
        ni_ *= 2;
        return false;
    }
}

/** @brief conjugate gradient with perconditioning
*  the jacobi PCG method
*/
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) 
{
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-6 * r0.norm();
    while (r1.norm() > threshold && i < n) {
        i++;
        VecX z1 = M_inv * r1;
        double r1z1 = r1.dot(z1);
        double belta = r1z1 / r0z0;
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        alpha = r1z1 / p.dot(w);
        x += alpha * p;
        r1 -= alpha * w;
    }
    return x;
}

}
}