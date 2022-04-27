#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>
#include "backend/problem.h"
#include "utils/tic_toc.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) 
{
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}


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
        // 添加该Vertex，按照其Id添加到verticies_中
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }
    // 如果是SLAM问题，且加入的Vertex是某个Pose
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        if (IsPoseVertex(vertex)) {
            ResizePoseHessiansWhenAddingPose(vertex);
        }
    }
    return true;
}

// 如果是SLAM问题，设置该Vertex对应的OrderingId
void Problem::AddOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v) 
{
    // 如果该Vertex是相机Vertex
    if (IsPoseVertex(v)) {
        // 为该相机Vertex设置OrderingId
        v->SetOrderingId(ordering_poses_);
        // 更新Pose部分的OrderingId
        ordering_poses_ += v->LocalDimension();
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    } 
    // 如果该Vertex是特征点Vertex
    else if (IsLandmarkVertex(v)) {
        // 为该特征点Vertex设置OrderingId
        v->SetOrderingId(ordering_landmarks_);
        // 更新Landmark部分的OrderingId
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
}

// 在新增一个Pose的Vertex后，需要调整几个hessian的大小
// 这里仅应用于滑动窗口等类似情形，在添加新的Vertex之后，旧的先验信息依旧想保留下来，不至于重构problem
void Problem::ResizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) 
{
    int size = H_prior_.rows() + v->LocalDimension();
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(v->LocalDimension()).setZero();
    H_prior_.rightCols(v->LocalDimension()).setZero();
    H_prior_.bottomRows(v->LocalDimension()).setZero();
}

// 检查是否为代表Pose的Vertex
bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) 
{
    string type = v->TypeInfo();
    return type == string("VertexPose");
}

// 检查是否为代表特征点的Vertex
bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) 
{
    string type = v->TypeInfo();
    return type == string("VertexPointXYZ") ||
           type == string("VertexInverseDepth");
}

// 添加一条Edge
bool Problem::AddEdge(shared_ptr<Edge> edge) 
{
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }
    for (auto& vertex : edge->Verticies()) {
        // 关联和当前Edge相关的所有Vertex
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }
    return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) 
{
    vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {

        // 并且这个edge还需要存在，而不是已经被remove了
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;

        edges.emplace_back(iter->second);
    }
    return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) 
{
    //check if the vertex is in map_verticies_
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
        return false;
    }

    // 这里要 remove 该顶点对应的 edge.
    vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        RemoveEdge(remove_edges[i]);
    }

    if (IsPoseVertex(vertex))
        idx_pose_vertices_.erase(vertex->Id());
    else
        idx_landmark_vertices_.erase(vertex->Id());

    vertex->SetOrderingId(-1);      // used to debug
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());
    return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) 
{
    //check if the edge is in map_edges_
    if (edges_.find(edge->Id()) == edges_.end()) {
        // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
        return false;
    }
    edges_.erase(edge->Id());
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

    // 进入迭代优化求解
    while (!stop && (iter < iterations)) {

        // 打印当前迭代中的信息：当前的总残差值和阻尼因子
        std::cout << "iter: " << iter << ", chi = " 
                  << currentChi_ << ", Lambda = " << currentLambda_ << std::endl;

        // 当前迭代的参数
        bool oneStepSuccess = false;
        int false_cnt = 0;

        // 进入当前的迭代求解
        // LM：不断尝试有效的阻尼系数Lambda, 直到当前的迭代求解过程可以被判断为是成功的
        while (!oneStepSuccess) {

            // 当前迭代的Hessian对角线加上当前的阻尼系数Lambda，对应LM的求解
            // AddLambdatoHessianLM();
            // 解线性方程 H * dx = b，即求解状态更新量delta_x_的值
            SolveLinearSystem();
            // Hessian对角线减去之前添加的阻尼系数Lambda
            // RemoveLambdaHessianLM();

            // 检查优化退出条件1： 当前迭代过程的更新量delta_x_很小，则退出
            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
                stop = true;
                break;
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
                // // 检查优化退出条件2： 如果残差 b_max 已经很小了，那就退出
                // stop = (b_max <= 1e-12);
                false_cnt = 0;
            } 
            // 如果当前的迭代求解被认为是失败的
            else {
                // 误差没下降，回滚到之前的待估计状态量，在当前迭代中重新尝试新的阻尼因子
                false_cnt ++;
                RollbackStates();
            }
        }
        // 当前的迭代的求解过程是成功的，退出当前的迭代

        // 更新迭代的计数
        iter++;

        // 检查优化退出条件3： 如果当前迭代完成后的残差currentChi_跟迭代初始时的残差相比下降了1e6倍，则退出
        if (sqrt(currentChi_) <= stopThresholdLM_) {
            stop = true;
        }
    }
    // 迭代优化求解完成
    // 输出相关信息
    std::cout << " problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << " makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
    return true;
}

// 设置各顶点的ordering_index
void Problem::SetOrdering() 
{
    // 初始化，每次构建problem时需要重新计数(比如对于每次求解problem时Vertex有变化的情况：滑动窗口)
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;
    int debug = 0;
    // 遍历加入到当前problem的所有Vertex
    // Note: verticies_是map类型的, 顺序是按照Vertex的普通Id号排序的
    for (auto vertex : verticies_) {
        // 累加，最终可得到所有优化变量的总维数
        ordering_generic_ += vertex.second->LocalDimension();
        if (IsPoseVertex(vertex.second)) {
            debug += vertex.second->LocalDimension();
        }
        // 如果是slam问题，还要分别统计pose和landmark的维数，后面会对他们进行排序
        if (problemType_ == ProblemType::SLAM_PROBLEM){    
            // 如果是slam问题，设置该Vertex在Problem中对应的OrderingId 
            // 这里设置的OrderingId在遍历处理每个Vertex之后是需要更新的
            AddOrderingSLAM(vertex.second);
        }
        if (IsPoseVertex(vertex.second)) {
            std::cout << vertex.second->Id() << " order: " << vertex.second->OrderingId() << std::endl;
        }
    }
    std::cout << "\n ordered_landmark_vertices_ size : " << idx_landmark_vertices_.size() << std::endl;
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        // 在AddOrderingSLAM()中，出于便利，OrderingId是对于Pose和Landmark是分开从零开始设置的
        // 这里要把所有Landmark的OrderingId加上所有Pose的总维度，
        // 这样保持了在构建Hessian时，由于依据OrderingId进行排序，所以Landmark在后，Pose在前
        ulong all_pose_dimension = ordering_poses_;     // 即得到Pose的总维度
        for (auto landmarkVertex : idx_landmark_vertices_) {
            landmarkVertex.second->SetOrderingId(       // 重新设置特征点Vertex的OrderingId
                landmarkVertex.second->OrderingId() + all_pose_dimension
            );
        }
    }
    // CHECK_EQ(CheckOrdering(), true);
}

// 检查OrderingId
bool Problem::CheckOrdering() 
{
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        // 在SLAM问题中，由于首先排列Pose的Vertex，再排列Landmark的Vertex
        // 因此也需要按照该顺序检查OrderingId
        int current_ordering = 0;
        for (auto v : idx_pose_vertices_) {     // idx_pose_vertices以Vertex的Id进行排序
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
        for (auto v: idx_landmark_vertices_) {  // idx_landmark_vertices以Vertex的Id进行排序
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
    }
    return true;
}

// 构造当前迭代过程中的H矩阵
void Problem::MakeHessian() 
{
    // 计时
    TicToc t_h;

    // 初始化当前迭代过程中的大 H 矩阵和 b 向量，注意维度
    ulong size = ordering_generic_;     // 得到所有待优化变量的总维度(使用各自Vertex局部参数化的维度)
    MatXX H(MatXX::Zero(size, size));   // 根据维度进行初始化大 Hessian 为全0矩阵
    VecX b(VecX::Zero(size));           // 根据维度进行初始化 b 为全0向量  
    
    // 遍历问题中的每一个Edge，并计算他们当前迭代中的残差和小雅克比，以得到最后的大 H = J^T * J 和 b
    for (auto& edge : edges_) {
        // 基于当前的待估状态量，计算并更新当前Edge在当前迭代中的残差和雅可比
        // 更新后的残差和雅可比会存储到Edge的成员变量中
        edge.second->ComputeResidual();     // 残差
        edge.second->ComputeJacobians();    // 所有小雅可比
        // 得到当前Edge所有相关的Vertex，由于取决于Edge的类型，一个Edge可能对应多个Vertex
        auto verticies = edge.second->Verticies();  // 所有的相关Vertex
        // 得到当前Edge对所有相关Vertex的雅可比
        auto jacobians = edge.second->Jacobians();  // 所有相关Vertex的小雅可比
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
            ulong index_i = v_i->OrderingId();      // 当前Vertex的OrderingId
            ulong dim_i = v_i->LocalDimension();    // 当前Vertex的局部参数化维度
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
                ulong index_j = v_j->OrderingId();      // 当前Vertex的OrderingId 
                ulong dim_j = v_j->LocalDimension();    // 当前Vertex的局部参数化维度
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
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // std::cout << svd.singularValues() <<std::endl;

    if (err_prior_.rows() > 0) {
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
    }
    Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_;
    b_.head(ordering_poses_) += b_prior_;
    // reset当前迭代中的更新量
    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}

// 解线性方程Hx = b
void Problem::SolveLinearSystem() 
{
    // 对于非SLAM问题
    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        // 得到当前迭代的Hessian
        MatXX H = Hessian_;
        for (ulong i = 0; i < Hessian_.cols(); ++i) {
            // Hessian加入当前迭代的阻尼系数
            H(i, i) += currentLambda_;
        }
        // PCG solver
        // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
        // 简单取逆计算
        delta_x_ = Hessian_.inverse() * b_;
    } 
    // 对于SLAM问题
    else {
        // SLAM问题采用舒尔补的方式以加速Hx = b的计算
        // 舒尔补求解Hx = b的思路：先marg调Landmark，只求解Pose
        // 确定需要被marg的维度，以及需要被保留的维度
        int reserve_size = ordering_poses_;     // 所有Pose的Vertex总维度
        int marg_size = ordering_landmarks_;    // 所有Landmark的Vertex总维度

        // 将完整的H矩阵分为四部分：Hpp, Hmm, Hpm, Hmp
        // 将完整的b向量分成两部分：bpp, bmm
        // A.block(r, c, n_rows, n_cols) == A(r:r+n_rows, c:c+n_cols)
        // Hmm：完整Hessian矩阵中需要被marg的部分，即代表Landmark的部分
        MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
        // Hpm：完整Hessian矩阵中右上侧部分
        MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
        // Hpm：完整Hessian矩阵中左下侧部分
        MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
        // bmm：完整b向量中需要被marg的部分，即代表Landmark的部分
        VecX bmm = b_.segment(reserve_size, marg_size);
        // bpp：完整b向量中代表Pose的部分
        VecX bpp = b_.segment(0, reserve_size); 

        // 对Hmm求逆：Hmm是对角块矩阵，对它求逆可以直接对它每个对角块分别求逆，
        // 如果Landmark是利用逆深度参数化的，则对角线块为1维的，求逆可以直接取对角线上元素的倒数，可以加速计算
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // 遍历所有的Landmark的Vertex
        for (auto landmarkVertex : idx_landmark_vertices_) {
            // 得到当前Landmark的Vertex在Hmm中对应的位置
            int idx = landmarkVertex.second->OrderingId() - reserve_size;
            int size = landmarkVertex.second->LocalDimension();
            // 对当前的对角块求逆
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
            // 最终完成对Hmm求逆
        }

        // 执行舒尔补，得到fill-in的Hpp, bpp，以构建marg后的解线性方程组
        MatXX tempH = Hpm * Hmm_inv;    // 中间变量
        // 舒尔补后的Hpp
        H_pp_schur_ = Hessian_.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
        // 舒尔补后的bpp
        b_pp_schur_ = bpp - tempH * bmm;

        // 舒尔补完成后，求解Pose部分的更新量，即求解H_pp_schur * delta_x = b_pp_schur
        VecX delta_x_pp(VecX::Zero(reserve_size));
        // 使用PCG Solver求解
        for (ulong i = 0; i < ordering_poses_; ++i) {
            // Hessian加入当前迭代的阻尼系数
            H_pp_schur_(i, i) += currentLambda_;
        }
        int n = H_pp_schur_.rows() * 2;                       // 设定迭代次数
        delta_x_pp = PCGSolver(H_pp_schur_, b_pp_schur_, n);  // 哈哈，小规模问题，搞pcg花里胡哨
        // 更新当前求解出来的Pose部分的更新量，之后还要求解Landmark部分的更新量
        delta_x_.head(reserve_size) = delta_x_pp;

        // 求解Landmark部分的更新量
        VecX delta_x_ll(marg_size);
        delta_x_ll = Hmm_inv * (bmm - Hmm * delta_x_pp);
        // 更新当前求解出来的Landmark部分的更新量，
        // 至此当前迭代中的状态量更新delta_x_已经求解完成
        delta_x_.tail(marg_size) = delta_x_ll;
    }
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
    if (err_prior_.rows() > 0) {
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
        err_prior_ = Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 6);
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
    if (err_prior_.rows() > 0) {
        b_prior_ += H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
        err_prior_ = Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 6);
    }
}

// 计算LM算法的初始阻尼系数Lambda
void Problem::ComputeLambdaInitLM() 
{
    // 初始化相关参数
    ni_ = 2.;               // 初始的阻尼因子缩放系数
    currentLambda_ = -1.;   // 初始的阻尼因子
    currentChi_ = 0.0;      // 初始的系统残差和

    // TODO:: robust cost chi2
    // 遍历problem当前的所有Edge，计算残差的和(考虑信息矩阵)
    for (auto edge: edges_) {
        currentChi_ += edge.second->Chi2();
    }
    // 遍历所有的prior_error，计算残差的和
    if (err_prior_.rows() > 0)      // marg prior residual
        currentChi_ += err_prior_.norm();
    // 以当前初始的残差和为标准，设置迭代完成时要求的残差和作为阀值
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
    // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
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

    // 在当前迭代完成并更新状态量后，再次统计所有的残差
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->Chi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.norm();
    // 将新旧残差进行对比，
    // 计算比例因子rho，其公式见L3-s17-(10)
    double rho = (currentChi_ - tempChi) / scale;

    // 根据计算出的比例因子判断当前迭代
    // 这里采用Nielsen策略，其公式见课件L3-s20-(13)
    // 如果 rho > 0，则说明状态量的更新导致了残差的下降，当前的迭代是有效的
    if (rho > 0 && isfinite(tempChi)) {
        double alpha = 1. - pow((2 * rho - 1), 3);
        alpha = std::min(alpha, 2. / 3.);
        double scaleFactor = (std::max)(1. / 3., alpha);
        currentLambda_ *= scaleFactor;  // 更新阻尼因子，用于下一次迭代
        ni_ = 2;
        currentChi_ = tempChi;          // 更新残差值
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
VecX Problem::PCGSolver(const MatXX& A, const VecX& b, int maxIter = -1) 
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

// marg所有和frame相连的edge: imu factor，projection factor，prior factor
bool Problem::Marginalize(const std::shared_ptr<Vertex> frameVertex) 
{
    return true;
}


// 测试marginalization
// 一个单独的函数，和之前创建的问题无关！
void Problem::TestMarginalize() 
{
    // 相关参数
    int marg_idx = 1;                // 需要marg的变量的idx，这里即3x3Hessian矩阵中间的那个元素
    int marg_dim = 1;                // marg变量自身占据的维度
    int allstate_size = 3;           // 总共变量的维度
    double delta1 = 0.1 * 0.1;
    double delta2 = 0.2 * 0.2;  
    double delta3 = 0.3 * 0.3;
    // 初始化Hessian矩阵
    int cols = 3;
    MatXX H_marg(MatXX::Zero(cols, cols));
    H_marg << 1./delta1, -1./delta1, 0,
              -1./delta1, 1./delta1 + 1./delta2 + 1./delta3, -1./delta3,
              0.,  -1./delta3, 1/delta3;
    // 显示初始的Hessian矩阵
    std::cout << "---------- TEST Marg: marg 之前的H矩阵------------"<< std::endl;
    std::cout << H_marg << std::endl;

    // 开始进行marginalization
    // 准备工作：将需要被marg的变量移动到Hessian矩阵的右下角
    // 首先，根据需要被marg的变量的idx，将对应的row(marg_idx)移动到Hessian矩阵的最下面
    // 提取row(marg_idx)
    Eigen::MatrixXd temp_rows = H_marg.block(marg_idx, 0, marg_dim, allstate_size);
    // 提取row(marg_idx)下面的所有rows
    Eigen::MatrixXd temp_botRows = H_marg.block(marg_idx + marg_dim, 0, 
                                                allstate_size - marg_idx - marg_dim, allstate_size);
    // 将row(marg_idx)下面的所有rows赋值到H矩阵的对应部分，即上移
    H_marg.block(marg_idx, 0, allstate_size - marg_idx - marg_dim, allstate_size) = temp_botRows;
    // 将row(marg_idx)移动到Hessian矩阵的最下面
    H_marg.block(allstate_size - marg_dim, 0, marg_dim, allstate_size) = temp_rows;

    // 其次，根据需要被marg的变量的idx，将对应的col(marg_idx)移动到Hessian矩阵的最右面
    // 提取col(marg_idx)
    Eigen::MatrixXd temp_cols = H_marg.block(0, marg_idx, allstate_size, marg_dim);
    // 提取col(marg_idx)右面的所有cols
    Eigen::MatrixXd temp_rightCols = H_marg.block(0, marg_idx + marg_dim, 
                                                  allstate_size, allstate_size - marg_idx - marg_dim);
    // 将col(marg_idx)右面的所有cols赋值到H矩阵的对应部分，即左移
    H_marg.block(0, marg_idx, allstate_size, allstate_size - marg_idx - marg_dim) = temp_rightCols;
    // 将col(marg_idx)移动到Hessian矩阵的最右面
    H_marg.block(0, allstate_size - marg_dim, allstate_size, marg_dim) = temp_cols;
    // 显示marg变量移动后的Hessian矩阵
    std::cout << "---------- TEST Marg: 将变量移动到右下角------------"<< std::endl;
    std::cout<< H_marg <<std::endl;

    // 开始marg操作，需要被marg的变量现在位于Hessian矩阵的右下角
    int m2 = marg_dim;                   // 被marg变量的维度    
    int n2 = allstate_size - marg_dim;   // marg后，剩余变量的维度
    // 计算Amm的逆，这里使用Eigen::SelfAdjointEigenSolver
    double eps = 1e-8;
    Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
            (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                              saes.eigenvectors().transpose();

    // TODO:: home work. 完成舒尔补操作
    //Eigen::MatrixXd Arm = H_marg.block(?,?,?,?);
    //Eigen::MatrixXd Amr = H_marg.block(?,?,?,?);
    //Eigen::MatrixXd Arr = H_marg.block(?,?,?,?);

    Eigen::MatrixXd tempB = Arm * Amm_inv;
    Eigen::MatrixXd H_prior = Arr - tempB * Amr;

    std::cout << "---------- TEST Marg: after marg------------"<< std::endl;
    std::cout << H_prior << std::endl;
}

}
}