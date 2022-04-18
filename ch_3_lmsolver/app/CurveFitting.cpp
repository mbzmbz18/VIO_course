#include <iostream>
#include <random>
#include "../backend/problem.h"

using namespace myslam::backend;
using namespace std;

// 子类：曲线拟合模型的Vertex
class CurveFittingVertex : public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数，调用Vertex的构造函数，这里Vertex是3维的，Vertex的类型是abc
    CurveFittingVertex(): Vertex(3) {}
    // 返回Vertex的类型信息，debug时使用
    virtual std::string TypeInfo() const { return "abc"; }
};

// 子类：曲线拟合模型的Edge
class CurveFittingEdge : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 构造函数，调用Edge的构造函数，这里观测值是1维，相关Vertex有1个，相关Vertex的类型是abc
    CurveFittingEdge(double x, double y): Edge(1, 1, std::vector<std::string>{"abc"}) 
    {
        x_ = x; // 观测值xi
        y_ = y; // 观测值yi
    }
    // 计算当前Edge的残差ei
    virtual void ComputeResidual() override
    {
        /*
        // 应用于曲线函数 y = exp(ax^2 + bx + c) + w
        Vec3 abc = verticies_[0]->Parameters(); // 得到当前待估计的参数值
        residual_(0) = std::exp(abc(0)*x_*x_ + abc(1)*x_ + abc(2)) - y_;  // 计算残差，这里残差的维度是1
        */
        // 应用于曲线函数 y = ax^2 + bx + c + w
        Vec3 abc = verticies_[0]->Parameters(); // 得到当前待估计的参数值
        residual_(0) = abc(0)*x_*x_ + abc(1)*x_ + abc(2) - y_;

    }
    // 计算当前Edge的残差对待优化变量的雅克比J
    virtual void ComputeJacobians() override
    {
        /*
        // 应用于曲线函数 y = exp(ax^2 + bx + c) + w
        Vec3 abc = verticies_[0]->Parameters(); // 得到当前待估计的参数值
        // 初始化雅可比，残差为1维，状态量3个，所以是1x3的雅克比矩阵
        Eigen::Matrix<double, 1, 3> jaco_abc;
        // 计算雅可比，代入实际的观测数据以及当前待估计的参数值
        double exp_y = std::exp(abc(0)*x_*x_ + abc(1)*x_ + abc(2));
        jaco_abc << x_ * x_ * exp_y, x_ * exp_y , 1 * exp_y;    
        jacobians_[0] = jaco_abc;   // 赋值雅可比J  
        */
        // 应用于曲线函数 y = ax^2 + bx + c + w
        Vec3 abc = verticies_[0]->Parameters(); // 得到当前待估计的参数值
        // 初始化雅可比，残差为1维，状态量3个，所以是1x3的雅克比矩阵
        Eigen::Matrix<double, 1, 3> jaco_abc;
        // 计算雅可比，代入实际的观测数据以及当前待估计的参数值
        jaco_abc << x_*x_, x_, 1;
        jacobians_[0] = jaco_abc;   // 赋值雅可比J
    }
    // 返回Edge的类型信息，debug时使用
    virtual std::string TypeInfo() const override { return "CurveFittingEdge"; }
public:
    double x_, y_;  // x，y为观测值(xi,yi)
};



int main()
{
    // 问题的相关参数
    double a = 1.0, b = 2.0, c = 1.0;       // 真实参数值
    int N = 500;                            // 提供观测点的数量
    double w_sigma= 1.0;                    // 观测高斯白噪声的标准差
    std::default_random_engine generator;   // 随机数生成器
    std::normal_distribution<double> noise(0., w_sigma);    // 高斯分布，均值为0，标准差为w_sigma

    // 构建最小二乘问题
    Problem problem(Problem::ProblemType::GENERIC_PROBLEM); // 设置为通用的最小二乘问题

    // 构建一个Vertex
    shared_ptr<CurveFittingVertex> vertex(new CurveFittingVertex());
    // 对该Vertex设定待估计状态 x = (a, b, c) 的初始值
    vertex->SetParameters(Eigen::Vector3d(0., 0., 0.));
    // 将该Vertex加入最小二乘问题
    problem.AddVertex(vertex);

    // 制作N次观测数据(xi, yi)
    for (int i = 0; i < N; i++) {
        // 制作当前的观测xi
        double xi = i/100.;
        double n = noise(generator);
        // 制作当前的观测yi
        // 应用于曲线函数 y = exp(ax^2 + bx + c) + w
        //double yi = std::exp(a*xi*xi + b*xi + c) + n;   // 加入高斯白噪声，以模拟真实的观测数据
        // 应用于曲线函数 y = ax^2 + bx + c + w
        double yi = a*xi*xi + b*xi + c;
        // 每个观测(xi, yi)对应的残差函数，构建一个Edge
        shared_ptr<CurveFittingEdge> edge(new CurveFittingEdge(xi, yi));
        // 设置与当前Edge相关的Vertex
        std::vector<std::shared_ptr<Vertex>> edge_vertex;   
        edge_vertex.push_back(vertex);
        edge->SetVertex(edge_vertex);
        // 把这个Edge添加到最小二乘问题
        problem.AddEdge(edge);
    }

    std::cout << "\nTest Curve Fitting start..." << std::endl;

    // ========================
    // 使用LM求解，最大迭代次数为30
    problem.Solve(30);
    // ========================

    std::cout << "------- After optimization, we got these parameters :" << std::endl;
    std::cout << vertex->Parameters().transpose() << std::endl;
    std::cout << "------- ground truth: " << std::endl;
    std::cout << "1.0,  2.0,  1.0" << std::endl;
    return 0;
}
