#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

// 当前的迭代中，构建 H * dx = b 的方法1：
// 使用累加的方法，遍历所有的观测数据，计算当前迭代中的雅可比J
void computeJacobianAcc(const vector<double>& x_data, const vector<double>& y_data, double inv_sigma,
                        double ae, double be, double ce,
                        Matrix3d& H, Vector3d& b, double& curr_cost)
{
    // 注意：无论用那种方法，都需要提取所有的数据点，且H和b的维度都是固定的！
    // 遍历所有的数据点
    for (int i = 0; i < x_data.size(); i++) {
        double xi = x_data[i], yi = y_data[i];              // 得到第i个数据点
        // 计算当前误差ei具体值
        // ***计算单独误差时，相应代入具体的观测值(xi, yi)以及当前迭代的估计值xk = (a, b, c)***
        double err_i = yi - exp(ae*xi*xi + be*xi + ce);     // 用当前的估计值计算当前的误差值err_i
        // 计算当前误差ei的雅可比Ji
        // ***首先计算Ji的解析形式，再相应代入具体的观测值(xi, yi)以及当前迭代的估计值xk = (a, b, c)作为线性化点***
        Vector3d Ji; // Ji实际应该是1x3向量，这里为了方便初始化为3x1
        Ji[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);// 计算dei/da的表达式，并带入当前的估计值以及数据(xi,yi)
        Ji[1] = -xi * exp(ae * xi * xi + be * xi + ce);     // 计算dei/db的表达式，并带入当前的估计值
        Ji[2] = -exp(ae * xi * xi + be * xi + ce);          // 计算dei/dc的表达式，并带入当前的估计值
        // 将当前误差e_i的雅可比Ji“加入”到H和b中
        H += inv_sigma * inv_sigma * Ji * Ji.transpose();   // H来自于所有误差ei计算的J^T * W^-1 * J的累加
        b += -inv_sigma * inv_sigma * err_i * Ji;           // b来自于所有误差ei计算的-J^T * W^-1 * e的累加
        // 累加当前误差ei值的平方到当前迭代中的cost
        curr_cost += err_i * err_i;
    }
}

// 当前的迭代中，构建 H * dx = b 的方法2：
// 使用向量的方法，将基于所有观测数据计算的雅可比J存储在向量中
void computeJacobianVec(const vector<double>& x_data, const vector<double>& y_data, double inv_sigma,
                        double ae, double be, double ce,
                        Matrix3d& H, Vector3d& b, double& curr_cost)
{
    // 注意：无论用那种方法，都需要提取所有的数据点，且H和b的维度都是固定的！
    // 初始化叠放的总雅可比J_all
    Eigen::MatrixXd J_all(x_data.size(), 3);
    Eigen::VectorXd err_all(x_data.size());
    // 遍历所有的数据点
    for (int i = 0; i < x_data.size(); i++) {
        double xi = x_data[i], yi = y_data[i];              // 得到第i个数据点
        // 计算当前误差ei具体值
        // ***计算单独误差时，相应代入具体的观测值(xi, yi)以及当前迭代的估计值xk = (a, b, c)***
        double err_i = yi - exp(ae*xi*xi + be*xi + ce);     // 用当前的估计值计算当前的误差值err_i
        // 计算当前误差ei的雅可比Ji
        // ***首先计算Ji的解析形式，再相应代入具体的观测值(xi, yi)以及当前迭代的估计值xk = (a, b, c)作为线性化点***
        Vector3d Ji; // Ji实际应该是1x3向量，这里为了方便初始化为3x1
        Ji[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce); // 计算dei/da的表达式，并带入当前的估计值以及数据(xi,yi)
        Ji[1] = -xi * exp(ae * xi * xi + be * xi + ce);      // 计算dei/db的表达式，并带入当前的估计值
        Ji[2] = -exp(ae * xi * xi + be * xi + ce);           // 计算dei/dc的表达式，并带入当前的估计值
        // 将Ji进行叠放，形成总雅可比J_all
        J_all.row(i) = Ji.transpose();
        // 将err_i进行叠放，形成总误差向量err_all
        err_all(i) = err_i;
        // 累加当前误差ei的平方到当前迭代中的cost
        curr_cost += err_i * err_i;
    }
    // 利用总雅可比J_all计算H和b
    H = inv_sigma * inv_sigma * J_all.transpose() * J_all;
    b = -inv_sigma * inv_sigma * J_all.transpose() * err_all;
}

// 手写的Gauss-Newton用于非线性最小二乘
int main(int argc, char** argv)
{
    // 定义优化问题：
    // 非线性曲线拟合，y = exp(ax^2 + bx + c) + w
    // 给定一组(xi, yi)的观测数据，总共有100个，求解最优的参数a,b,c，使得误差的平方和最小化
    // 因此，待优化的状态量可定义为： x = (a, b, c)，为三维向量(m=3)
    //                           ***x通过迭代进行优化，每次迭代时，记为xk
    //      单独的误差函数可定义为： ei = yi - exp(axi^2 + bxi + c)，为标量(n=1)
    //                           ***计算单独误差时，相应代入具体的观测值(xi, yi)以及当前迭代的估计值xk = (a, b, c)***
    //      单独的误差函数的雅可比： Ji = [dei/da, dei/db, dei/dc]
    //                           ***首先计算Ji的解析形式，再相应代入具体的观测值(xi, yi)以及当前迭代的估计值xk = (a, b, c)作为线性化点***
    //      可构建误差的数量等于观测数据的数量，即(xi, yi)，i的取之范围为[1, N](N=100)
    
    // 优化问题的相关参数
    double ar = 1.0, br = 2.0, cr = 1.0;   // 真实的参数值
    double ae = 2.0, be = -1.0, ce = 5.0;  // 估计的参数值(这里设置成初始值，需要稍后迭代求解)
    double w_sigma = 1.0;                  // 高斯白噪声的标准差，这里是标量
    double inv_w_sigma = 1.0 / w_sigma;    // 标准差的逆
    cv::RNG rng;                           // OpenCV随机数产生器
    cout << "real abc = " << ar << ", " << br << ", " << cr << endl;
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;

    // 制作一些模拟的观测数据(xi, yi)，i ~ [1, N]，作为测量值
    int n_data = 100;                      // 观测数据(xi, yi)的数量，i ~ [1, N]，作为测量值
    vector<double> x_data;
    vector<double> y_data;
    for (int i = 0; i < n_data; i++) {
        double x = i / 100.0;                               // 制作当前的x观测数据
        x_data.push_back(x);
        double y_noise = rng.gaussian(w_sigma*w_sigma);     // 制作高斯白噪声，均值为0，方差为w_sigma*w_sigma
        double y = (exp(ar*x*x + br*x + cr) + y_noise);     // 制作当前的y观测数据，附加高斯噪声以模拟真实情形
        y_data.push_back(y);
    }

    // 开始Gauss-Newton优化
    int n_iterations = 100;              // 给定优化时的最大迭代次数
    double curr_cost = 0, last_cost = 0; // 本次迭代的cost和上一次迭代的cost，
                                         // cost即当前迭代中，根据观测值计算的所有error的平方和
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    // 进行给定迭代次数的优化过程
    // 在实际优化过程中，迭代次数可能会小于设定值（触发了迭代优化的停止条件）
    for (int iter = 0; iter < n_iterations; iter++) {

        // 每一次迭代，
        // 初始化当前迭代的相关参数
        Matrix3d H = Matrix3d::Zero();      // 当前迭代的Hessian-matrix(H) = J^T * W^-1 * J
        Vector3d b = Vector3d::Zero();      // 当前迭代的bias-vector(b) = -J^T * W^-1 * e
                                            // 注意：在每次迭代中，由于状态量x = (a, b, c)的更新，雅可比J的线性化点也会发生变化，
                                            //      因此每次迭代中需要重新计算雅可比J，以及H和b
        curr_cost = 0;                      // 当前迭代中的cost，即根据观测值计算的所有error的平方和

        // 计算当前迭代的雅可比J，并计算H和b
        // 计算方法1：
        //computeJacobianAcc(x_data, y_data, inv_sigma, ae, be, ce, H, b, curr_cost);
        // 计算方法2：
        computeJacobianVec(x_data, y_data, inv_w_sigma, ae, be, ce, H, b, curr_cost);
        
        // 求解线性方程 H * dx = b，得到当前迭代中的增量delta_x
        // 使用cholesky分解计算
        Vector3d dx = H.ldlt().solve(b);

        // 查看迭代的停止条件
        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }
        if (iter > 0 && curr_cost >= last_cost) {
            cout << "cost: " << curr_cost << ">= last cost: " << last_cost << ", break." << endl;
            break;
        }

        // 更新当前迭代中待估计的参数值
        ae += dx[0];
        be += dx[1];
        ce += dx[2];
        last_cost = curr_cost;  // 记录当前迭代的cost
        // 打印当前迭代中的信息
        cout << "***current iteration*** total cost: " << curr_cost  <<
                "\testimated params: " << ae << "," << be << "," << ce << endl;
        // 结束当前的迭代，进入下一迭代
    } 
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    // 输出最后的结果
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}