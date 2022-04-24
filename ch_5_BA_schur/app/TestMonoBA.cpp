#include <iostream>
#include <random>
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/edge_reprojection.h"
#include "backend/problem.h"

using namespace myslam::backend;
using namespace std;

// Frame：保存每帧的姿态和观测
struct Frame 
{
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;
    unordered_map<int, Eigen::Vector3d> featurePerId; // 该帧观测到的特征点id和相应特征点的归一化坐标
};

// 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
void GetSimDataInWordFrame(vector<Frame>& cameraPoses, vector<Eigen::Vector3d>& points) 
{
    int featureNums = 20;   // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 3;       // 相机数目

    // 根据相机的轨迹制作相机帧
    double radius = 8;      // 轨迹半径
    for (int n = 0; n < poseNums; ++n) {
        // 当前的角度
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
        // 绕z轴旋转当前的角度
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        // 相应的平移部分
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        // 添加当前的相机帧
        cameraPoses.push_back(Frame(R, t));
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1./1000.);  // 相机观测的像素噪声：2pixel / focal
    for (int j = 0; j < featureNums; ++j) {
        // 当前的特征点的3D坐标，其id为j
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);
        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        // 加入到特征点的3D坐标
        points.push_back(Pw);
        // 遍历所有的相机帧
        for (int i = 0; i < poseNums; ++i) {
            // 当前的相机帧
            // 计算当前特征点在当前相机中的坐标，构成一次当前的观测
            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            Pc = Pc / Pc.z();               // 归一化图像平面
            Pc[0] += noise_pdf(generator);  // 添加像素噪声
            Pc[1] += noise_pdf(generator);  // 添加像素噪声
            // 当前的相机帧中加入特征点j的观测信息，假设每帧都能观测到所有的特征
            cameraPoses[i].featurePerId.insert(make_pair(j, Pc));
        }
    }
}

int main() 
{
    // Step 0: 相关的数据准备工作
    // 初始化
    vector<Frame> cameras;                      // 所有相机帧，共有3个相机Pose
    vector<Eigen::Vector3d> points;             // 所有特征点，共有20个特征点
    // 准备相机和观测数据
    GetSimDataInWordFrame(cameras, points);     
    // IMU相机外参
    Eigen::Quaterniond qic(1, 0, 0, 0); 
    Eigen::Vector3d tic(0, 0, 0);
    // 构建problem
    Problem problem(Problem::ProblemType::SLAM_PROBLEM);

    // Step 1: 构建最小二乘问题的Vertex
    // Step 1.1: 根据所有相机Pose创建Vertex
    vector<shared_ptr<VertexPose>> vertexCams_vec;  // 存储所有相机Pose的Vertex
    // 共有3个相机Pose
    for (size_t i = 0; i < cameras.size(); ++i) {
        // Step 1.2: 初始化当前相机Pose的Vertex
        shared_ptr<VertexPose> vertexCam(new VertexPose());
        Eigen::VectorXd pose(7);
        pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(), cameras[i].qwc.z(), cameras[i].qwc.w();
        vertexCam->SetParameters(pose);     // 设置Vertex的初始状态
        // if(i < 2) vertexCam->SetFixed();
        // 当前相机Vertex加入到problem
        problem.AddVertex(vertexCam);
        // 当前相机Vertex加入到容器
        vertexCams_vec.push_back(vertexCam);
    }
    // Step 1.3: 根据所有特征点Pos创建Vertex
    // 随机数
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0, 1.);
    double noise = 0;
    vector<double> noise_invd;  // 存储所有特征点的逆深度
    vector<shared_ptr<VertexInverseDepth>> allPoints;   // 存储所有特征点逆深度的Vertex
    // 共有20个特征点
    for (size_t i = 0; i < points.size(); ++i) {
        // 假设所有特征点的初始观测均为第0帧，计算相应的逆深度
        Eigen::Vector3d Pw = points[i];
        Eigen::Vector3d Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc);
        noise = noise_pdf(generator);
        double inverse_depth = 1. / (Pc.z() + noise);   // 得到带有噪声的逆深度
        noise_invd.push_back(inverse_depth);
        // Step 1.4: 初始化当前特征点Pos的Vertex，利用逆深度参数化
        shared_ptr<VertexInverseDepth> verterxPoint(new VertexInverseDepth());
        // 设置特征点Vertex的逆深度
        VecX inv_d(1);
        inv_d << inverse_depth; 
        verterxPoint->SetParameters(inv_d); // 设置Vertex的初始状态 
        // 当前特征点Vertex加入到problem
        problem.AddVertex(verterxPoint);
        // 当前特征点Vertex加入到容器
        allPoints.push_back(verterxPoint);

        // Step 2: 根据不同相机Pose对特征点的观测数据，创建Edge
        // 计算每个特征点在各个相机帧中的投影误差
        // 注意：第0帧为起始帧，即各个特征点的逆深度初始化帧，不用考虑其观测
        for (size_t j = 1; j < cameras.size(); ++j) {
            // 当前特征点在第0帧的归一化坐标，这里假设所有特征点的逆深度初始化帧均为第0帧
            Eigen::Vector3d pt_i = cameras[0].featurePerId.find(i)->second;
            // 当前特征点在当前帧中的归一化坐标
            Eigen::Vector3d pt_j = cameras[j].featurePerId.find(i)->second;
            // Step 2.1: 初始化当前特征点观测的Edge
            // 初始化当前Edge，需要用到观测到该特征点的第0帧和当前观测帧
            shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
            // 设置该Edge的IMU相机外参
            edge->SetTranslationImuFromCamera(qic, tic);
            // 设置与当前Edge相关的Vertex
            // 注意：视觉重投影观测Edge为三元边，三个相关的Vertex需要按照指定的顺序添加
            std::vector<std::shared_ptr<Vertex>> edge_vertex;
            edge_vertex.push_back(verterxPoint);        // 特征点Vertex
            edge_vertex.push_back(vertexCams_vec[0]);   // 第0帧相机pose，即逆深度初始化帧
            edge_vertex.push_back(vertexCams_vec[j]);   // 当前观测帧相机pose
            edge->SetVertex(edge_vertex);               // 设置与该Edge相关的Vertex
            // 当前观测Edge加入到problem
            problem.AddEdge(edge);
        }
    }

    // Step 3: 求解problem
    // ====================
    // 调用函数Solve()
    problem.Solve(5);
    // ====================
    // 输出结果
    std::cout << "\nCompare MonoBA results after opt..." << std::endl;
    for (size_t k = 0; k < allPoints.size(); k+=1) {
        std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z() << " , noise "
                  << noise_invd[k] << " , opt " << allPoints[k]->Parameters() << std::endl;
    }
    std::cout << "------------ pose translation ----------------" << std::endl;
    for (int i = 0; i < vertexCams_vec.size(); ++i) {
        std::cout << "translation after opt: " << i << " :" << vertexCams_vec[i]->Parameters().head(3).transpose() 
                  << " || gt: " << cameras[i].twc.transpose() << std::endl;
    }
    // 优化完成后，第一帧相机的 pose 平移（x,y,z）不再是原点 0,0,0. 说明向零空间发生了漂移。
    // 解决办法： fix 第一帧和第二帧，固定 7 自由度。 或者加上非常大的先验值。

    // Step 4: 测试marginalization
    // 一个单独的函数，和之前创建的问题无关！
    problem.TestMarginalize();

    return 0;
}

