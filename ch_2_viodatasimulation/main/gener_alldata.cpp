#include <fstream>
#include <sys/stat.h>
#include "../src/imu.h"
#include "../src/utilities.h"

using Point = Eigen::Vector4d;                                      // 点
using Points = std::vector<Point, Eigen::aligned_allocator<Point>>; // 点集
using Line = std::pair<Eigen::Vector4d, Eigen::Vector4d>;           // 线段，由两个点组成
using Lines = std::vector<Line, Eigen::aligned_allocator<Line>>;    // 线段集


// 读取文件，用来创建三维空间中的一些点和直线
void CreatePointsLines(Points& points, Lines& lines)
{
    // 打开文件，文件存储的是模拟house的所有顶点
    std::ifstream f;
    f.open("house_model/house.txt");
    while (!f.eof()) {
        std::string s;
        std::getline(f,s);
        if (!s.empty()) {
            // 按行读取文件中的数据开始
            // 每行包括6个数字
            std::stringstream ss;
            ss << s;
            double x, y, z;
            ss >> x;
            ss >> y;
            ss >> z;
            Eigen::Vector4d pt0(x, y, z, 1); // 读取前3个数字，创建构成线段的第一个点
            ss >> x;
            ss >> y;
            ss >> z;
            Eigen::Vector4d pt1(x, y, z, 1); // 读取后3个数字，创建构成线段的第二个点
            // 按行读取文件中的数据结束
            bool isHistoryPoint = false;
            for (int i = 0; i < points.size(); ++i) {
                Eigen::Vector4d pt = points[i];
                if (pt == pt0) {
                    isHistoryPoint = true;
                }
            }
            if (!isHistoryPoint) {
                points.push_back(pt0);      // 创建点并添加
            }
            isHistoryPoint = false;
            for (int i = 0; i < points.size(); ++i) {
                Eigen::Vector4d pt = points[i];
                if (pt == pt1) {
                    isHistoryPoint = true;
                }
            }
            if (!isHistoryPoint) {
                points.push_back(pt1);      // 创建点并添加
            }
            // pt0 = Twl * pt0;
            // pt1 = Twl * pt1;
            lines.emplace_back(pt0, pt1);   // 创建线段并添加
        }
    }
    // 创建更多的点
    int n = points.size();
    for (int j = 0; j < n; ++j) {
        Eigen::Vector4d p = points[j] + Eigen::Vector4d(0.5, 0.5, -0.5, 0);
        points.push_back(p);
    }
    // 存储所有的点
    save_points("all_points.txt", points);
}

// 主函数
int main()
{
    // 建立keyframe文件夹
    mkdir("keyframe", 0777);

    // Step 0: 读取文件中的数据，用于生成一些三维空间中的点和线段
    Points points;
    Lines lines;
    CreatePointsLines(points, lines);

    // Step 1: IMU的运动数据
    // 生成IMU参数对象
    Param params;
    // 生成IMU对象
    IMU imuGen(params);
    // 创建IMU运动数据
    std::vector<MotionData> imudata;        // IMU的所有运动数据
    std::vector<MotionData> imudata_noise;  // 有噪声的IMU所有运动数据
    // 遍历所有的模拟时间帧
    for (float t = params.t_start; t < params.t_end;) {
        // 计算得到当前帧的IMU运动数据，作为真值
        MotionData data = imuGen.MotionModel(t);
        imudata.push_back(data);
        // 对当前帧的IMU运动数据加入噪声，作为测量值
        MotionData data_noise = data;
        imuGen.addIMUnoise(data_noise);
        imudata_noise.push_back(data_noise);
        // 更新时间帧
        t += 1.0/params.imu_frequency;
    }
    // 设置IMU的初始状态
    imuGen.init_velocity_ = imudata[0].imu_velocity;    // 初始速度
    imuGen.init_twb_ = imudata.at(0).twb;               // 初始平移
    imuGen.init_Rwb_ = imudata.at(0).Rwb;               // 初始旋转
    // 存储模拟的IMU所有运动数据到文件
    save_Pose("imu_pose.txt", imudata);         
    // 存储模拟的IMU所有带有噪声的运动数据到文件 
    save_Pose("imu_pose_noise.txt", imudata_noise);
    // 测试IMU, 对相应的测量值进行积分以得到运动轨迹
    imuGen.testImu("imu_pose.txt", "imu_int_pose.txt");                 // imu_integrate_pose
    imuGen.testImu("imu_pose_noise.txt", "imu_int_pose_noise.txt");     // imu_integrate_pose_noise

    // Step 2: 相机的运动数据
    // 创建相机运动数据
    std::vector<MotionData> camdata;    // 初始化
    // 遍历所有的模拟时间帧
    for (float t = params.t_start; t < params.t_end;) {
        // 计算得到当前帧的IMU运动数据，作为真值
        MotionData imu = imuGen.MotionModel(t);
        // 计算当前帧的相机运动数据
        // 注意：相机和IMU之间存在外参，因此数据需要通过转换
        MotionData cam;
        cam.timestamp = imu.timestamp;              // 时间戳
        cam.Rwb = imu.Rwb * params.R_bc;            // 旋转
        cam.twb = imu.twb + imu.Rwb * params.t_bc;  // 平移
        camdata.push_back(cam);
        // 更新时间帧
        // 注意：相机的工作频率要慢于IMU，为30Hz
        t += 1.0/params.cam_frequency;
    }
    // 存储模拟的相机所有运动数据到文件，作为真值
    save_Pose("cam_pose.txt", camdata);
    // 存储模拟的相机所有运动数据到文件(TUM格式)，作为真值
    save_Pose_asTUM("cam_pose_tum.txt", camdata);

    // 遍历所有的相加运动数据
    // 在每一帧处，尝试将空间中的三维点投影到相机中，产生空间点的观测数据
    for (int n = 0; n < camdata.size(); ++n) {
        // 得到当前帧的相机位姿Twc
        MotionData data = camdata[n];
        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block(0, 0, 3, 3) = data.Rwb;
        Twc.block(0, 3, 3, 1) = data.twb;
        // 初始化
        std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_cam;    // 在当前帧相机视野里的空间点
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> features_cam;  // 空间点对应的二维图像坐标
        // 遍历所有的空间点，查看哪些点在当前帧的视野里
        for (int i = 0; i < points.size(); ++i) {
            Eigen::Vector4d pw = points[i];
            pw[3] = 1;                                          // 设置齐次坐标最后一位
            Eigen::Vector4d pc1 = Twc.inverse() * pw;           // 计算点在当前帧相机系下的坐标
            if (pc1(2) < 0) { continue; }                       // 深度z必须大于0
            Eigen::Vector2d obs(pc1(0)/pc1(2), pc1(1)/pc1(2));  // 计算点的归一化坐标
            // if ((obs(0)*460 + 255) < params.image_h && ( obs(0) * 460 + 255) > 0 &&
            //     (obs(1)*460 + 255) > 0 && ( obs(1)* 460 + 255) < params.image_w )
            {
                // 添加在当前帧相机视野里的点，以及其观测的二维图像坐标
                points_cam.push_back(points[i]);    // 空间点的世界坐标
                features_cam.push_back(obs);        // 空间点在当前帧中的二维图像坐标
            }
        }
        // 存储这些在当前帧视野里的特征点
        std::stringstream filename1;
        filename1 << "keyframe/all_points_" << n << ".txt";
        save_features(filename1.str(), points_cam, features_cam);
    }

    // 遍历所有的相加运动数据
    // 在每一帧处，尝试将空间中的三维点投影到相机中，产生空间点的观测数据
    for (int n = 0; n < camdata.size(); ++n) {
        // 得到当前帧的相机位姿Twc
        MotionData data = camdata[n];
        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block(0, 0, 3, 3) = data.Rwb;
        Twc.block(0, 3, 3, 1) = data.twb;
        // 初始化
        // std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_cam; // 在当前帧相机视野里的空间点
        std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> features_cam;  // 空间点对应的二维图像坐标
        // 遍历所有的空间线段，查看哪些特征点在当前帧的视野里
        for (int i = 0; i < lines.size(); ++i) {
            Line linept = lines[i];
            Eigen::Vector4d pc1 = Twc.inverse() * linept.first;     // 计算点在当前帧相机系下的坐标
            Eigen::Vector4d pc2 = Twc.inverse() * linept.second;    // 计算点在当前帧相机系下的坐标
            if (pc1(2) < 0 || pc2(2) < 0) continue;                 // 深度z必须大于0
            Eigen::Vector4d obs(pc1(0)/pc1(2), pc1(1)/pc1(2),       // 计算点的归一化坐标
                                pc2(0)/pc2(2), pc2(1)/pc2(2));
            // if(obs(0) < params.image_h && obs(0) > 0 && obs(1)> 0 && obs(1) < params.image_w)
            {
                // 添加在当前帧相机视野里的线段(两个端点)观测的二维图像坐标
                features_cam.push_back(obs);
            }
        }
        // 存储这些在当前帧视野里的线段(两个端点)
        std::stringstream filename1;
        filename1 << "keyframe/all_lines_" << n << ".txt";
        save_lines(filename1.str(), features_cam);
    }
    return 0;
}