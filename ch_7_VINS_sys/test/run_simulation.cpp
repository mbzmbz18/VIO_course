//
// Author: Arthur <shihezichen@live.cn>
// Date:  2019.08.10
// Description: 
//    Read the IMU and Image data from data files and do the 
// optimization of them , then draw the final trajectory.


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;


std::shared_ptr<System> pSystem;//宏定义整个VIO系统类

// 获取IMU数据
void PubImuData(string  sData_path)
{
 
    string sImu_data_file = sData_path + "imu_pose_noise.txt";
    cout << "1 PubImuData start sImu_data_filea: " << sImu_data_file << endl;
    ifstream fsImu;
    fsImu.open(sImu_data_file.c_str());
    if (!fsImu.is_open())
    {
        cerr << "Failed to open imu file! " << sImu_data_file << endl;
        return;
    }//读取文件

    std::string sImu_line;
    double dStampNSec = 0.0;//时间戳
    Vector3d vAcc;//加速度
    Vector3d vGyr;//陀螺仪
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
    {
        std::istringstream ssImuData(sImu_line);
        ssImuData >> dStampNSec >> q.w() >> q.x() >> q.y() >> q.z() >> t(0) >> t(1) >> t(2) >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
        //cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << endl;
        pSystem->PubImuData(dStampNSec, vGyr, vAcc);
        usleep(5000*nDelayTimes);//将调用usleep函数的线程挂起一段时间,单位是微秒
    }
    fsImu.close();
}

// 获取图像数据
void PubImageData(string sData_path)
{
    // 由于 keyframe 文件夹下的文件中每一行的image的point没有时间戳, 
    // 因此通过读取图像位姿文件cam_pose_tum.txt, 它对应行的第一列时间戳可作为point的的时间戳
    string sImage_file = sData_path + "cam_pose_tum.txt";
    cout << "1 PubImageData start sImage_file image timestamp: " << sImage_file << endl;
    ifstream fsImage;  //图像位姿文件,这里主要是需要图像的时间戳
    fsImage.open(sImage_file.c_str());
    if (!fsImage.is_open())
    {
        cerr << "Failed to open image file! " << sImage_file << endl;
        return;
    }

    std::string sImage_line;
    double dStampNSec;
    int file_count = 0;//标记keyframe下对应的feature文件

    // cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
    {
        std::istringstream ssImuData(sImage_line);//读取图像位姿文件,为了得到时间戳
        ssImuData >> dStampNSec;
        //cout << "Image t : " << fixed << dStampNSec << endl;
        std::stringstream featurePath;
        featurePath<<sData_path<<"keyframe/all_points_"<<file_count<<".txt";
        string sfeaturePath = featurePath.str();
        //cout<<sfeaturePath<<endl;
        pSystem->PubImageData(dStampNSec, sfeaturePath);
        usleep(50000*nDelayTimes);
        file_count++;
    }
    fsImage.close();
}

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << "./run_simulation PATH_TO_FOLDER/bin/ PATH_TO_CONFIG/config \n"
             << "For example: ./run_simulation ./ ../config/"<< endl;
        return -1;
    }
    // 保存数据的文件夹, 例如 bin/data/
    string sData_path = argv[1];
    string sConfig_path = argv[2];
    pSystem.reset(new System(sConfig_path));//核心类
    // std::thread 初始化为函数,表示直接执行函数,初始化为类函数,则后边应该跟着类
    std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);//后端优化最重要的

    // sleep(5);
    std::thread thd_PubImuData(PubImuData, sData_path);//获取IMU数据线程

    std::thread thd_PubImageData(PubImageData, sData_path);//获取图像数据线程

    std::thread thd_Draw(&System::Draw, pSystem);//显示轨迹线程

    // 数据运行完就把线程停掉,但画图和主线程仍然在运行
    thd_PubImuData.join();
    thd_PubImageData.join();

     thd_BackEnd.join();
     thd_Draw.join();

    cout << "main end... see you ... <Ctrl>+C to exit." << endl;
    return 0;
}
