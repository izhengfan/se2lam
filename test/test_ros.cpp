/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#include <geometry_msgs/Vector3Stamped.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>

#include "OdoSLAM.h"


using namespace std;
using namespace cv;
using namespace Eigen;

class SensorHandler{
public:
    SensorHandler(se2lam::OdoSLAM* slam){
        _slam = slam;
    }

    ~SensorHandler(){}

    inline void updateImg(const sensor_msgs::ImageConstPtr& msg)
    {
        _slam->receiveImgData(cv_bridge::toCvShare(msg, "mono8")->image.clone());
    }

    inline void updateOdo(const geometry_msgs::Vector3Stamped& msg)
    {
        _slam->receiveOdoData(msg.vector.x, msg.vector.y, msg.vector.z);
    }

private:
    se2lam::OdoSLAM* _slam;

};

geometry_msgs::Pose toRosPose(const cv::Mat T4x4)
{
    geometry_msgs::Pose rosPose;
    Eigen::Matrix<double,3,3> eigMat = se2lam::toMatrix3d(T4x4.rowRange(0,3).colRange(0,3));
    Eigen::Quaterniond quaterd(eigMat);
    rosPose.position.x = T4x4.at<float>(0,3);
    rosPose.position.y = T4x4.at<float>(1,3);
    rosPose.position.z = T4x4.at<float>(2,3);
    rosPose.orientation.w = quaterd.w();
    rosPose.orientation.x = quaterd.x();
    rosPose.orientation.y = quaterd.y();
    rosPose.orientation.z = quaterd.z();
    return rosPose;
}

int main(int argc, char **argv)
{

    //! Initialize
    ros::init(argc, argv, "test_ros");
    ros::start();

    if(argc != 3){
        cerr << "Input data_path and PATH_TO_ORBvoc.bin!" << endl;
        ros::shutdown();
        return 1;
    }


    se2lam::OdoSLAM system;

    SensorHandler sensor(&system);

    string path = argv[1];
    string strVoc = argv[2];
    system.setVocFileBin(strVoc.c_str());
    system.setDataPath(path.c_str());

    system.start();

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber img_sub = it.subscribe("/camera/image_gray", 1, &SensorHandler::updateImg, &sensor);
    ros::Subscriber odo_sub = nh.subscribe("/odo_raw", 1, &SensorHandler::updateOdo, &sensor);
    ros::Publisher posev_pub = nh.advertise<geometry_msgs::PoseStamped>("/vehicle_pose_w_b", 1);
    ros::Publisher posec_pub = nh.advertise<geometry_msgs::PoseStamped>("/vehicle_pose_w_c", 1);

    while(nh.ok() && system.ok()) {

        geometry_msgs::PoseStamped pose_wb_stamped, pose_wc_stamped;
        pose_wb_stamped.pose = toRosPose(system.getCurrentVehiclePose());
        pose_wb_stamped.header.stamp = ros::Time::now();
        pose_wc_stamped.pose = toRosPose(system.getCurrentCameraPoseWC());
        pose_wc_stamped.header.stamp = pose_wb_stamped.header.stamp;
        posev_pub.publish(pose_wb_stamped);
        posec_pub.publish(pose_wc_stamped);

        ros::spinOnce();

    }

    cerr << "Exit system.. " << endl;

    system.waitForFinish();

    ros::shutdown();

    return 0;
}


