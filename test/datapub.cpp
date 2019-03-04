/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/Vector3Stamped.h>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "datapub");
    ros::start();

    const char* ImgTopic = "/camera/image_gray";
    const char* OdoTopic = "/odo_raw";

    std::string path = argv[1];
    int N = atoi(argv[2]); // Number of images

    string fullOdoName = path + "/odo_raw.txt";
    ifstream rec(fullOdoName);
    float x,y,theta;
    string line;

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub = it.advertise(ImgTopic, 1);
    ros::Publisher odo_pub = nh.advertise<geometry_msgs::Vector3Stamped>(OdoTopic, 100);


    ros::Rate rate(30);
    for(int i = 0; i < N && ros::ok(); i++)
    {
        string fullImgName = path + "/image/" + to_string(i) + ".bmp";
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;
        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img).toImageMsg();
        geometry_msgs::Vector3Stamped odo_msg;

        img_msg->header.stamp = ros::Time::now();
        odo_msg.header.stamp = img_msg->header.stamp;
        odo_msg.vector.x = x;
        odo_msg.vector.y = y;
        odo_msg.vector.z = theta;

        img_pub.publish(img_msg);
        odo_pub.publish(odo_msg);
        rate.sleep();
    }

    ros::shutdown();
    return 0;

}
