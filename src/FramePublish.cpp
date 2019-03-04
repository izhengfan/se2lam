/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "FramePublish.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "cvutil.h"
#include "Track.h"
#include "GlobalMapper.h"
#include "Localizer.h"

namespace se2lam{

using namespace cv;
using std::vector;

typedef lock_guard<mutex> locker;

FramePublish::FramePublish(){

}

FramePublish::FramePublish(Track* pTR, GlobalMapper* pGM){
    mpTrack = pTR;
    mpGM = pGM;
    mbIsLocalize = false;
}

FramePublish::~FramePublish(){

}

void FramePublish::run(){

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/framepub",1);

    float fps = Config::FPS;
    ros::Rate rate(fps);

    while(nh.ok() && ros::ok()){

        if (!mbIsLocalize) {
            if( mpTrack->copyForPub(kpRef, kp, mImgRef, mImg, matches) ){

                WorkTimer timer;
                timer.start();

                Mat imgCurr = drawMatchesInOneImg(kpRef, mImg, kp, matches);
                Mat imgRef = drawKeys(kpRef, mImgRef, matches);
                Mat imgMatch;
                mpGM->mImgMatch.copyTo(imgMatch);
                Size sizeImgCurr = imgCurr.size();
                Size sizeImgMatch = imgMatch.size();

                Mat imgOut(sizeImgCurr.height*2, sizeImgCurr.width*2, imgCurr.type(), Scalar(0));

                imgCurr.copyTo(imgOut(cv::Rect(0,0,sizeImgCurr.width,sizeImgCurr.height)));
                imgRef.copyTo(imgOut(cv::Rect(0,sizeImgCurr.height,sizeImgCurr.width,sizeImgCurr.height)));
                if (sizeImgMatch.width != 0) {
                    imgMatch.copyTo(imgOut(cv::Rect(sizeImgCurr.width,0,sizeImgMatch.width,sizeImgMatch.height)));
                }

                timer.stop();
                cv::resize(imgOut, imgOut, cv::Size(640,480));
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgOut).toImageMsg();
                pub.publish(msg);

            }
        }
        else {

            locker lockImg(mpLocalizer->mMutexImg);

            if (mpLocalizer == NULL) continue;
            if (mpLocalizer->mpKFCurr == NULL) continue;
            if (mpLocalizer->mImgCurr.cols == 0) continue;

            Mat imgCurr;
            mpLocalizer->mImgCurr.copyTo(imgCurr);
            Size sizeImgCurr = imgCurr.size();

            Mat imgOut(sizeImgCurr.height*2, sizeImgCurr.width*2, imgCurr.type(), Scalar(0));
            imgCurr.copyTo(imgOut(cv::Rect(0, 0, sizeImgCurr.width, sizeImgCurr.height)));

            if (mpLocalizer->mImgLoop.cols != 0) {
                Mat imgLoop;
                mpLocalizer->mImgLoop.copyTo(imgLoop);
                imgLoop.copyTo(imgOut(cv::Rect(0, sizeImgCurr.height, sizeImgCurr.width, sizeImgCurr.height)));
            }

            Mat imgMatch;
            mpLocalizer->mImgMatch.copyTo(imgMatch);
            Size sizeImgMatch = imgMatch.size();
            if (sizeImgMatch.width != 0) {
                imgMatch.copyTo(imgOut(cv::Rect(sizeImgCurr.width,0,sizeImgMatch.width,sizeImgMatch.height)));
            }

            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgOut).toImageMsg();
            pub.publish(msg);
        }

        rate.sleep();
    }
}

cv::Mat FramePublish::drawMatchesInOneImg(const vector<KeyPoint> queryKeys, const Mat &trainImg,
                                          const vector<KeyPoint> trainKeys, const vector<int> &matches){
    Mat out = trainImg.clone();
    if (trainImg.channels() == 1)
        cvtColor(trainImg, out, CV_GRAY2BGR);
    for (unsigned i = 0; i < matches.size(); i++) {

        if (matches[i] < 0) {
            continue;
//            Point2f ptRef = queryKeys[i].pt;
//            circle(out, ptRef, 5, Scalar(255, 0, 0), 1);
        }
        else {
            Point2f ptRef = queryKeys[i].pt;
            Point2f ptCurr = trainKeys[matches[i]].pt;
            circle(out, ptCurr, 5, Scalar(0, 255, 0), 1);
            circle(out, ptRef, 5, Scalar(0, 0, 255), 1);
            line(out, ptRef, ptCurr, Scalar(0, 255, 0));
        }
    }
    return out.clone();
}

cv::Mat FramePublish::drawKeys(const vector<KeyPoint> keys, const Mat &img, vector<int> matched){
    Mat out = img.clone();
    if (img.channels() == 1)
        cvtColor(img, out, CV_GRAY2BGR);
    for (unsigned i = 0; i < matched.size(); i++) {
        Point2f pt1 = keys[i].pt;
        if (matched[i] < 0) {
            circle(out, pt1, 5, Scalar(255, 0, 0), 1);
        }
        else {
            circle(out, pt1, 5, Scalar(0, 0, 255), 1);
        }
    }
    return out.clone();
}

Mat FramePublish::drawFrame() {

    if (!mbIsLocalize) {
        if (mpTrack->copyForPub(kpRef, kp, mImgRef, mImg, matches)){

            Mat imgCurr = drawMatchesInOneImg(kpRef, mImg, kp, matches);
            Mat imgRef = drawKeys(kpRef, mImgRef, matches);
            Mat imgMatch;
            mpGM->mImgMatch.copyTo(imgMatch);
            Size sizeImgCurr = imgCurr.size();
            Size sizeImgMatch = imgMatch.size();

            Mat imgOut(sizeImgCurr.height * 2, sizeImgCurr.width * 2, imgCurr.type(), Scalar(0));

            imgCurr.copyTo(imgOut(cv::Rect(0, 0, sizeImgCurr.width, sizeImgCurr.height)));
            imgRef.copyTo(imgOut(cv::Rect(0, sizeImgCurr.height, sizeImgCurr.width, sizeImgCurr.height)));
            if (sizeImgMatch.width != 0) {
                imgMatch.copyTo(imgOut(cv::Rect(sizeImgCurr.width, 0, sizeImgMatch.width, sizeImgMatch.height)));
            }

            imgOut.copyTo(mImgOut);
        }
    }
    else if (mpLocalizer != NULL && mpLocalizer->mpKFCurr != NULL && mpLocalizer->mImgCurr.cols != 0){

        locker lockImg(mpLocalizer->mMutexImg);

        Mat imgCurr;
        mpLocalizer->mImgCurr.copyTo(imgCurr);
        Size sizeImgCurr = imgCurr.size();

        Mat imgOut(sizeImgCurr.height * 2, sizeImgCurr.width * 2, imgCurr.type(), Scalar(0));
        imgCurr.copyTo(imgOut(cv::Rect(0, 0, sizeImgCurr.width, sizeImgCurr.height)));

        if (mpLocalizer->mImgLoop.cols != 0) {
            Mat imgLoop;
            mpLocalizer->mImgLoop.copyTo(imgLoop);
            imgLoop.copyTo(imgOut(cv::Rect(0, sizeImgCurr.height, sizeImgCurr.width, sizeImgCurr.height)));
        }

        Mat imgMatch;
        mpLocalizer->mImgMatch.copyTo(imgMatch);
        Size sizeImgMatch = imgMatch.size();
        if (sizeImgMatch.width != 0) {
            imgMatch.copyTo(imgOut(cv::Rect(sizeImgCurr.width, 0, sizeImgMatch.width, sizeImgMatch.height)));
        }

        imgOut.copyTo(mImgOut);
    }

    return mImgOut.clone();
}


void FramePublish::setLocalizer(Localizer* localizer){
    mpLocalizer = localizer;
}


}// namespace se2lam
