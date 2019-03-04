/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef MAPPUBLISH_H
#define MAPPUBLISH_H

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "LocalMapper.h"
#include "Localizer.h"
#include "FramePublish.h"


namespace se2lam {

class Map;

class MapPublish{
public:
    MapPublish(Map* pMap);
    ~MapPublish();
    Map* mpMap;
    LocalMapper* mpLocalMapper;
    Localizer* mpLocalize;
    FramePublish* mpFramePub;

    void run();


    void setFramePub(FramePublish* pFP);
    void setMap(Map* pMap) { mpMap = pMap; }
    void setLocalMapper(LocalMapper* pLocal) { mpLocalMapper = pLocal; }
    void setLocalizer(Localizer* pLocalize) { mpLocalize = pLocalize; }

    void PublishMapPoints();
    void PublishKeyFrames();
    void PublishCameraCurr(const cv::Mat &Twc);

private:

    ros::NodeHandle nh;
    ros::Publisher publisher;
    tf::TransformBroadcaster tfb;

    visualization_msgs::Marker mMPsNeg;
    visualization_msgs::Marker mMPsAct;
    visualization_msgs::Marker mMPsNow;

    visualization_msgs::Marker mKFsNeg;
    visualization_msgs::Marker mKFsAct;
    visualization_msgs::Marker mKFNow;

    visualization_msgs::Marker mCovisGraph;
    visualization_msgs::Marker mFeatGraph;
    visualization_msgs::Marker mOdoGraph;
    visualization_msgs::Marker mMST;


    float mPointSize;
    float mCameraSize;
    float mScaleRatio;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

public:
    bool mbIsLocalize;

    void RequestFinish();
    bool isFinished();

};// class MapPublisher

} // namespace se2lam

#endif // MAPPUBLISH_H
