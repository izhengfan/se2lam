/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef TRACK_H
#define TRACK_H
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "Config.h"
#include "Frame.h"
#include "Sensors.h"

namespace se2lam {

class KeyFrame;
typedef std::shared_ptr<KeyFrame> PtrKeyFrame;
class Map;
class LocalMapper;

class Track{
public:
    Track();
    ~Track();

    void run();

    void setMap(Map* pMap);

    void setLocalMapper(LocalMapper* pLocalMapper);

    void setSensors(Sensors* pSensors);

    static void calcOdoConstraintCam(const Se2 &dOdo, cv::Mat &cTc, g2o::Matrix6d &Info_se3);

    static void calcSE3toXYZInfo(cv::Point3f xyz1, const cv::Mat& Tcw1, const cv::Mat& Tcw2, Eigen::Matrix3d& info1, Eigen::Matrix3d& info2);

    // for frame publisher
    std::vector<int> mMatchIdx;
    int copyForPub(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& img1, cv::Mat& img2, std::vector<int> &vMatches12);

    void requestFinish();
    bool isFinished();

private:
    // only useful when odo time not sync with img time
    //float mTimeOdo;
    //float mTimeImg;

    static bool mbUseOdometry;
    Map* mpMap;
    LocalMapper* mpLocalMapper;

    Sensors* mpSensors;

    ORBextractor* mpORBextractor;

    std::vector<cv::Point3f> mLocalMPs;
    int mnGoodPrl; // count number of mLocalMPs with good parallax
    std::vector<bool> mvbGoodPrl;

    int nMinFrames;
    int nMaxFrames;

    Frame mFrame;
    Frame mRefFrame;
    PtrKeyFrame mpKF;
    std::vector<cv::Point2f> mPrevMatched;

    void mCreateFrame(const cv::Mat& img, const Se2& odo);
    void mTrack(const cv::Mat& img, const Se2& odo);
    void resetLocalTrack();
    void updateFramePose();
    int removeOutliers(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2, std::vector<int>& matches);
    bool needNewKF(int nTrackedOldMP, int nMatched);
    int doTriangulate();

    std::mutex mMutexForPub;

    // preintegration on SE2
    PreSE2 preSE2;
    Se2 lastOdom;

public:

    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

};


} // namespace se2lam

#endif // TRACK_H
