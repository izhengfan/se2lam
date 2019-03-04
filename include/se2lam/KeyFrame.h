/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef KEYFRAME_H
#define KEYFRAME_H


#include <opencv2/calib3d/calib3d.hpp>
#include "Frame.h"
#include "Config.h"
#include "converter.h"

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"

namespace se2lam{

struct SE3Constraint{
public:
    cv::Mat measure;
    cv::Mat info;
    SE3Constraint(){}
    SE3Constraint(const cv::Mat& _mea, const cv::Mat& _info){
        _mea.copyTo(measure);
        _info.copyTo(info);
    }
    ~SE3Constraint(){}
};


typedef shared_ptr<SE3Constraint> PtrSE3Cstrt;

class MapPoint;
typedef shared_ptr<MapPoint> PtrMapPoint;

class KeyFrame: public Frame{
public:
    KeyFrame();
    KeyFrame(const Frame &frame);
    ~KeyFrame();

    cv::Mat getPose();
    void setPose(const cv::Mat& _Tcw);
    void setPose(const Se2& _Twb);

    int mIdKF;
    static int mNextIdKF;

    struct IdLessThan{
        bool operator() (const shared_ptr<KeyFrame>& lhs, const shared_ptr<KeyFrame>& rhs) const{
            return lhs->mIdKF < rhs->mIdKF;
        }
    };


    void setNull(const shared_ptr<KeyFrame> &pThis);
    bool isNull();

    std::set<shared_ptr<KeyFrame> > getAllCovisibleKFs();
    void eraseCovisibleKF(const shared_ptr<KeyFrame> pKF);
    void addCovisibleKF(const shared_ptr<KeyFrame> pKF);

    //! Functions for observation operations
    std::set<PtrMapPoint> getAllObsMPs(bool checkParallax = true);

    void addObservation(PtrMapPoint pMP, int idx);

    // Return all observations as a std::map
    std::map<PtrMapPoint, int> getObservations();

    // Count how many observed MP
    int getSizeObsMP();

    void eraseObservation(const PtrMapPoint pMP);
    void eraseObservation(int idx);

    // Whether a MP is observed by this KF
    bool hasObservation(const PtrMapPoint& pMP);

    // Whether the index in image KeyPoints corresponds to an observed MP
    bool hasObservation(int idx);

    // Get an observed MP by an index
    PtrMapPoint getObservation(int id);

    // Get the corresponding index of an observed MP
    int getFtrIdx(const PtrMapPoint& pMP);

    // Set a new MP in location index (used in MP merge)
    void setObservation(const PtrMapPoint& pMP, int idx);

    vector<cv::Point3f> mViewMPs;
    vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > mViewMPsInfo;
    void setViewMP(cv::Point3f pt3f, int idx, Eigen::Matrix3d info);

    // KeyFrame contraints: From this or To this
    std::map< shared_ptr<KeyFrame>, SE3Constraint > mFtrMeasureFrom;
    std::map< shared_ptr<KeyFrame>, SE3Constraint > mFtrMeasureTo;
    std::pair< shared_ptr<KeyFrame>, SE3Constraint > mOdoMeasureFrom;
    std::pair< shared_ptr<KeyFrame>, SE3Constraint > mOdoMeasureTo;

    std::pair< shared_ptr<KeyFrame>, PreSE2 > preOdomFromSelf;
    std::pair< shared_ptr<KeyFrame>, PreSE2 > preOdomToSelf;

    void addFtrMeasureFrom(shared_ptr<KeyFrame> pKF, const cv::Mat& _mea, const cv::Mat& _info);
    void addFtrMeasureTo(shared_ptr<KeyFrame> pKF, const cv::Mat& _mea, const cv::Mat& _info);
    void eraseFtrMeasureFrom(shared_ptr<KeyFrame> pKF);
    void eraseFtrMeasureTo(shared_ptr<KeyFrame> pKF);

    void setOdoMeasureFrom(shared_ptr<KeyFrame> pKF, const cv::Mat& _mea, const cv::Mat& _info);
    void setOdoMeasureTo(shared_ptr<KeyFrame> pKF, const cv::Mat& _mea, const cv::Mat& _info);

    // ORB BoW by THB:
    void ComputeBoW(ORBVocabulary* _pVoc);
    DBoW2::FeatureVector GetFeatureVector();
    DBoW2::BowVector GetBowVector();
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;
    bool mbBowVecExist;
    vector<PtrMapPoint> GetMapPointMatches();

    // Copy Image
//    void copyImgTo(cv::Mat & imgRet);

protected:

    std::map<PtrMapPoint, int> mObservations;
    std::map<int, PtrMapPoint> mDualObservations;

    bool mbNull;
    std::set<shared_ptr<KeyFrame> > mCovisibleKFs;
    std::mutex mMutexPose;
    std::mutex mMutexObs;
//    std::mutex mMutexImg;
};

typedef shared_ptr<KeyFrame> PtrKeyFrame;



} // namespace se2lam
#endif
