/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <mutex>
#include <map>
#include <memory>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace se2lam{
class KeyFrame;

typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

class MapPoint
{
public:
    MapPoint();
    MapPoint(cv::Point3f pos, bool goodPrl);
    ~MapPoint();

    std::set<PtrKeyFrame> getObservations();

    bool hasObservation(const PtrKeyFrame& pKF);

    void eraseObservation(const PtrKeyFrame& pKF);

    int countObservation();

    // Do pKF.setViewMP() before use this
    void addObservation(const PtrKeyFrame &pKF, int idx);

    int getOctave(const PtrKeyFrame pKF);

    bool isGoodPrl();
    void setGoodPrl(bool value);

    bool isNull();
    void setNull(const std::shared_ptr<MapPoint> &pThis);

    cv::Point3f getPos();
    void setPos(const cv::Point3f& pt3f);

    float getInvLevelSigma2(const PtrKeyFrame &pKF);

    // The descriptor with least median distance to the rest
    cv::Mat mMainDescriptor;
    PtrKeyFrame mMainKF;
    int mMainOctave;
    float mLevelScaleFactor;
    cv::Point2f getMainMeasure();
    void updateMainKFandDescriptor();

    bool acceptNewObserve(cv::Point3f posKF, const cv::KeyPoint kp);

    void updateParallax(const PtrKeyFrame& pKF);

    void updateMeasureInKFs();

    int mId;
    static int mNextId;

    struct IdLessThan{
        bool operator() (const std::shared_ptr<MapPoint>& lhs, const std::shared_ptr<MapPoint>& rhs) const{
            return lhs->mId < rhs->mId;
        }
    };


    void revisitFailCount();

    // This MP would be replaced and abandoned later by
    void mergedInto(const std::shared_ptr<MapPoint>& pMP);

    int getFtrIdx(PtrKeyFrame pKF);

protected:
    std::map<PtrKeyFrame, int> mObservations;

    cv::Point3f mPos;

    void setNull();
    bool mbNull;
    bool mbGoodParallax;

    float mMinDist;
    float mMaxDist;

    // Mean view direction
    cv::Point3f mNormalVector;

    std::mutex mMutexPos;
    std::mutex mMutexObs;

};

typedef std::shared_ptr<MapPoint> PtrMapPoint;


} //namespace se2lam

#endif // MAPPOINT_H
