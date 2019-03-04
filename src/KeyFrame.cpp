/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "cvutil.h"

namespace se2lam {
using namespace cv;
using namespace std;

typedef lock_guard<mutex> locker;


int KeyFrame::mNextIdKF = 0;

KeyFrame::KeyFrame(){
    mbNull = false;
    PtrKeyFrame pKF = NULL;
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint());
    mOdoMeasureTo = make_pair(pKF, SE3Constraint());

    preOdomFromSelf = make_pair(pKF, PreSE2());
    preOdomToSelf = make_pair(pKF, PreSE2());

    //Scale Levels Info
    mnScaleLevels = Config::MaxLevel;
    mfScaleFactor = Config::ScaleFactor;

    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for(int i=1; i<mnScaleLevels; i++)
    {
        mvScaleFactors[i]=mvScaleFactors[i-1]*mfScaleFactor;
        mvLevelSigma2[i]=mvScaleFactors[i]*mvScaleFactors[i];
    }

    mvInvLevelSigma2.resize(mvLevelSigma2.size());
    for(int i=0; i<mnScaleLevels; i++)
        mvInvLevelSigma2[i] = 1.0/mvLevelSigma2[i];

}


KeyFrame::KeyFrame(const Frame& frame):
    Frame(frame),
    mbBowVecExist(false),
    mbNull(false)
{
    size_t sz = frame.keyPoints.size();
    mViewMPs = vector<Point3f>(sz, Point3f(-1,-1,-1));
    mViewMPsInfo = vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> >(sz, Eigen::Matrix3d::Identity()*-1);
    mNextIdKF++;
    mIdKF = mNextIdKF;
    PtrKeyFrame pKF = NULL;
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint());
    mOdoMeasureTo = make_pair(pKF, SE3Constraint());

    preOdomFromSelf = make_pair(pKF, PreSE2());
    preOdomToSelf = make_pair(pKF, PreSE2());
}

KeyFrame::~KeyFrame(){}


// Please handle odometry based constraints after calling this function
void KeyFrame::setNull(const shared_ptr<KeyFrame>& pThis){

    lock_guard<mutex> lckImg(mMutexImg);
    lock_guard<mutex> lckPose(mMutexPose);
    lock_guard<mutex> lckObs(mMutexObs);
    lock_guard<mutex> lckDes(mMutexDes);

    mbNull = true;
    descriptors.release();
    img.release();
    keyPoints.clear();
    keyPointsUn.clear();


    // Handle Feature based constraints
    for(auto it = mFtrMeasureFrom.begin(), iend = mFtrMeasureFrom.end();
        it != iend; it++){
        it->first->mFtrMeasureTo.erase(pThis);
    }
    for(auto it = mFtrMeasureTo.begin(), iend = mFtrMeasureTo.end();
        it != iend; it++){
        it->first->mFtrMeasureFrom.erase(pThis);
    }

    mFtrMeasureFrom.clear();
    mFtrMeasureTo.clear();

    // Handle observations in MapPoints
    for(auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++ ){
        PtrMapPoint pMP = it->first;
        pMP->eraseObservation(pThis);
    }

    // Handle Covisibility
    for(auto it = mCovisibleKFs.begin(), iend = mCovisibleKFs.end(); it != iend; it++){
        (*it)->eraseCovisibleKF(pThis);
    }
    mObservations.clear();
    mDualObservations.clear();
    mViewMPs.clear();
    mViewMPsInfo.clear();
    mCovisibleKFs.clear();
}

//void KeyFrame::copyImgTo(cv::Mat & imgRet) {
//    locker lock(mMutexImg);
//    img.copyTo(imgRet);
//}

int KeyFrame::getSizeObsMP(){
    locker lock(mMutexObs);
    return mObservations.size();
}

void KeyFrame::setViewMP(Point3f pt3f, int idx, Eigen::Matrix3d info){
    locker lock(mMutexObs);
    mViewMPs[idx] = pt3f;
    mViewMPsInfo[idx] = info;
}

void KeyFrame::eraseCovisibleKF(const shared_ptr<KeyFrame> pKF){
    mCovisibleKFs.erase(pKF);
}

void KeyFrame::addCovisibleKF(const shared_ptr<KeyFrame> pKF){
    mCovisibleKFs.insert(pKF);
}

set<PtrKeyFrame> KeyFrame::getAllCovisibleKFs(){
    return mCovisibleKFs;
}


set<PtrMapPoint> KeyFrame::getAllObsMPs(bool checkParallax){
    locker lock(mMutexObs);
    set<PtrMapPoint> spMP;
    auto i = mObservations.begin(), iend = mObservations.end();
    for(; i != iend; i++){
        PtrMapPoint pMP = i->first;
        if( pMP->isNull() )
            continue;
        if( checkParallax && !pMP->isGoodPrl() )
            continue;
        spMP.insert(pMP);
    }
    return spMP;
}


bool KeyFrame::isNull(){
    return mbNull;
}

bool KeyFrame::hasObservation(const PtrMapPoint &pMP){
    locker lock(mMutexObs);
    map<PtrMapPoint,int>::iterator it = mObservations.find(pMP);
    return(it!=mObservations.end());
}

bool KeyFrame::hasObservation(int idx) {
    locker lock(mMutexObs);
    auto it = mDualObservations.find(idx);
    return(it != mDualObservations.end());
}

Mat KeyFrame::getPose(){
    locker lock(mMutexPose);
    return Tcw.clone();
}

void KeyFrame::setPose(const Mat &_Tcw){
    locker lock(mMutexPose);
    _Tcw.copyTo(Tcw);
    Twb.fromCvSE3(cvu::inv(Tcw) * Config::cTb);
}
void KeyFrame::setPose(const Se2 &_Twb)
{
    locker lock(mMutexPose);
    Twb = _Twb;
    Tcw = Config::cTb * Twb.inv().toCvSE3();
}

void KeyFrame::addObservation(PtrMapPoint pMP, int idx){
    locker lock(mMutexObs);
    if(pMP->isNull())
        return;
    mObservations[pMP] = idx;
    mDualObservations[idx] = pMP;
}

map<PtrMapPoint,int> KeyFrame::getObservations(){
    locker lock(mMutexObs);
    return mObservations;
}

void KeyFrame::eraseObservation(const PtrMapPoint pMP){
    locker lock(mMutexObs);
    int idx = mObservations[pMP];
    mObservations.erase(pMP);
    mDualObservations.erase(idx);
}

void KeyFrame::eraseObservation(int idx){
    locker lock(mMutexObs);
    mObservations.erase(mDualObservations[idx]);
    mDualObservations.erase(idx);
}

void KeyFrame::addFtrMeasureFrom(shared_ptr<KeyFrame> pKF, const Mat &_mea, const Mat &_info){
    mFtrMeasureFrom.insert(make_pair(pKF, SE3Constraint(_mea, _info)));
}

void KeyFrame::eraseFtrMeasureFrom(shared_ptr<KeyFrame> pKF) {
    mFtrMeasureFrom.erase(pKF);
}

void KeyFrame::addFtrMeasureTo(shared_ptr<KeyFrame> pKF, const Mat &_mea, const Mat &_info){
    mFtrMeasureTo.insert(make_pair(pKF, SE3Constraint(_mea, _info)));
}

void KeyFrame::eraseFtrMeasureTo(shared_ptr<KeyFrame> pKF) {
    mFtrMeasureTo.erase(pKF);
}

void KeyFrame::setOdoMeasureFrom(shared_ptr<KeyFrame> pKF, const Mat &_mea, const Mat &_info){
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint(_mea, _info));
}
void KeyFrame::setOdoMeasureTo(shared_ptr<KeyFrame> pKF, const Mat &_mea, const Mat &_info){
    mOdoMeasureTo = make_pair(pKF, SE3Constraint(_mea, _info));
}

void KeyFrame::ComputeBoW(ORBVocabulary* _pVoc)
{
    lock_guard<mutex> lck(mMutexDes);
    if(mBowVec.empty() || mFeatVec.empty()) {
        vector<cv::Mat> vCurrentDesc = toDescriptorVector(descriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        _pVoc->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
    mbBowVecExist = true;
}

DBoW2::FeatureVector KeyFrame::GetFeatureVector()
{
//    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mFeatVec;
}

DBoW2::BowVector KeyFrame::GetBowVector()
{
//    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mBowVec;
}

vector<PtrMapPoint> KeyFrame::GetMapPointMatches(){
    vector<PtrMapPoint> ret;
    int numKPs = keyPointsUn.size();
    for (int i=0; i<numKPs; i++) {
        PtrMapPoint pMP;
        std::map<int, PtrMapPoint>::iterator iter;
        iter = mDualObservations.find(i);

        if (iter == mDualObservations.end()) {
            ret.push_back(pMP);
        }
        else {
            pMP = iter->second;
            ret.push_back(pMP);
        }
    }

    return ret;
}

void KeyFrame::setObservation(const PtrMapPoint &pMP, int idx) {
    locker lock(mMutexObs);

    if(mDualObservations.find(idx) == mDualObservations.end())
        return;

    mObservations.erase(mDualObservations[idx]);
    mObservations[pMP] = idx;
    mDualObservations[idx] = pMP;
}

PtrMapPoint KeyFrame::getObservation(int idx){
    locker lock(mMutexObs);

    if(!mDualObservations[idx]){
        printf("This is NULL! /in MP\n");
    }
    return mDualObservations[idx];
}

int KeyFrame::getFtrIdx(const PtrMapPoint &pMP){
    locker lock(mMutexObs);
    if(mObservations.find(pMP) == mObservations.end())
        return -1;
    return mObservations[pMP];
}

}// namespace se2lam
