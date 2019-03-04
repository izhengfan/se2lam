/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Map.h"
#include "cvutil.h"
#include "converter.h"
#include "optimizer.h"
#include "LocalMapper.h"
#include "GlobalMapper.h"
#include "Track.h"

namespace se2lam {
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace g2o;
typedef lock_guard<mutex> locker;

Map::Map():
    isEmpty(true)
{
    mCurrentFramePose = cv::Mat::eye(4,4,CV_32FC1);
}
Map::~Map(){

}

bool Map::empty(){
    return isEmpty;
}


void Map::insertKF(const PtrKeyFrame& pkf){
    locker lock(mMutexGraph);
    mKFs.insert(pkf);
    isEmpty = false;
    mCurrentKF = pkf;
}

PtrKeyFrame Map::getCurrentKF(){
    locker lock(mMutexCurrentKF);
    return mCurrentKF;
}

void Map::setCurrentKF(const PtrKeyFrame &pKF){
    locker lock(mMutexCurrentKF);
    mCurrentKF = pKF;
}



vector<PtrKeyFrame> Map::getAllKF(){
    locker lock(mMutexGraph);
    return vector<PtrKeyFrame>(mKFs.begin(), mKFs.end());
}


vector<PtrMapPoint> Map::getAllMP(){
    locker lock(mMutexGraph);
    return vector<PtrMapPoint>(mMPs.begin(), mMPs.end());
}

void Map::clear(){
    locker lock(mMutexGraph);
    mKFs.clear();
    mMPs.clear();
    mLocalGraphKFs.clear();
    mLocalGraphMPs.clear();
    mRefKFs.clear();
    isEmpty = true;
    KeyFrame::mNextIdKF = 0;
    MapPoint::mNextId = 0;
}

void Map::insertMP(const PtrMapPoint& pmp){
    locker lock(mMutexGraph);
    mMPs.insert(pmp);
}

Mat Map::getCurrentFramePose(){
    locker lock(mMutexCurrentFrame);
    return mCurrentFramePose.clone();
}


void Map::setCurrentFramePose(const Mat &pose){
    locker lock(mMutexCurrentFrame);
    pose.copyTo(mCurrentFramePose);
}

size_t Map::countKFs(){
    locker lock(mMutexGraph);
    return mKFs.size();
}

size_t Map::countMPs(){
    locker lock(mMutexGraph);
    return mMPs.size();
}

void Map::eraseKF(const PtrKeyFrame &pKF) {
    locker lock(mMutexGraph);
    mKFs.erase(pKF);
}

void Map::eraseMP(const PtrMapPoint &pMP) {
    locker lock(mMutexGraph);
    mMPs.erase(pMP);
}


void Map::mergeMP(PtrMapPoint &toKeep, PtrMapPoint &toDelete) {

    std::set<PtrKeyFrame> pKFs = toKeep->getObservations();
    for(auto it = pKFs.begin(), itend = pKFs.end(); it != itend; it++){
        PtrKeyFrame pKF = *it;
        if(pKF->hasObservation(toKeep) && pKF->hasObservation(toDelete)){
            return;
        }
    }
    pKFs = toDelete->getObservations();
    for(auto it = pKFs.begin(), itend = pKFs.end(); it != itend; it++){
        PtrKeyFrame pKF = *it;
        if(pKF->hasObservation(toKeep) && pKF->hasObservation(toDelete)){
            return;
        }
    }

    toDelete->mergedInto(toKeep);
    mMPs.erase(toDelete);

    auto it = std::find(mLocalGraphMPs.begin(), mLocalGraphMPs.end(), toDelete);
    if(it != mLocalGraphMPs.end()) {
        *it = toKeep;
    }

}

void Map::setLocalMapper(LocalMapper *pLocalMapper){
    mpLocalMapper = pLocalMapper;
}

bool Map::pruneRedundantKF(){

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGraph);

    bool pruned = false;
    int prunedIdxLocalKF = -1;
    //int count;

    if (mLocalGraphKFs.size() <= 3) {
        return pruned;
    }

    vector<int> goodIdxLocalKF;
    vector<int> goodIdxLocalMP;

    for(int i = 0, iend = mLocalGraphKFs.size(); i < iend; i++){

        bool prunedThis = false;

        if (!pruned) {

            PtrKeyFrame thisKF = mLocalGraphKFs[i];

            // The first and the current KF should not be pruned
            if(thisKF->isNull() || mCurrentKF->mIdKF == thisKF->mIdKF || thisKF->mIdKF <= 1)
                continue;

            // Count how many KF share same MPs more than 80% of each's observation
//            count = 0;
//            set<PtrKeyFrame> covKFs = thisKF->getAllCovisibleKFs();
//            for(auto j = covKFs.begin(), jend = covKFs.end(); j!=jend; j++ ) {
//                PtrKeyFrame pKFj = *j;
//                if(pKFj->isNull())
//                    continue;
//                set<PtrMapPoint> spMPs;
//                Point2f ratio = compareViewMPs(thisKF, pKFj, spMPs);
//                //            if(ratio.x > 0.8f && ratio.y > 0.8f) {
//                //                count++;
//                //            }
//                if (ratio.x > 0.6f) {
//                    count++;
//                }
//            }
//            bool bIsThisKFRedundant = (count >= 2);

            // Count MPs in thisKF observed by covKFs 2 times
            set<PtrMapPoint> spMPs;
            set<PtrKeyFrame> covKFs = thisKF->getAllCovisibleKFs();
            double ratio = compareViewMPs(thisKF, covKFs, spMPs, 2);
            bool bIsThisKFRedundant = (ratio >= 0.8);


            // Do Prune if pass threashold test
            if(bIsThisKFRedundant) {
                PtrKeyFrame lastKF = thisKF->mOdoMeasureTo.first;
                PtrKeyFrame nextKF = thisKF->mOdoMeasureFrom.first;

                bool bIsInitKF = (thisKF->mIdKF == 0);
                bool bHasFeatEdge = (thisKF->mFtrMeasureFrom.size() != 0);

                // Prune this KF and link a new odometry constrait
                if(lastKF && nextKF && !bIsInitKF && !bHasFeatEdge) {

                    Se2 dOdoLastThis = thisKF->odom - lastKF->odom;
                    Se2 dOdoThisNext = nextKF->odom - thisKF->odom;

                    double theshl = 10000;
                    double thesht = 45*3.1415926/180;

                    double dl1 = sqrt(dOdoLastThis.x*dOdoLastThis.x
                                      + dOdoLastThis.y*dOdoLastThis.y);
                    double dt1 = abs(dOdoLastThis.theta);
                    double dl2 = sqrt(dOdoThisNext.x*dOdoThisNext.x
                                      + dOdoThisNext.y*dOdoThisNext.y);
                    double dt2 = abs(dOdoThisNext.theta);

                    if (dl1 < theshl &&  dl2 < theshl && dt1 < thesht && dt2 < thesht) {
                        mKFs.erase(thisKF);
                        thisKF->setNull(thisKF);

                        Mat measure;
                        g2o::Matrix6d info;
                        Track::calcOdoConstraintCam(nextKF->odom - lastKF->odom, measure, info);
                        nextKF->setOdoMeasureTo(lastKF, measure, toCvMat6f(info));
                        lastKF->setOdoMeasureFrom(nextKF, measure, toCvMat6f(info));

                        nextKF->addCovisibleKF(lastKF);
                        lastKF->addCovisibleKF(nextKF);

                        printf("!! INFO MP: Prune KF %d\n", thisKF->mIdKF);

                        pruned = true;
                        prunedThis = true;
                        prunedIdxLocalKF = i;
                    }
                }
            }
        }

        if (!prunedThis) {
            goodIdxLocalKF.push_back(i);
        }
    }

    // Remove useless MP
    if(pruned) {
        for(int i = 0, iend = mLocalGraphMPs.size(); i < iend; i++) {
            PtrMapPoint pMP = mLocalGraphMPs[i];
            if (pMP->isNull()) {
                mMPs.erase(pMP);
            }
            else {
                goodIdxLocalMP.push_back(i);
            }
        }
    }

    if(pruned) {
        vector<PtrMapPoint> vpMPs;
        vector<PtrKeyFrame> vpKFs;
        vpMPs.reserve(goodIdxLocalMP.size());
        vpKFs.reserve(goodIdxLocalKF.size());

        for(int i = 0, iend = goodIdxLocalKF.size(); i < iend; i++) {
            vpKFs.push_back(mLocalGraphKFs[goodIdxLocalKF[i]]);
        }

        for(int i = 0, iend = goodIdxLocalMP.size(); i < iend; i++) {
            vpMPs.push_back(mLocalGraphMPs[goodIdxLocalMP[i]]);
        }

        std::swap(vpMPs, mLocalGraphMPs);
        std::swap(vpKFs, mLocalGraphKFs);
    }

    return pruned;
}

void Map::updateLocalGraph(){

    locker lock(mMutexLocalGraph);

    mLocalGraphKFs.clear();
    mRefKFs.clear();
    mLocalGraphMPs.clear();

    std::set<PtrKeyFrame, KeyFrame::IdLessThan> setLocalKFs;
    std::set<PtrKeyFrame, KeyFrame::IdLessThan> setRefKFs;
    std::set<PtrMapPoint, MapPoint::IdLessThan> setLocalMPs;

    setLocalKFs.insert(mCurrentKF);

    int searchLevel = 3;
    while(searchLevel>0){
        std::set<PtrKeyFrame, KeyFrame::IdLessThan> currentLocalKFs = setLocalKFs;
        for(auto i = currentLocalKFs.begin(), iend = currentLocalKFs.end(); i != iend; i++){
            PtrKeyFrame pKF = (*i);
            std::set<PtrKeyFrame> pKFs = pKF->getAllCovisibleKFs();
            setLocalKFs.insert(pKFs.begin(), pKFs.end());
        }
        searchLevel--;
    }

    for(auto i = setLocalKFs.begin(), iend = setLocalKFs.end(); i != iend; i++){
        PtrKeyFrame pKF = *i;
        bool checkPrl = false;
        set<PtrMapPoint> pMPs = pKF->getAllObsMPs(checkPrl);
        setLocalMPs.insert(pMPs.begin(), pMPs.end());
    }

    setRefKFs.clear();
    for(auto i = setLocalMPs.begin(), iend = setLocalMPs.end(); i != iend; i++){
        PtrMapPoint pMP = (*i);
        std::set<PtrKeyFrame> pKFs = pMP->getObservations();
        for(auto j = pKFs.begin(), jend = pKFs.end(); j != jend; j++){
            if(setLocalKFs.find((*j)) != setLocalKFs.end() || setRefKFs.find((*j)) != setRefKFs.end())
                continue;
            setRefKFs.insert((*j));
        }
    }

    mLocalGraphKFs = vector<PtrKeyFrame>(setLocalKFs.begin(), setLocalKFs.end());
    mRefKFs = vector<PtrKeyFrame>(setRefKFs.begin(), setRefKFs.end());
    mLocalGraphMPs = vector<PtrMapPoint>(setLocalMPs.begin(), setLocalMPs.end());
}

void Map::mergeLoopClose(const std::map<int, int> &mapMatchMP, PtrKeyFrame& pKFCurr, PtrKeyFrame& pKFLoop){
    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGraph);

    pKFCurr->addCovisibleKF(pKFLoop);
    pKFLoop->addCovisibleKF(pKFCurr);

    for (auto iter = mapMatchMP.begin(); iter != mapMatchMP.end(); iter++) {
        int idKPCurr = iter->first;
        int idKPLoop = iter->second;

        if (pKFCurr->hasObservation(idKPCurr) && pKFLoop->hasObservation(idKPLoop)) {
            PtrMapPoint pMPCurr = pKFCurr->getObservation(idKPCurr);
            if(!pMPCurr) cerr << "This is NULL /in M::mergeLoopClose \n";
            PtrMapPoint pMPLoop = pKFLoop->getObservation(idKPLoop);
            if(!pMPLoop) cerr << "This is NULL /in M::mergeLoopClose \n";
            mergeMP(pMPLoop, pMPCurr);
        }
    }
}

Point2f Map::compareViewMPs(const PtrKeyFrame &pKF1, const PtrKeyFrame &pKF2, set<PtrMapPoint> &spMPs){
    int nSameMP = 0;
    spMPs.clear();
    bool checkPrl = false;
    set<PtrMapPoint> setMPs = pKF1->getAllObsMPs(checkPrl);
    for(auto i = setMPs.begin(), iend = setMPs.end(); i != iend; i++){
        PtrMapPoint pMP = *i;
        if(pKF2->hasObservation(pMP)){
            spMPs.insert(pMP);
            nSameMP++;
        }
    }

    return Point2f( (float)nSameMP / (float)(pKF1->getSizeObsMP()),
                    (float)nSameMP / (float)(pKF2->getSizeObsMP()) );
}

// Find MPs in pKF which are observed by vpKFs for k times
// MPs are returned by vpMPs
double Map::compareViewMPs(const PtrKeyFrame & pKFNow, const set<PtrKeyFrame> & spKFsRef, set<PtrMapPoint> & spMPsRet, int k) {

    spMPsRet.clear();
    set<PtrMapPoint> spMPsAll = pKFNow->getAllObsMPs();
    if (spMPsAll.size() == 0) {
        return -1;
    }

    for (auto iter = spMPsAll.begin(); iter != spMPsAll.end(); iter++) {

        PtrMapPoint pMP = *iter;
        int count = 0;

        for (auto iter2 = spKFsRef.begin(); iter2 != spKFsRef.end(); iter2++) {
            PtrKeyFrame pKFRef = *iter2;
            if (pKFRef->hasObservation(pMP)) {
                count++;
            }
        }

        if (count >= k) {
            spMPsRet.insert(pMP);
        }
    }

    double ratio = spMPsRet.size()*1.0/spMPsAll.size();
    return ratio;
}

int Map::countLocalKFs()
{
    locker lock(mMutexLocalGraph);
    return mLocalGraphKFs.size();
}

int Map::countLocalMPs()
{
    locker lock(mMutexLocalGraph);
    return mLocalGraphMPs.size();
}

void Map::loadLocalGraph(SlamOptimizer &optimizer, vector< vector<EdgeProjectXYZ2UV*> > &vpEdgesAll, vector< vector<int> >& vnAllIdx){

    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    int maxKFid = -1;
    int minKFid = -1;
    // If no reference KF, the KF with minId should be fixed

    if(mRefKFs.empty()){
        minKFid = (*(mLocalGraphKFs.begin()))->mIdKF;
        for(auto i = mLocalGraphKFs.begin(), iend = mLocalGraphKFs.end(); i != iend; i++){
            PtrKeyFrame pKF = *i;
            if(pKF->isNull())
                continue;
            if( pKF->mIdKF <  minKFid)
                minKFid = pKF->mIdKF;
        }
    }

    const int nLocalKFs = mLocalGraphKFs.size();
    const int nRefKFs = mRefKFs.size();

    // Add local graph KeyFrames
    for(int i = 0; i < nLocalKFs; i++){

        PtrKeyFrame pKF = mLocalGraphKFs[i];

        if(pKF->isNull())
            continue;

        int vertexIdKF = i;

        bool fixed = (pKF->mIdKF == minKFid) || pKF->mIdKF == 1;
        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, fixed);
        addPlaneMotionSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, Config::bTc);

    }

    // Add odometry based constraints
    for(int i = 0; i < nLocalKFs; i++){

        PtrKeyFrame pKF = mLocalGraphKFs[i];

        if(pKF->isNull())
            continue;

        PtrKeyFrame pKF1 = pKF->mOdoMeasureFrom.first;
        auto it = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF1);
        if( it == mLocalGraphKFs.end() || pKF1->isNull())
            continue;

        int id1 = it - mLocalGraphKFs.begin();

        g2o::Matrix6d info = toMatrix6d(pKF->mOdoMeasureFrom.second.info);
        addEdgeSE3Expmap(optimizer, toSE3Quat(pKF->mOdoMeasureFrom.second.measure), id1, i, info);

    }

    // Add Reference KeyFrames as fixed
    for(int i = 0; i < nRefKFs; i++){

        PtrKeyFrame pKF = mRefKFs[i];

        if(pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs + 1;

    // Store xyz2uv edges
    const int N = mLocalGraphMPs.size();
    vpEdgesAll.clear();
    vpEdgesAll.reserve(N);
    vnAllIdx.clear();
    vnAllIdx.reserve(N);

    const float delta = Config::TH_HUBER;

    // Add local graph MapPoints
    for(int i = 0; i < N; i++){

        PtrMapPoint pMP = mLocalGraphMPs[i];
        assert(!pMP->isNull());

        int vertexIdMP = maxKFid + i;

        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), vertexIdMP );

        std::set<PtrKeyFrame> pKFs = pMP->getObservations();
        vector<EdgeProjectXYZ2UV*> vpEdges;
        vector<int> vnIdx;

        if(!pMP->isGoodPrl()) {
            vpEdgesAll.push_back(vpEdges);
            vnAllIdx.push_back(vnIdx);
            continue;
        }

        for(auto j = pKFs.begin(), jend = pKFs.end(); j!=jend;  j++){

            PtrKeyFrame pKF = (*j);
            if(pKF->isNull())
                continue;

            if(checkAssociationErr(pKF, pMP) ) {
                printf("!! ERR MP: Wrong Association! [LocalGraph loading] \n");
                continue;
            }

            int ftrIdx = pMP->getFtrIdx(pKF);
            int octave = pMP->getOctave(pKF);
            const float invSigma2 = pKF->mvInvLevelSigma2[octave];
            Eigen::Vector2d uv = toVector2d( pKF->keyPointsUn[ftrIdx].pt );
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

            int vertexIdKF = -1;

            auto it1 = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF);
            if(it1 != mLocalGraphKFs.end()) {
                vertexIdKF =  it1 - mLocalGraphKFs.begin();
            }
            else {
                auto it2 = std::find(mRefKFs.begin(), mRefKFs.end(), pKF);
                if(it2 != mRefKFs.end())
                    vertexIdKF = it2 - mRefKFs.begin() + nLocalKFs;
            }

            if(vertexIdKF == -1)
                continue;

            EdgeProjectXYZ2UV* ei = addEdgeXYZ2UV(optimizer, uv, vertexIdMP, vertexIdKF, camParaId, info, delta);
            ei->setLevel(0);

            optimizer.addEdge(ei);
            vpEdges.push_back(ei);

            vnIdx.push_back(vertexIdKF);
        }

        vpEdgesAll.push_back(vpEdges);
        vnAllIdx.push_back(vnIdx);
    }
    assert(vpEdgesAll.size() == mLocalGraphMPs.size());
    assert(vnAllIdx.size() == mLocalGraphMPs.size());

}

void Map::loadLocalGraphOnlyBa(SlamOptimizer &optimizer, vector< vector<EdgeProjectXYZ2UV*> > &vpEdgesAll, vector< vector<int> >& vnAllIdx){

    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    int maxKFid = -1;
    int minKFid = -1;
    // If no reference KF, the KF with minId should be fixed

    if(mRefKFs.empty()){
        minKFid = (*(mLocalGraphKFs.begin()))->mIdKF;
        for(auto i = mLocalGraphKFs.begin(), iend = mLocalGraphKFs.end(); i != iend; i++){
            PtrKeyFrame pKF = *i;
            if(pKF->isNull())
                continue;
            if( pKF->mIdKF <  minKFid)
                minKFid = pKF->mIdKF;
        }
    }

    const int nLocalKFs = mLocalGraphKFs.size();
    const int nRefKFs = mRefKFs.size();

    // Add local graph KeyFrames
    for(int i = 0; i < nLocalKFs; i++){

        PtrKeyFrame pKF = mLocalGraphKFs[i];

        if(pKF->isNull())
            continue;

        int vertexIdKF = i;

        bool fixed = (pKF->mIdKF == minKFid) || pKF->mIdKF == 1;
        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, fixed);
    }

    // Add Reference KeyFrames as fixed
    for(int i = 0; i < nRefKFs; i++){

        PtrKeyFrame pKF = mRefKFs[i];

        if(pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs + 1;

    // Store xyz2uv edges
    const int N = mLocalGraphMPs.size();
    vpEdgesAll.clear();
    vpEdgesAll.reserve(N);
    vnAllIdx.clear();
    vnAllIdx.reserve(N);

    const float delta = Config::TH_HUBER;

    // Add local graph MapPoints
    for(int i = 0; i < N; i++){

        PtrMapPoint pMP = mLocalGraphMPs[i];
        assert(!pMP->isNull());

        int vertexIdMP = maxKFid + i;

        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), vertexIdMP );

        std::set<PtrKeyFrame> pKFs = pMP->getObservations();
        vector<EdgeProjectXYZ2UV*> vpEdges;
        vector<int> vnIdx;

        if(!pMP->isGoodPrl()) {
            vpEdgesAll.push_back(vpEdges);
            vnAllIdx.push_back(vnIdx);
            continue;
        }

        for(auto j = pKFs.begin(), jend = pKFs.end(); j!=jend;  j++){

            PtrKeyFrame pKF = (*j);
            if(pKF->isNull())
                continue;

            if(checkAssociationErr(pKF, pMP) ) {
                printf("!! ERR MP: Wrong Association! [LocalGraph loading] \n");
                continue;
            }

            int ftrIdx = pMP->getFtrIdx(pKF);
            int octave = pMP->getOctave(pKF);
            const float invSigma2 = pKF->mvInvLevelSigma2[octave];
            Eigen::Vector2d uv = toVector2d( pKF->keyPointsUn[ftrIdx].pt );
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

            int vertexIdKF = -1;

            auto it1 = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF);
            if(it1 != mLocalGraphKFs.end()) {
                vertexIdKF =  it1 - mLocalGraphKFs.begin();
            }
            else {
                auto it2 = std::find(mRefKFs.begin(), mRefKFs.end(), pKF);
                if(it2 != mRefKFs.end())
                    vertexIdKF = it2 - mRefKFs.begin() + nLocalKFs;
            }

            if(vertexIdKF == -1)
                continue;

            EdgeProjectXYZ2UV* ei = addEdgeXYZ2UV(optimizer, uv, vertexIdMP, vertexIdKF, camParaId, info, delta);
            ei->setLevel(0);

            optimizer.addEdge(ei);
            vpEdges.push_back(ei);

            vnIdx.push_back(vertexIdKF);
        }

        vpEdgesAll.push_back(vpEdges);
        vnAllIdx.push_back(vnIdx);
    }
    assert(vpEdgesAll.size() == mLocalGraphMPs.size());
    assert(vnAllIdx.size() == mLocalGraphMPs.size());

}

int Map::removeLocalOutlierMP(const vector< vector<int> > &vnOutlierIdxAll){

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGraph);

    assert(vnOutlierIdxAll.size() == mLocalGraphMPs.size());

    const int nLocalKFs = mLocalGraphKFs.size();
    const int N = mLocalGraphMPs.size();
    int nBadMP = 0;

    for(int i = 0, iend = N; i < iend; i++) {

        PtrMapPoint pMP = mLocalGraphMPs[i];

        for(int j = 0, jend = vnOutlierIdxAll[i].size(); j < jend ; j++) {
            PtrKeyFrame pKF = NULL;
            int idxKF = vnOutlierIdxAll[i][j];

            if(idxKF < nLocalKFs){
                pKF = mLocalGraphKFs[idxKF];
            } else {
                pKF = mRefKFs[idxKF - nLocalKFs];
            }

            if(pKF == NULL) {
                printf("!! ERR MP: KF In Outlier Edge Not In LocalKF or RefKF! at line number %d in file %s\n", __LINE__, __FILE__);
                exit(-1);
                continue;
            }

            if(!pKF->hasObservation(pMP))
                continue;

            if( checkAssociationErr(pKF, pMP) ){
                printf("!! ERR MP: Wrong Association [Outlier removal] ! at line number %d in file %s\n", __LINE__, __FILE__);
                exit(-1);
                continue;
            }

            pMP->eraseObservation(pKF);
            pKF->eraseObservation(pMP);
        }

        if(pMP->countObservation() < 2){
            mMPs.erase(pMP);
            pMP->setNull(pMP);
            nBadMP++;
        }

    }
    return nBadMP;
}

void Map::optimizeLocalGraph(SlamOptimizer &optimizer){

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGraph);

    const int nLocalKFs = mLocalGraphKFs.size();
    const int nRefKFs = mRefKFs.size();
    const int N = mLocalGraphMPs.size();
    const int maxKFid = nLocalKFs + nRefKFs + 1;

    for(int i = 0; i < nLocalKFs; i++) {
        PtrKeyFrame pKF = mLocalGraphKFs[i];
        if(pKF->isNull())
            continue;
        Eigen::Vector3d vp = estimateVertexSE2(optimizer, i).toVector();
        pKF->setPose(Se2(vp(0), vp(1), vp(2)));

    }

    for(int i = 0; i < N; i++) {
        PtrMapPoint pMP = mLocalGraphMPs[i];
        if(pMP->isNull() || !pMP->isGoodPrl())
            continue;

        Point3f pos = toCvPt3f( estimateVertexSBAXYZ(optimizer, i+maxKFid) );
        pMP->setPos(pos);
        pMP->updateMeasureInKFs();
    }

}

void Map::updateCovisibility(PtrKeyFrame &pNewKF){

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGraph);

    for(auto i = mLocalGraphKFs.begin(), iend = mLocalGraphKFs.end(); i!=iend; i++){
        set<PtrMapPoint> spMPs;
        PtrKeyFrame pKFi = *i;
        compareViewMPs(pNewKF, pKFi, spMPs);
        if(spMPs.size() > 0.3f * pNewKF->getSizeObsMP()){
            pNewKF->addCovisibleKF(pKFi);
            pKFi->addCovisibleKF(pNewKF);
        }
    }
}

vector<PtrKeyFrame> Map::getLocalKFs(){
    locker lock(mMutexLocalGraph);
    return vector<PtrKeyFrame>(mLocalGraphKFs.begin(), mLocalGraphKFs.end());
}

vector<PtrMapPoint> Map::getLocalMPs(){
    locker lock(mMutexLocalGraph);
    return vector<PtrMapPoint>(mLocalGraphMPs.begin(), mLocalGraphMPs.end());
}

vector<PtrKeyFrame> Map::getRefKFs(){
    locker lock(mMutexLocalGraph);
    return vector<PtrKeyFrame>(mRefKFs.begin(), mRefKFs.end());
}

bool Map::checkAssociationErr(const PtrKeyFrame &pKF, const PtrMapPoint &pMP){
    int ftrIdx0 = pKF->getFtrIdx(pMP);
    int ftrIdx1 = pMP->getFtrIdx(pKF);
    if(ftrIdx0 != ftrIdx1){
        printf("!! ERR AS: pKF->pMP / pMP->pKF: %d / %d\n", ftrIdx0, ftrIdx1);
    }
    return (ftrIdx0 != ftrIdx1);
}

// Select KF pairs to creat feature constraint between which
vector <pair <PtrKeyFrame, PtrKeyFrame> > Map::SelectKFPairFeat(const PtrKeyFrame& _pKF) {

    vector<pair <PtrKeyFrame, PtrKeyFrame> > _vKFPairs;

    // Smallest distance between KFs in covis-graph to create a new feature edge
    int threshCovisGraphDist = 5;

    set<PtrKeyFrame> sKFSelected;
    set<PtrKeyFrame> sKFCovis = _pKF->getAllCovisibleKFs();
    set<PtrKeyFrame> sKFLocal = GlobalMapper::GetAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);

    for (auto iter = sKFCovis.begin(); iter != sKFCovis.end(); iter++) {

        PtrKeyFrame _pKFCand = *iter;
        if (sKFLocal.count(_pKFCand) == 0) {
            sKFSelected.insert(*iter);
            sKFLocal = GlobalMapper::GetAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);
        }
        else {
            continue;
        }
    }

    _vKFPairs.clear();
    for (auto iter = sKFSelected.begin(); iter != sKFSelected.end(); iter++) {
        _vKFPairs.push_back(make_pair(_pKF, *iter));
    }

    return _vKFPairs;
}

bool Map::UpdateFeatGraph(const PtrKeyFrame& _pKF) {

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGraph);

    vector<pair<PtrKeyFrame, PtrKeyFrame> > _vKFPairs = SelectKFPairFeat(_pKF);

    if (_vKFPairs.empty())
        return false;

    int numPairKFs = _vKFPairs.size();
    for (int i = 0; i<numPairKFs; i++) {
        pair<PtrKeyFrame, PtrKeyFrame> pairKF = _vKFPairs[i];
        PtrKeyFrame ptKFFrom = pairKF.first;
        PtrKeyFrame ptKFTo = pairKF.second;
        SE3Constraint ftrCnstr;

        if (GlobalMapper::CreateFeatEdge(ptKFFrom, ptKFTo, ftrCnstr) == 0) {
            ptKFFrom->addFtrMeasureFrom(ptKFTo, ftrCnstr.measure, ftrCnstr.info);
            ptKFTo->addFtrMeasureTo(ptKFFrom, ftrCnstr.measure, ftrCnstr.info);
            if (Config::GLOBAL_PRINT) {
                cerr << "## DEBUG GM: add feature constraint from " << ptKFFrom->id
                    << " to " << ptKFTo->id << endl;
            }
        }
        else {
            if (Config::GLOBAL_PRINT)
                cerr << "## DEBUG GM: add feature constraint failed" << endl;
        }
    }

    return true;
}

void Map::loadLocalGraph(SlamOptimizer &optimizer)
{

    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    CamPara* campr = addCamPara(optimizer, (Config::Kcam), camParaId);

    int maxKFid = -1;
    int minKFid = -1;
    // If no reference KF, the KF with minId should be fixed

    if(mRefKFs.empty()){
        minKFid = (*(mLocalGraphKFs.begin()))->id;
        for(auto i = mLocalGraphKFs.begin(), iend = mLocalGraphKFs.end(); i != iend; i++){
            PtrKeyFrame pKF = *i;
            if(pKF->isNull())
                continue;
            if( pKF->id <  minKFid)
                minKFid = pKF->id;
        }
    }

    const int nLocalKFs = mLocalGraphKFs.size();
    const int nRefKFs = mRefKFs.size();

    // Add local graph KeyFrames
    for(int i = 0; i < nLocalKFs; i++){

        PtrKeyFrame pKF = mLocalGraphKFs[i];

        if(pKF->isNull())
            continue;

        int vertexIdKF = i;

        bool fixed = (pKF->id == minKFid) || pKF->id == 1;

        g2o::SE2 pose(pKF->Twb.x, pKF->Twb.y, pKF->Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, fixed);

    }

    // Add odometry based constraints
    for(int i = 0; i < nLocalKFs; i++){

        PtrKeyFrame pKF = mLocalGraphKFs[i];

        if(pKF->isNull())
            continue;

        PtrKeyFrame pKF1 = pKF->preOdomFromSelf.first;
        PreSE2 meas = pKF->preOdomFromSelf.second;
        auto it = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF1);
        if( it == mLocalGraphKFs.end() || pKF1->isNull())
            continue;

        int id1 = it - mLocalGraphKFs.begin();

        {
            Eigen::Map<Eigen::Matrix3d, RowMajor> info(meas.cov);
            addEdgeSE2(optimizer, Vector3D(meas.meas), i, id1, info);
        }

    }

    // Add Reference KeyFrames as fixed
    for(int i = 0; i < nRefKFs; i++){

        PtrKeyFrame pKF = mRefKFs[i];

        if(pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        g2o::SE2 pose(pKF->Twb.x, pKF->Twb.y, pKF->Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs + 1;

    // Store xyz2uv edges
    const int N = mLocalGraphMPs.size();

    const float delta = Config::TH_HUBER;

    // Add local graph MapPoints
    for(int i = 0; i < N; i++){

        PtrMapPoint pMP = mLocalGraphMPs[i];
        assert(!pMP->isNull());

        int vertexIdMP = maxKFid + i;
        Vector3d lw = toVector3d(pMP->getPos());

        addVertexSBAXYZ(optimizer, lw, vertexIdMP );

        std::set<PtrKeyFrame> pKFs = pMP->getObservations();

        for(auto j = pKFs.begin(), jend = pKFs.end(); j!=jend;  j++){

            PtrKeyFrame pKF = (*j);
            if(pKF->isNull())
                continue;

            if(checkAssociationErr(pKF, pMP) ) {
                printf("!! ERR MP: Wrong Association! [LocalGraph loading] \n");
                continue;
            }

            int ftrIdx = pMP->getFtrIdx(pKF);
            int octave = pMP->getOctave(pKF);
            const float Sigma2 = pKF->mvLevelSigma2[octave];
            Eigen::Vector2d uv = toVector2d( pKF->keyPointsUn[ftrIdx].pt );


            int vertexIdKF = -1;

            auto it1 = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF);
            if(it1 != mLocalGraphKFs.end()) {
                vertexIdKF =  it1 - mLocalGraphKFs.begin();
            }
            else {
                auto it2 = std::find(mRefKFs.begin(), mRefKFs.end(), pKF);
                if(it2 != mRefKFs.end())
                    vertexIdKF = it2 - mRefKFs.begin() + nLocalKFs;
            }

            if(vertexIdKF == -1)
                continue;

            // compute covariance/information
            Matrix2d Sigma_u = Eigen::Matrix2d::Identity() * Sigma2;
            Vector3d lc = toVector3d( pKF->mViewMPs[ftrIdx] );

            double zc = lc(2);
            double zc_inv = 1. / zc;
            double zc_inv2 = zc_inv * zc_inv;
                    const float& fx = Config::fxCam;
            Matrix23d J_pi;
            J_pi << fx * zc_inv, 0, -fx*lc(0)*zc_inv2,
                    0, fx * zc_inv, -fx*lc(1)*zc_inv2;
            Matrix3d Rcw = toMatrix3d(pKF->Tcw.rowRange(0,3).colRange(0,3));
            Vector3d pi(pKF->Twb.x, pKF->Twb.y, 0);


            Matrix23d J_pi_Rcw = J_pi * Rcw;

            Matrix2d J_rotxy = (J_pi_Rcw * skew(lw-pi)).block<2,2>(0,0);
            Matrix<double,2,1> J_z = -J_pi_Rcw.block<2,1>(0,2);
            float Sigma_rotxy = 1./Config::PLANEMOTION_XROT_INFO;
            float Sigma_z = 1./Config::PLANEMOTION_Z_INFO;
            Matrix2d Sigma_all = Sigma_rotxy*J_rotxy*J_rotxy.transpose() +
                                 Sigma_z* J_z*J_z.transpose() + Sigma_u;

            addEdgeSE2XYZ(optimizer, uv, vertexIdKF, vertexIdMP, campr,
                          toSE3Quat(Config::bTc), Sigma_all.inverse(), delta);

        }
    }
}

}// namespace se2lam
