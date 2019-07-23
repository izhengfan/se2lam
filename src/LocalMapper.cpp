/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "LocalMapper.h"
#include <condition_variable>
#include <ros/ros.h>
#include "GlobalMapper.h"
#include "cvutil.h"
#include "optimizer.h"
#include "Track.h"
#include "converter.h"
#include "ORBmatcher.h"
#include <fstream>

namespace se2lam {

using namespace std;
using namespace cv;
using namespace g2o;


#ifdef TIME_TO_LOG_LOCAL_BA
std::ofstream local_ba_time_log;
#endif
typedef lock_guard<mutex> locker;

LocalMapper::LocalMapper(){
    mbUpdated = false;
    mbAbortBA = false;
    mbAcceptNewKF = true;
    mbGlobalBABegin = false;
    mbPrintDebugInfo = false;    

    mbFinished = false;
    mbFinishRequested = false;
}

void LocalMapper::setMap(Map *pMap) {
    mpMap = pMap;
    mpMap->setLocalMapper(this);
}

void LocalMapper::setGlobalMapper(GlobalMapper *pGlobalMapper){
    mpGlobalMapper = pGlobalMapper;
}


void LocalMapper::addNewKF(PtrKeyFrame& pKF, const vector<Point3f> &localMPs,
                           const vector<int> &vMatched12, const vector<bool>& vbGoodPrl){

    mNewKF = pKF;

    findCorrespd(vMatched12, localMPs, vbGoodPrl);

    mpMap->updateCovisibility(mNewKF);

    {
        PtrKeyFrame pKF0 = mpMap->getCurrentKF();        

        // Add KeyFrame-KeyFrame relation
        {
            // There must be covisibility between NewKF and PrevKF
            pKF->addCovisibleKF(pKF0);
            pKF0->addCovisibleKF(pKF);
            Mat measure;
            g2o::Matrix6d info;
            Track::calcOdoConstraintCam(pKF->odom - pKF0->odom, measure, info);

            pKF0->setOdoMeasureFrom(pKF, measure, toCvMat6f(info));
            pKF->setOdoMeasureTo(pKF0, measure, toCvMat6f(info));

        }

        mpMap->insertKF(pKF);
        mbUpdated = true;

    }

    mbAbortBA = false;
    mbAcceptNewKF = false;

}

void LocalMapper::findCorrespd(const vector<int> &vMatched12, const vector<Point3f> &localMPs, const vector<bool>& vbGoodPrl){

    bool bNoMP = ( mpMap->countMPs() == 0 );

    // Identify tracked map points
    PtrKeyFrame pPrefKF = mpMap->getCurrentKF();
    if(!bNoMP) {

        for(int i = 0, iend = pPrefKF->N; i < iend; i++) {
            if(pPrefKF->hasObservation(i) && vMatched12[i]>=0){
                PtrMapPoint pMP = pPrefKF->getObservation(i);
                if(!pMP){
                    printf("This is NULL. /in LM\n");
                }
                Eigen::Matrix3d xyzinfo, xyzinfo0;
                Track::calcSE3toXYZInfo(pPrefKF->mViewMPs[i], cv::Mat::eye(4,4,CV_32FC1), mNewKF->Tcr, xyzinfo0, xyzinfo);
                mNewKF->setViewMP( cvu::se3map(mNewKF->Tcr, pPrefKF->mViewMPs[i]), vMatched12[i], xyzinfo );
                mNewKF->addObservation(pMP, vMatched12[i]);
                pMP->addObservation(mNewKF, vMatched12[i]);
            }
        }
    }


    // Match features of MapPoints with those in NewKF
    if(!bNoMP) {

        //vector<PtrMapPoint> vLocalMPs(mLocalGraphMPs.begin(), mLocalGraphMPs.end());
        vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        matcher.MatchByProjection(mNewKF, vLocalMPs, 15, 2, vMatchedIdxMPs);
        for(int i = 0; i < mNewKF->N; i++){
            if(vMatchedIdxMPs[i] < 0)
                continue;
            PtrMapPoint pMP = vLocalMPs[vMatchedIdxMPs[i]];

            // We do triangulation here because we need to produce constraint of
            // mNewKF to the matched old MapPoint.
            Point3f x3d =
            cvu::triangulate(pMP->getMainMeasure(), mNewKF->keyPointsUn[i].pt,
                             Config::Kcam*pMP->mMainKF->Tcw.rowRange(0,3),
                             Config::Kcam*mNewKF->Tcw.rowRange(0,3));
            Point3f posNewKF = cvu::se3map(mNewKF->Tcw, x3d);
            if(!pMP->acceptNewObserve(posNewKF, mNewKF->keyPoints[i])){
                continue;
            }
            if(posNewKF.z > Config::UPPER_DEPTH || posNewKF.z < Config::LOWER_DEPTH)
                continue;
            Eigen::Matrix3d infoNew, infoOld;
            Track::calcSE3toXYZInfo(posNewKF, mNewKF->Tcw, pMP->mMainKF->Tcw, infoNew, infoOld);
            mNewKF->setViewMP(posNewKF, i, infoNew);
            mNewKF->addObservation(pMP, i);
            pMP->addObservation(mNewKF, i);
        }
    }

    // Add new points from mNewKF to the map
    for(int i = 0, iend = pPrefKF->N; i < iend; i++){
        if(!pPrefKF->hasObservation(i) && vMatched12[i]>=0){
            if(mNewKF->hasObservation(vMatched12[i]))
                continue;

            Point3f posW = cvu::se3map(cvu::inv(pPrefKF->Tcw), localMPs[i]);
            Point3f posKF = cvu::se3map(mNewKF->Tcr, localMPs[i]);
            Eigen::Matrix3d xyzinfo, xyzinfo0;
            Track::calcSE3toXYZInfo(localMPs[i], pPrefKF->Tcw, mNewKF->Tcw, xyzinfo0, xyzinfo);

            mNewKF->setViewMP(posKF, vMatched12[i], xyzinfo);
            pPrefKF->setViewMP(localMPs[i], i, xyzinfo0);
            //PtrMapPoint pMP = make_shared<MapPoint>(mNewKF, vMatched12[i], posW, vbGoodPrl[i]);
            PtrMapPoint pMP = make_shared<MapPoint>(posW, vbGoodPrl[i]);

            pMP->addObservation(pPrefKF, i);
            pMP->addObservation(mNewKF, vMatched12[i]);
            pPrefKF->addObservation(pMP, i);
            mNewKF->addObservation(pMP, vMatched12[i]);

            mpMap->insertMP(pMP);

        }
    }

}

void LocalMapper::removeOutlierChi2() {
    std::unique_lock<mutex> lockmapper(mutexMapper);

    SlamOptimizer optimizer;
    initOptimizer(optimizer);

    vector< vector<EdgeProjectXYZ2UV*> > vpEdgesAll;
    vector< vector<int> > vnAllIdx;
    mpMap->loadLocalGraph(optimizer, vpEdgesAll, vnAllIdx);

    WorkTimer timer;
    timer.start();

    const float chi2 = 25;

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    const int nAllMP = vpEdgesAll.size();
    int nBadMP = 0;
    vector< vector<int> > vnOutlierIdxAll;

    for(int i = 0; i < nAllMP; i++ ) {

        vector<int> vnOutlierIdx;
        for(int j = 0, jend = vpEdgesAll[i].size(); j < jend; j++) {

            EdgeProjectXYZ2UV* eij = vpEdgesAll[i][j];

            if(eij->level() > 0)
                continue;

            eij->computeError();
            bool chi2Bad = eij->chi2() > chi2;

            int idKF = vnAllIdx[i][j];

            if(chi2Bad) {
                eij->setLevel(1);
                vnOutlierIdx.push_back(idKF);
            }
        }

        vnOutlierIdxAll.push_back(vnOutlierIdx);

    }

    timer.stop();

    nBadMP = mpMap->removeLocalOutlierMP(vnOutlierIdxAll);

    vpEdgesAll.clear();
    vnAllIdx.clear();

    if (mbPrintDebugInfo) {
        printf("-- DEBUG LM: Remove Outlier Time %f\n", timer.time );
        printf("-- DEBUG LM: Outliers: %d; totally %d\n", nBadMP, nAllMP);
    }
}

void LocalMapper::localBA(){

    if(mbGlobalBABegin)
        return;

    std::unique_lock<mutex> lockmapper(mutexMapper);

    SlamOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(Config::LOCAL_VERBOSE);
#ifndef TIME_TO_LOG_LOCAL_BA
    optimizer.setForceStopFlag(&mbAbortBA);
#endif
    mpMap->loadLocalGraph(optimizer);

    WorkTimer timer;
#ifdef TIME_TO_LOG_LOCAL_BA
    int numKf = mpMap->countLocalKFs();
    int numMp = mpMap->countLocalMPs();

    timer.start();
#endif
    //assert(optimizer.verifyInformationMatrices(true));

    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LOCAL_ITER);

#ifdef TIME_TO_LOG_LOCAL_BA
    timer.stop();

    local_ba_time_log << numKf << " " << numMp << " " << timer.time;

    optimizer.clear();
    optimizer.clearParameters();

    mpMap->loadLocalGraphOnlyBa(optimizer, vpEdgesAll, vnAllIdx);
    timer.start();
    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LOCAL_ITER);
    timer.stop();
    local_ba_time_log << " " << timer.time << std::endl;
#endif

    if (mbPrintDebugInfo) {
    cerr << "-- DEBUG LM: time " << timer.time
         << ", number of KF " << mpMap->getLocalKFs().size()
     //    << ", number of MP " << vpEdgesAll.size()
         << endl;
    }

#ifdef REJECT_IF_LARGE_LAMBDA
    if(solver->currentLambda() > 100.0) {
        cerr << "-- DEBUG LM: current lambda too large "
             << solver->currentLambda()
             << " , reject optimized result" << endl;
        return;
    }
#endif

    if(mbGlobalBABegin){
        return;
    }

#ifndef TIME_TO_LOG_LOCAL_BA
    mpMap->optimizeLocalGraph(optimizer);
#endif

}

void LocalMapper::run(){

    if(Config::LOCALIZATION_ONLY)
        return;

    mbPrintDebugInfo = Config::LOCAL_PRINT;


#ifdef TIME_TO_LOG_LOCAL_BA
    local_ba_time_log.open("/home/fzheng/Documents/se2lam_lobal_time.txt");
#endif

    ros::Rate rate(Config::FPS * 10);

    while(ros::ok()){

        if(mbUpdated) {

            WorkTimer timer;
            timer.start();

            updateLocalGraphInMap();

            pruneRedundantKfInMap();

            //removeOutlierChi2();

            updateLocalGraphInMap();

            localBA();

            timer.stop();

            if (mbPrintDebugInfo) {
               cerr << "-- DEBUG LM: time " << timer.time << " ms." << endl;
            }

            mbUpdated = false;

            mpGlobalMapper->waitIfBusy();

            updateLocalGraphInMap();

        }

        mbAcceptNewKF = true;

        if(checkFinish())
            break;

        rate.sleep();
    }

#ifdef TIME_TO_LOG_LOCAL_BA
    local_ba_time_log.close();
#endif

    cerr << "Exiting localmapper .." << endl;

    setFinish();
}

void LocalMapper::setAbortBA(){
    mbAbortBA = true;
}

bool LocalMapper::acceptNewKF(){
    return mbAcceptNewKF;
}

void LocalMapper::printOptInfo(const SlamOptimizer & _optimizer) {

    // for odometry edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it ++) {
        g2o::EdgeSE3Expmap *pEdge = static_cast<g2o::EdgeSE3Expmap*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (max(id0,id1) > (mNewKF->mIdKF)) {
                // Not odometry edge
                continue;
            }
            cerr << "odometry edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "id1 = " << id1 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";
            cerr << "err = ";
            for (int i=0; i<6; i++) {
                cerr << pEdge->error()(i) << "; ";
            }
            cerr << endl;
        }
    }

    // for plane motion edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it ++) {
        g2o::EdgeSE3Expmap *pEdge = static_cast<g2o::EdgeSE3Expmap*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 1) {

            int id0 = vVertices[0]->id();

            cerr << "plane motion edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";
            cerr << "err = ";
            for (int i=0; i<6; i++) {
                cerr << pEdge->error()(i) << "; ";
            }
            cerr << endl;
        }
    }

    // for XYZ2UV edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it ++) {
        g2o::EdgeProjectXYZ2UV *pEdge = static_cast<g2o::EdgeProjectXYZ2UV*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (max(id0,id1) > (mNewKF->mIdKF)) {
                if (pEdge->chi2() < 10)
                    continue;
                cerr << "XYZ2UV edge: ";
                cerr << "id0 = " << id0 << "; ";
                cerr << "id1 = " << id1 << "; ";
                cerr << "chi2 = " << pEdge->chi2() << "; ";
                cerr << "err = ";
                for (int i=0; i<2; i++) {
                    cerr << pEdge->error()(i) << "; ";
                }
                cerr << endl;
            }
        }
    }
}

void LocalMapper::updateLocalGraphInMap()
{
    unique_lock<mutex> lock(mutexMapper);
    mpMap->updateLocalGraph();
}

void LocalMapper::pruneRedundantKfInMap()
{
    std::unique_lock<mutex> lockmapper(mutexMapper);
    bool bPruned = false;
    int countPrune = 0;
    do
    {
        bPruned = mpMap->pruneRedundantKF();
        countPrune++;
    }
    while (bPruned && countPrune < 5);
}

void LocalMapper::setGlobalBABegin(bool value){
    locker lock(mMutexLocalGraph);
    mbGlobalBABegin = value;
    if(value)
        mbAbortBA = true;
}

void LocalMapper::requestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapper::checkFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool LocalMapper::isFinished() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapper::setFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

} // namespace se2lam

