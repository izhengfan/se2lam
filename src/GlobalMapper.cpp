/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "GlobalMapper.h"
#include <ros/ros.h>
#include "LocalMapper.h"
#include "Map.h"
#include "sparsifier.h"
#include "cvutil.h"

#include <g2o/core/optimizable_graph.h>
#include <opencv2/calib3d/calib3d.hpp>

namespace se2lam {
using namespace cv;
using namespace std;
using namespace g2o;

GlobalMapper::GlobalMapper(){
    mdeqPairKFs.clear();
    mbNewKF = false;
    mbGlobalBALastLoop = false;
    mbExit = false;
    mbIsBusy = false;

    mbFinished = false;
    mbFinishRequested = false;
}

void GlobalMapper::setUpdated(bool val){
    mbUpdated = val;
    mpKFCurr = mpMap->getCurrentKF();
}

void GlobalMapper::setLocalMapper(LocalMapper *pLocalMapper){
    mpLocalMapper = pLocalMapper;
}

void GlobalMapper::setORBVoc(ORBVocabulary* pORBVoc) {
    mpORBVoc = pORBVoc;
}

void GlobalMapper::setMap(Map* pMap) {
    mpMap = pMap;
}

bool GlobalMapper::CheckGMReady() {

    if (mpMap->empty())
        return false;

    PtrKeyFrame pKFCurr = mpMap->getCurrentKF();
    if (pKFCurr != mpKFCurr && pKFCurr != NULL && !pKFCurr->isNull()) {
        mbNewKF = true;
        mpKFCurr = pKFCurr;
        return true;
    }
    else {
        return false;
    }
}

void GlobalMapper::run() {

    mbExit = false;

    if(Config::LOCALIZATION_ONLY)
        return;

    ros::Rate rate(Config::FPS * 10);


    while(ros::ok() && !mbExit){

        if(checkFinish())
            break;

        //! Check if everything is ready for global mapping
        if (!CheckGMReady()) {
            rate.sleep();
            continue;
        }

        bool bIfFeatGraphRenewed = false;
        bool bIfLoopCloseDetected = false;
        bool bIfLoopCloseVerified = false;

        WorkTimer timer;
        timer.start();

        setBusy(true);

        //! Update FeatGraph with Covisibility-Graph
        bIfFeatGraphRenewed = mpMap->UpdateFeatGraph(mpKFCurr);


//        vector<pair<PtrKeyFrame, PtrKeyFrame>> vKFPairs = SelectKFPairFeat(mpKFCurr);
//        if (!vKFPairs.empty()) {
//            UpdataFeatGraph(vKFPairs);
//            bIfFeatGraphRenewed = true;
//        }

        timer.stop();
        double t1 = timer.time;
        timer.start();

        //! Refresh BowVec for all KFs
        ComputeBowVecAll();

        timer.stop();
        double t2 = timer.time;
        timer.start();

        //! Detect loop close
        bIfLoopCloseDetected = DetectLoopClose();

        timer.stop();
        double t3 = timer.time;
        timer.start();

        //! Verify loop close
        map<int,int> mapMatchMP, mapMatchGood, mapMatchRaw;
        if (bIfLoopCloseDetected) {
            std::unique_lock<std::mutex> lock(mpLocalMapper->mutexMapper);
            bIfLoopCloseVerified = VerifyLoopClose(mapMatchMP, mapMatchGood, mapMatchRaw);
        }

        //! Create feature edge from loop close
        // TODO...

        //! Draw Matches
        DrawMatch(mapMatchGood);

        timer.stop();
        double t4 = timer.time;
        timer.start();

        //! Do Global Correction if Needed
        if (!mbGlobalBALastLoop && (bIfLoopCloseVerified || bIfFeatGraphRenewed) ) {
            std::unique_lock<std::mutex> lock(mpLocalMapper->mutexMapper);
#ifndef TIME_TO_LOG_LOCAL_BA
            GlobalBA();
#endif
            mbGlobalBALastLoop = true;
            cerr << "## INFO GM: Loop closed!"
                 << " numKFs = " << mpMap->countKFs() << ", numMPs = " << mpMap->countMPs()
                 << endl;
        }
        else {
            mbGlobalBALastLoop = false;
        }

        timer.stop();
        double t5 = timer.time;

        //! Return
        if (Config::GLOBAL_PRINT) {
            cerr << "## DEBUG GM: " << "loopTime = " << t1+t2+t3+t4+t5
                 << ", numKFs = " << mpMap->countKFs() << ", numMPs = " << mpMap->countMPs()
                 << endl;
        }

        mbNewKF = false;

        setBusy(false);

        rate.sleep();
    }
    cerr << "Exiting globalmapper .." << endl;

    setFinish();
}


void GlobalMapper::UpdataFeatGraph(vector<pair<PtrKeyFrame, PtrKeyFrame>> & _vKFPairs) {
    int numPairKFs = _vKFPairs.size();
    for (int i=0; i<numPairKFs; i++) {
        pair<PtrKeyFrame, PtrKeyFrame> pairKF = _vKFPairs[i];
        PtrKeyFrame ptKFFrom = pairKF.first;
        PtrKeyFrame ptKFTo = pairKF.second;
        SE3Constraint ftrCnstr;

        if (CreateFeatEdge(ptKFFrom, ptKFTo, ftrCnstr) == 0) {
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
}

bool GlobalMapper::DetectLoopClose()
{
    // Loop closure detection with ORB-BOW method

    bool bDetected = false;
    int minKFIdOffset = Config::GM_DCL_MIN_KFID_OFFSET;
    double minScoreBest = Config::GM_DCL_MIN_SCORE_BEST;

    PtrKeyFrame pKFCurr = mpMap->getCurrentKF();
    if (pKFCurr == NULL) {
        return bDetected;
    }
    if (mpLastKFLoopDetect == pKFCurr) {
        return bDetected;
    }

    DBoW2::BowVector BowVecCurr = pKFCurr->mBowVec;
    int idKFCurr = pKFCurr->mIdKF;

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKF();
    int numKFs = vpKFsAll.size();
    PtrKeyFrame pKFBest;
    double scoreBest = 0;

    for(int i=0; i<numKFs; i++) {
        PtrKeyFrame pKF = vpKFsAll[i];
        DBoW2::BowVector BowVec = pKF->mBowVec;

        int idKF = pKF->mIdKF;
        //        int id = pKF->id;

        // Omit neigbor KFs
        if (abs(idKF-idKFCurr) < minKFIdOffset) {
            continue;
        }

        double score = mpORBVoc->score(BowVecCurr, BowVec);
        if (score > scoreBest) {
            scoreBest = score;
            pKFBest = pKF;
        }
    }

    // Loop CLosing Threshold ...
    if (pKFBest != NULL && scoreBest > minScoreBest) {
        mpKFLoop = pKFBest;
        bDetected = true;
    }
    else {
        mpKFLoop.reset();
    }

    return bDetected;
}

bool GlobalMapper::VerifyLoopClose(map<int,int> & _mapMatchMP, map<int,int> & _mapMatchGood, map<int,int> & _mapMatchRaw) {

    _mapMatchMP.clear();
    _mapMatchGood.clear();
    _mapMatchRaw.clear();
    map<int,int> mapMatch;

    bool bVerified = false;
    int numMinMatchMP = Config::GM_VCL_NUM_MIN_MATCH_MP;
    int numMinMatchKP = Config::GM_VCL_NUM_MIN_MATCH_KP;
    double ratioMinMatchMP = Config::GM_VCL_RATIO_MIN_MATCH_MP;

    if (mpKFCurr == NULL || mpKFLoop == NULL) {
        //        cerr << "## DEBUG GM: " << "No good match candidate found!!!" << endl;
        return false;
    }

    //! Match ORB KPs
    ORBmatcher matcher;
    bool bIfMatchMPOnly = false;
    matcher.SearchByBoW(mpKFCurr, mpKFLoop, mapMatch, bIfMatchMPOnly);
    _mapMatchRaw = mapMatch;

    //! Remove Outliers: by RANSAC of Fundamental
    RemoveMatchOutlierRansac(mpKFCurr, mpKFLoop, mapMatch);
    _mapMatchGood = mapMatch;
    int numGoodMatch = mapMatch.size();

    //! Remove all KPs matches
    RemoveKPMatch(mpKFCurr, mpKFLoop, mapMatch);
    _mapMatchMP = mapMatch;
    int numGoodMPMatch = mapMatch.size();

    //! Show Match Info
    std::set<PtrMapPoint> spMPsCurrent = mpKFCurr->getAllObsMPs();
    int numMPsCurrent = spMPsCurrent.size();
    int numKPsCurrent = mpKFCurr->keyPoints.size();

    //! Create New Feature based Constraint
    double ratioMPMatched = numGoodMPMatch*1.0/numMPsCurrent;
    if (numGoodMPMatch >= numMinMatchMP && numGoodMatch >= numMinMatchKP
            && ratioMPMatched >= ratioMinMatchMP) {
        // Generate feature based constraint
        SE3Constraint Se3_Curr_Loop;
        bool bFtrCnstrErr = CreateFeatEdge(mpKFCurr, mpKFLoop, _mapMatchMP, Se3_Curr_Loop);
        if (!bFtrCnstrErr) {
            mpKFCurr->addFtrMeasureFrom(mpKFLoop, Se3_Curr_Loop.measure, Se3_Curr_Loop.info);
            mpKFLoop->addFtrMeasureTo(mpKFCurr, Se3_Curr_Loop.measure, Se3_Curr_Loop.info);
            bVerified = true;
        }

        if (Config::GLOBAL_PRINT) {
            cerr << "## DEBUG GM: " << "add feature constraint from "
                 << mpKFCurr->mIdKF << " to " << mpKFLoop->mIdKF <<  ", MPGood/KPGood/MPNow/KPNow="
                 << numGoodMPMatch << "/" << numGoodMatch << "/" << numMPsCurrent << "/" << numKPsCurrent << endl;
        }
    }
    else {
        if (Config::GLOBAL_PRINT) {
            cerr << "## DEBUG GM: " << "no enough MPs matched, MPGood/KPGood/MPNow/KPNow="
                 << numGoodMPMatch << "/" << numGoodMatch << "/" << numMPsCurrent << "/" << numKPsCurrent << endl;
        }
    }

    //! Renew Co-Visibility Graph and Merge MPs
    if (bVerified) {
        mpMap->mergeLoopClose(_mapMatchMP, mpKFCurr, mpKFLoop);
    }

    return bVerified;
}

void GlobalMapper::GlobalBA() {

    mpLocalMapper->setGlobalBABegin(true);

#ifdef PRE_REJECT_FTR_OUTLIER
    double threshFeatEdgeChi2 = 30.0;
    double threshFeatEdgeChi2Pre = 1000.0;
#endif

    std::vector<PtrKeyFrame> vecKFs = mpMap->getAllKF();

    SlamOptimizer optimizer;
    //initOptimizer(optimizer);
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);

    int SE3OffsetParaId = 0;
    addParaSE3Offset(optimizer, g2o::Isometry3D::Identity(), SE3OffsetParaId);

    int maxKFid = -1;

    // Add all KFs
    map<int, PtrKeyFrame> mapId2pKF;
    vector<g2o::EdgeSE3Prior*> vpEdgePlane;

    for (auto it = vecKFs.begin(); it != vecKFs.end(); it++) {
        PtrKeyFrame pKF = (*it);

        if(pKF->isNull())
            continue;

        Mat T_w_c = cvu::inv(pKF->Tcw);
        bool bIfFix = (pKF->mIdKF == 0);

        //        addVertexSE3(optimizer, toIsometry3D(T_w_c), pKF->mIdKF, bIfFix);
        g2o::EdgeSE3Prior* pEdge = addVertexSE3PlaneMotion(optimizer, toIsometry3D(T_w_c), pKF->mIdKF, Config::bTc, SE3OffsetParaId, bIfFix);
        vpEdgePlane.push_back(pEdge);
        //        pEdge->setLevel(1);

        mapId2pKF[pKF->mIdKF] = pKF;

        if(pKF->mIdKF > maxKFid)
            maxKFid = pKF->mIdKF;
    }

    // Add odometry based constraints
    int numOdoCnstr = 0;
    vector<g2o::EdgeSE3*> vpEdgeOdo;
    for (auto it = vecKFs.begin(); it != vecKFs.end(); it++) {
        PtrKeyFrame pKF = (*it);
        if(pKF->isNull())
            continue;
        if(pKF->mOdoMeasureFrom.first == NULL)
            continue;

        g2o::Matrix6d info = toMatrix6d(pKF->mOdoMeasureFrom.second.info);

        g2o::EdgeSE3* pEdgeOdoTmp = addEdgeSE3(optimizer, toIsometry3D(pKF->mOdoMeasureFrom.second.measure),
                                               pKF->mIdKF, pKF->mOdoMeasureFrom.first->mIdKF, info);
        vpEdgeOdo.push_back(pEdgeOdoTmp);

        numOdoCnstr ++;
    }

    // Add feature based constraints
    int numFtrCnstr = 0;
    vector<g2o::EdgeSE3*> vpEdgeFeat;
    for (auto it = vecKFs.begin(); it != vecKFs.end(); it++) {
        PtrKeyFrame ptrKFFrom = (*it);
        if(ptrKFFrom->isNull())
            continue;

        for (auto it2 = ptrKFFrom->mFtrMeasureFrom.begin();
             it2 != ptrKFFrom->mFtrMeasureFrom.end(); it2++) {

            PtrKeyFrame ptrKFTo = (*it2).first;
            if(std::find(vecKFs.begin(), vecKFs.end(), ptrKFTo)
                    == vecKFs.end())
                continue;

            Mat meas = (*it2).second.measure;
            g2o::Matrix6d info = toMatrix6d((*it2).second.info);

            g2o::EdgeSE3* pEdgeFeatTmp = addEdgeSE3(optimizer, toIsometry3D(meas),
                                                    ptrKFFrom->mIdKF, ptrKFTo->mIdKF, info);
            vpEdgeFeat.push_back(pEdgeFeatTmp);

            numFtrCnstr ++;
        }
    }

#ifdef PRE_REJECT_FTR_OUTLIER
    // Pre-reject outliers in feature edges
    {
        vector<g2o::EdgeSE3*> vpEdgeFeatGood;
        for (auto iter = vpEdgeFeat.begin(); iter != vpEdgeFeat.end(); iter++) {
            g2o::EdgeSE3* pEdge = *iter;
            pEdge->computeError();
            double chi2 = pEdge->chi2();
            if (chi2 > threshFeatEdgeChi2Pre) {
                int id0 = pEdge->vertex(0)->id();
                int id1 = pEdge->vertex(1)->id();
                PtrKeyFrame pKF0 = mapId2pKF[id0];
                PtrKeyFrame pKF1 = mapId2pKF[id1];
                pKF0->eraseFtrMeasureFrom(pKF1);
                pKF0->eraseCovisibleKF(pKF1);
                pKF1->eraseFtrMeasureTo(pKF0);
                pKF1->eraseCovisibleKF(pKF0);
                pEdge->setLevel(1);
            }
            else {
                vpEdgeFeatGood.push_back(pEdge);
            }
        }
        vpEdgeFeat.swap(vpEdgeFeatGood);
    }
#endif

    optimizer.setVerbose(Config::GLOBAL_VERBOSE);
    optimizer.initializeOptimization();
    optimizer.optimize(Config::GLOBAL_ITER);

#ifdef PRE_REJECT_FTR_OUTLIER

    if (Config::GLOBAL_VERBOSE) {
        PrintOptInfo(vpEdgeOdo, vpEdgeFeat, vpEdgePlane, 1.0, false);
    }

    // Reject outliers in feature edges
    bool bFindOutlier = false;
    vector<g2o::EdgeSE3*> vpEdgeFeatGood;
    for (auto iter = vpEdgeFeat.begin(); iter != vpEdgeFeat.end(); iter++) {
        g2o::EdgeSE3* pEdge = *iter;
        double chi2 = pEdge->chi2();
        if (chi2 > threshFeatEdgeChi2) {
            int id0 = pEdge->vertex(0)->id();
            int id1 = pEdge->vertex(1)->id();
            PtrKeyFrame pKF0 = mapId2pKF[id0];
            PtrKeyFrame pKF1 = mapId2pKF[id1];
            pKF0->eraseFtrMeasureFrom(pKF1);
            pKF0->eraseCovisibleKF(pKF1);
            pKF1->eraseFtrMeasureTo(pKF0);
            pKF1->eraseCovisibleKF(pKF0);
            pEdge->setLevel(1);
            bFindOutlier = true;
        }
        else {
            vpEdgeFeatGood.push_back(pEdge);
        }
    }
    vpEdgeFeat.swap(vpEdgeFeatGood);

    if (!bFindOutlier) {
        break;
    }

#endif

#ifdef REJECT_IF_LARGE_LAMBDA
    if(solver->currentLambda() > 100) {
        mpLocalMapper->setGlobalBABegin(false);
        return;
    }
#endif

    // Update local graph KeyFrame poses
    for(auto it = vecKFs.begin(), iend = vecKFs.end();
        it != iend; it++){
        PtrKeyFrame pKF = (*it);
        if(pKF->isNull()) {
            continue;
        }
        Mat Twc = toCvMat(estimateVertexSE3(optimizer, pKF->mIdKF));
        pKF->setPose( cvu::inv(Twc) );
    }

    // Update local graph MapPoint positions
    vector<PtrMapPoint>  vMPsAll = mpMap->getAllMP();
    for(auto it = vMPsAll.begin(); it != vMPsAll.end(); it++) {
        PtrMapPoint pMP = (*it);

        if(pMP->isNull()){
            continue;
        }

        PtrKeyFrame pKF = pMP->mMainKF;
        Mat Twc = pKF->getPose().inv();
        Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
        Mat twc = Twc.rowRange(0,3).colRange(3,4);

        if(!pKF->hasObservation(pMP)) {
            continue;
        }

        int idx = pKF->getFtrIdx(pMP);

        Point3f Pt3_MP_KF = pKF->mViewMPs[idx];
        Mat t3_MP_KF = (Mat_<float>(3,1) << Pt3_MP_KF.x, Pt3_MP_KF.y, Pt3_MP_KF.z);
        Mat t3_MP_w = Rwc * t3_MP_KF + twc;
        Point3f Pt3_MP_w(t3_MP_w);
        pMP->setPos(Pt3_MP_w);
    }

    mpLocalMapper->setGlobalBABegin(false);

}

void GlobalMapper::PrintOptInfo(const vector<g2o::EdgeSE3*> & vpEdgeOdo,
                                const vector<g2o::EdgeSE3*> & vpEdgeFeat,
                                const vector<g2o::EdgeSE3Prior*> & vpEdgePlane,
                                double threshChi2, bool bPrintMatInfo) {


    cerr << "Edges with large chi2: " << endl;
    // print odometry edges
    for (auto it = vpEdgeOdo.begin(); it != vpEdgeOdo.end(); it ++) {
        g2o::EdgeSE3 *pEdge = *it;
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();

        int id0 = vVertices[0]->id();
        int id1 = vVertices[1]->id();

        if (pEdge->chi2() > threshChi2 || pEdge->chi2() < 0) {
            cerr << "odometry edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "id1 = " << id1 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";

            Matrix6d info = pEdge->information();
            Matrix3D infoTrans = info.block<3,3>(0,0);
            Matrix3D infoRot = info.block<3,3>(3,3);
            Vector6d err = pEdge->error();
            Vector3D errTrans = err.block<3,1>(0,0);
            Vector3D errRot = err.block<3,1>(3,0);

            cerr << "chi2Trans = " << errTrans.dot(infoTrans * errTrans) << "; ";
            cerr << "chi2Rot = " << errRot.dot(infoRot * errRot) << "; ";

            cerr << "err = ";
            for (int i=0; i<6; i++) {
                cerr << pEdge->error()(i) << "; ";
            }

            if (bPrintMatInfo) {
                cerr << endl;
                cerr << "info = " << endl << pEdge->information();
            }
            cerr << endl;
        }
    }

    // print loop closing edge
    for (auto it = vpEdgeFeat.begin(); it != vpEdgeFeat.end(); it ++) {
        g2o::EdgeSE3 *pEdge = *it;
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();

        int id0 = vVertices[0]->id();
        int id1 = vVertices[1]->id();

        if (pEdge->chi2() > threshChi2 || pEdge->chi2() < 0) {
            cerr << "feature edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "id1 = " << id1 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";

            Matrix6d info = pEdge->information();
            Matrix3D infoTrans = info.block<3,3>(0,0);
            Matrix3D infoRot = info.block<3,3>(3,3);
            Vector6d err = pEdge->error();
            Vector3D errTrans = err.block<3,1>(0,0);
            Vector3D errRot = err.block<3,1>(3,0);

            cerr << "chi2Trans = " << errTrans.dot(infoTrans * errTrans) << "; ";
            cerr << "chi2Rot = " << errRot.dot(infoRot * errRot) << "; ";

            cerr << "err = ";
            for (int i=0; i<6; i++) {
                cerr << pEdge->error()(i) << "; ";
            }

            if (bPrintMatInfo) {
                cerr << endl;
                cerr << "info = "<< endl << pEdge->information();
            }
            cerr << endl;
        }
    }

    // print plane motion edge
    for (auto it = vpEdgePlane.begin(); it != vpEdgePlane.end(); it ++) {
        g2o::EdgeSE3Prior *pEdge = *it;
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();

        int id0 = vVertices[0]->id();

        pEdge->computeError();

        if (pEdge->chi2() > threshChi2 || pEdge->chi2() < 0) {
            cerr << "plane edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";

            Matrix6d info = pEdge->information();
            Matrix3D infoTrans = info.block<3,3>(0,0);
            Matrix3D infoRot = info.block<3,3>(3,3);
            Vector6d err = pEdge->error();
            Vector3D errTrans = err.block<3,1>(0,0);
            Vector3D errRot = err.block<3,1>(3,0);

            cerr << "chi2Trans = " << errTrans.dot(infoTrans * errTrans) << "; ";
            cerr << "chi2Rot = " << errRot.dot(infoRot * errRot) << "; ";

            cerr << "err = ";
            for (int i=0; i<6; i++) {
                cerr << pEdge->error()(i) << "; ";
            }

            if (bPrintMatInfo) {
                cerr << endl;
                cerr << "info = " << endl << pEdge->information();
            }
            cerr << endl;
        }
    }
}

void GlobalMapper::PrintOptInfo(const SlamOptimizer & _optimizer) {

    double threshChi2 = 5.0;

    // print odometry edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it ++) {
        g2o::EdgeSE3 *pEdge = static_cast<g2o::EdgeSE3*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (abs(id1-id0) < 5) {
                if (pEdge->chi2() > threshChi2) {
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
        }
    }

    // print loop closing edge
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it ++) {
        g2o::EdgeSE3 *pEdge = static_cast<g2o::EdgeSE3*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (abs(id1-id0) >= 5) {
                if (pEdge->chi2() > threshChi2) {
                    cerr << "loop close edge: ";
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
        }
    }

    // print plane motion edge
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it ++) {
        g2o::EdgeSE3 *pEdge = static_cast<g2o::EdgeSE3*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 1) {
            if (pEdge->chi2() > threshChi2) {
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
    }
}

// Interface function:
// Called by LocalMapper when feature constraint generation is needed
void GlobalMapper::SetKFPairFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo) {

    std::pair<PtrKeyFrame, PtrKeyFrame> pairKF;
    pairKF.first = _pKFFrom;
    pairKF.second = _pKFTo;
    mdeqPairKFs.push_back(pairKF);
}

// Generate feature constraint between 2 KFs
int GlobalMapper::CreateFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo,
                                 SE3Constraint & SE3CnstrOutput)
{
    // Find co-observed MPs
    std::set<PtrMapPoint> spMPs;
    Map::compareViewMPs(_pKFFrom, _pKFTo, spMPs);

    //FindCoObsMPs(_pKFFrom, _pKFTo, setMPs);

    // Return when lack of co-observed MPs
    unsigned int numMinMPs = 10;
    if(spMPs.size() < numMinMPs) {
        return 1;
    }

    // Local BA with only 2 KFs and the co-observed MPs
    std::vector<PtrKeyFrame> vPtrKFs;
    vPtrKFs.push_back(_pKFFrom);
    vPtrKFs.push_back(_pKFTo);

    std::vector<PtrMapPoint> vPtrMPs;
    for (auto iter = spMPs.begin(); iter != spMPs.end(); iter++) {
        vPtrMPs.push_back(*iter);
    }

    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > vSe3KFs;
    vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D> > vPt3MPs;
    OptKFPair(vPtrKFs, vPtrMPs, vSe3KFs, vPt3MPs);

    // Generate feature based constraint between 2 KFs
    vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ> > vMeasSE3XYZ;
    g2o::SE3Quat meas_out;
    g2o::Matrix6d info_out;

    CreateVecMeasSE3XYZ(vPtrKFs, vPtrMPs, vMeasSE3XYZ);
    Sparsifier::DoMarginalizeSE3XYZ(
                vSe3KFs, vPt3MPs, vMeasSE3XYZ, meas_out, info_out);

    // Return
    SE3CnstrOutput.measure = toCvMat(meas_out);
    SE3CnstrOutput.info = toCvMat6f(info_out);
    return 0;
}

int GlobalMapper::CreateFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo, map<int,int> & mapMatch,
                                 SE3Constraint & SE3CnstrOutput)
{
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > vSe3KFs;
    vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D> > vPt3MPs;
    set<int> sIdMPin1Outlier;

    //! Optimize Local Graph with Only 2 KFs and MPs matched, and remove outlier by 3D measurements
    OptKFPairMatch(_pKFFrom, _pKFTo, mapMatch, vSe3KFs, vPt3MPs, sIdMPin1Outlier);
    if (mapMatch.size() < 3) {
        return 1;
    }

    // Generate feature based constraint between 2 KFs
    vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ> > vMeasSE3XYZ;

    int count = 0;
    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {
        if (sIdMPin1Outlier.count(iter->first))
            continue;

        int idxMPin1 = iter->first;
        //        PtrMapPoint pMPin1 = _pKFFrom->mDualObservations[idxMPin1];
        MeasSE3XYZ Meas1;
        Meas1.idKF = 0;
        Meas1.idMP = count;
        Meas1.z = toVector3d(_pKFFrom->mViewMPs[idxMPin1]);
        Meas1.info = _pKFFrom->mViewMPsInfo[idxMPin1];

        int idxMPin2 = iter->second;
        //        PtrMapPoint pMPin2 = _pKFTo->mDualObservations[idxMPin2];
        MeasSE3XYZ Meas2;
        Meas2.idKF = 1;
        Meas2.idMP = count;
        Meas2.z = toVector3d(_pKFTo->mViewMPs[idxMPin2]);
        Meas2.info = _pKFTo->mViewMPsInfo[idxMPin2];


        // DEBUG ON NAN
        double d = Meas1.info(0,0);
        if(std::isnan(d)) {
            cerr << "ERROR!!!" << endl;
        }
        d = Meas2.info(0,0);
        if(std::isnan(d)) {
            cerr << "ERROR!!!" << endl;
        }

        vMeasSE3XYZ.push_back(Meas1);
        vMeasSE3XYZ.push_back(Meas2);

        count ++;
    }

    g2o::SE3Quat meas_out;
    g2o::Matrix6d info_out;
    Sparsifier::DoMarginalizeSE3XYZ(vSe3KFs, vPt3MPs, vMeasSE3XYZ, meas_out, info_out);

    // Return
    SE3CnstrOutput.measure = toCvMat(meas_out);
    SE3CnstrOutput.info = toCvMat6f(info_out);
    return 0;
}


// Do localBA
void GlobalMapper::OptKFPair(const vector<PtrKeyFrame> & _vPtrKFs, const vector<PtrMapPoint> & _vPtrMPs,
                             vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > & _vSe3KFs, vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D> > & _vPt3MPs)
{
    // Init optimizer
    SlamOptimizer optimizer;
    initOptimizer(optimizer, true);
    addParaSE3Offset(optimizer, g2o::Isometry3D::Identity(), 0);

    // Init KF vertex
    int vertexId = 0;
    int numKFs = _vPtrKFs.size();
    for(int i=0; i<numKFs; i++) {
        PtrKeyFrame PtrKFi = _vPtrKFs[i];
        Mat T3_kf_w = PtrKFi->getPose();
        g2o::Isometry3D Iso3_w_kf = toIsometry3D(T3_kf_w.inv());

        if (i == 0) {
            addVertexSE3PlaneMotion(optimizer, Iso3_w_kf, vertexId, Config::bTc, 0, true);
        }
        else {
            addVertexSE3PlaneMotion(optimizer, Iso3_w_kf, vertexId, Config::bTc, 0, false);
        }

        vertexId ++;
    }

    // Init MP vertex
    int numMPs = _vPtrMPs.size();
    for (int i=0; i<numMPs; i++) {
        PtrMapPoint PtrMPi = _vPtrMPs[i];
        g2o::Vector3D Pt3MPi = toVector3d(PtrMPi->getPos());

        addVertexXYZ(optimizer, Pt3MPi, vertexId, true);
        vertexId++;
    }

    // Set SE3XYZ edges
    for (int i=0; i<numKFs; i++) {
        int vertexIdKF = i;
        PtrKeyFrame PtrKFi = _vPtrKFs[i];

        for (int j=0; j<numMPs; j++) {
            int vertexIdMP = j+numKFs;
            PtrMapPoint PtrMPj = _vPtrMPs[j];

            if(!PtrKFi->hasObservation(PtrMPj)) {
                continue;
            }

            int idx = PtrKFi->getFtrIdx(PtrMPj);

            g2o::Vector3D meas = toVector3d(PtrKFi->mViewMPs[idx]);
            g2o::Matrix3D info = PtrKFi->mViewMPsInfo[idx];

            addEdgeSE3XYZ(optimizer, meas, vertexIdKF, vertexIdMP, 0, info, 5.99);
        }
    }

    // Do optimize with g2o
    WorkTimer timer;
    timer.start();

    optimizer.initializeOptimization();
    optimizer.setVerbose(false);
    optimizer.optimize(15);

    timer.stop();

    // Return optimize results
    _vSe3KFs.clear();
    for (int i=0; i<numKFs; i++) {
        g2o::SE3Quat Se3KFi = toSE3Quat(estimateVertexSE3(optimizer, i));
        _vSe3KFs.push_back(Se3KFi);
    }

    _vPt3MPs.clear();
    for (int j=0; j<numMPs; j++) {
        g2o::Vector3D Pt3MPj = estimateVertexXYZ(optimizer, j+numKFs);
        _vPt3MPs.push_back(Pt3MPj);
    }
}

void GlobalMapper::OptKFPairMatch(PtrKeyFrame _pKF1, PtrKeyFrame _pKF2, map<int,int> & mapMatch,
                                  vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > & _vSe3KFs, 
                                  vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D> > & _vPt3MPs,
                                  set<int> &sIdMPin1Outlier)
{
    // Init optimizer
    SlamOptimizer optimizer;
    initOptimizer(optimizer, true);
    addParaSE3Offset(optimizer, g2o::Isometry3D::Identity(), 0);

    // Set vertex KF1
    Mat T3_kf1_w = _pKF1->getPose();
    g2o::Isometry3D Iso3_w_kf1 = toIsometry3D(T3_kf1_w.inv());
    addVertexSE3PlaneMotion(optimizer, Iso3_w_kf1, 0, Config::bTc, 0, false);

    // Set vertex KF2
    Mat T3_kf2_w = _pKF2->getPose();
    g2o::Isometry3D Iso3_w_kf2 = toIsometry3D(T3_kf2_w.inv());
    addVertexSE3PlaneMotion(optimizer, Iso3_w_kf2, 1, Config::bTc, 0, false);

    int vertexId = 2;

    // Init MP vertex from KF1 and Set edge
    int numMatch = mapMatch.size();
    vector<int> vIdMPin1;
    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {

        int idMPin1 = iter->first;
        vIdMPin1.push_back(idMPin1);
        int idMPin2 = iter->second;

        PtrMapPoint pMPin1 = _pKF1->getObservation(idMPin1);
        if(!pMPin1) cerr << "This is NULL /in GM::OptKFPairMatch 1\n";
        if(!pMPin1) continue;
        PtrMapPoint pMPin2 = _pKF2->getObservation(idMPin2);
        if(!pMPin2) cerr << "This is NULL /in GM::OptKFPairMatch 2\n";
        if(!pMPin2) continue;


        g2o::Vector3D Pt3MP = toVector3d(pMPin1->getPos());
        addVertexXYZ(optimizer, Pt3MP, vertexId, true);

        g2o::Vector3D meas1 = toVector3d(_pKF1->mViewMPs[idMPin1]);
        g2o::Matrix3D info1 = _pKF1->mViewMPsInfo[idMPin1];
        addEdgeSE3XYZ(optimizer, meas1, 0, vertexId, 0, info1, 5.99);

        g2o::Vector3D meas2 = toVector3d(_pKF2->mViewMPs[idMPin2]);
        g2o::Matrix3D info2 = _pKF2->mViewMPsInfo[idMPin2];
        addEdgeSE3XYZ(optimizer, meas2, 1, vertexId, 0, info2, 5.99);

        vertexId++;
    }

    // Do optimize with g2o
    optimizer.initializeOptimization();
    optimizer.setVerbose(false);
    optimizer.optimize(30);


    // Remove outliers by checking 3D measurement error
    double dThreshChi2 = 5.0;
    // set<int> sIdMPin1Outlier;

    for (auto it = optimizer.edges().begin(); it != optimizer.edges().end(); it ++) {
        g2o::EdgeSE3PointXYZ *pEdge = static_cast<g2o::EdgeSE3PointXYZ*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();

        if (vVertices.size() == 2) {
            int id1 = vVertices[1]->id();
            double chi2 = pEdge->chi2();
            if (chi2 > dThreshChi2) {
                sIdMPin1Outlier.insert(vIdMPin1[id1-2]);
            }
        }
    }

    if (Config::GLOBAL_PRINT) {
        cerr << "## DEBUG GM: " << "Find " << sIdMPin1Outlier.size()
             << " outliers by 3D MP to KF measurements." << endl;
    }

    // Return optimize results
    _vSe3KFs.clear();
    for (int i=0; i<2; i++) {
        g2o::SE3Quat Se3KFi = toSE3Quat(estimateVertexSE3(optimizer, i));
        _vSe3KFs.push_back(Se3KFi);
    }

    _vPt3MPs.clear();
    vertexId = 1;
    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {
        vertexId++;
        if (sIdMPin1Outlier.count(iter->first))
            continue;
        g2o::Vector3D Pt3MPj = estimateVertexXYZ(optimizer, vertexId);
        _vPt3MPs.push_back(Pt3MPj);
    }
	/*
    for (int j=0; j<numMatch; j++) {
        g2o::Vector3D Pt3MPj = estimateVertexXYZ(optimizer, j+2);
        _vPt3MPs.push_back(Pt3MPj);
    }
	 */
}

void GlobalMapper::PrintSE3(const g2o::SE3Quat se3) {

    Eigen::Vector3d _t = se3.translation();
    Eigen::Quaterniond _r = se3.rotation();

    cerr << "t = ";
    cerr << _t[0] << " " ;
    cerr << _t[1] << " " ;
    cerr << _t[2] << "; " ;
    cerr << "q = ";
    cerr << _r.w() << " " ;
    cerr << _r.x() << " " ;
    cerr << _r.y() << " " ;
    cerr << _r.z() << "; " ;
    cerr << endl;
}

void GlobalMapper::CreateVecMeasSE3XYZ(const vector<PtrKeyFrame> _vpKFs, const vector<PtrMapPoint> _vpMPs,
                                       vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ> > & vMeas)
{
    int numKFs = _vpKFs.size();
    int numMPs = _vpMPs.size();
    vMeas.clear();

    for (int i=0; i<numKFs; i++) {
        PtrKeyFrame PtrKFi = _vpKFs[i];
        for (int j=0; j<numMPs; j++) {
            PtrMapPoint PtrMPj = _vpMPs[j];

            MeasSE3XYZ Meas_ij;
            Meas_ij.idKF = i;
            Meas_ij.idMP = j;

            if(!PtrKFi->hasObservation(PtrMPj)){
                continue;
            }

            int idxMPinKF = PtrKFi->getFtrIdx(PtrMPj);

            Meas_ij.z = toVector3d(PtrKFi->mViewMPs[idxMPinKF]);
            Meas_ij.info = PtrKFi->mViewMPsInfo[idxMPinKF];

            vMeas.push_back(Meas_ij);
        }
    }
}

void GlobalMapper::ComputeBowVecAll()
{
    // Compute BowVector for all KFs, when BowVec does not exist
    vector<PtrKeyFrame> vpKFs;
    vpKFs = mpMap->getAllKF();
    int numKFs = vpKFs.size();
    for(int i=0; i<numKFs; i++) {
        PtrKeyFrame pKF = vpKFs[i];
        if (pKF->mbBowVecExist) {
            continue;
        }
        pKF->ComputeBoW(mpORBVoc);
    }
}

void GlobalMapper::DrawMatch(const map<int, int> & mapMatch) {

    //! Renew images

    if (mpKFCurr == NULL || mpKFCurr->isNull()) {
        return;
    }

    mpKFCurr->copyImgTo(mImgCurr);

    if (mpKFLoop == NULL || mpKFLoop->isNull()) {
        mImgLoop.setTo(cv::Scalar(0));
        return;
    }
    else {
        mpKFLoop->copyImgTo(mImgLoop);
    }

    if(mImgCurr.channels() == 1) {
        Mat imgTemp = mImgCurr.clone();
        cvtColor(mImgCurr, imgTemp, CV_GRAY2BGR);
        imgTemp.copyTo(mImgCurr);
    }
    if(mImgLoop.channels() == 1) {
        Mat imgTemp = mImgLoop.clone();
        cvtColor(mImgLoop, imgTemp, CV_GRAY2BGR);
        imgTemp.copyTo(mImgLoop);
    }
    Size sizeImgCurr = mImgCurr.size();
    Size sizeImgLoop = mImgLoop.size();

    Mat imgMatch(sizeImgCurr.height*2, sizeImgCurr.width, mImgCurr.type());
    mImgCurr.copyTo(imgMatch(cv::Rect(0,0,sizeImgCurr.width,sizeImgCurr.height)));
    mImgLoop.copyTo(imgMatch(cv::Rect(0,sizeImgCurr.height,sizeImgLoop.width,sizeImgLoop.height)));
    imgMatch.copyTo(mImgMatch);

    //! Draw Features
    for (int i=0, iend=mpKFCurr->keyPoints.size(); i<iend; i++) {
        KeyPoint kpCurr = mpKFCurr->keyPoints[i];
        Point2f ptCurr = kpCurr.pt;
        bool ifMPCurr = bool(mpKFCurr->hasObservation(i));
        Scalar colorCurr;
        if (ifMPCurr) {
            colorCurr = Scalar(0,255,0);
        }
        else {
            colorCurr = Scalar(255,0,0);
        }
        circle(mImgMatch, ptCurr, 5, colorCurr, 1);
    }

    for (int i=0, iend=mpKFLoop->keyPoints.size(); i<iend; i++) {
        KeyPoint kpLoop = mpKFLoop->keyPoints[i];
        Point2f ptLoop = kpLoop.pt;
        Point2f ptLoopMatch = ptLoop;
        ptLoopMatch.y += 480;

        bool ifMPLoop = bool(mpKFLoop->hasObservation(i));
        Scalar colorLoop;
        if (ifMPLoop) {
            colorLoop = Scalar(0,255,0);
        }
        else {
            colorLoop = Scalar(255,0,0);
        }
        circle(mImgMatch, ptLoopMatch, 5, colorLoop, 1);
    }

    //! Draw Matches
    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {

        int idxCurr = iter->first;
        KeyPoint kpCurr = mpKFCurr->keyPoints[idxCurr];
        Point2f ptCurr = kpCurr.pt;

        int idxLoop = iter->second;
        KeyPoint kpLoop = mpKFLoop->keyPoints[idxLoop];
        Point2f ptLoop = kpLoop.pt;
        Point2f ptLoopMatch = ptLoop;
        ptLoopMatch.y += 480;

        bool ifMPCurr = bool(mpKFCurr->hasObservation(idxCurr));
        bool ifMPLoop = bool(mpKFLoop->hasObservation(idxLoop));

        Scalar colorCurr, colorLoop;
        if (ifMPCurr) {
            colorCurr = Scalar(0,255,0);
        }
        else {
            colorCurr = Scalar(255,0,0);
        }
        if (ifMPLoop) {
            colorLoop = Scalar(0,255,0);
        }
        else {
            colorLoop = Scalar(255,0,0);
        }

        circle(mImgMatch, ptCurr, 5, colorCurr, 1);
        circle(mImgMatch, ptLoopMatch, 5, colorLoop, 1);
        if (ifMPCurr && ifMPLoop) {
            line(mImgMatch, ptCurr, ptLoopMatch, Scalar(0,97,255), 2);
        }
        else {
            line(mImgMatch, ptCurr, ptLoopMatch, colorCurr, 1);
        }

    }

}

void GlobalMapper::RemoveMatchOutlierRansac(PtrKeyFrame _pKFCurr, PtrKeyFrame _pKFLoop,
                                            map<int, int> & mapMatch)
{
    int numMinMatch = 10;

    // Initialize
    int numMatch = mapMatch.size();
    if (numMatch < numMinMatch) {
        mapMatch.clear();
        return; // return when small number of matches
    }

    map<int, int> mapMatchGood;
    vector<int> vIdxCurr, vIdxLoop;
    vector<Point2f> vPtCurr, vPtLoop;

    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {

        int idxCurr = iter->first;
        int idxLoop = iter->second;

        vIdxCurr.push_back(idxCurr);
        vIdxLoop.push_back(idxLoop);

        vPtCurr.push_back(_pKFCurr->keyPointsUn[idxCurr].pt);
        vPtLoop.push_back(_pKFLoop->keyPointsUn[idxLoop].pt);
    }

    // RANSAC with fundemantal matrix
    vector<uchar> vInlier; // 1 when inliers, 0 when outliers
    findFundamentalMat(vPtCurr, vPtLoop, FM_RANSAC, 3.0, 0.99, vInlier);
    for (unsigned int i=0; i<vInlier.size(); i++) {
        int idxCurr = vIdxCurr[i];
        int idxLoop = vIdxLoop[i];
        if(vInlier[i] == true) {
            mapMatchGood[idxCurr] = idxLoop;
        }
    }

    // Return good Matches
    mapMatch = mapMatchGood;
}

// Remove match pair with KP
void GlobalMapper::RemoveKPMatch(PtrKeyFrame _pKFCurr, PtrKeyFrame _pKFLoop,
                                 map<int, int> & mapMatch)
{
    vector<int> vIdxToErase;

    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {

        int idxCurr = iter->first;
        int idxLoop = iter->second;

        bool ifMPCurr = _pKFCurr->hasObservation(idxCurr);
        bool ifMPLoop = _pKFLoop->hasObservation(idxLoop);

        if(ifMPCurr && ifMPLoop) {
            continue;
        }
        else {
            vIdxToErase.push_back(idxCurr);
        }
    }

    int numToErase = vIdxToErase.size();
    for (int i=0; i<numToErase; i++) {
        mapMatch.erase(vIdxToErase[i]);
    }
}

// Return all connected KFs from a given KF
set<PtrKeyFrame> GlobalMapper::GetAllConnectedKFs(const PtrKeyFrame _pKF, set<PtrKeyFrame> _sKFSelected) {
    set<PtrKeyFrame> sKFConnected;

    PtrKeyFrame pKFOdoChild = _pKF->mOdoMeasureFrom.first;
    if (pKFOdoChild != NULL) {
        sKFConnected.insert(pKFOdoChild);
    }

    PtrKeyFrame pKFOdoParent = _pKF->mOdoMeasureTo.first;
    if (pKFOdoParent != NULL) {
        sKFConnected.insert(pKFOdoParent);
    }

    for (auto iter = _pKF->mFtrMeasureFrom.begin();
         iter != _pKF->mFtrMeasureFrom.end(); iter ++) {
        sKFConnected.insert(iter->first);
    }

    for (auto iter = _pKF->mFtrMeasureTo.begin();
         iter != _pKF->mFtrMeasureTo.end(); iter ++) {
        sKFConnected.insert(iter->first);
    }

    for (auto iter = _sKFSelected.begin();
         iter != _sKFSelected.end(); iter ++) {
        sKFConnected.insert(*iter);
    }

    return sKFConnected;
}

set<PtrKeyFrame> GlobalMapper::GetAllConnectedKFs_nLayers(const PtrKeyFrame _pKF, int numLayers, set<PtrKeyFrame> _sKFSelected) {

    set<PtrKeyFrame> sKFLocal;      // Set of KFs whose distance from _pKF smaller than maxDist
    set<PtrKeyFrame> sKFActive;     // Set of KFs who are active for next loop;
    sKFActive.insert(_pKF);

    for (int i=0; i<numLayers; i++) {

        set<PtrKeyFrame> sKFNew;

        for (auto iter = sKFActive.begin(); iter != sKFActive.end(); iter++) {
            PtrKeyFrame pKFTmp = *iter;
            set<PtrKeyFrame> sKFAdjTmp = GetAllConnectedKFs(pKFTmp, _sKFSelected);
            for (auto iter2 = sKFAdjTmp.begin(); iter2 != sKFAdjTmp.end(); iter2 ++) {
                if (sKFLocal.count(*iter2) == 0) {
                    sKFNew.insert(*iter2);
                }
            }
        }

        sKFLocal.insert(sKFNew.begin(), sKFNew.end());
        sKFActive.swap(sKFNew);
    }

    return sKFLocal;
}



// Select KF pairs to creat feature constraint between which
vector <pair <PtrKeyFrame, PtrKeyFrame> > GlobalMapper::SelectKFPairFeat(const PtrKeyFrame _pKF) {

    vector<pair <PtrKeyFrame, PtrKeyFrame> > _vKFPairs;

    // Smallest distance between KFs in covis-graph to create a new feature edge
    int threshCovisGraphDist = 5;

    set<PtrKeyFrame> sKFSelected;
    set<PtrKeyFrame> sKFCovis = _pKF->getAllCovisibleKFs();
    set<PtrKeyFrame> sKFLocal = GetAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);

    for (auto iter = sKFCovis.begin(); iter != sKFCovis.end(); iter++) {

        PtrKeyFrame _pKFCand = *iter;
        if (sKFLocal.count(_pKFCand) == 0) {
            sKFSelected.insert(*iter);
            sKFLocal = GetAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);
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

void GlobalMapper::setBusy(bool v){
    std::unique_lock<std::mutex> lock(mMutexBusy);
    mbIsBusy = v;
    if(!v) {
        mcIsBusy.notify_one();
    }
}


void GlobalMapper::waitIfBusy(){
    std::unique_lock<std::mutex> lock(mMutexBusy);
    while(mbIsBusy){
        mcIsBusy.wait(lock);
    }
}

void GlobalMapper::requestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool GlobalMapper::checkFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool GlobalMapper::isFinished() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void GlobalMapper::setFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}


} // namespace se2lam
