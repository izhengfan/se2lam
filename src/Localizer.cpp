/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Localizer.h"
#include <ros/ros.h>
#include "ORBmatcher.h"
#include "optimizer.h"

namespace se2lam {
using namespace std;
using namespace cv;
using namespace g2o;

typedef lock_guard<mutex> locker;

Localizer::Localizer() {

    mpORBextractor = new ORBextractor(Config::MaxFtrNumber,Config::ScaleFactor,Config::MaxLevel);
    mbIsTracked = false;

    mbFinished = false;
    mbFinishRequested = false;
}

Localizer::~Localizer() {

}

void Localizer::run() {
    //! Init
//    string fullOdoName = Config::DataPath + "/odo_raw.txt";
//    ifstream odo_rec(fullOdoName);
//    string odo_line;
//    float odo_x, odo_y, odo_theta;
//    int id_raw = 0;

//    for (int i=0; i < Config::ImgIndexLocalSt; i++) {
//        getline(odo_rec, odo_line);
//    }

    ComputeBowVecAll();

    // traj log
    ofstream fileOutTraj(se2lam::Config::WRITE_TRAJ_FILE_PATH + se2lam::Config::WRITE_TRAJ_FILE_NAME);
    // traj log


    ros::Rate rate(Config::FPS);

    //! Main loop
    while(ros::ok()) {

        WorkTimer timer;
        timer.start();

        //! Get new measurement: image and odometry
//        string fullImgName = Config::DataPath + "/image/" + to_string(id_raw) + ".bmp";
//        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);

//        getline(odo_rec, odo_line);
//        istringstream iss(odo_line);
//        iss >> odo_x >> odo_y >> odo_theta;
//        Se2 odo(odo_x, odo_y, odo_theta);


//        id_raw++;


        cv::Mat img;
        Se2 odo;
        bool sensorUpdated = mpSensors->update();
        if (sensorUpdated) {
            Point3f odo_3f;
            mpSensors->readData(odo_3f, img);
            odo = Se2(odo_3f.x, odo_3f.y, odo_3f.z);

            ReadFrameInfo(img, odo);

            if (mpKFRef == NULL) continue;

            UpdatePoseCurr();

            //! Tracking process
            // Tracking good, do localize based on last loop
            if (mbIsTracked) {

                //            MatchLastFrame();
                //            DoLocalBA();

                MatchLocalMap();

                int numMPCurr = mpKFCurr->getSizeObsMP();
                if (numMPCurr > 30) {
                    DoLocalBA();
                }

                UpdateCovisKFCurr();

                UpdateLocalMap(1);

                DrawImgCurr();
                mImgMatch = Mat::zeros(mImgMatch.rows, mImgMatch.cols, mImgMatch.type());

                DetectIfLost();
            }
            // Tracking lost, need loop close
            else {
                bool bIfLoopCloseDetected = false;
                bool bIfLoopCloseVerified = false;

                bIfLoopCloseDetected = DetectLoopClose();

                if (bIfLoopCloseDetected) {

                    map<int,int> mapMatchMP, mapMatchGood, mapMatchRaw;
                    bIfLoopCloseVerified = VerifyLoopClose(mapMatchMP, mapMatchGood, mapMatchRaw);

                    if (bIfLoopCloseVerified) {

                        mpKFCurr->setPose(mpKFLoop->getPose());
                        mpKFCurr->addCovisibleKF(mpKFLoop);

                        // Update local map from KFLoop
                        UpdateLocalMap();

                        // Set MP observation of KFCurr from KFLoop
                        MatchLoopClose(mapMatchGood);

                        // Do Local BA and Do outlier rejection
                        DoLocalBA();

                        // Set MP observation of KFCurr from local map
                        MatchLocalMap();

                        // Do local BA again and do outlier rejection
                        DoLocalBA();
                    }
                    else {
                        ResetLocalMap();
                    }

                    DrawImgCurr();
                    DrawImgMatch(mapMatchGood);

                }
                else {
                    DrawImgCurr();
                    mImgMatch = Mat::zeros(mImgMatch.rows, mImgMatch.cols, mImgMatch.type());
                }

                DetectIfLost();
            }

        }

        if (checkFinish()) {
            break;
        }

        mpKFCurrRefined = mpKFCurr;
        WriteTrajFile(fileOutTraj);


        timer.stop();
        cerr << "-- DEBUG LOCAL: " << "loopTime = " << timer.time << endl << endl;
        rate.sleep();
    }

    cerr << "Exiting locaizer .." << endl;

    ros::shutdown();
    setFinish();
}

void Localizer::WriteTrajFile(ofstream & file) {

    if (mpKFCurrRefined == NULL || mpKFCurrRefined->isNull()) {
        return;
    }

    Mat wTb = cvu::inv(se2lam::Config::bTc * mpKFCurrRefined->getPose());
    Mat wRb = wTb.rowRange(0, 3).colRange(0, 3);
    g2o::Vector3D euler = g2o::internal::toEuler(se2lam::toMatrix3d(wRb));

    file << mpKFCurrRefined->id << "," <<
        wTb.at<float>(0, 3) << "," <<
        wTb.at<float>(1, 3) << "," <<
        euler(2) << endl;

}

void Localizer::ReadFrameInfo(const Mat &img, const Se2& odo) {

    mFrameRef = mFrameCurr;
    mpKFRef = mpKFCurr;

    mFrameCurr = Frame(img, odo, mpORBextractor, Config::Kcam, Config::Dcam);
    mFrameCurr.Tcw = Config::cTb.clone();
    mpKFCurr = make_shared<KeyFrame>(mFrameCurr);
    mpKFCurr->ComputeBoW(mpORBVoc);

}

void Localizer::MatchLastFrame() {

}

void Localizer::MatchLocalMap() {

    //! Match in local map
    vector<PtrMapPoint> vpMPLocal = GetLocalMPs();
    vector<int> vIdxMPMatched;
    ORBmatcher matcher;
    int numMPMatched = matcher.MatchByProjection(mpKFCurr, vpMPLocal, 15, 2, vIdxMPMatched);

    //! Renew KF observation
    for (int idxKPCurr=0, idend=vIdxMPMatched.size(); idxKPCurr < idend; idxKPCurr++) {
        int idxMPLocal = vIdxMPMatched[idxKPCurr];

        if (idxMPLocal == -1) continue;

        PtrMapPoint pMP = vpMPLocal[idxMPLocal];
        mpKFCurr->addObservation(pMP,idxKPCurr);
    }

    cerr << "-- DEBUG LOCAL: numMPMatchLocal = " << numMPMatched << endl;
}


void Localizer::DoLocalBA() {

    SlamOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    int maxKFid = -1;

    // Add KFCurr
    addVertexSE3Expmap(optimizer,
                       toSE3Quat(mpKFCurr->getPose()), mpKFCurr->mIdKF, false);
    addPlaneMotionSE3Expmap(optimizer,
                            toSE3Quat(mpKFCurr->getPose()), mpKFCurr->mIdKF, Config::bTc);
    maxKFid = mpKFCurr->mIdKF;

    // Add MPs in local map as fixed
    const float delta = Config::TH_HUBER;
    set<PtrMapPoint> setMPs = mpKFCurr->getAllObsMPs();

    map<PtrMapPoint, int> Observations = mpKFCurr->getObservations();


    for (auto iter = setMPs.begin(); iter != setMPs.end(); iter++) {
        PtrMapPoint pMP = *iter;
        if(pMP->isNull() || !pMP->isGoodPrl())
            continue;

        bool marginal = false;
        bool fixed = true;
        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()),
                        maxKFid + pMP->mId, marginal, fixed);

        int ftrIdx = Observations[pMP];
        int octave = pMP->getOctave(mpKFCurr);
        const float invSigma2 = mpKFCurr->mvInvLevelSigma2[octave];
        Eigen::Vector2d uv = toVector2d( mpKFCurr->keyPointsUn[ftrIdx].pt );
        Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

        EdgeProjectXYZ2UV* ei = new EdgeProjectXYZ2UV();
        ei->setVertex(0, dynamic_cast<OptimizableGraph::Vertex*>(optimizer.vertex(maxKFid + pMP->mId)));
        ei->setVertex(1, dynamic_cast<OptimizableGraph::Vertex*>(optimizer.vertex(mpKFCurr->mIdKF)));
        ei->setMeasurement(uv);
        ei->setParameterId(0, camParaId);
        ei->setInformation(info);
        RobustKernelHuber* rk = new RobustKernelHuber;
        ei->setRobustKernel(rk);
        rk->setDelta(delta);
        ei->setLevel(0);
        optimizer.addEdge(ei);
    }

    WorkTimer timer;
    timer.start();

    optimizer.initializeOptimization(0);
    optimizer.optimize(30);

    timer.stop();

    Mat Twc = toCvMat(estimateVertexSE3Expmap(optimizer, mpKFCurr->mIdKF));
    mpKFCurr->setPose(Twc);

    cerr << "-- DEBUG LOCAL: localBATime = " << timer.time << endl;
}

void Localizer::DetectIfLost() {

    int numKFLocal = GetLocalKFs().size();
    if (numKFLocal > 0) {
        mbIsTracked = true;
    }
    else {
        mbIsTracked = false;
    }
}

void Localizer::setMap(Map *pMap) {
    mpMap = pMap;
}

void Localizer::setORBVoc(ORBVocabulary* pORBVoc) {
    mpORBVoc = pORBVoc;
}

void Localizer::ComputeBowVecAll() {
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

bool Localizer::DetectLoopClose() {

    // Loop closure detection with ORB-BOW method
    bool bDetected = false;
    double minScoreBest = 0.05;

    PtrKeyFrame pKFCurr = mpKFCurr;
    if (pKFCurr == NULL) {
        return bDetected;
    }

    DBoW2::BowVector BowVecCurr = pKFCurr->mBowVec;

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKF();
    int numKFs = vpKFsAll.size();
    PtrKeyFrame pKFBest;
    double scoreBest = 0;

    for(int i=0; i<numKFs; i++) {

        PtrKeyFrame pKF = vpKFsAll[i];
        DBoW2::BowVector BowVec = pKF->mBowVec;

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

    //! DEBUG: Print loop closing info
    if (bDetected) {
        cerr << "-- DEBUG LOCAL:"
             << " idCurr = " << pKFCurr->id
             << ", idLoop = " << mpKFLoop->id
             << ", bestScore = " << scoreBest
             << endl;
    }
    else {
        cerr << "-- DEBUG LOCAL:"
             << " idCurr = " << pKFCurr->id
             << ", NO good loop close detected."
             << endl;
    }

    return bDetected;
}

bool Localizer::VerifyLoopClose(map<int,int> & mapMatchMP, map<int,int> & mapMatchGood, map<int,int> & mapMatchRaw) {

    mapMatchMP.clear();
    mapMatchGood.clear();
    mapMatchRaw.clear();
    map<int,int> mapMatch;

    bool bVerified = false;
    //int numMinMPMatch = 15;
    int numMinMatch = 45;
    //double ratioMinMPMatch = 0.1;

    if (mpKFCurr == NULL || mpKFLoop == NULL) {
        return false;
    }

    //! Match ORB KPs
    ORBmatcher matcher;
    bool bIfMatchMPOnly = false;
    matcher.SearchByBoW(mpKFCurr, mpKFLoop, mapMatch, bIfMatchMPOnly);
    mapMatchRaw = mapMatch;

    //! Remove Outliers: by RANSAC of Fundamental
    RemoveMatchOutlierRansac(mpKFCurr, mpKFLoop, mapMatch);
    mapMatchGood = mapMatch;
    int numGoodMatch = mapMatch.size();

    if (numGoodMatch >= numMinMatch) {
        cerr << "-- DEBUG LOCAL: " << "loop close verification PASSED, "
             << "numGoodMatch = " << numGoodMatch << endl;
        bVerified = true;
    }
    else {
        cerr << "-- DEBUG LOCAL: " << "loop close verification FAILED, "
             << "numGoodMatch = " << numGoodMatch << endl;
    }
    return bVerified;
}

vector<PtrKeyFrame> Localizer::GetLocalKFs() {
    locker lock(mMutexKFLocal);
    return vector<PtrKeyFrame>(mspKFLocal.begin(), mspKFLocal.end());
}

vector<PtrMapPoint> Localizer::GetLocalMPs() {
    locker lock(mMutexMPLocal);
    return vector<PtrMapPoint>(mspMPLocal.begin(), mspMPLocal.end());
}

void Localizer::DrawImgCurr() {

    locker lockImg(mMutexImg);

    if(mpKFCurr == NULL)
        return;

    mpKFCurr->copyImgTo(mImgCurr);
    if(mImgCurr.channels() == 1)
        cvtColor(mImgCurr, mImgCurr, CV_GRAY2BGR);

    for (int i=0, iend=mpKFCurr->keyPoints.size(); i<iend; i++) {

        KeyPoint kpCurr = mpKFCurr->keyPoints[i];
        Point2f ptCurr = kpCurr.pt;

        bool ifMPCurr = mpKFCurr->hasObservation(i);
        Scalar colorCurr;
        if (ifMPCurr) {
            colorCurr = Scalar(0,255,0);
        }
        else {
            colorCurr = Scalar(255,0,0);
        }

        circle(mImgCurr, ptCurr, 5, colorCurr, 1);
    }

}

void Localizer::DrawImgMatch(const map<int, int> & mapMatch) {

    locker lockImg(mMutexImg);

    //! Renew images

    if(mpKFCurr == NULL || mpKFLoop == NULL) {
        return;
    }

    if (mpKFLoop != NULL) {
        mpKFLoop->copyImgTo(mImgLoop);
    }
    else {
        mImgCurr.copyTo(mImgLoop);
        mImgLoop.setTo(cv::Scalar(0));
    }

    if(mImgLoop.channels() == 1)
        cvtColor(mImgLoop, mImgLoop, CV_GRAY2BGR);
    if(mImgMatch.channels() == 1)
        cvtColor(mImgMatch, mImgMatch, CV_GRAY2BGR);
    vconcat(mImgCurr, mImgLoop, mImgMatch);

    //! Draw Features
    for (int i=0, iend=mpKFCurr->keyPoints.size(); i<iend; i++) {
        KeyPoint kpCurr = mpKFCurr->keyPoints[i];
        Point2f ptCurr = kpCurr.pt;
        bool ifMPCurr = mpKFCurr->hasObservation(i);
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

        bool ifMPLoop = mpKFLoop->hasObservation(i);
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

        bool ifMPCurr = mpKFCurr->hasObservation(idxCurr);
        bool ifMPLoop = mpKFLoop->hasObservation(idxLoop);

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
            line(mImgMatch, ptCurr, ptLoopMatch, colorLoop, 1);
        }

    }
}

void Localizer::RemoveMatchOutlierRansac(PtrKeyFrame _pKFCurr, PtrKeyFrame _pKFLoop,
                                         map<int, int> & mapMatch) {

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

void Localizer::UpdatePoseCurr() {
    Se2 dOdo = mpKFRef->odom - mpKFCurr->odom;
    //mpKFCurr->Tcr = Config::cTb * toT4x4(dOdo.x, dOdo.y, dOdo.theta) * Config::bTc;
    mpKFCurr->Tcr = Config::cTb * Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3() * Config::bTc;
    mpKFCurr->Tcw = mpKFCurr->Tcr * mpKFRef->Tcw;
}

void Localizer::ResetLocalMap() {
    mspKFLocal.clear();
    mspMPLocal.clear();
}

void Localizer::UpdateLocalMapTrack() {
    mspKFLocal.clear();
    mspMPLocal.clear();
}

void Localizer::UpdateLocalMap(int searchLevel) {

    locker lock(mMutexLocalMap);

    mspKFLocal.clear();
    mspMPLocal.clear();

    mspKFLocal = mpKFCurr->getAllCovisibleKFs();

    while(searchLevel > 0) {
        std::set<PtrKeyFrame> currentLocalKFs = mspKFLocal;
        for(auto iter = currentLocalKFs.begin(); iter != currentLocalKFs.end(); iter++) {
            PtrKeyFrame pKF = *iter;
            std::set<PtrKeyFrame> spKF = pKF->getAllCovisibleKFs();
            mspKFLocal.insert(spKF.begin(), spKF.end());
        }
        searchLevel--;
    }

    for(auto iter = mspKFLocal.begin(), iend = mspKFLocal.end(); iter != iend; iter++) {
        PtrKeyFrame pKF = *iter;
        set<PtrMapPoint> spMP = pKF->getAllObsMPs();
        mspMPLocal.insert(spMP.begin(), spMP.end());
    }
}


void Localizer::MatchLoopClose(map<int,int> mapMatchGood) {
    // mapMatchGood: KP index map from KFCurr to KFLoop

    //! Set MP observation in KFCurr
    for (auto iter = mapMatchGood.begin(); iter != mapMatchGood.end(); iter++) {

        int idxCurr = iter->first;
        int idxLoop = iter->second;
        bool isMPLoop = mpKFLoop->hasObservation(idxLoop);

        if (isMPLoop) {
            PtrMapPoint pMP = mpKFLoop->getObservation(idxLoop);
            mpKFCurr->addObservation(pMP, idxCurr);
        }
    }
}

void Localizer::UpdateCovisKFCurr() {

    for(auto iter = mspKFLocal.begin(); iter != mspKFLocal.end(); iter++) {
        set<PtrMapPoint> spMPs;
        PtrKeyFrame pKF = *iter;

        FindCommonMPs(mpKFCurr, pKF, spMPs);

        if(spMPs.size() > 0.1 * mpKFCurr->getSizeObsMP()){
            mpKFCurr->addCovisibleKF(pKF);
        }
    }
}

int Localizer::FindCommonMPs(const PtrKeyFrame pKF1, const PtrKeyFrame pKF2, set<PtrMapPoint>& spMPs) {
    spMPs.clear();
    mpMap->compareViewMPs(pKF1, pKF2, spMPs);
    return spMPs.size();
}

void Localizer::Test(int a, int b) {
    //int c = a + b;
}

void Localizer::setSensors(Sensors* pSensors){
    mpSensors = pSensors;
}

void Localizer::requestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Localizer::checkFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool Localizer::isFinished() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Localizer::setFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

}
