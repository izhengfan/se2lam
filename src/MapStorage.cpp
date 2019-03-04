/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "MapStorage.h"

#include <iostream>
#include "converter.h"
#include <opencv2/highgui/highgui.hpp>

namespace se2lam {

using namespace std;
using namespace cv;

MapStorage::MapStorage(){
    mvKFs.clear();
    mvMPs.clear();
}

void MapStorage::setFilePath(const string path, const string file){
    mMapPath = path;
    mMapFile = file;
}

void MapStorage::setMap(Map *pMap){
    mpMap = pMap;
}

void MapStorage::loadMap(){

    loadKeyFrames();

    loadMapPoints();

    loadObservations();

    loadCovisibilityGraph();

    loadOdoGraph();

    loadFtrGraph();

    loadToMap();

    printf("&& INFO MS: Map loaded from __%s__.\n\n", (mMapPath+mMapFile).c_str());

}

void MapStorage::saveMap(){

    mvKFs = mpMap->getAllKF();
    mvMPs = mpMap->getAllMP();

    sortKeyFrames();

    sortMapPoints();

    saveKeyFrames();

    saveMapPoints();

    saveObservations();

    saveCovisibilityGraph();

    saveOdoGraph();

    saveFtrGraph();

    printf("&& INFO MS: Map saved to __%s__.\n\n", (mMapPath+mMapFile).c_str());

}

void MapStorage::sortKeyFrames(){
    // Remove null KFs
    {
        vector<PtrKeyFrame> vKFs;
        vKFs.reserve(mvKFs.size());
        for(int i = 0, iend = mvKFs.size(); i < iend; i++) {
            PtrKeyFrame pKF = mvKFs[i];
            if(pKF->isNull())
                continue;
            vKFs.push_back(pKF);
        }
        std::swap(vKFs, mvKFs);
    }

    // Change Id of KF to be vector index
    for(int i = 0, iend = mvKFs.size(); i < iend; i++) {
        PtrKeyFrame pKF = mvKFs[i];
        pKF->mIdKF  = i;
    }
}

void MapStorage::sortMapPoints() {
    // Remove null MPs
    {
        vector<PtrMapPoint> vMPs;
        vMPs.reserve(mvMPs.size());
        for(int i = 0, iend = mvMPs.size(); i < iend; i++) {
            PtrMapPoint pMP = mvMPs[i];
            if(pMP->isNull() || !(pMP->isGoodPrl()) )
                continue;
            vMPs.push_back(pMP);
        }
        std::swap(vMPs, mvMPs);
    }

    // Change Id of MP to be vector index
    for(int i = 0, iend = mvMPs.size(); i < iend; i++) {
        PtrMapPoint pMP = mvMPs[i];
        pMP->mId = i;
    }

}

void MapStorage::saveKeyFrames(){

    // Save images to individual files
    for(int i = 0, iend = mvKFs.size(); i < iend; i++) {
        PtrKeyFrame pKF = mvKFs[i];
        imwrite(mMapPath + to_string(i) + ".bmp", pKF->img);
    }

    // Write data to file
    FileStorage file(mMapPath + mMapFile, FileStorage::WRITE);
    file << "KeyFrames" << "[";
    for(int i = 0, iend = mvKFs.size(); i < iend; i++) {

        PtrKeyFrame pKF = mvKFs[i];

        file << "{";

        file << "Id" << i;

        file << "KeyPoints" << "[";
        for(int j = 0, jend = pKF->keyPoints.size(); j < jend; j++) {
            KeyPoint kp  = pKF->keyPoints[j];
            file << "{";
            file << "pt" << kp.pt;
            file << "octave" << kp.octave;
            file << "angle" << kp.angle;
            file << "response" << kp.response;
            file << "}";
        }
        file << "]";

        file << "KeyPointsUn" << "[";
        for(int j = 0, jend = pKF->keyPointsUn.size(); j < jend; j++) {
            KeyPoint kp  = pKF->keyPointsUn[j];
            file << "{";
            file << "pt" << kp.pt;
            file << "octave" << kp.octave;
            file << "angle" << kp.angle;
            file << "response" << kp.response;
            file << "}";
        }
        file << "]";

        file << "Descriptor" << pKF->descriptors;

        if (pKF->mViewMPs.size() != pKF->keyPoints.size())
            cout << "Wrong size of KP in saving" << endl;

        file << "ViewMPs" << "[";
        for(int j = 0, jend = pKF->mViewMPs.size(); j < jend; j++) {
            file << pKF->mViewMPs[j];
        }
        file << "]";

        file << "ViewMPInfo" << "[";
        for(int j = 0, jend = pKF->mViewMPsInfo.size(); j < jend; j++) {
            file << toCvMat( pKF->mViewMPsInfo[j] );
        }
        file << "]";

        file << "Pose" << pKF->getPose();

        file << "Odometry" << Point3f(pKF->odom.x, pKF->odom.y, pKF->odom.theta);

        file << "ScaleFactor" << pKF->mfScaleFactor;

        file << "}";

    }
    file << "]";

    file.release();

}

void MapStorage::saveMapPoints(){

    // Write data to file
    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);
    file << "MapPoints" << "[";

    for(int i = 0, iend = mvMPs.size(); i < iend; i++) {
        PtrMapPoint pMP = mvMPs[i];
        file << "{";

        file << "Id" << i;
        file << "Pos" << pMP->getPos();

        file << "}";
    }
    file << "]";

    file.release();

}

void MapStorage::saveObservations() {

    int sizeKF = mvKFs.size();
    int sizeMP = mvMPs.size();

    cv::Mat obs(sizeKF, sizeMP, CV_32SC1, Scalar(0));
    cv::Mat Index(sizeKF, sizeMP, CV_32SC1, Scalar(-1));

    //mObservations = Mat_<int>(sizeKF, sizeMP, 0);

    //Mat_<int> Index(sizeKF, sizeMP, -1);

    for(int i = 0; i < sizeKF; i++) {
        PtrKeyFrame pKF = mvKFs[i];
        for(int j = 0; j < sizeMP; j++) {
            PtrMapPoint pMP = mvMPs[j];
            if(pKF->hasObservation(pMP)) {
                obs.at<int>(i,j) = 1;
                Index.at<int>(i,j) = pMP->getFtrIdx(pKF);
            }
        }
    }

    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);

    file << "Observations" << obs;

    file << "ObservationIndex" << Index;

    file.release();
}

void MapStorage::saveCovisibilityGraph(){

    int sizeKF = mvKFs.size();

    cv::Mat CovisibilityGraph(sizeKF, sizeKF, CV_32SC1, Scalar(0));

    for(int i = 0; i < sizeKF; i++) {
        PtrKeyFrame pKF = mvKFs[i];
        set<PtrKeyFrame> sKFs = pKF->getAllCovisibleKFs();

        for(auto it = sKFs.begin(), itend = sKFs.end(); it != itend; it++){
            PtrKeyFrame covKF = *it;
            CovisibilityGraph.at<int>(i,covKF->mIdKF) = 1;
        }
    }

    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);
    file << "CovisibilityGraph" << CovisibilityGraph;
    file.release();
}

void MapStorage::saveOdoGraph(){

    int sizeKF = mvKFs.size();

    mOdoNextId = vector<int>(sizeKF, -1);
    for(int i = 0; i < sizeKF; i++) {
        PtrKeyFrame pKF = mvKFs[i];
        PtrKeyFrame nextKF = pKF->mOdoMeasureFrom.first;

        if(nextKF != NULL) {
            mOdoNextId[i] = nextKF->mIdKF;
        }
    }

    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);

    file << "OdoGraphNextKF" << "[";
    for(int i = 0; i < sizeKF; i++) {
        PtrKeyFrame pKF = mvKFs[i];
        Mat measure = pKF->mOdoMeasureFrom.second.measure;
        Mat info = ( pKF->mOdoMeasureFrom.second.info );

        file << "{";

        file << "NextId" << mOdoNextId[i];
        file << "Measure" << measure;
        file << "Info" << info;

        file << "}";
    }

}

void MapStorage::saveFtrGraph(){

    int sizeKF = mvKFs.size();

    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);

    file << "FtrGraphPairs" << "[";
    for(int i = 0; i < sizeKF; i++) {
        PtrKeyFrame pKFi = mvKFs[i];
        int idi = pKFi->mIdKF;

        for(auto it = pKFi->mFtrMeasureFrom.begin(), itend = pKFi->mFtrMeasureFrom.end(); it != itend; it++){
            PtrKeyFrame pKFj = it->first;
            int idj = pKFj->mIdKF;
            Mat measure = it->second.measure;
            Mat info = ( it->second.info );

            file << "{";

            file << "PairId" << Point2i(idi, idj);
            file << "Measure" << measure;
            file << "Info" << info;

            file << "}";

        }
    }
    file << "]";

    file.release();

}

void MapStorage::loadKeyFrames(){

    mvKFs.clear();

    FileStorage file(mMapPath + mMapFile, FileStorage::READ);
    FileNode nodeKFs = file["KeyFrames"];
    FileNodeIterator it = nodeKFs.begin(), itend = nodeKFs.end();

    for(; it != itend; it++) {
        PtrKeyFrame pKF = make_shared<KeyFrame>();

        FileNode nodeKF = *it;

        pKF->mIdKF = (int)nodeKF["Id"];

        pKF->keyPoints.clear();
        FileNode nodeKP = nodeKF["KeyPoints"];
        {
            vector<KeyPoint> vKPs;

            FileNodeIterator itKP = nodeKP.begin(), itKPend = nodeKP.end();
            for(; itKP != itKPend; itKP++) {
                KeyPoint kp;
                (*itKP)["pt"] >> kp.pt;
                kp.octave = (int)(*itKP)["octave"];
                kp.angle = (float)(*itKP)["angle"];
                kp.response = (float)(*itKP)["response"];
                vKPs.push_back(kp);
            }
            pKF->keyPoints = vKPs;
        }

        pKF->keyPointsUn.clear();
        FileNode nodeKPUn = nodeKF["KeyPointsUn"];
        {
            vector<KeyPoint> vKPs;

            FileNodeIterator itKP = nodeKPUn.begin(), itKPend = nodeKPUn.end();
            for(; itKP != itKPend; itKP++) {
                KeyPoint kp;
                (*itKP)["pt"] >> kp.pt;
                kp.octave = (int)(*itKP)["octave"];
                kp.angle = (float)(*itKP)["angle"];
                kp.response = (float)(*itKP)["response"];
                vKPs.push_back(kp);
            }
            pKF->keyPointsUn = vKPs;
        }

        nodeKF["Descriptor"] >> pKF->descriptors;

        pKF->mViewMPs.clear();
        FileNode nodeViewMP = nodeKF["ViewMPs"];
        if (nodeKP.size() != nodeViewMP.size())
            cout << "Wrong KP size in loading " << endl;

        {
            vector<Point3f> vPos;
            FileNodeIterator itMP = nodeViewMP.begin(), itMPend = nodeViewMP.end();
            for( ; itMP != itMPend; itMP++) {
                Point3f pos;
                (*itMP) >> pos;
                vPos.push_back(pos);
            }
            pKF->mViewMPs = vPos;
        }
        if (pKF->keyPoints.size() != pKF->mViewMPs.size())
            cout << "Wrong KP size after loading " << endl;

        pKF->mViewMPsInfo.clear();
        FileNode nodeMPInfo = nodeKF["ViewMPInfo"];
        {
            vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > vInfo;
            FileNodeIterator itInfo = nodeMPInfo.begin(), itInfoend = nodeMPInfo.end();
            for( ; itInfo != itInfoend; itInfo++) {
                Mat info;
                (*itInfo) >> info;
                vInfo.push_back( toMatrix3d(info) );
            }
            pKF->mViewMPsInfo = (vInfo);
        }

        Mat pose;
        nodeKF["Pose"] >> pose;
        pKF->setPose(pose);

        Point3f odo;
        nodeKF["Odometry"] >> odo;
        pKF->odom = Se2(odo.x, odo.y, odo.z);

        pKF->mfScaleFactor = (float)nodeKF["ScaleFactor"];

        mvKFs.push_back(pKF);
    }

    file.release();

    for(int i = 0, iend = mvKFs.size(); i < iend; i++) {
        PtrKeyFrame pKF = mvKFs[i];
        Mat img = imread(mMapPath + to_string(i) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
        img.copyTo(pKF->img);
    }

}

void MapStorage::loadMapPoints(){

    mvMPs.clear();

    FileStorage file(mMapPath + mMapFile, FileStorage::READ);
    FileNode nodeMPs = file["MapPoints"];
    FileNodeIterator it = nodeMPs.begin(), itend = nodeMPs.end();

    for(; it != itend; it++) {
        PtrMapPoint pMP = make_shared<MapPoint>();

        FileNode nodeMP = *it;

        pMP->mId = (int)nodeMP["Id"];

        Point3f pos;
        nodeMP["Pos"] >> pos;
        pMP->setPos(pos);

        pMP->setGoodPrl(true);

        mvMPs.push_back(pMP);
    }

    file.release();

}

void MapStorage::loadObservations(){
    FileStorage file(mMapPath + mMapFile, FileStorage::READ);
    Mat Index, Obs;

    file["Observations"] >> Obs;

    file["ObservationIndex"] >> Index;

    //mObservations = Obs;
    //Mat_<int> Index = Index_;

    int sizeKF = Obs.rows;
    int sizeMP = Obs.cols;

    for(int i = 0; i < sizeKF; i++) {
        for( int j = 0; j < sizeMP; j++) {
            if( Obs.at<int>(i,j) ) {
                PtrKeyFrame pKF = mvKFs[i];
                PtrMapPoint pMP = mvMPs[j];
                int idx = Index.at<int>(i,j);

                pKF->addObservation(pMP, idx);
                pMP->addObservation(pKF, idx);
            }
        }
    }

    file.release();
}

void MapStorage::loadCovisibilityGraph(){
    FileStorage file(mMapPath + mMapFile, FileStorage::READ);

    Mat mCovisibilityGraph;

    file["CovisibilityGraph"] >> mCovisibilityGraph;

    int sizeKF = mCovisibilityGraph.rows;
    int sizeMP = mCovisibilityGraph.cols;

    for(int i = 0; i < sizeKF; i++) {
        for( int j = 0; j < sizeMP; j++) {
            if( mCovisibilityGraph.at<int>(i,j) ) {
                PtrKeyFrame pKFi = mvKFs[i];
                PtrKeyFrame pKFj = mvKFs[j];
                pKFi->addCovisibleKF(pKFj);
                pKFj->addCovisibleKF(pKFi);
            }
        }
    }

    file.release();

}

void MapStorage::loadOdoGraph(){
    FileStorage file(mMapPath + mMapFile, FileStorage::READ);

    FileNode nodeOdos = file["OdoGraphNextKF"];
    FileNodeIterator it = nodeOdos.begin(), itend = nodeOdos.end();

    for(int i = 0; it != itend; it++, i++) {

        FileNode nodeOdo = *it;

        int j = (int)nodeOdo["NextId"];

        if(j < 0)
            continue;

        Mat measure, info;
        nodeOdo["Measure"] >> measure;
        nodeOdo["Info"] >> info;

        cv::Mat Info = (info);

        PtrKeyFrame pKFi = mvKFs[i];
        PtrKeyFrame pKFj = mvKFs[j];

        pKFi->setOdoMeasureFrom(pKFj, measure, Info);
        pKFj->setOdoMeasureTo(pKFi, measure, Info);

    }
    file.release();

}

void MapStorage::loadFtrGraph(){
    FileStorage file(mMapPath + mMapFile, FileStorage::READ);

    FileNode nodeFtrs = file["FtrGraphPairs"];
    FileNodeIterator it = nodeFtrs.begin(), itend = nodeFtrs.end();

    for(; it != itend; it++) {
        FileNode nodeFtr = (*it);

        Point2i pairId;
        Mat measure, info;
        nodeFtr["PairId"] >> pairId;
        nodeFtr["Measure"] >> measure;
        nodeFtr["Info"] >> info;
        cv::Mat Info = (info);

        PtrKeyFrame pKFi = mvKFs[pairId.x];
        PtrKeyFrame pKFj = mvKFs[pairId.y];

        pKFi->addFtrMeasureFrom(pKFj, measure, Info);
        pKFj->addFtrMeasureTo(pKFi, measure, Info);

    }

    file.release();
}

void MapStorage::loadToMap(){
    mpMap->clear();
    for(int i = 0, iend = mvKFs.size(); i < iend; i++) {
        mpMap->insertKF(mvKFs[i]);
    }
    for(int i = 0, iend = mvMPs.size(); i < iend; i++) {
        mpMap->insertMP(mvMPs[i]);
    }
}

void MapStorage::clearData(){
    mvKFs.clear();
    mvMPs.clear();
    //mCovisibilityGraph.release();
    //mObservations.release();
    mOdoNextId.clear();
}

} // namespace se2lam
