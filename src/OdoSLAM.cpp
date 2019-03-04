/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef ODOSLAM_CPP
#define ODOSLAM_CPP
#include "OdoSLAM.h"
#include <opencv2/highgui/highgui.hpp>

#endif // ODOSLAM_CPP

namespace se2lam {
using namespace std;
using namespace cv;

std::mutex mMutexFinish;

OdoSLAM::~OdoSLAM(){
    delete mpMapPub;
    delete mpLocalizer;
    delete mpTrack;
    delete mpLocalMapper;
    delete mpGlobalMapper;
    delete mpMap;
    delete mpMapStorage;
    delete mpFramePub;
    delete mpSensors;
}

OdoSLAM::OdoSLAM(){

}

void OdoSLAM::setVocFileBin(const char *strVoc){
    cerr << "\n###\n"
         << "###  se2lam: On-SE(2) Localization and Mapping with SE(2)-XYZ Constraints.\n"
         << "###\n" << endl;

    //Init ORB BoW
    cerr << endl << "Loading ORB Vocabulary. This could take a while." << endl;
    string strVocFile = strVoc;
    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
    if (!bVocLoad) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        return;
    }
    cerr << "Vocabulary loaded!" << endl << endl;
}

void OdoSLAM::setDataPath(const char *strDataPath){

    Config::readConfig(strDataPath);

}

cv::Mat OdoSLAM::getCurrentVehiclePose()
{
    return cvu::inv( mpMap->getCurrentFramePose() ) * Config::cTb;
}

cv::Mat OdoSLAM::getCurrentCameraPoseWC()
{
    return cvu::inv( mpMap->getCurrentFramePose() );
}

cv::Mat OdoSLAM::getCurrentCameraPoseCW()
{
    return mpMap->getCurrentFramePose();
}

void OdoSLAM::start() {

    // Construct the system
    mpMap = new Map;
    mpSensors = new Sensors;
    mpTrack = new Track;
    mpLocalMapper = new LocalMapper;
    mpGlobalMapper = new GlobalMapper;
    mpFramePub = new FramePublish(mpTrack, mpGlobalMapper);
    mpMapStorage = new MapStorage();
    mpMapPub = new MapPublish(mpMap);
    mpLocalizer = new Localizer();

    mpTrack->setLocalMapper(mpLocalMapper);
    mpTrack->setMap(mpMap);
    mpTrack->setSensors(mpSensors);

    mpLocalMapper->setMap(mpMap);
    mpLocalMapper->setGlobalMapper(mpGlobalMapper);

    mpGlobalMapper->setMap(mpMap);
    mpGlobalMapper->setLocalMapper(mpLocalMapper);
    mpGlobalMapper->setORBVoc(mpVocabulary);

    mpMapStorage->setMap(mpMap);

    mpMapPub->setFramePub(mpFramePub);

    mpLocalizer->setMap(mpMap);
    mpLocalizer->setORBVoc(mpVocabulary);
    mpLocalizer->setSensors(mpSensors);


    mpFramePub->setLocalizer(mpLocalizer);
    mpMapPub->setLocalizer(mpLocalizer);


    if (Config::USE_PREV_MAP){
        mpMapStorage->setFilePath(Config::READ_MAP_FILE_PATH, Config::READ_MAP_FILE_NAME);
        mpMapStorage->loadMap();
    }

    mbFinishRequested = false;
    mbFinished = false;

    if (se2lam::Config::LOCALIZATION_ONLY) {

        thread threadLocalizer(&se2lam::Localizer::run, mpLocalizer);

        mpFramePub->mbIsLocalize = true;
        mpMapPub->mbIsLocalize = true;

        thread threadMapPub(&se2lam::MapPublish::run, mpMapPub);

        threadLocalizer.detach();
        threadMapPub.detach();

    }
    // SLAM case
    else {

        cout << "Running SLAM" << endl;

        mpMapPub->mbIsLocalize = false;
        mpFramePub->mbIsLocalize = false;


        thread threadTracker(&se2lam::Track::run, mpTrack);
        thread threadLocalMapper(&se2lam::LocalMapper::run, mpLocalMapper);
        thread threadGlobalMapper(&se2lam::GlobalMapper::run, mpGlobalMapper);
        thread threadMapPub(&se2lam::MapPublish::run, mpMapPub);

        threadTracker.detach();
        threadLocalMapper.detach();
        threadGlobalMapper.detach();
        threadMapPub.detach();

    }

    thread threadWait(&wait, this);
    threadWait.detach();

}

void OdoSLAM::wait(OdoSLAM* system){

    ros::Rate rate(Config::FPS * 10);
    cv::Mat empty(100, 640, CV_8U, cv::Scalar(0));

    //cv::namedWindow("Press q on this window to exit...");
    while (1) {
        if (system->checkFinish()) {

            system->sendRequestFinish();

            break;
        }
        //cv::imshow("Press q on this window to exit...", empty);
        if(cv::waitKey(5) == 'q'){
            system->requestFinish();
        }
        rate.sleep();
    }
    //cv::destroyAllWindows();

    system->saveMap();

    system->checkAllExit();

    system->clear();

    system->mbFinished = true;

    cerr << "System is cleared .." << endl;

}

void OdoSLAM::saveMap() {
    if (se2lam::Config::SAVE_NEW_MAP){
        mpMapStorage->setFilePath(se2lam::Config::WRITE_MAP_FILE_PATH, se2lam::Config::WRITE_MAP_FILE_NAME);
        printf("&& DBG MS: Begin save map.\n");
        mpMapStorage->saveMap();
    }

    // Save keyframe trajectory
    cerr << "\n# Finished. Saving keyframe trajectory ..." << endl;
    ofstream towrite(se2lam::Config::WRITE_MAP_FILE_PATH  + "/se2lam_kf_trajectory.txt");
    vector<se2lam::PtrKeyFrame> vct = mpMap->getAllKF();
    for (size_t i = 0; i<vct.size(); i++){
        if (!vct[i]->isNull()){
            Mat wTb = cvu::inv(se2lam::Config::bTc * vct[i]->getPose());
            Mat wRb = wTb.rowRange(0, 3).colRange(0, 3);
            g2o::Vector3D euler = g2o::internal::toEuler(se2lam::toMatrix3d(wRb));
            towrite << vct[i]->id << " " <<
                       wTb.at<float>(0, 3) << " " <<
                       wTb.at<float>(1, 3) << " " <<
                       wTb.at<float>(2, 3) << " " <<
                       euler(2) << endl;
        }
    }
}

void OdoSLAM::requestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool OdoSLAM::checkFinish(){
    unique_lock<mutex> lock(mMutexFinish);
    if(se2lam::Config::LOCALIZATION_ONLY){
        if(mpLocalizer->isFinished() || mpMapPub->isFinished()){
            mbFinishRequested = true;
            return true;
        }
    } else {
        if(mpTrack->isFinished() || mpLocalMapper->isFinished() ||
                mpGlobalMapper->isFinished() || mpMapPub->isFinished()) {
            mbFinishRequested = true;
            return true;
        }
    }

    return mbFinishRequested;
}

void OdoSLAM::sendRequestFinish(){
    if (Config::LOCALIZATION_ONLY) {
        mpLocalizer->requestFinish();
        mpMapPub->RequestFinish();
    }
    else {
        mpTrack->requestFinish();
        mpLocalMapper->requestFinish();
        mpGlobalMapper->requestFinish();
        mpMapPub->RequestFinish();
    }
}

void OdoSLAM::checkAllExit() {
    if (Config::LOCALIZATION_ONLY) {
        while (1) {
            if (mpLocalizer->isFinished() && mpMapPub->isFinished())
                break;
            else
                std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    }
    else {
        while (1) {
            if (mpTrack->isFinished() && mpLocalMapper->isFinished() &&
                    mpGlobalMapper->isFinished() && mpMapPub->isFinished()) {
                break;
            }
            else {
                std::this_thread::sleep_for(std::chrono::microseconds(2));
            }
        }
    }
}

void OdoSLAM::clear() {

}

void OdoSLAM::waitForFinish(){
    while (1) {
        if (mbFinished) {
            break;
        }
        else {
            std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
    cerr << "wait for finish finished..." << endl;
}

bool OdoSLAM::ok(){
    unique_lock<mutex> lock(mMutexFinish);
    return !mbFinishRequested;
}

} // namespace se2lam
