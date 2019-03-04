/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef LOCALMAPPER_H
#define LOCALMAPPER_H

#include "Map.h"
#include "optimizer.h"
namespace se2lam{

//#define TIME_TO_LOG_LOCAL_BA

class GlobalMapper;

class LocalMapper{
public:
    LocalMapper();

    void run();

    void setMap(Map *pMap);

    void setGlobalMapper(GlobalMapper* pGlobalMapper);

    void addNewKF(PtrKeyFrame &pKF, const std::vector<cv::Point3f>& localMPs, const std::vector<int> &vMatched12, const std::vector<bool>& vbGoodPrl);

    void findCorrespd(const std::vector<int> &vMatched12, const std::vector<cv::Point3f> &localMPs, const std::vector<bool>& vbGoodPrl);

    void removeOutlierChi2();

    void localBA();

    void setAbortBA();

    bool acceptNewKF();

    void setGlobalBABegin(bool value);

    void printOptInfo(const SlamOptimizer & _optimizer);    // For debugging by hbtang
    bool mbPrintDebugInfo;


    void requestFinish();
    bool isFinished();

    std::mutex mutexMapper;

    void updateLocalGraphInMap();

    void pruneRedundantKfInMap();


protected:
    Map* mpMap;
    GlobalMapper* mpGlobalMapper;
    PtrKeyFrame mNewKF;

    bool mbUpdated;
    bool mbAbortBA;
    bool mbAcceptNewKF;
    bool mbGlobalBABegin;

    mutex mMutexLocalGraph;

    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;



};

}


#endif // LOCALMAPPER_H
