/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef MAP_H
#define MAP_H

#include "KeyFrame.h"
#include "MapPoint.h"
#include "Config.h"
#include "optimizer.h"
#include <unordered_map>
#include <set>

namespace se2lam {

class LocalMapper;

class Map{

public:
    Map();
    ~Map();

    void insertKF(const PtrKeyFrame& pkf);
    void insertMP(const PtrMapPoint& pmp);

    void eraseKF(const PtrKeyFrame& pKF);
    void eraseMP(const PtrMapPoint& pMP);

    std::vector<PtrKeyFrame> getAllKF();
    std::vector<PtrMapPoint> getAllMP();
    size_t countKFs();
    size_t countMPs();

    void clear();
    bool empty();

    PtrKeyFrame getCurrentKF();
    void setCurrentKF(const PtrKeyFrame &pKF);

    void setCurrentFramePose(const cv::Mat& pose);
    cv::Mat getCurrentFramePose();

    cv::SparseMat mFtrBasedGraph;
    cv::SparseMat mOdoBasedGraph;
    std::unordered_map<int, SE3Constraint> mFtrBasedEdges;
    std::unordered_map<int, SE3Constraint> mOdoBasedEdges;
    std::vector<int> mIdxFtrBased;
    std::vector<int> mIdxOdoBased;

    void mergeMP(PtrMapPoint& toKeep, PtrMapPoint& toDelete);


    bool locked();
    void lock();
    void unlock();

    static cv::Point2f compareViewMPs(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2, std::set<PtrMapPoint>& spMPs);

    static double compareViewMPs(const PtrKeyFrame & pKF, const set<PtrKeyFrame> & vpKFs, std::set<PtrMapPoint> & vpMPs, int k = 2);

    static bool checkAssociationErr(const PtrKeyFrame& pKF, const PtrMapPoint& pMP);


    //! For LocalMapper
    void setLocalMapper(LocalMapper* pLocalMapper);

    void updateLocalGraph();

    bool pruneRedundantKF();

    void loadLocalGraph(SlamOptimizer& optimizer);

    void loadLocalGraph(SlamOptimizer& optimizer, std::vector< std::vector<g2o::EdgeProjectXYZ2UV*> > &vpEdgesAll, std::vector< std::vector<int> >& vnAllIdx);

    void loadLocalGraphOnlyBa(SlamOptimizer& optimizer, std::vector< std::vector<g2o::EdgeProjectXYZ2UV*> > &vpEdgesAll, std::vector< std::vector<int> >& vnAllIdx);

    int removeLocalOutlierMP(const vector<vector<int> > &vnOutlierIdxAll);

    void optimizeLocalGraph(SlamOptimizer& optimizer);

    void updateCovisibility(PtrKeyFrame& pNewKF);

    std::vector<PtrKeyFrame> getLocalKFs();
    std::vector<PtrMapPoint> getLocalMPs();
    std::vector<PtrKeyFrame> getRefKFs();

    int countLocalKFs();
    int countLocalMPs();


    //! For GlobalMapper
    void mergeLoopClose(const std::map<int, int>& mapMatchMP, PtrKeyFrame& pKFCurr, PtrKeyFrame& pKFLoop);

    //! Set KF pair waiting for feature constraint generation, called by localmapper
    std::vector<pair<PtrKeyFrame, PtrKeyFrame>> SelectKFPairFeat(const PtrKeyFrame &_pKF);

    //! Update feature constraint graph, on KFs pairs given by LocalMapper
    bool UpdateFeatGraph(const PtrKeyFrame &_pKF);


protected:

    PtrKeyFrame mCurrentKF;

    bool isEmpty;

    //! Global Map
    std::set<PtrMapPoint, MapPoint::IdLessThan> mMPs;
    std::set<PtrKeyFrame, KeyFrame::IdLessThan> mKFs;

    //! Local Map
    std::vector<PtrMapPoint> mLocalGraphMPs;
    std::vector<PtrKeyFrame> mLocalGraphKFs;
    std::vector<PtrKeyFrame> mRefKFs;
    LocalMapper* mpLocalMapper;

    cv::Mat mCurrentFramePose;

    std::mutex mMutexGraph;
    std::mutex mMutexLocalGraph;
    std::mutex mMutexCurrentKF;
    std::mutex mMutexCurrentFrame;

}; //class Map

}// namespace se2lam

#endif
