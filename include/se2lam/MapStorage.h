/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef MAPSTORAGE_H
#define MAPSTORAGE_H

#include "Map.h"


//! The storage file structure would be like:

/**
    - /somewhere/set/as/MapPath/
     |
     |---[MapFileName]
     |---0.bmp
     |---1.bmp
     |---2.bmp
     |---...
**/

namespace se2lam {

class MapStorage
{

public:

    MapStorage();

    void setMap(Map* pMap);

    void setFilePath(const string path, const string file);

    // Save map to file
    void saveMap();

    // Load map from file
    void loadMap();

    void clearData();


protected:

    void sortKeyFrames();

    void sortMapPoints();

    void saveKeyFrames();

    void saveMapPoints();

    void saveObservations();

    void saveCovisibilityGraph();

    void saveOdoGraph();

    void saveFtrGraph();

    void loadKeyFrames();

    void loadMapPoints();

    void loadObservations();

    void loadCovisibilityGraph();

    void loadOdoGraph();

    void loadFtrGraph();

    void loadToMap();

    Map* mpMap;

    string mMapPath;
    string mMapFile;

    vector<PtrKeyFrame> mvKFs;

    vector<PtrMapPoint> mvMPs;

    cv::Mat_<int> mObservations;

    cv::Mat_<int> mCovisibilityGraph;

    vector<int> mOdoNextId;

};

}// namespace se2lam

#endif // MAPSTORAGE_H
