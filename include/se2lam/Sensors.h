/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef SENSORS_H
#define SENSORS_H

#include <opencv2/core/core.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace se2lam {

class Sensors {

public:

    Sensors();

    ~Sensors();

    bool update();

    void setUpdated(bool val);

    void updateOdo(double x_, double y_, double theta_, double time_ = 0);

    void updateImg(const cv::Mat &img_, double time_ = 0);

    // After readData(), img_updatd and odo_updated would be set false
    void readData(cv::Point3f& dataOdo, cv::Mat& dataImg);

    void forceSetUpdate(bool val);

protected:

    cv::Mat mImg;
    cv::Point3f mOdo;

    std::mutex mutex_odo;
    std::mutex mutex_img;

    std::atomic_bool odo_updated;
    std::atomic_bool img_updated;

    std::condition_variable cndvSensorUpdate;

    // reserve for sync
    double time_odo;
    double time_img;

};


}


#endif
