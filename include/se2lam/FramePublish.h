/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef FRAMEPUBLISH_H
#define FRAMEPUBLISH_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace se2lam {

class Track;
class GlobalMapper;
class Localizer;

class FramePublish{
public:

    FramePublish();
    FramePublish(Track* pTR, GlobalMapper* pGM);
    ~FramePublish();

    void run();

    cv::Mat drawMatchesInOneImg(const std::vector<cv::KeyPoint> queryKeys,
                                const cv::Mat &trainImg, const std::vector<cv::KeyPoint> trainKeys,
                                const std::vector<int> &matches);

    cv::Mat drawKeys(const std::vector<cv::KeyPoint> keys, const cv::Mat &mImg,
                     std::vector<int> matched);
    cv::Mat drawFrame();

    void setLocalizer(Localizer* localizer);

    bool mbIsLocalize;
    Localizer* mpLocalizer;    

private:

    Track* mpTrack;
    GlobalMapper* mpGM;

    std::vector<cv::KeyPoint> kp, kpRef;
    std::vector<int> matches;

    cv::Mat mImg, mImgRef;
    cv::Mat mImgOut;

};


} // namespace se2lam

#endif // FRAMEPUBLISH_H
