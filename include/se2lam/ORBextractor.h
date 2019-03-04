/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace se2lam
{

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures = 1000, float scaleFactor = 1.2f, int nlevels = 8, int scoreType=FAST_SCORE, int fastTh = 20);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image
    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}


protected:

    void ComputePyramid(cv::Mat image, cv::Mat Mask=cv::Mat());
    void ComputeKeyPoints(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int scoreType;
    int fastTh;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<cv::Mat> mvMaskPyramid;

};

} //namespace se2lam

#endif

