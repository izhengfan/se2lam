/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan)
*/

#ifndef CONVERTER_H
#define CONVERTER_H

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "Config.h"

namespace se2lam{

//cv::Mat toT4x4(cv::Mat R, cv::Mat T);
//cv::Mat toT4x4(float x, float y, float theta);

Eigen::Vector2d toVector2d(const cv::Point2f &cvVector);
g2o::Isometry3D toIsometry3D(const cv::Mat& T);
cv::Mat toCvMat(const g2o::Isometry3D& t);
cv::Point2f toCvPt2f(const Eigen::Vector2d& vec);
cv::Point3f toCvPt3f(const Eigen::Vector3d& vec);
g2o::Isometry3D toIsometry3D(const g2o::SE3Quat& se3quat);
g2o::SE3Quat toSE3Quat(const g2o::Isometry3D&  iso);

// below from ORB_SLAM: https://github.com/raulmur/ORB_SLAM
std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);
g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
cv::Mat toCvMat(const g2o::SE3Quat &SE3);
cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
cv::Mat toCvMat(const Eigen::Matrix3d &m);
cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
cv::Mat toCvMat6f(const g2o::Matrix6d& m);
cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);
Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
Eigen::Matrix<double,2,1> toVector2d(const cv::Mat &cvVector);
Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
g2o::Matrix6d toMatrix6d(const cv::Mat& cvMat6d);
std::vector<float> toQuaternion(const cv::Mat &M);

} // namespace se2lam
#endif
