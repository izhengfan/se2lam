/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "converter.h"
#include <g2o/types/slam3d/isometry3d_mappings.h>
#include <cassert>

namespace se2lam{


Eigen::Vector2d toVector2d(const cv::Point2f &cvVector){
    Eigen::Vector2d v;
    v << cvVector.x, cvVector.y;
    return v;
}

g2o::Isometry3D toIsometry3D(const cv::Mat &T){
    Eigen::Matrix<double,3,3> R;
    R << T.at<float>(0,0), T.at<float>(0,1), T.at<float>(0,2),
         T.at<float>(1,0), T.at<float>(1,1), T.at<float>(1,2),
         T.at<float>(2,0), T.at<float>(2,1), T.at<float>(2,2);
    g2o::Isometry3D ret = (g2o::Isometry3D) Eigen::Quaterniond(R);
    Eigen::Vector3d t(T.at<float>(0,3), T.at<float>(1,3), T.at<float>(2,3));
    ret.translation() = t;
    return ret;
}

cv::Mat toCvMat(const g2o::Isometry3D &t){
    return toCvMat(g2o::internal::toSE3Quat(t));
}

cv::Point2f toCvPt2f(const Eigen::Vector2d& vec){
    return cv::Point2f(vec(0),vec(1));
}

cv::Point3f toCvPt3f(const Eigen::Vector3d& vec){
    return cv::Point3f(vec(0), vec(1), vec(2));
}

g2o::Isometry3D toIsometry3D(const g2o::SE3Quat &se3quat){
    return g2o::internal::fromSE3Quat(se3quat);
}

g2o::SE3Quat toSE3Quat(const g2o::Isometry3D &iso){
    return g2o::internal::toSE3Quat(iso);
}


cv::Mat toCvMat6f(const g2o::Matrix6d& m) {
    cv::Mat mat(6, 6, CV_32FC1);
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 6; j++)
            mat.at<float>(i,j) = (float)m(i,j);
    return mat;
}

g2o::Matrix6d toMatrix6d(const cv::Mat &cvMat6d){
    g2o::Matrix6d m = g2o::Matrix6d::Zero();
    for(int i = 0; i < 6; i++)
        for(int j = 0; j < 6; j++)
            m(i,j) = cvMat6d.at<float>(i,j);
    return m;
}


// below from ORB_SLAM: https://github.com/raulmur/ORB_SLAM
std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

g2o::SE3Quat toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
            cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
            cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

cv::Mat toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}


cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32FC1);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}


cv::Mat toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32FC1);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32FC1);
    for(int i=0;i<3;i++)
        cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}

cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32FC1);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}

Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,2,1> toVector2d(const cv::Mat &cvVector)
{
    assert((cvVector.rows==2 && cvVector.cols==1) || (cvVector.cols==2 && cvVector.rows==1));

    Eigen::Matrix<double,2,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1);

    return v;
}

Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<float> toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);

    v[0] = q.w();
    v[1] = q.x();
    v[2] = q.y();
    v[3] = q.z();

    return v;
}


}
