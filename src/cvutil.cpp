/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "cvutil.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace cvu{

using namespace cv;
using namespace std;

Mat inv(const Mat &T4x4){
    assert(T4x4.cols == 4 && T4x4.rows == 4);
    Mat RT = T4x4.rowRange(0,3).colRange(0,3).t();
    Mat t = -RT * T4x4.rowRange(0,3).col(3);
    Mat T = Mat::eye(4,4,CV_32FC1);
    RT.copyTo(T.rowRange(0,3).colRange(0,3));
    t.copyTo(T.rowRange(0,3).col(3));
    return T;
}

void pts2Ftrs(const vector<KeyPoint>& _orgnFtrs, const vector<Point2f>& _points, vector<KeyPoint>& _features) {
    _features.resize(_points.size());
    for (size_t i = 0; i < _points.size(); i ++) {
        _features[i] = _orgnFtrs[i];
        _features[i].pt = _points[i];
    }
}


Mat sk_sym(const Point3f _v){
    Mat mat(3,3,CV_32FC1, Scalar(0));
    mat.at<float>(0,1) = -_v.z;
    mat.at<float>(0,2) =  _v.y;
    mat.at<float>(1,0) =  _v.z;
    mat.at<float>(1,2) = -_v.x;
    mat.at<float>(2,0) = -_v.y;
    mat.at<float>(2,1) =  _v.x;
    return mat;
}


Point3f triangulate(const Point2f &pt1, const Point2f &pt2, const Mat &P1, const Mat &P2){
    Mat A(4,4,CV_32FC1);

    A.row(0) = pt1.x*P1.row(2)-P1.row(0);
    A.row(1) = pt1.y*P1.row(2)-P1.row(1);
    A.row(2) = pt2.x*P2.row(2)-P2.row(0);
    A.row(3) = pt2.y*P2.row(2)-P2.row(1);

    Mat u, w, vt, x3D;
    SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A|SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

    return Point3f(x3D);

#ifdef USE_EIGEN_FOR_SVD
    Eigen::Matrix4d A;
    Mtrx34d _P1, _P2;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 4; j++) {
            _P1(i,j) = P1.at<float>(i,j);
            _P2(i,j) = P2.at<float>(i,j);
        }


    A.row(0).noalias() = pt1.x*_P1.row(2)-_P1.row(0);
    A.row(1).noalias() = pt1.y*_P1.row(2)-_P1.row(1);
    A.row(2).noalias() = pt2.x*_P2.row(2)-_P2.row(0);
    A.row(3).noalias() = pt2.y*_P2.row(2)-_P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix4d V = svd.matrixV();
    Vtr3d x3d = (1.0/V(3,3))*V.block<3,1>(0,3);

    x3D = Mat(3,1,CV_32FC1);
    for(int i = 0;i < 3; i++)
        x3D.at<float>(i) = x3d(i);
#endif
}

Point2f camprjc(const Mat &_K, const Point3f &_pt)
{
    Point3f uvw = Matx33f(_K) * _pt;
    return Point2f(uvw.x/uvw.z, uvw.y/uvw.z);
}

bool checkParallax(const Point3f &o1, const Point3f &o2, const Point3f &pt3, int minDegree){
    float minCos[4] = {0.9998, 0.9994, 0.9986, 0.9976};
    Point3f p1 = pt3 - o1;
    Point3f p2 = pt3 - o2;
    float cosParallax = cv::norm(p1.dot(p2)) / ( cv::norm(p1) * cv::norm(p2) );
    return cosParallax < minCos[minDegree-1];
}

Point3f se3map(const Mat &_Tcw, const Point3f &_pt)
{
    Matx33f R(_Tcw.rowRange(0,3).colRange(0,3));
    Point3f t(_Tcw.rowRange(0,3).col(3));
    return (R*_pt + t);
}


} // namespace scv
