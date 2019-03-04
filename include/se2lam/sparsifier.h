/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef SPARSIFIER_H
#define SPARSIFIER_H

#include <vector>
#include <g2o/types/sba/types_six_dof_expmap.h>

struct MeasSE3XYZ {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    g2o::Vector3D z;
    g2o::Matrix3D info;
    int idMP = -1;
    int idKF = -1;
};

struct MeasXYZ2UV {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    g2o::Vector2D z;
    g2o::Matrix2D info;
    int idMP = -1;
    int idKF = -1;
};

struct MeasSE3Expmap {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    g2o::SE3Quat z;
    g2o::Matrix6d info;
    int id1 = -1;
    int id2 = -1;
};

class Sparsifier
{
public:
    Sparsifier();

    static void HessianXYZ2UV(g2o::SE3Quat KF, g2o::Vector3D MP, MeasXYZ2UV measure, g2o::CameraParameters* pCamParam,
                              Eigen::Matrix<double, 9, 9>  & H );

    static void JacobianXYZ2UV(g2o::SE3Quat KF, g2o::Vector3D MP, g2o::CameraParameters* pCamParam,
                               Eigen::Matrix<double, 2, 9>  & J);

    static void HessianSE3XYZ(const g2o::SE3Quat KF, const g2o::Vector3D MP, const g2o::Matrix3D info,
                              Eigen::Matrix<double, 9, 9>  & H);

    static void JacobianSE3XYZ(const g2o::SE3Quat KF, const g2o::Vector3D MP,
                               Eigen::Matrix<double, 3, 9>  & J);

    static void DoMarginalizeSE3XYZ(const std::vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > & vKF, const std::vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D> > & vMP,
                                    const std::vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ> > & vMeasure,
                                    g2o::SE3Quat & z_out, g2o::Matrix6d & info_out);

    static void JacobianSE3(const g2o::SE3Quat KF1, const g2o::SE3Quat KF2,
                            Eigen::Matrix<double, 6, 12> & J);

    static void InfoSE3(const g2o::SE3Quat KF1, const g2o::SE3Quat KF2, const Eigen::Matrix<double, 12,12> & info,
                           Eigen::Matrix<double, 6, 6> & H);

};

#endif // SPARSIFIER_H
