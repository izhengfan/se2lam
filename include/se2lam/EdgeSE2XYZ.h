/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef EDGE_SE2_XYZ_H
#define EDGE_SE2_XYZ_H

#pragma once

#include <Eigen/Dense>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam2d/vertex_se2.h>

#define CUSTOMIZE_JACOBIAN_SE2XYZ

namespace g2o
{
typedef Eigen::Matrix<double, 2, 3> Matrix23d;
typedef Eigen::Matrix<double, 3, 2> Matrix32d;

Eigen::Matrix3d d_inv_d_se2(const g2o::SE2& _se2);

g2o::SE3Quat SE2ToSE3(const g2o::SE2& _se2);

g2o::SE2 SE3ToSE2(const g2o::SE3Quat& _se3);

class EdgeSE2XYZ : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE2, g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE2XYZ();
    ~EdgeSE2XYZ();

    // Useless functions we don't care
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;


    // Note: covariance are set here. Just set information to identity outside.
    void computeError();


    virtual void linearizeOplus();

    inline void setCameraParameter(g2o::CameraParameters* _cam){cam = _cam;}

    inline void setExtParameter(const g2o::SE3Quat& _Tbc) { Tbc = _Tbc; Tcb = Tbc.inverse(); }

private:
    g2o::SE3Quat Tbc;
    g2o::SE3Quat Tcb;

    g2o::CameraParameters * cam;
};

class PreEdgeSE2 : public BaseBinaryEdge<3, Vector3D, VertexSE2, VertexSE2>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PreEdgeSE2(){}

    void computeError()
    {
        const VertexSE2* v1 = static_cast<const VertexSE2*>(_vertices[0]);
        const VertexSE2* v2 = static_cast<const VertexSE2*>(_vertices[1]);
        Matrix2D Ri = v1->estimate().rotation().toRotationMatrix();
        Vector2D ri = v1->estimate().translation();
        double ai = v1->estimate().rotation().angle();
        double aj = v2->estimate().rotation().angle();
        Vector2D rj = v2->estimate().translation();

        _error.head<2>() = Ri.transpose() * (rj-ri) - _measurement.head<2>();
        _error[2] = aj - ai - _measurement[2];

    }
    virtual void linearizeOplus()
    {
        const VertexSE2* v1 = static_cast<const VertexSE2*>(_vertices[0]);
        const VertexSE2* v2 = static_cast<const VertexSE2*>(_vertices[1]);
        Matrix2D Ri = v1->estimate().rotation().toRotationMatrix();
        Vector2D ri = v1->estimate().translation();
        Vector2D rj = v2->estimate().translation();
        Vector2D rij = rj-ri;
        Vector2D rij_x(-rij[1], rij[0]);

        _jacobianOplusXi.block<2,2>(0,0) = -Ri.transpose();
        _jacobianOplusXi.block<2,1>(0,2) = -Ri.transpose() * rij_x;
        _jacobianOplusXi.block<1,2>(2,0).setZero();
        _jacobianOplusXi(2,2) = -1;

        _jacobianOplusXj.setIdentity();
        _jacobianOplusXj.block<2,2>(0,0) = Ri.transpose();
    }
    virtual bool read(std::istream& is) {return true;}
    virtual bool write(std::ostream& os) const {return true;}
};
}

#endif
