/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <g2o/core/eigen_types.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/dquat2mat.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam2d/vertex_se2.h>
#include <g2o/types/slam2d/edge_se2.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/factory.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include "Config.h"
#include "KeyFrame.h"
#include "EdgeSE2XYZ.h"

namespace se2lam{

typedef g2o::BlockSolverX SlamBlockSolver;
typedef g2o::LinearSolverCholmod<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
typedef g2o::OptimizationAlgorithmLevenberg SlamAlgorithm;
typedef g2o::SparseOptimizer SlamOptimizer;
typedef g2o::CameraParameters CamPara;

inline Eigen::Quaterniond toQuaterniond(const Eigen::Vector3d &rot_vector)
{
    double angle = rot_vector.norm();
    if(angle <= 1e-14)
        return Eigen::Quaterniond(1, 0, 0, 0);
    else
        return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rot_vector.normalized()));
}

inline Eigen::Vector3d toRotationVector(const Eigen::Quaterniond &q_)
{
    Eigen::AngleAxisd angle_axis(q_);
    return angle_axis.angle() * angle_axis.axis();
}

class G2O_TYPES_SBA_API EdgeSE3ExpmapPrior: public g2o::BaseUnaryEdge<6, g2o::SE3Quat, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3ExpmapPrior();

    // Useless functions we don't care
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    void computeError();

    void setMeasurement(const g2o::SE3Quat& m);

    virtual void linearizeOplus();
};

g2o::Matrix3D
Jl(const g2o::Vector3D& v3d);

g2o::Matrix3D
invJl(const g2o::Vector3D& v3d);


g2o::Matrix6d
invJJl(const g2o::Vector6d& v6d);

void
initOptimizer(SlamOptimizer &opt, bool verbose=false);

EdgeSE3ExpmapPrior*
addPlaneMotionSE3Expmap(SlamOptimizer& opt, const g2o::SE3Quat &pose, int vId, const cv::Mat& extPara);


CamPara*
addCamPara(SlamOptimizer &opt, const cv::Mat& K, int id);

void
addVertexSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat& pose, int id, bool fixed=false);

void
addVertexSBAXYZ(SlamOptimizer &opt, const Eigen::Vector3d &xyz, int id, bool marginal=true, bool fixed=false);

void
addEdgeSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat& measure, int id0, int id1, const g2o::Matrix6d& info);

g2o::EdgeProjectXYZ2UV*
addEdgeXYZ2UV(SlamOptimizer &opt, const Eigen::Vector2d& measure, int id0, int id1, int paraId, const Eigen::Matrix2d &info, double thHuber);

g2o::EdgeSE2XYZ*
addEdgeSE2XYZ(SlamOptimizer &opt, const g2o::Vector2D& meas, int id0, int id1,
              CamPara* campara, const g2o::SE3Quat &_Tbc, const g2o::Matrix2D &info, double thHuber);

g2o::VertexSE2*
addVertexSE2(SlamOptimizer &opt, const g2o::SE2& pose, int id, bool fixed = false);

g2o::SE2
estimateVertexSE2(SlamOptimizer &opt, int id);

g2o::PreEdgeSE2 *addEdgeSE2(SlamOptimizer &opt, const g2o::Vector3D& meas,
           int id0, int id1, const g2o::Matrix3D& info);

g2o::ParameterSE3Offset*
addParaSE3Offset(SlamOptimizer &opt, const g2o::Isometry3D& se3offset, int id);

void
addVertexSE3(SlamOptimizer &opt, const g2o::Isometry3D &pose, int id, bool fixed=false);

g2o::EdgeSE3Prior*
addVertexSE3PlaneMotion(SlamOptimizer &opt, const g2o::Isometry3D &pose, int id, const cv::Mat& extPara, int paraSE3OffsetId, bool fixed=false);

void
addVertexXYZ(SlamOptimizer &opt, const g2o::Vector3D &xyz, int id, bool marginal=true);

g2o::EdgeSE3*
addEdgeSE3(SlamOptimizer &opt, const g2o::Isometry3D &measure, int id0, int id1, const g2o::Matrix6d& info);

g2o::EdgeSE3PointXYZ*
addEdgeSE3XYZ(SlamOptimizer &opt, const g2o::Vector3D& measure, int idse3, int idxyz, int paraSE3OffsetId, const g2o::Matrix3D &info, double thHuber);


g2o::Isometry3D
estimateVertexSE3(SlamOptimizer &opt, int id);

Eigen::Vector3d
estimateVertexXYZ(SlamOptimizer &opt, int id);

g2o::SE3Quat
estimateVertexSE3Expmap(SlamOptimizer &opt, int id);

g2o::Vector3D
estimateVertexSBAXYZ(SlamOptimizer &opt, int id);

bool
verifyInfo(const g2o::Matrix6d& info);

bool
verifyInfo(const Eigen::Matrix3d& info);


}// namespace se2lam

#endif // OPTIMIZER_H
