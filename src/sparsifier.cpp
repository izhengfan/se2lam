/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "sparsifier.h"

using namespace std;

Sparsifier::Sparsifier()
{
}


// compute Hessian from edge XYZ2UV
void Sparsifier::HessianXYZ2UV(g2o::SE3Quat KF, g2o::Vector3D MP, MeasXYZ2UV measure,
                               g2o::CameraParameters* pCamParam,
                               Eigen::Matrix<double, 9, 9>  & H )
{
    // ...
}

// compute Jacobian matrix of image measurement (UV) w.r.t. KF (SE3) and MP (XYZ)
// state vector of KF (SE3) is defined as (x;y;z;qx;qy;qz), from toMinimalVector()
void Sparsifier::JacobianXYZ2UV(g2o::SE3Quat KF, g2o::Vector3D MP, g2o::CameraParameters* pCamParam,
                                Eigen::Matrix<double, 2, 9>  & J)
{
    g2o::Vector2D z_ref = pCamParam->cam_map(KF.map(MP));
    g2o::Vector6d v6KF = KF.toMinimalVector();

    for (int i=0; i<6; i++) {
        g2o::Vector6d v6KF_delta = v6KF;
        v6KF_delta[i] += 0.000001;

        g2o::SE3Quat KF_delta;
        KF_delta.fromMinimalVector(v6KF_delta);

        g2o::Vector2D z_delta = pCamParam->cam_map(KF_delta.map(MP));
        g2o::Vector2D dz = z_delta - z_ref;

        J(i,0) = dz(0)/0.000001;
        J(i,1) = dz(1)/0.000001;
    }

    for (int i=0; i<3; i++) {
        Eigen::Vector3d MP_delta = MP;
        MP_delta[i] += 0.000001;

        g2o::Vector2D z_delta = pCamParam->cam_map(KF.map(MP_delta));
        g2o::Vector2D dz = z_delta - z_ref;

        J(i+6,0) = dz(0)/0.000001;
        J(i+6,1) = dz(1)/0.000001;
    }
}

// compute Jacobian matrix of edge SE3XYZ
void Sparsifier::JacobianSE3XYZ(const g2o::SE3Quat KF, const g2o::Vector3D MP,
                                Eigen::Matrix<double, 3, 9>  & J)
{
    g2o::Vector3D z_ref = KF.inverse() * MP;
    g2o::Vector6d v6KF = KF.toMinimalVector();
    double delta = 1e-6;

    g2o::Vector3D z_delta;
    g2o::Vector3D dz;

    for (int i=0; i<9; i++) {

        if (i<6) {
            g2o::Vector6d v6KF_delta = v6KF;
            v6KF_delta[i] += delta;

            g2o::SE3Quat KF_delta;
            KF_delta.fromMinimalVector(v6KF_delta);

            z_delta = KF_delta.inverse() * MP;
            dz = z_delta - z_ref;
        }
        else {
            Eigen::Vector3d MP_delta = MP;
            MP_delta[i-6] += delta;

            z_delta = KF.inverse() * MP_delta;
            dz = z_delta - z_ref;
        }

        J(0,i) = dz(0)/delta;
        J(1,i) = dz(1)/delta;
        J(2,i) = dz(2)/delta;
    }
}

// compute Hessian matrix of edge SE3XYZ
void Sparsifier::HessianSE3XYZ(const g2o::SE3Quat KF, const g2o::Vector3D MP, const g2o::Matrix3D info,
                               Eigen::Matrix<double, 9, 9>  & H)
{
    Eigen::Matrix<double, 3, 9> J;
    JacobianSE3XYZ(KF, MP, J);
    H = J.transpose() * info * J;
}

// do marginalize and return SE3 constraint between KFs
void Sparsifier::DoMarginalizeSE3XYZ(const std::vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > & vKF, const std::vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D> > & vMP,
                                     const std::vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ> > & vMeasure,
                                     g2o::SE3Quat & z_out, g2o::Matrix6d & info_out)
{
    //    int idxKF1, idxKF2;
    //    idxKF1 = 0; idxKF2 = 1;

    vector<int> vIdMP;
    vector<MeasSE3XYZ> vMeasureRelated;

    int numMeas = vMeasure.size();
    for (int i=0; i<numMeas; i++) {
        MeasSE3XYZ measure = vMeasure.at(i);
        if (measure.idKF != 0 && measure.idKF != 1) {
            continue;
        }

        vMeasureRelated.push_back(measure);

        if (find(vIdMP.begin(), vIdMP.end(), measure.idMP) == vIdMP.end()) {
            vIdMP.push_back(measure.idMP);
        }
    }


    const int dim = 12+vIdMP.size()*3;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

    int numMeasRel = vMeasureRelated.size();
    for (int i=0; i<numMeasRel; i++) {
        MeasSE3XYZ measure = vMeasureRelated.at(i);

        int idKF = measure.idKF;
        int idMP = measure.idMP;

        vector<int>::iterator itr;
        itr = find(vIdMP.begin(), vIdMP.end(), idMP);
        int idMPBlk = vIdMP[itr - vIdMP.begin()];

        g2o::Matrix3D info = measure.info;

        // DEBUG ON NAN
        double d = info(0,0);
        if(std::isnan(d)) {
            cerr << "ERROR!!!" << endl;
        }

        Eigen::Matrix<double, 9, 9> H_local;

        HessianSE3XYZ(vKF.at(idKF), vMP.at(idMPBlk), info, H_local);

        H.block(idKF*6,idKF*6,6,6) += H_local.block(0,0,6,6);
        H.block(12+3*idMPBlk,12+3*idMPBlk,3,3) += H_local.block(6,6,3,3);
        H.block(idKF*6,12+3*idMPBlk,6,3) += H_local.block(0,6,6,3);
        H.block(12+3*idMPBlk,idKF*6,3,6) += H_local.block(6,0,3,6);

    }

    H.block(0,0,12,12) += Eigen::MatrixXd::Identity(12,12) * 1e-6;

    // marginalize hessian

    Eigen::MatrixXd H11 = H.block(0,0,12,12);
    Eigen::MatrixXd H12 = H.block(0,12,12,dim-12);
    Eigen::MatrixXd H21 = H.block(12,0,dim-12,12);
    Eigen::MatrixXd H22 = H.block(12,12,dim-12,dim-12);

    Eigen::MatrixXd T = H22.ldlt().solve(H21);
    Eigen::MatrixXd H_marginal = H11 - H12*T;

    InfoSE3(vKF[0], vKF[1], H_marginal, info_out);
    z_out = vKF[0].inverse() * vKF[1];
}

void Sparsifier::JacobianSE3(const g2o::SE3Quat KF1, const g2o::SE3Quat KF2,
                             Eigen::Matrix<double, 6, 12> & J)
{
    g2o::SE3Quat z_se3_ref = KF1.inverse() * KF2;
    g2o::Vector6d z_ref = z_se3_ref.toMinimalVector();

    g2o::Vector6d v6_KF1 = KF1.toMinimalVector();
    g2o::Vector6d v6_KF2 = KF2.toMinimalVector();

    double delta = 1e-6;

    g2o::Vector6d z_delta;
    g2o::Vector6d dz;

    for (int i=0; i<12; i++) {
        if (i<6) {
            g2o::Vector6d v6_KF1_delta = v6_KF1;
            v6_KF1_delta[i] += delta;

            g2o::SE3Quat se3_KF1_delta;
            se3_KF1_delta.fromMinimalVector(v6_KF1_delta);

            z_delta = (se3_KF1_delta.inverse()*KF2).toMinimalVector();
            dz = z_delta - z_ref;
        }
        else {
            g2o::Vector6d v6_KF2_delta = v6_KF2;
            v6_KF2_delta[i-6] += delta;

            g2o::SE3Quat se3_KF2_delta;
            se3_KF2_delta.fromMinimalVector(v6_KF2_delta);

            z_delta = (KF1.inverse()*se3_KF2_delta).toMinimalVector();
            dz = z_delta - z_ref;
        }

        J.col(i) = dz/delta;
    }
}

void Sparsifier::InfoSE3(const g2o::SE3Quat KF1, const g2o::SE3Quat KF2, const Eigen::Matrix<double, 12,12> & H,
                         Eigen::Matrix<double, 6, 6> & I)
{
    Eigen::Matrix<double, 6, 12> J;
    JacobianSE3(KF1, KF2, J);

//    cerr << H.eigenvalues() << endl;
//    Eigen::Matrix<double, 12, 12> H_inv = H.inverse();
//    cerr << H_inv.eigenvalues() << endl;
//    Eigen::Matrix<double, 6, 6> I_inv = J * H.inverse() * J.transpose();
//    cerr << I_inv.eigenvalues() << endl;
//    Eigen::Matrix<double, 6, 6> I = I_inv.inverse();
//    cerr << I.eigenvalues() << endl;

    I = (J * H.inverse() * J.transpose()).inverse();

    // Set symmetric
    I = (I + I.transpose())/2;

    // Compute SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(I, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 6, 1> s = svd.singularValues();
    Eigen::Matrix<double, 6, 6> U = svd.matrixU();
    Eigen::Matrix<double, 6, 6> V = svd.matrixV();

    for (int i=0; i<6; i++) {
        Eigen::Matrix<double, 6, 1> ui = U.block<6,1>(0,i);
        Eigen::Matrix<double, 6, 1> vi = V.block<6,1>(0,i);

        double norm = ui.dot(vi);
        if (norm >= 0) {
            s(i,0) = max(s(i,0), 1e-6);
            s(i,0) = min(s(i,0), 1e4);
        }
        else {
            s(i,0) = -1e-6;
        }
    }

    // Set limit on eigenvalues and refine info matrix
    Eigen::Matrix<double, 6, 6> S = Eigen::MatrixXd::Zero(6,6);
    for (int i=0; i<6; i++) {
        S(i,i) = s(i,0);
    }
    I = U*S*(V.transpose());

//    cerr << "S:" << endl << S << endl;
//    cerr << "U:" << endl << U << endl;
//    cerr << "V:" << endl << V << endl;
//    cerr << "I:" << endl << I << endl;

    // Set symmetric
    I = (I + I.transpose())/2;

//    cerr << "I_final:" << endl << I << endl;
}





