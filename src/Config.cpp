/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Config.h"
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>

namespace se2lam{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

std::string Config::DataPath;
int Config::ImgIndex;
int Config::ImgIndexLocalSt;
cv::Size Config::ImgSize;
cv::Mat Config::bTc; // camera extrinsic
cv::Mat Config::cTb; // inv of bTc
cv::Mat Config::Kcam; // camera intrinsic
float Config::fxCam;
float Config::fyCam;
cv::Mat Config::Dcam; // camera distortion

float Config::UPPER_DEPTH;
float Config::LOWER_DEPTH;

int Config::NUM_FILTER_LAST_SEVERAL_MU;
int Config::FILTER_CONVERGE_CONTINUE_COUNT;
float Config::DEPTH_FILTER_THRESHOLD;

float Config::ScaleFactor; // scalefactor in detecting features
int Config::MaxLevel; // level number of pyramid in detecting features
int Config::MaxFtrNumber; // max feature number to detect
float Config::FEATURE_SIGMA;

float Config::ODO_X_UNCERTAIN, Config::ODO_Y_UNCERTAIN, Config::ODO_T_UNCERTAIN;
float Config::ODO_X_NOISE, Config::ODO_Y_NOISE, Config::ODO_T_NOISE;

float Config::PLANEMOTION_XROT_INFO = 1e6;
float Config::PLANEMOTION_YROT_INFO = 1e6;
float Config::PLANEMOTION_Z_INFO = 1;

int Config::LOCAL_FRAMES_NUM;
float Config::TH_HUBER;
int Config::LOCAL_ITER;
bool Config::LOCAL_VERBOSE = false;
int Config::GLOBAL_ITER = 15;
bool Config::GLOBAL_VERBOSE = false;
bool Config::LOCAL_PRINT = false;
bool Config::GLOBAL_PRINT = false;

int Config::FPS;

bool Config::USE_PREV_MAP = false;
bool Config::LOCALIZATION_ONLY = false;
bool Config::SAVE_NEW_MAP = false;
std::string Config::READ_MAP_FILE_NAME;
std::string Config::READ_MAP_FILE_PATH;
std::string Config::WRITE_MAP_FILE_NAME = "se2lam.map";
std::string Config::WRITE_MAP_FILE_PATH = "/home/se2lam/";

std::string Config::WRITE_TRAJ_FILE_NAME;
std::string Config::WRITE_TRAJ_FILE_PATH;

cv::Mat Config::PrjMtrxEye;

int Config::MAPPUB_SCALE_RATIO = 300;

int Config::GM_VCL_NUM_MIN_MATCH_MP = 15;
int Config::GM_VCL_NUM_MIN_MATCH_KP = 30;
double Config::GM_VCL_RATIO_MIN_MATCH_MP = 0.05;

int Config::GM_DCL_MIN_KFID_OFFSET = 20;
double Config::GM_DCL_MIN_SCORE_BEST = 0.005;

void Config::readConfig(const std::string &path){
    DataPath = path;
    std::string camParaPath = path + "/config/CamConfig.yml";
    cv::FileStorage camPara(camParaPath, cv::FileStorage::READ);
    assert(camPara.isOpened());
    cv::Mat _mK, _mD, _rvec, rvec, _T, T, R;
    float height, width;
    camPara["image_height"] >> height;
    camPara["image_width"] >> width;
    camPara["camera_matrix"] >> _mK;
    camPara["distortion_coefficients"] >> _mD;
    camPara["rvec_b_c"] >> _rvec;
    camPara["tvec_b_c"] >> _T;
    _mK.convertTo(Kcam,CV_32FC1);
    _mD.convertTo(Dcam,CV_32FC2);
    _rvec.convertTo(rvec,CV_32FC1);
    _T.convertTo(T,CV_32FC1);
    fxCam = Kcam.at<float>(0,0);
    fyCam = Kcam.at<float>(1,1);
    ImgSize.height = height;
    ImgSize.width = width;
    std::cerr << "# Load camera config ..." << std::endl;
    std::cerr << "- Camera matrix: " << std::endl << " " <<
            Kcam << std::endl <<
            "- Camera distortion: " << std::endl << " " <<
            Dcam << std::endl <<
            "- Img size: " << std::endl << " " <<
            ImgSize << std::endl << std::endl;
    // bTc: camera extrinsic
    cv::Rodrigues(rvec,R);
    bTc = cv::Mat::eye(4,4,CV_32FC1);
    R.copyTo(bTc.rowRange(0,3).colRange(0,3));
    T.copyTo(bTc.rowRange(0,3).col(3));
    cv::Mat RT = R.t();
    cv::Mat t = -RT * T;
    cTb = cv::Mat::eye(4,4,CV_32FC1);
    RT.copyTo(cTb.rowRange(0,3).colRange(0,3));
    t.copyTo(cTb.rowRange(0,3).col(3));


    PrjMtrxEye = Kcam * cv::Mat::eye(3,4,CV_32FC1);
    camPara.release();

    std::string settingsPath = path + "/config/Settings.yml";
    cv::FileStorage settings(settingsPath, cv::FileStorage::READ);
    assert(settings.isOpened());

    ImgIndex = (int)settings["img_num"];
    ImgIndexLocalSt = (int)settings["img_id_local_st"];
    UPPER_DEPTH = (float)settings["upper_depth"];
    LOWER_DEPTH = (float)settings["lower_depth"];
    NUM_FILTER_LAST_SEVERAL_MU = (int)settings["depth_filter_avrg_count"];
    FILTER_CONVERGE_CONTINUE_COUNT = (int)settings["depth_filter_converge_count"];
    DEPTH_FILTER_THRESHOLD = (float)settings["depth_filter_thresh"];
    ScaleFactor = (float)settings["scale_facotr"];
    MaxLevel = (int)settings["max_level"];
    MaxFtrNumber = (int)settings["max_feature_num"];
    FEATURE_SIGMA = (float)settings["feature_sigma"];

    ODO_X_UNCERTAIN = (float)settings["odo_x_uncertain"];
    ODO_Y_UNCERTAIN = (float)settings["odo_y_uncertain"];
    ODO_T_UNCERTAIN = (float)settings["odo_theta_uncertain"];
    ODO_X_NOISE = (float)settings["odo_x_steady_noise"];
    ODO_Y_NOISE = (float)settings["odo_y_steady_noise"];
    ODO_T_NOISE = (float)settings["odo_theta_steady_noise"];
    if(!settings["plane_motion_xrot_info"].empty())
        PLANEMOTION_XROT_INFO = (float)settings["plane_motion_xrot_info"];
    if(!settings["plane_motion_yrot_info"].empty())
        PLANEMOTION_YROT_INFO = (float)settings["plane_motion_yrot_info"];
    if(!settings["plane_motion_z_info"].empty())
        PLANEMOTION_Z_INFO = (float)settings["plane_motion_z_info"];
    LOCAL_FRAMES_NUM = (int)settings["frame_num"];
    TH_HUBER = sqrt((float)settings["th_huber2"]);
    LOCAL_ITER = (int)settings["local_iter"];
    LOCAL_VERBOSE = (bool)(int)(settings["local_verbose"]);
    LOCAL_PRINT = (bool)(int)(settings["local_print"]);
    if((int)settings["global_iter"]){
        GLOBAL_ITER = (int)settings["global_iter"];
    }
    GLOBAL_VERBOSE = (bool)(int)(settings["global_verbose"]);
    GLOBAL_PRINT = (bool)(int)(settings["global_print"]);
    FPS = (int)settings["fps"];

    USE_PREV_MAP = (bool)(int)(settings["use_prev_map"]);
    SAVE_NEW_MAP = (bool)(int)(settings["save_new_map"]);
    LOCALIZATION_ONLY = (bool)(int)(settings["localization_only"]);
    settings["read_map_file_name"] >> READ_MAP_FILE_NAME;
    settings["write_map_file_name"] >> WRITE_MAP_FILE_NAME;
    settings["read_map_file_path"] >> READ_MAP_FILE_PATH;
    settings["write_map_file_path"] >> WRITE_MAP_FILE_PATH;
    settings["write_traj_file_name"] >> WRITE_TRAJ_FILE_NAME;
    settings["write_traj_file_path"] >> WRITE_TRAJ_FILE_PATH;

    MAPPUB_SCALE_RATIO = (int)(settings["mappub_scale_ratio"]);

    GM_VCL_NUM_MIN_MATCH_MP = (int)(settings["gm_vcl_num_min_match_mp"]);
    GM_VCL_NUM_MIN_MATCH_KP = (int)(settings["gm_vcl_num_min_match_kp"]);
    GM_VCL_RATIO_MIN_MATCH_MP = (double)(settings["gm_vcl_ratio_min_match_kp"]);

    GM_DCL_MIN_KFID_OFFSET = (int)(settings["gm_dcl_min_kfid_offset"]);
    GM_DCL_MIN_SCORE_BEST = (double)(settings["gm_dcl_min_score_best"]);

    settings.release();
}

bool Config::acceptDepth(float depth){
    return (depth >= LOWER_DEPTH && depth <= UPPER_DEPTH);
}


Se2::Se2(){}
Se2::Se2(float _x, float _y ,float _theta):
    x(_x), y(_y), theta(normalize_angle(_theta)){}
Se2::~Se2(){}

Se2 Se2::inv() const
{
    float c = std::cos(theta);
    float s = std::sin(theta);
    return Se2(-c*x-s*y, s*x-c*y, -theta);
}

Se2 Se2::operator +(const Se2& that) const{
    float c = std::cos(theta);
    float s = std::sin(theta);
    float _x = x + that.x*c - that.y*s;
    float _y = y + that.x*s + that.y*c;
    float _theta = normalize_angle(theta + that.theta);
    return Se2(_x, _y, _theta);
}

// Same as: that.inv() + *this
Se2 Se2::operator -(const Se2& that) const{
    float dx = x - that.x;
    float dy = y - that.y;
    float dth = normalize_angle(theta - that.theta);

    float c = std::cos(that.theta);
    float s = std::sin(that.theta);
    return Se2(c*dx+s*dy, -s*dx+c*dy, dth);
}

cv::Mat Se2::toCvSE3()const
{
    float c = cos(theta);
    float s = sin(theta);

    return (cv::Mat_<float>(4,4) <<
            c,-s, 0, x,
            s, c, 0, y,
            0, 0, 1, 0,
            0, 0, 0, 1);
}


Se2& Se2::fromCvSE3(const cv::Mat &mat)
{
    float yaw = std::atan2(mat.at<float>(1,0), mat.at<float>(0,0));
    theta = normalize_angle(yaw);
    x = mat.at<float>(0,3);
    y = mat.at<float>(1,3);
    return *this;
}

}
