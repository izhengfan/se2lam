/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef CONFIG_H
#define CONFIG_H

#include <opencv2/core/core.hpp>

namespace se2lam{

struct Se2{
    float x;
    float y;
    float theta;
    Se2();
    Se2(float _x, float _y ,float _theta);
    ~Se2();
    Se2 inv() const;
    Se2 operator -(const Se2& that) const;
    Se2 operator +(const Se2& that) const;
    cv::Mat toCvSE3() const;
    Se2& fromCvSE3(const cv::Mat& mat);
};
inline double normalize_angle(double theta)
{
  if (theta >= -M_PI && theta < M_PI)
    return theta;

  double multiplier = floor(theta / (2*M_PI));
  theta = theta - multiplier*2*M_PI;
  if (theta >= M_PI)
    theta -= 2*M_PI;
  if (theta < -M_PI)
    theta += 2*M_PI;

  return theta;
}
class WorkTimer
{
private:
    int64 tickBegin, tickEnd;
public:
    WorkTimer(){}
    ~WorkTimer(){}
    double time;
    void start(){
        tickBegin = cv::getTickCount();
    }

    void stop(){
        tickEnd = cv::getTickCount();
        time = (double)(tickEnd- tickBegin) / ((double)cv::getTickFrequency()) * 1000.;
    }
};


class Config{
public:
    static std::string DataPath;
    static int ImgIndex;
    static int ImgIndexLocalSt;
    static cv::Size ImgSize;
    static cv::Mat bTc; // camera extrinsic
    static cv::Mat cTb; // inv of bTc
    static cv::Mat Kcam; // camera intrinsic
    static float fxCam, fyCam;
    static cv::Mat Dcam; // camera distortion

    static float UPPER_DEPTH;
    static float LOWER_DEPTH;

    static int NUM_FILTER_LAST_SEVERAL_MU;
    static int FILTER_CONVERGE_CONTINUE_COUNT;
    static float DEPTH_FILTER_THRESHOLD;

    static float ScaleFactor; // scalefactor in detecting features
    static int MaxLevel; // level number of pyramid in detecting features
    static int MaxFtrNumber; // max feature number to detect
    static float FEATURE_SIGMA;

    static float ODO_X_UNCERTAIN, ODO_Y_UNCERTAIN, ODO_T_UNCERTAIN;
    static float ODO_X_NOISE, ODO_Y_NOISE, ODO_T_NOISE;

    static float PLANEMOTION_Z_INFO;
    static float PLANEMOTION_XROT_INFO;
    static float PLANEMOTION_YROT_INFO;

    static int LOCAL_FRAMES_NUM;
    static float TH_HUBER;
    static int LOCAL_ITER;
    static bool LOCAL_VERBOSE;
    static int GLOBAL_ITER;
    static bool GLOBAL_VERBOSE;

    static bool LOCAL_PRINT;
    static bool GLOBAL_PRINT;

    static int FPS;
    static cv::Mat PrjMtrxEye;

    static bool USE_PREV_MAP;
    static bool LOCALIZATION_ONLY;
    static bool SAVE_NEW_MAP;
    static std::string READ_MAP_FILE_NAME;
    static std::string WRITE_MAP_FILE_NAME;
    static std::string READ_MAP_FILE_PATH;
    static std::string WRITE_MAP_FILE_PATH;

    static std::string WRITE_TRAJ_FILE_NAME;
    static std::string WRITE_TRAJ_FILE_PATH;

    static int MAPPUB_SCALE_RATIO;

    static int GM_VCL_NUM_MIN_MATCH_MP;
    static int GM_VCL_NUM_MIN_MATCH_KP;
    static double GM_VCL_RATIO_MIN_MATCH_MP;

    static int GM_DCL_MIN_KFID_OFFSET;
    static double GM_DCL_MIN_SCORE_BEST;

    static void readConfig(const std::string& path);
    static bool acceptDepth(float depth);

};

}//namespace se2lam

#endif // CONFIG_H
