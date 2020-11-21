/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

/*** This file is part of ORB-SLAM.
* It is based on the file orb.cpp from the OpenCV library (see BSD license below)
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

#include "ORBmatcher.h"

#include<limits.h>

#include<ros/ros.h>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "cvutil.h"


namespace se2lam
{
using namespace cv;
using namespace std;

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 75;
const int ORBmatcher::HISTO_LENGTH = 30;


ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}



void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

int ORBmatcher::SearchByBoW(PtrKeyFrame pKF1, PtrKeyFrame pKF2,
                map<int, int> &mapMatches12, bool bIfMPOnly)
{
    mapMatches12.clear();

    if (pKF1 == NULL || pKF1->isNull() || pKF2 == NULL || pKF2->isNull()) {
        return 0;
    }

    vector<cv::KeyPoint> vKeysUn1 = pKF1->keyPointsUn;
    DBoW2::FeatureVector vFeatVec1 = pKF1->GetFeatureVector();
    vector<PtrMapPoint> vpMapPoints1 = pKF1->GetMapPointMatches();
    cv::Mat Descriptors1 = pKF1->descriptors;

    vector<cv::KeyPoint> vKeysUn2 = pKF2->keyPointsUn;
    DBoW2::FeatureVector vFeatVec2 = pKF2->GetFeatureVector();
    vector<PtrMapPoint> vpMapPoints2 = pKF2->GetMapPointMatches();
    cv::Mat Descriptors2 = pKF2->descriptors;



    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = (float)HISTO_LENGTH / 360.0f ;

    int nmatches = 0;

    DBoW2::FeatureVector::iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                size_t idx1 = f1it->second[i1];

                PtrMapPoint pMP1 = vpMapPoints1[idx1];

                if(bIfMPOnly) {
                    if(!pMP1)
                        continue;
                    if(pMP1->isNull())
                        continue;
                }

                cv::Mat d1 = Descriptors1.row(idx1);

                int bestDist1=INT_MAX;
                int bestIdx2 =-1 ;
                int bestDist2=INT_MAX;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    PtrMapPoint pMP2 = vpMapPoints2[idx2];

                    if(bIfMPOnly){
                        if(!pMP2)
                            continue;
                        if(pMP2->isNull())
                            continue;
                    }

                    if(vbMatched2[idx2])
                        continue;

                    cv::Mat d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        mapMatches12[idx1] = bestIdx2;
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                mapMatches12.erase(rotHist[i][j]);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::MatchByWindow(const Frame &frame1, Frame &frame2,
                              vector<Point2f> &vbPrevMatched, const int winSize,  vector<int> &vnMatches12,
                              const int levelOffset, const int minLevel, const int maxLevel) {
    int nmatches = 0;
    vnMatches12 = vector<int>(frame1.N, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = (float)HISTO_LENGTH / 360.0f ;

    vector<int> vMatchesDistance(frame2.N, INT_MAX);
    vector<int> vnMatches21(frame2.N, -1);

    for(int i1 = 0, iend1 = frame1.N; i1 < iend1; i1++){
        KeyPoint kp1 = frame1.keyPointsUn[i1];
        int level1 = kp1.octave;
        if(level1 > maxLevel || level1 < minLevel)
            continue;
        int minLevel2 = level1-levelOffset>0 ? level1-levelOffset : 0;
        vector<size_t> vIndices2 = frame2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, winSize, minLevel2, level1+levelOffset);
        if(vIndices2.empty())
            continue;

        cv::Mat d1 = frame1.descriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(auto vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++){
            size_t i2 = *vit;

            cv::Mat d2 = frame2.descriptors.row(i2);

            int dist = DescriptorDistance(d1, d2);

            if(vMatchesDistance[i2] <= dist)
                continue;

            if(dist < bestDist){
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            } else if(dist < bestDist2){
                bestDist2 = dist;
            }
        }

        if(bestDist <= TH_LOW){
            if(bestDist < (float)bestDist2*mfNNratio){
                if(vnMatches21[bestIdx2] >= 0){
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                vnMatches12[i1] = bestIdx2;
                vnMatches21[bestIdx2] = i1;
                vMatchesDistance[bestIdx2] = bestDist;
                nmatches++;

                // for orientation check
                float rot = frame1.keyPointsUn[i1].angle - frame2.keyPointsUn[bestIdx2].angle;
                if(rot < 0.0)
                    rot += 360.f;
                int bin = round(rot*factor);
                if(bin == HISTO_LENGTH)
                    bin = 0;
                rotHist[bin].push_back(i1);
            }
        }
    }

    // orientation check
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }
    }

    // update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=frame2.keyPointsUn[vnMatches12[i1]].pt;

    return nmatches;

}

int ORBmatcher::MatchByProjection(PtrKeyFrame &pNewKF, std::vector<PtrMapPoint> &localMPs,
                                  const int winSize, const int levelOffset, std::vector<int>& vMatchesIdxMP){
    int nmatches = 0;

    vMatchesIdxMP = vector<int>(pNewKF->N, -1);
    vector<int> vMatchesDistance(pNewKF->N, INT_MAX);

    for(int i = 0, iend = localMPs.size(); i < iend; i++){
        PtrMapPoint pMP = localMPs[i];
        if(pMP->isNull() || !pMP->isGoodPrl())
            continue;
        if(pNewKF->hasObservation(pMP))
            continue;
        //Point2f predictUV = scv::prjcPt2Cam(Config::Kcam, pNewKF->Tcw, pMP->getPos());
        Point2f predictUV = cvu::camprjc(Config::Kcam, cvu::se3map(pNewKF->Tcw, pMP->getPos()));
        if( !pNewKF->inImgBound(predictUV) )
            continue;
        const int predictLevel = pMP->mMainOctave;
        const int levelWinSize = predictLevel * winSize;
        const int minLevel = predictLevel > levelOffset? predictLevel-levelOffset : 0;

        vector<size_t> vNearIndices = pNewKF->GetFeaturesInArea(predictUV.x, predictUV.y, levelWinSize, minLevel, predictLevel+levelOffset);
        if(vNearIndices.empty())
            continue;

        int bestDist = INT_MAX;
        int bestLevel = -1;
        int bestDist2 = INT_MAX;
        int bestLevel2 = -1;
        int bestIdx = -1 ;

        // Get best and second matches with near keypoints
        for(auto it = vNearIndices.begin(), iend = vNearIndices.end(); it != iend; it++){
            int idx = *it;
            if(pNewKF->hasObservation(idx))
                continue;
            cv::Mat d = pNewKF->descriptors.row(idx);

            const int dist = DescriptorDistance(pMP->mMainDescriptor, d);

            if(vMatchesDistance[idx] <= dist)
                continue;

            if(dist < bestDist){
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = pNewKF->keyPoints[idx].octave;
                bestIdx = idx;
            } else if(dist < bestDist2){
                bestLevel2 = pNewKF->keyPoints[idx].octave;
                bestDist2 = dist;
            }

        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist <= TH_HIGH){
            if(bestLevel == bestLevel2 && bestDist > mfNNratio*bestDist2)
                continue;
            if(vMatchesIdxMP[bestIdx] >= 0){
                vMatchesIdxMP[bestIdx] = -1;
                nmatches--;
            }
            vMatchesIdxMP[bestIdx] = i;
            vMatchesDistance[bestIdx] = bestDist;
            nmatches++;
        }
    }

    return nmatches;
}


}//namespace ORB_SLAM
