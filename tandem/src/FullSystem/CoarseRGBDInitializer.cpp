/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 *
 *  Created on: Jun, 2020
 *      Author: nan
 */

#include "FullSystem/CoarseRGBDInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseRGBDInitializer::CoarseRGBDInitializer(int ww, int hh) :
        width(ww),
        height(hh),
        thisToNext_aff(0,0),
        thisToNext(SE3()),
        newFrame(nullptr),
        tracked_frame(0)
{
    frameID = -1;
    coarse_tracker = new CoarseTracker(width, height);
}

CoarseRGBDInitializer::~CoarseRGBDInitializer()
{
	delete coarse_tracker;
}


bool CoarseRGBDInitializer::trackFrame(FrameHessian* newFrameHessian, CalibHessian* HCalib, RGBDepth* rgbd_depth, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(!snapped)
    {
        std::vector<FrameHessian*> frameHessians;
        frameHessians.push_back(cur_ref);
        coarse_tracker -> setCoarseTrackingRef(frameHessians, cur_ref, true);
    }

    std::cout << "Track Frame: " <<  newFrameHessian->shell->id  << std::endl;
    std::cout << "Track Frame Ref: " <<  cur_ref->shell->id  << std::endl;

    SE3 currBestPose = SE3();
    AffLight aff_g2l = AffLight(0,0);
    Vec5 achievedResL = Vec5::Constant(NAN);

    bool trackingIsGood = coarse_tracker->trackNewestCoarse(newFrameHessian,
                                                              currBestPose,
                                                              aff_g2l,
                                                              pyrLevelsUsed-1,
                                                              achievedResL);

    if(!trackingIsGood)
    {
        printf("---------------------------------------------\n");
        printf("Initialization Coarse Tracking failed!\n");

        untracked_frames.push_back(newFrameHessian);
        setFirst(HCalib, newFrameHessian, rgbd_depth);
        return false;
    }

    snapped = true;

    if(newFrameHessian->shell->id > 1)
    {
        SE3 framePose = SE3();
        for(int i = 0; i < untracked_frames.size(); ++i)
        {
            framePose = framePose * currBestPose;
            untracked_frames[i]->shell->camToWorld = framePose.inverse();
            untracked_frames[i]->shell->trackingRef = newFrameHessian->shell;
            untracked_frames[i]->shell->camToTrackingRef = untracked_frames[i]->shell->camToWorld;
        }

        thisToNext = framePose;
    }else
    {
        thisToNext = currBestPose;
    }

    return true;
}

void CoarseRGBDInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian, RGBDepth* rgbd_detph)
{
    std::cout << "Set First Frame." << std::endl;

    coarse_tracker->makeK(HCalib);

    if(newFrameHessian->shell->id == 0)
    {
        firstFrame = newFrameHessian;
    }
    cur_ref = newFrameHessian;

    const int xPadding = patternPadding + 2;
    const int yPadding = patternPadding + 2;

    for (int y = yPadding; y < height -yPadding; y++)
    {
        for (int x = xPadding; x < width -xPadding; x++)
        {
            int idx = x + y * width;

            if(!rgbd_detph->valid_mask[idx]) continue;
            float depth = 1.0f / rgbd_detph->depth[idx];

            ImmaturePoint* immaturePoint = new ImmaturePoint(x+0.5f, y+0.5f, newFrameHessian, 1, HCalib);

            if( !std::isfinite(immaturePoint->energyTH) )
            {
                delete immaturePoint;
                continue;
            }
            immaturePoint->idepth_max = depth;
            immaturePoint->idepth_min = depth;

            PointHessian* ph = new PointHessian(immaturePoint, HCalib);

            if(!std::isfinite(ph->energyTH)) {delete ph; delete immaturePoint; continue;}

            ph->idepth_hessian = 0.002;

            ph->setIdepth(depth);
            ph->setIdepthZero(ph->idepth);
            ph->hasDepthPrior=true;
            ph->setPointStatus(PointHessian::ACTIVE);

            newFrameHessian->pointHessians.push_back(ph);
            newFrameHessian->immaturePoints.push_back(immaturePoint);
        }
    }

    printf("Initialize RGBD Frame Done.\n");

    snapped = false;
    frameID = 0;
}
}