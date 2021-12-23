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


#pragma once
#include <pangolin/pangolin.h>
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"
#include <map>
#include <deque>


namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class KeyFrameDisplay;
class DrFrameDisplay;

struct GraphConnection
{
	KeyFrameDisplay* from;
	KeyFrameDisplay* to;
	int fwdMarg, bwdMarg, fwdAct, bwdAct;
};


class PangolinDSOViewer : public Output3DWrapper
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PangolinDSOViewer(int w, int h, bool startRunThread=true);
	virtual ~PangolinDSOViewer();

	void run();
	void close();

	void addImageToDisplay(std::string name, MinimalImageB3* image);
	void clearAllImagesToDisplay();


	// ==================== Output3DWrapper Functionality ======================
    virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override;
    virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override;
    virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override;
    virtual void publishDrframes(unsigned char* bgr, float* depth, float* confidence, SE3 camToWorld, CalibHessian* HCalib) override;

    virtual void pushLiveFrame(FrameHessian* image) override;
    virtual void pushDepthImage(MinimalImageB3* image) override;
    virtual bool needPushDepthImage() override;

    virtual void pushDrKfImage(unsigned char * bgr) override;
    virtual void pushDrKfDepth(float const* image, float depth_min, float depth_max) override;
    virtual void pushDrKfConfidence(float const* image) override;

    virtual void pushFusionKfImage(unsigned char const* image) override;
    virtual void pushFusionKfDepth(float const* image, float depth_min, float depth_max) override;

    virtual void pushDrMesh(size_t num, float const* vert, float const* cols) override;

    virtual void join() override;

    virtual void reset() override;

    virtual void save_mesh() override;
    virtual bool should_save_mesh() override;

private:

	bool needReset;
	bool shouldSaveMesh=false;
	void reset_internal();
	void drawConstraints();

	boost::thread runThread;
	bool running;
	int w,h;



	// images rendering
	boost::mutex openImagesMutex;
	MinimalImageB3* internalVideoImg;
	MinimalImageB3* internalKFImg;
	MinimalImageB3* internalResImg;
	bool videoImgChanged, kfImgChanged, resImgChanged;

    MinimalImageB3* internalDrKfImage;
    MinimalImageB3* internalDrKfDepth;
    MinimalImageB3* internalDrKfConfidence;
    MinimalImageB3* internalFusionKfImage;
    MinimalImageB3* internalFusionKfDepth;
    bool drKfImgChanged, drKfDepthChanged, drKfConfidenceChanged;
    bool drFusionImgChanged, drFusionDepthChanged;


	// 3D model rendering
	boost::mutex model3DMutex;
	KeyFrameDisplay* currentCam;
	std::vector<KeyFrameDisplay*> keyframes;
    std::vector<DrFrameDisplay*> drframes;
	std::vector<Vec3f,Eigen::aligned_allocator<Vec3f>> allFramePoses;
	std::map<int, KeyFrameDisplay*> keyframesByKFID;
	std::vector<GraphConnection,Eigen::aligned_allocator<GraphConnection>> connections;


    // Mesh Stuff
    boost::mutex drMeshMutex;
    size_t dr_mesh_num = 0;
    const size_t dr_mesh_num_max = 60000000;
    float* dr_mesh_vert;
    float* dr_mesh_cols;
    bool dr_mesh_changed = false;


	// render settings
	bool settings_showKFCameras;
	bool settings_showCurrentCamera;
	bool settings_showTrajectory;
	bool settings_showFullTrajectory;
	bool settings_showActiveConstraints;
	bool settings_showAllConstraints;

	float settings_scaledVarTH;
	float settings_drConfTH;
	float settings_absVarTH;
	int settings_pointCloudMode;
	float settings_minRelBS;
	int settings_sparsity;


	// timings
	struct timeval last_track;
	struct timeval last_map;


	std::deque<float> lastNTrackingMs;
	std::deque<float> lastNMappingMs;
};



}



}
