#pragma once

#include "util/NumType.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include "util/settings.h"
#include "vector"
#include <math.h>
#include "util/ImageAndExposure.h"
#include "CoarseInitializer.h"
#include "ImmaturePoint.h"
#include "CoarseTracker.h"


namespace dso
{
struct CalibHessian;
struct FrameHessian;

class CoarseRGBDInitializer {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CoarseRGBDInitializer(int w, int h);
	~CoarseRGBDInitializer();


	void setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian, RGBDepth* rgbd_detph);
	bool trackFrame(FrameHessian* newFrameHessian, CalibHessian* HCalib, RGBDepth* rgbd_depth, std::vector<IOWrap::Output3DWrapper*> &wraps);

	int frameID;

	AffLight thisToNext_aff;
	SE3 thisToNext;


	FrameHessian* firstFrame;
	FrameHessian* newFrame;
	FrameHessian* cur_ref;

private:
	bool snapped;
    int width;
    int height;
    int tracked_frame;
    std::vector<FrameHessian*> untracked_frames;
    CoarseTracker* coarse_tracker;
};

}