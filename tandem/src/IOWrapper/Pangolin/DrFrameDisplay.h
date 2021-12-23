#pragma once

#undef Success
#include <Eigen/Core>
#include "util/NumType.h"
#include <pangolin/pangolin.h>

#include <sstream>
#include <fstream>

namespace dso
{
class CalibHessian;
class FrameHessian;
class FrameShell;

namespace IOWrap
{

struct DrPoint
{
	float u;
	float v;
	float depth;
	float confidence;
	unsigned char color[3];
	unsigned char status;
};

struct DrVertex
{
	float point[3];
	unsigned char color[4];
};

// stores a pointcloud associated to a Keyframe.
class DrFrameDisplay
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	DrFrameDisplay(const unsigned char* bgr, const float* depth, float* confidence, const SE3& pose, CalibHessian* HCalib);
	~DrFrameDisplay();

	// copies & filters internal data to GL buffer for rendering. if nothing to do: does nothing.
	bool refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity);

	// renders cam & pointcloud.
//	void drawCam(float lineWidth = 1, float* color = 0, float sizeFactor=1);
	void drawPC(float pointSize);

	bool active;
	SE3 camToWorld;

private:
	float fx,fy,cx,cy;
	float fxi,fyi,cxi,cyi;
	int width, height;
    float my_confTH;

	int my_displayMode;
	bool needRefresh;


	int numSparsePoints;
	int numSparseBufferSize;
    DrPoint* originalInputSparse;


	bool bufferValid;
	int numGLBufferPoints;
	int numGLBufferGoodPoints;
	pangolin::GlBuffer vertexBuffer;
	pangolin::GlBuffer colorBuffer;
};

}
}

