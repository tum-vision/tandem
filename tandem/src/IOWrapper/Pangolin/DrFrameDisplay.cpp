#include <stdio.h>
#include "util/settings.h"

#include <pangolin/pangolin.h>

#include <utility>
#include "DrFrameDisplay.h"
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"



namespace dso
{
namespace IOWrap
{


DrFrameDisplay::DrFrameDisplay(const unsigned char* bgr, const float* depth, float* confidence, const SE3& pose, CalibHessian* HCalib)
{
	originalInputSparse = 0;
	numSparseBufferSize=0;
	numSparsePoints=0;

	active= true;

    camToWorld = std::move(pose);

	my_displayMode = 1;

	numGLBufferPoints=0;
	bufferValid = false;

    fx = HCalib->fxl();
    fy = HCalib->fyl();
    cx = HCalib->cxl();
    cy = HCalib->cyl();
    width = wG[0];
    height = hG[0];
    fxi = 1/fx;
    fyi = 1/fy;
    cxi = -cx / fx;
    cyi = -cy / fy;

    // add all traces, inlier and outlier points.
    // TODO
    int npoints = width * height;

    if(numSparseBufferSize < npoints)
    {
        if(originalInputSparse != 0) delete originalInputSparse;
        numSparseBufferSize = npoints+ width * height;
        originalInputSparse = new DrPoint[numSparseBufferSize];
    }

    DrPoint* pc = originalInputSparse;
    numSparsePoints=0;
    for(int v = 0; v < height; ++v)
        for(int u = 0; u < width; ++u){
            // TODO: Assuming row major now
            int ind = width * v + u;
            if(depth[ind] < 1e-4) continue;
            // BGR -> RGB
            pc[numSparsePoints].color[0] = bgr[ind * 3 + 2] ;
            pc[numSparsePoints].color[1] = bgr[ind * 3 + 1] ;
            pc[numSparsePoints].color[2] = bgr[ind * 3 + 0] ;
            pc[numSparsePoints].u = u;
            pc[numSparsePoints].v = v;
            pc[numSparsePoints].depth = depth[ind];
            pc[numSparsePoints].confidence = confidence[ind];
            pc[numSparsePoints].status = 1;
            numSparsePoints++;
        }

    assert(numSparsePoints <= npoints);
    needRefresh=true;
}

DrFrameDisplay::~DrFrameDisplay()
{
	if(originalInputSparse != 0)
		delete[] originalInputSparse;
}

bool DrFrameDisplay::refreshPC(bool canRefresh, float confTH, float absTH, int mode, float minBS, int sparsity)
{
	if(canRefresh)
	{
		needRefresh = needRefresh || my_confTH != confTH ||
				my_displayMode != mode;
	}

	if(!needRefresh) return false;
	needRefresh=false;

	my_displayMode = mode;
    my_confTH = confTH;


	// if there are no vertices, done!
	if(numSparsePoints == 0)
		return false;

	// make data
	Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints];
	Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints];
	int vertexBufferNumPoints=0;

	for(int i=0;i<numSparsePoints;i++)
	{
		/* display modes:
		 * my_displayMode==0 - all pts, color-coded
		 * my_displayMode==1 - normal points
		 * my_displayMode==2 - active only
		 * my_displayMode==3 - nothing
		 */

		if(my_displayMode==1 && originalInputSparse[i].status != 1 && originalInputSparse[i].status!= 2) continue;
		if(my_displayMode==2 && originalInputSparse[i].status != 1) continue;
		if(my_displayMode>2) continue;

		if(originalInputSparse[i].depth < 0) continue;
        if(originalInputSparse[i].confidence < my_confTH) continue;

		float depth = originalInputSparse[i].depth;

        tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u)*fxi + cxi) * depth;
        tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v)*fyi + cyi) * depth;
        tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi);



        if(my_displayMode==0)
        {
            if(originalInputSparse[i].status==0)
            {
                tmpColorBuffer[vertexBufferNumPoints][0] = 0;
                tmpColorBuffer[vertexBufferNumPoints][1] = 255;
                tmpColorBuffer[vertexBufferNumPoints][2] = 255;
            }
            else if(originalInputSparse[i].status==1)
            {
                tmpColorBuffer[vertexBufferNumPoints][0] = 0;
                tmpColorBuffer[vertexBufferNumPoints][1] = 255;
                tmpColorBuffer[vertexBufferNumPoints][2] = 0;
            }
            else if(originalInputSparse[i].status==2)
            {
                tmpColorBuffer[vertexBufferNumPoints][0] = 0;
                tmpColorBuffer[vertexBufferNumPoints][1] = 0;
                tmpColorBuffer[vertexBufferNumPoints][2] = 255;
            }
            else if(originalInputSparse[i].status==3)
            {
                tmpColorBuffer[vertexBufferNumPoints][0] = 255;
                tmpColorBuffer[vertexBufferNumPoints][1] = 0;
                tmpColorBuffer[vertexBufferNumPoints][2] = 0;
            }
            else
            {
                tmpColorBuffer[vertexBufferNumPoints][0] = 255;
                tmpColorBuffer[vertexBufferNumPoints][1] = 255;
                tmpColorBuffer[vertexBufferNumPoints][2] = 255;
            }

        }
        //TODO
        else
        {
            tmpColorBuffer[vertexBufferNumPoints][0] = originalInputSparse[i].color[0];
            tmpColorBuffer[vertexBufferNumPoints][1] = originalInputSparse[i].color[1];
            tmpColorBuffer[vertexBufferNumPoints][2] = originalInputSparse[i].color[2];
        }
        vertexBufferNumPoints++;


        assert(vertexBufferNumPoints <= numSparsePoints*patternNum);
	}

	if(vertexBufferNumPoints==0)
	{
		delete[] tmpColorBuffer;
		delete[] tmpVertexBuffer;
		return true;
	}

	numGLBufferGoodPoints = vertexBufferNumPoints;
	if(numGLBufferGoodPoints > numGLBufferPoints)
	{
		numGLBufferPoints = vertexBufferNumPoints*1.3;
		vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
		colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
	}
	vertexBuffer.Upload(tmpVertexBuffer, sizeof(float)*3*numGLBufferGoodPoints, 0);
	colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char)*3*numGLBufferGoodPoints, 0);
	bufferValid=true;
	delete[] tmpColorBuffer;
	delete[] tmpVertexBuffer;


	return true;
}


// TODO
//void DrFrameDisplay::drawCam(float lineWidth, float* color, float sizeFactor)
//{
//	if(width == 0)
//		return;
//
//	float sz=sizeFactor;
//
//	glPushMatrix();
//
//		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
//		glMultMatrixf((GLfloat*)m.data());
//
//		if(color == 0)
//		{
//			glColor3f(1,0,0);
//		}
//		else
//			glColor3f(color[0],color[1],color[2]);
//
//		glLineWidth(lineWidth);
//		glBegin(GL_LINES);
//		glVertex3f(0,0,0);
//		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
//		glVertex3f(0,0,0);
//		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
//		glVertex3f(0,0,0);
//		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
//		glVertex3f(0,0,0);
//		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
//
//		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
//		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
//
//		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
//		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
//
//		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
//		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
//
//		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
//		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
//
//		glEnd();
//	glPopMatrix();
//}


void DrFrameDisplay::drawPC(float pointSize)
{

	if(!bufferValid || numGLBufferGoodPoints==0)
		return;


	glDisable(GL_LIGHTING);

	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
		glMultMatrixf((GLfloat*)m.data());

		glPointSize(pointSize);


		colorBuffer.Bind();
		glColorPointer(colorBuffer.count_per_element, colorBuffer.datatype, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);

		vertexBuffer.Bind();
		glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
		glDisableClientState(GL_VERTEX_ARRAY);
		vertexBuffer.Unbind();

		glDisableClientState(GL_COLOR_ARRAY);
		colorBuffer.Unbind();

	glPopMatrix();
}

}
}
