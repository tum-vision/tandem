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



#include "IOWrapper/ImageRW.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace dso
{

namespace IOWrap
{
MinimalImageB* readImageBW_8U(std::string filename)
{
	cv::Mat m = cv::imread(filename, cv::ImreadModes::IMREAD_GRAYSCALE);
	if(m.rows*m.cols==0)
	{
		printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
		return 0;
	}
	if(m.type() != CV_8U)
	{
		printf("cv::imread did something strange! this may segfault. \n");
		return 0;
	}
	MinimalImageB* img = new MinimalImageB(m.cols, m.rows);
	memcpy(img->data, m.data, m.rows*m.cols);
	return img;
}

MinimalImageB3* readImageRGB_8U(std::string filename)
{
	cv::Mat m = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
	if(m.rows*m.cols==0)
	{
		printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
		return 0;
	}
	if(m.type() != CV_8UC3)
	{
		printf("cv::imread did something strange! this may segfault. \n");
		return 0;
	}
	MinimalImageB3* img = new MinimalImageB3(m.cols, m.rows);
	memcpy(img->data, m.data, 3*m.rows*m.cols);
	return img;
}

float *readImageRGB_32F3(std::string filename, int w_out, int h_out) {
    // TODO: Resizing here is a hack but the undistorter used in  DatasetReader doesn't support multiple channels.
    cv::Mat m_out(h_out, w_out, CV_8UC3);
    {
        // m only needed inside this scope
        cv::Mat m = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
        if (m.rows * m.cols == 0) {
            printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
            return 0;
        }
        if (m.type() != CV_8UC3) {
            printf("cv::imread did something strange! this may segfault. \n");
            return 0;
        }

        if (m.rows % h_out != 0) {
            printf("DRMVSNET: ERROR: resizing by non-integer factor: w_old = %d, w_new = %d\n", m.rows, h_out);
            exit(EXIT_FAILURE);
        }
        if (m.cols % w_out != 0) {
            printf("DRMVSNET: ERROR: resizing by non-integer factor: h_old = %d, h_new = %d\n", m.cols, w_out);
            exit(EXIT_FAILURE);
        }
        if (m.rows / h_out != m.cols / w_out) {
            printf("DRMVSNET: ERROR: resizing and changing aspect ration.\n");
            exit(EXIT_FAILURE);
        }
        cv::resize(m, m_out, m_out.size(), cv::InterpolationFlags::INTER_NEAREST);
    }
    cv::cvtColor(m_out, m_out, cv::COLOR_BGR2RGB);
    m_out.convertTo(m_out, CV_32F, 1.0/255.0);
    float* data = (float*) malloc(sizeof(float) * m_out.cols * m_out.rows * 3);
    memcpy(data, m_out.data, sizeof(float) * 3 * m_out.rows * m_out.cols);
        // Debug

//        cv::Mat m2(h_out, w_out, CV_8UC3);
//
//        for (int _h = 0; _h < h_out; _h++) {
//            for (int _w = 0; _w < w_out; _w++) {
//                const int i = _h * w_out + _w;
//                // RGB to BGR
//                m2.data[3 * i + 0] = (unsigned char) (255.0 * data[3 * i + 2]);
//                m2.data[3 * i + 1] = (unsigned char) (255.0 * data[3 * i + 1]);
//                m2.data[3 * i + 2] = (unsigned char) (255.0 * data[3 * i + 0]);
//            }
//        }
//
//        printf("DRMVSNET test2.png: h_out = %d, w_out = %d\n", h_out, w_out);
//        cv::imwrite("test2.png", m2);
    return data;
}


unsigned char *readImageBGR_8UC3(std::string filename, int w_out, int h_out) {
    // TODO: Resizing here is a hack but the undistorter used in  DatasetReader doesn't support multiple channels.
    unsigned char* data_out = (unsigned char*) malloc(sizeof(unsigned char) * w_out * h_out * 3);
    cv::Mat m_out(h_out, w_out, CV_8UC3, data_out);
    {
        // m only needed inside this scope
        cv::Mat m = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
        if (m.rows * m.cols == 0) {
            printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
            return 0;
        }
        if (m.type() != CV_8UC3) {
            printf("cv::imread did something strange! this may segfault. \n");
            return 0;
        }

        if (m.rows % h_out != 0) {
            printf("DRMVSNET: ERROR: resizing by non-integer factor: w_old = %d, w_new = %d\n", m.rows, h_out);
            exit(EXIT_FAILURE);
        }
        if (m.cols % w_out != 0) {
            printf("DRMVSNET: ERROR: resizing by non-integer factor: h_old = %d, h_new = %d\n", m.cols, w_out);
            exit(EXIT_FAILURE);
        }
        if (m.rows / h_out != m.cols / w_out) {
            printf("DRMVSNET: ERROR: resizing and changing aspect ration.\n");
            exit(EXIT_FAILURE);
        }
        cv::resize(m, m_out, m_out.size(), cv::InterpolationFlags::INTER_NEAREST);
    }

    return data_out;
}

MinimalImage<unsigned short>* readImageBW_16U(std::string filename)
{
	cv::Mat m = cv::imread(filename, cv::ImreadModes::IMREAD_UNCHANGED);
	if(m.rows*m.cols==0)
	{
		printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
		return 0;
	}
	if(m.type() != CV_16U)
	{
		printf("readImageBW_16U called on image that is not a 16bit grayscale image. this may segfault. \n");
		return 0;
	}
	MinimalImage<unsigned short>* img = new MinimalImage<unsigned short>(m.cols, m.rows);
	memcpy(img->data, m.data, 2*m.rows*m.cols);
	return img;
}

MinimalImageB* readStreamBW_8U(char* data, int numBytes)
{
	cv::Mat m = cv::imdecode(cv::Mat(numBytes,1,CV_8U, data), cv::ImreadModes::IMREAD_GRAYSCALE);
	if(m.rows*m.cols==0)
	{
		printf("cv::imdecode could not read stream (%d bytes)! this may segfault. \n", numBytes);
		return 0;
	}
	if(m.type() != CV_8U)
	{
		printf("cv::imdecode did something strange! this may segfault. \n");
		return 0;
	}
	MinimalImageB* img = new MinimalImageB(m.cols, m.rows);
	memcpy(img->data, m.data, m.rows*m.cols);
	return img;
}



void writeImage(std::string filename, MinimalImageB* img)
{
	cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8U, img->data));
}
void writeImage(std::string filename, MinimalImageB3* img)
{
	cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8UC3, img->data));
}
void writeImage(std::string filename, MinimalImageF* img)
{
	cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32F, img->data));
}
void writeImage(std::string filename, MinimalImageF3* img)
{
	cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32FC3, img->data));
}

}

}
