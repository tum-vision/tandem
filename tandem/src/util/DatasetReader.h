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
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <string>

#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"


#if HAS_ZIPLIB
	#include "zip.h"
#endif

#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dvo/core/rgbd_image.h>
#include <dvo/core/surface_pyramid.h>

using namespace dso;

inline int filename_to_int(std::string fname){
    std::string int_str = fname.substr(0, fname.find_last_of("."));
    return std::stoi(int_str);
}

inline int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != "..")
    		files.push_back(name);
    }
    closedir(dp);

    if (use_int_sorting) {
        std::sort(files.begin(), files.end(),
                  [](std::string s1, std::string s2) { return filename_to_int(s1) < filename_to_int(s2); });
    }else{
        std::sort(files.begin(), files.end());
    }

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
	for(unsigned int i=0;i<files.size();i++)
	{
		if(files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

    return files.size();
}


struct PrepImageItem
{
	int id;
	bool isQueud;
	ImageAndExposure* pt;

	inline PrepImageItem(int _id)
	{
		id=_id;
		isQueud = false;
		pt=0;
	}

	inline void release()
	{
		if(pt!=0) delete pt;
		pt=0;
	}
};




class ImageFolderReader
{
public:
    ImageFolderReader() {}

  int getWidth() const {return width;};
  int getHeight() const {return height;};

	ImageFolderReader(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile)
	{
		this->path = path;
		this->calibfile = calibFile;

#if HAS_ZIPLIB
		ziparchive=0;
		databuffer=0;
#endif

		isZipped = (path.length()>4 && path.substr(path.length()-4) == ".zip");





		if(isZipped)
		{
#if HAS_ZIPLIB
			int ziperror=0;
			ziparchive = zip_open(path.c_str(),  ZIP_RDONLY, &ziperror);
			if(ziperror!=0)
			{
				printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
				exit(1);
			}

			files.clear();
			int numEntries = zip_get_num_entries(ziparchive, 0);
			for(int k=0;k<numEntries;k++)
			{
				const char* name = zip_get_name(ziparchive, k,  ZIP_FL_ENC_STRICT);
				std::string nstr = std::string(name);
				if(nstr == "." || nstr == "..") continue;
				files.push_back(name);
			}

			printf("got %d entries and %d files!\n", numEntries, (int)files.size());
			std::sort(files.begin(), files.end());
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
		else
			getdir (path, files);


		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);


		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width=undistort->getSize()[0];
		height=undistort->getSize()[1];


		// load timestamps if possible.
		loadTimestamps();
		printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());

	}
	~ImageFolderReader()
	{
#if HAS_ZIPLIB
		if(ziparchive!=0) zip_close(ziparchive);
		if(databuffer!=0) delete databuffer;
#endif


		delete undistort;
	};

	Eigen::VectorXf getOriginalCalib()
	{
		return undistort->getOriginalParameter().cast<float>();
	}
	Eigen::Vector2i getOriginalDimensions()
	{
		return  undistort->getOriginalSize();
	}

	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}

	void setGlobalCalibration()
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		getCalibMono(K, w_out, h_out);
		setGlobalCalib(w_out, h_out, K);
	}

	int getNumImages()
	{
		return files.size();
	}

	double getTimestamp(int id)
	{
//		if(timestamps.size()==0) return id*0.1f;
        // if(timestamps.size()==0) return id*0.0303030303f;
        if(timestamps.size()==0) return id*0.066f;
		if(id >= (int)timestamps.size()) return 0;
		if(id < 0) return 0;
		return timestamps[id];
	}


	void prepImage(int id, bool as8U=false)
	{

	}


	MinimalImageB* getImageRaw(int id)
	{
			return getImageRaw_internal(id,0);
	}

	ImageAndExposure* getImage(int id, bool forceLoadDirectly=false)
	{
		return getImage_internal(id, 0);
	}

    virtual RGBDepth* getRGBDetph(int id, float png_scale){
        return nullptr;
	};

    virtual dvo::core::RgbdImagePyramid* getDVORGBDetph(int id, float png_scale){
        return nullptr;
    };

    float* getImageRGB_32F(int id)
    {
        return getImageRGB_32F_internal(id);
    }

    unsigned char* getImageBGR_8UC3(int id)
    {
        return getImageBGR_8UC3_internal(id);
    }

    unsigned char* getImageBGR_8UC3_undis(int id, const float* remapX, const float* remapY)
    {
        return getImageBGR_8UC3_undistort_internal(id, remapX, remapY);
    }


	inline float* getPhotometricGamma()
	{
		if(undistort==0 || undistort->photometricUndist==0) return 0;
		return undistort->photometricUndist->getG();
	}


	// undistorter. [0] always exists, [1-2] only when MT is enabled.
	Undistort* undistort;
protected:


	MinimalImageB* getImageRaw_internal(int id, int unused)
	{
		if(!isZipped)
		{
			// CHANGE FOR ZIP FILE
			return IOWrap::readImageBW_8U(files[id]);
		}
		else
		{
#if HAS_ZIPLIB
			if(databuffer==0) databuffer = new char[widthOrg*heightOrg*6+10000];
			zip_file_t* fle = zip_fopen(ziparchive, files[id].c_str(), 0);
			long readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*6+10000);

			if(readbytes > (long)widthOrg*heightOrg*6)
			{
				printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,(long)widthOrg*heightOrg*6+10000, files[id].c_str());
				delete[] databuffer;
				databuffer = new char[(long)widthOrg*heightOrg*30];
				fle = zip_fopen(ziparchive, files[id].c_str(), 0);
				readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*30+10000);

				if(readbytes > (long)widthOrg*heightOrg*30)
				{
					printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,(long)widthOrg*heightOrg*30+10000);
					exit(1);
				}
			}

			return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
	}

    float* getImageRGB_32F_internal(int id)
    {
        if(!isZipped)
        {
            // CHANGE FOR ZIP FILE
            return IOWrap::readImageRGB_32F3(files[id], width, height);
        }
        else
        {
            printf("ERROR: cannot read .zip archive, as not implemented!\n");
            exit(1);
        }
    }

    unsigned char* getImageBGR_8UC3_internal(int id)
    {
        if(!isZipped)
        {
            // CHANGE FOR ZIP FILE
            // we get original image size here.
            return IOWrap::readImageBGR_8UC3(files[id], width, height);
        }
        else
        {
            printf("ERROR: cannot read .zip archive, as not implemented!\n");
            exit(1);
        }
    }

    unsigned char* getImageBGR_8UC3_undistort_internal(int id, const float* remapX, const float* remapY)
    {
        if(!isZipped)
        {
            // CHANGE FOR ZIP FILE
            // we get original image size here.
            auto in_data = IOWrap::readImageBGR_8UC3(files[id], widthOrg, heightOrg);
            unsigned char* out_data = (unsigned char*) malloc(sizeof(unsigned char) * width * height * 3);
            for(int idx = width*height-1;idx>=0;idx--)
            {
                // get interp. values
                float xx = remapX[idx];
                float yy = remapY[idx];

                if(xx<0)
                    out_data[3*idx] = out_data[3*idx+1] = out_data[3*idx+2] = 0;
                else
                {
                    // get integer and rational parts
                    int xxi = xx;
                    int yyi = yy;
                    xx -= xxi;
                    yy -= yyi;
                    float xxyy = xx*yy;

                    // get array base pointer
                    for(int bgr = 0; bgr < 3; ++bgr)
                    {
                        const unsigned char* src = in_data + (3 * xxi + bgr) + 3 * yyi * widthOrg;

                        // interpolate (bilinear)
                        out_data[3 * idx + bgr] = static_cast<unsigned char>(xxyy * src[3+3 * widthOrg]
                                         + (yy-xxyy) * src[3 * widthOrg]
                                         + (xx-xxyy) * src[3]
                                         + (1-xx-yy+xxyy) * src[0]);
                    }
                }
            }

            return out_data;
        }
        else
        {
            printf("ERROR: cannot read .zip archive, as not implemented!\n");
            exit(1);
        }
    }

	ImageAndExposure* getImage_internal(int id, int unused)
	{
		MinimalImageB* minimg = getImageRaw_internal(id, 0);
		ImageAndExposure* ret2 = undistort->undistort<unsigned char>(
				minimg,
				(exposures.size() == 0 ? 1.0f : exposures[id]),
//				(timestamps.size() == 0 ? 0.0 : timestamps[id]));
                (timestamps.size() == 0 ? (float)id : timestamps[id]));
		delete minimg;
		return ret2;
	}

	inline void loadTimestamps()
	{
		std::ifstream tr;
		std::string timesFile = path.substr(0,path.find_last_of('/')) + "/rgb.txt";
		tr.open(timesFile.c_str());
		while(!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int id;
			double stamp;
			float exposure = 0;
            char tmp_str[1000];

			if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if(1 == sscanf(buf, "%lf %*s", &stamp)){
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }

			else if(2 == sscanf(buf, "%d %lf", &id, &stamp))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}
		}
		tr.close();

		// check if exposures are correct, (possibly skip)
		bool exposuresGood = ((int)exposures.size()==(int)getNumImages()) ;
		for(int i=0;i<(int)exposures.size();i++)
		{
			if(exposures[i] == 0)
			{
				// fix!
				float sum=0,num=0;
				if(i>0 && exposures[i-1] > 0) {sum += exposures[i-1]; num++;}
				if(i+1<(int)exposures.size() && exposures[i+1] > 0) {sum += exposures[i+1]; num++;}

				if(num>0)
					exposures[i] = sum/num;
			}

			if(exposures[i] == 0) exposuresGood=false;
		}


		if((int)getNumImages() != (int)timestamps.size())
		{
			printf("set timestamps and exposures to zero!\n");
			exposures.clear();
			timestamps.clear();
		}

		if((int)getNumImages() != (int)exposures.size() || !exposuresGood)
		{
			printf("set EXPOSURES to zero!\n");
			exposures.clear();
		}

		printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(), (int)exposures.size());
	}




	std::vector<ImageAndExposure*> preloadedImages;
	std::vector<std::string> files;
	std::vector<double> timestamps;
	std::vector<float> exposures;

	int width, height;
	int widthOrg, heightOrg;

	std::string path;
	std::string calibfile;

	bool isZipped;

#if HAS_ZIPLIB
	zip_t* ziparchive;
	char* databuffer;
#endif
};

class RGBDReader : public ImageFolderReader
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RGBDReader(std::string img_path, std::string depth_path, std::string calib_file, std::string gamma_file,
               std::string vignette_file)
    {
        this->path = img_path;
        this->rgbd_depth_path = depth_path;
        this->calibfile = calibfile;
        getdir (path, files);
        getdir (depth_path, rgbd_depth_files);

        undistort = Undistort::getUndistorterForFile(calib_file, gamma_file, vignette_file);

        widthOrg = undistort->getOriginalSize()[0];
        heightOrg = undistort->getOriginalSize()[1];
        width=undistort->getSize()[0];
        height=undistort->getSize()[1];

        // load timestamps if possible.
        loadTimestamps();
        printf("RGBDReader: got %d rgb files in %s!\n", (int)files.size(), path.c_str());
        printf("RGBDReader: got %d depth files in %s!\n", (int)rgbd_depth_files.size(), rgbd_depth_path.c_str());
    }

    ~RGBDReader()
    {
        delete undistort;
    }

    RGBDepth* getDepth_internal(int id, float png_scale){
        // TODO: change to SSE
        cv::Mat depth_mat_ori = cv::imread(rgbd_depth_files[id], -1);
        float* tmp_conf = new float[depth_mat_ori.rows * depth_mat_ori.cols];
        for(int i = 0; i < depth_mat_ori.rows*depth_mat_ori.cols; ++i){
            unsigned short tmp = depth_mat_ori.at<unsigned short >(i);
            if(tmp > 0) tmp_conf[i] = 1.0;
            else tmp_conf[i] = 0.0;
        }
        cv::Mat depth_mat(height, width, depth_mat_ori.type());
        cv::Mat conf_mat_in = cv::Mat(depth_mat_ori.size(), CV_32FC1, tmp_conf);
        cv::Mat conf_mat = cv::Mat(depth_mat.size(), CV_32FC1);
        cv::resize(conf_mat_in, conf_mat, conf_mat.size(), cv::InterpolationFlags::INTER_LINEAR);
        // Resize image
        // TODO: use original DSO remapping?
        cv::resize(depth_mat_ori, depth_mat, depth_mat.size(), cv::InterpolationFlags::INTER_LINEAR);
        RGBDepth* ret = new RGBDepth(width, height);
        const unsigned short* dp = depth_mat.ptr<unsigned short>();
        for(int i = 0; i < width*height; ++i){
            if(dp[i] > 0){
                ret->depth[i] = static_cast<float>(dp[i]) / png_scale;
            }
            else{
                ret->depth[i] = 0.0;
            }
            if(fabs(conf_mat.at<float>(i) - 1) > 1e-6){
                ret->confidence[i] = 0.0;
                ret->valid_mask[i] = false;
            } else{
                ret->valid_mask[i] = true;
            }
        }
        delete [] tmp_conf;
        return ret;
    }

    dvo::core::RgbdImagePyramid* getDVODepth_internal(int id, float png_scale){
        cv::Mat depth_u16 =
                cv::imread(rgbd_depth_files[id], cv::IMREAD_ANYDEPTH);
        cv::Mat depth_u16_resized(height, width, depth_u16.type());
        cv::resize(depth_u16, depth_u16_resized, depth_u16_resized.size(), cv::InterpolationFlags::INTER_LINEAR);
        cv::Mat depth_float;
        dvo::core::SurfacePyramid::convertRawDepthImageSse(depth_u16_resized, depth_float,
                                                           1.0f / 5000.0f);

        // Load the gray image.
        cv::Mat rgb_image = cv::imread(files[id]);
        cv::Mat rgb_image_resized(height, width, rgb_image.type());
        cv::resize(rgb_image, rgb_image_resized, rgb_image_resized.size(), cv::InterpolationFlags::INTER_LINEAR);
        cv::Mat img, img_float;
        cv::cvtColor(rgb_image_resized, img, cv::COLOR_BGR2GRAY);
        img.convertTo(img_float, CV_32F);

        // Build Rgbd image pyramid for this frame.
        // TODO: Change intrinsics
//        float fx = 726.28741455078 / 739.0 * 640.0;
//        float fy = 726.28741455078 / 458.0 * 480.0;
//        float cx = 354.6496887207 / 739 * 640;
//        float cy = 186.46566772461 / 458.0 * 480.0;

        float fx = 726.21081542969 / 743.0 * 640.0;
        float fy = 726.21081542969 / 465.0 * 480.0;
        float cx = 359.2048034668 / 743 * 640;
        float cy = 202.47247314453 / 465.0 * 480.0;
        dvo::core::IntrinsicMatrix intrinsics = dvo::core::IntrinsicMatrix::create(fx, fy, cx, cy);
        dvo::core::RgbdCameraPyramid camera(640, 480, intrinsics);
        camera.build(3);
        dvo::core::RgbdImagePyramid* new_frame = new dvo::core::RgbdImagePyramid(camera, img_float, depth_float);
        rgb_image.convertTo(new_frame->level(0).rgb, CV_32FC3);
        return new_frame;
    }

    RGBDepth* getRGBDetph(int id, float png_scale)
    {
        return getDepth_internal(id, png_scale);
    }

    dvo::core::RgbdImagePyramid* getDVORGBDetph(int id, float png_scale)
    {
        return getDVODepth_internal(id, png_scale);
    }

private:
    std::string rgbd_depth_path;
    std::vector<std::string> rgbd_depth_files;
};

