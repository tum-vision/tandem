/**
 *  This file is part of dvo.
 *
 *  Copyright 2012 Christian Kerl <christian.kerl@in.tum.de> (Technical University of Munich)
 *  For more information see <http://vision.in.tum.de/data/software/dvo>.
 *
 *  dvo is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  dvo is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with dvo.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef INTERPOLATION_H_
#define INTERPOLATION_H_

#include <opencv2/opencv.hpp>

#include <dvo/core/datatypes.h>

namespace dvo
{
namespace core
{

struct Interpolation
{
  static IntensityType none(const cv::Mat& img, float x, float y);
  static IntensityType bilinear(const cv::Mat& img, float x, float y);
  static IntensityType bilinearWithDepthBuffer(const cv::Mat& intensity, const cv::Mat& depth, const float& x, const float& y, const float& z);
};

} /* namespace core */
} /* namespace dvo */
#endif /* INTERPOLATION_H_ */
