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

#ifndef SURFACE_PYRAMID_H_
#define SURFACE_PYRAMID_H_

#include <mmintrin.h>
#include <emmintrin.h>

#include <opencv2/core/core.hpp>

namespace dvo
{
namespace core
{
// TODO: move to rgbd_image
class SurfacePyramid
{
public:

  /**
   * Converts the given raw depth image (type CV_16U) to a CV_32F image rescaling every pixel with the given scale
   * and replacing 0 with NaNs.
   */
  static void convertRawDepthImage(const cv::Mat& input, cv::Mat& output, float scale);

  static void convertRawDepthImageSse(const cv::Mat& input, cv::Mat& output, float scale);

  SurfacePyramid();
  virtual ~SurfacePyramid();
};

} /* namespace core */
} /* namespace dvo */

#endif /* SURFACE_PYRAMID_H_ */
