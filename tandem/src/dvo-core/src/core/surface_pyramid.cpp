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

#include <dvo/core/surface_pyramid.h>

#include <iostream>

namespace dvo
{
namespace core
{

SurfacePyramid::SurfacePyramid()
{
  // TODO Auto-generated constructor stub

}

SurfacePyramid::~SurfacePyramid()
{
  // TODO Auto-generated destructor stub
}

/**
 * Converts the given raw depth image (type CV_16UC1) to a CV_32FC1 image rescaling every pixel with the given scale
 * and replacing 0 with NaNs.
 */
void SurfacePyramid::convertRawDepthImage(const cv::Mat& input, cv::Mat& output, float scale)
{
  output.create(input.rows, input.cols, CV_32FC1);

  const unsigned short* input_ptr = input.ptr<unsigned short>();
  float* output_ptr = output.ptr<float>();

  for(int idx = 0; idx < input.size().area(); idx++, input_ptr++, output_ptr++)
  {
    if(*input_ptr == 0)
    {
      *output_ptr = std::numeric_limits<float>::quiet_NaN();
    }
    else
    {
      *output_ptr = ((float) *input_ptr) * scale;
    }
  }
}

void SurfacePyramid::convertRawDepthImageSse(const cv::Mat& input, cv::Mat& output, float scale)
{
  output.create(input.rows, input.cols, CV_32FC1);

  const unsigned short* input_ptr = input.ptr<unsigned short>();
  float* output_ptr = output.ptr<float>();

  __m128 _scale = _mm_set1_ps(scale);
  __m128 _zero  = _mm_setzero_ps();
  __m128 _nan   = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());

  for(int idx = 0; idx < input.size().area(); idx += 8, input_ptr += 8, output_ptr += 8)
  {
    __m128 _input, mask;
    __m128i _inputi = _mm_load_si128((__m128i*) input_ptr);

    // load low shorts and convert to float
    _input = _mm_cvtepi32_ps(_mm_unpacklo_epi16(_inputi, _mm_setzero_si128()));

    mask = _mm_cmpeq_ps(_input, _zero);

    // zero to nan
    _input = _mm_or_ps(_input, _mm_and_ps(mask, _nan));
    // scale
    _input = _mm_mul_ps(_input, _scale);
    // save
    _mm_store_ps(output_ptr + 0, _input);

    // load high shorts and convert to float
    _input = _mm_cvtepi32_ps(_mm_unpackhi_epi16(_inputi, _mm_setzero_si128()));

    mask = _mm_cmpeq_ps(_input, _zero);

    // zero to nan
    _input = _mm_or_ps(_input, _mm_and_ps(mask, _nan));
    // scale
    _input = _mm_mul_ps(_input, _scale);
    // save
    _mm_store_ps(output_ptr + 4, _input);
  }
}

} /* namespace core */
} /* namespace dvo */
