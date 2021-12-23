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

#include <dvo/core/datatypes.h>
#include <dvo/core/rgbd_image.h>
#include <dvo/core/interpolation.h>

#include <immintrin.h>
#include <pmmintrin.h>

//#include "../stopwatch.h"

#define ALIGN __attribute__((__aligned__(16)))


namespace dvo
{
namespace core
{

static inline void dump(const char* prefix, __m128 v)
{
  ALIGN float data[4];

  _mm_store_ps(data, v);

  std::cerr << prefix << " " << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << std::endl;
}

static const __m128 ZEROS = _mm_setzero_ps();
static const __m128 ONES = _mm_set1_ps(1.0f);
static const __m128 NANS = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());

static inline __m128 interpolateBilinearWithDepthBufferSse(const cv::Mat& intensity, const cv::Mat& depth, const __m128& xyxy_proj, const __m128& zzzz, const __m128& img_upper_bound)
{
  ALIGN int address[4];

  // TODO: this calculation can be done for both points at the same time!
  __m128i x0y0i = _mm_cvtps_epi32(xyxy_proj);
  __m128 x0y0   = _mm_cvtepi32_ps(x0y0i);      // [x0, y0, ?, ?]
  __m128 x1y1   = _mm_add_ps(x0y0, ONES);      // [x1, y1, ?, ?]
  __m128 x1wy1w = _mm_sub_ps(xyxy_proj, x0y0); // [x1_weight, y1_weight, ?, ?]
  __m128 x0wy0w = _mm_sub_ps(ONES, x1wy1w);    // [x0_weight, y0_weight, ?, ?]

  __m128 zeps = _mm_sub_ps(zzzz, _mm_set1_ps(0.05f));

  // check image bounds
  __m128 inimage = _mm_cmplt_ps(x1y1, img_upper_bound);

  int inimage_mask = _mm_movemask_ps(inimage);

  if((inimage_mask & 1) == 0 || (inimage_mask & 2) == 0)
  {
    return NANS;
  }

  x1wy1w = _mm_and_ps(inimage, x1wy1w); // [x1weight if x1 < width else 0, y1weight if y1 < width else 0, ?, ?]

  // weights
  __m128 tmp = _mm_shuffle_ps(x0wy0w, x1wy1w, _MM_SHUFFLE(1, 0, 1, 0));

  __m128 x0101w = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(2, 0, 2, 0)); // [x0_weight, x1_weight, x0_weight, x1_weight]
  __m128 y0011w = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(3, 3, 1, 1)); // [y0_weight, y0_weight, y1_weight, y1_weight]
  __m128 w = _mm_mul_ps(x0101w, y0011w); // [x0_weight * y0_weight, x1_weight * y0_weight, x0_weight * y1_weight, x1_weight * y1_weight]

  _mm_store_si128((__m128i*) address, x0y0i);

  const float* p_intensity1 = intensity.ptr<float>(address[1], address[0]);
  const float* p_intensity2 = intensity.ptr<float>(address[1] + 1, address[0]);
  const float* p_depth1 = depth.ptr<float>(address[1], address[0]);
  const float* p_depth2 = depth.ptr<float>(address[1] + 1, address[0]);

  __m128 p_intensity, p_depth;

  p_intensity = _mm_loadl_pi(NANS, (__m64*) p_intensity1);
  p_intensity = _mm_loadh_pi(p_intensity, (__m64*) p_intensity2);

  p_depth = _mm_loadl_pi(NANS, (__m64*) p_depth1);
  p_depth = _mm_loadh_pi(p_depth, (__m64*) p_depth2);

  //w = _mm_and_ps(_mm_and_ps(_mm_cmpord_ps(p_depth, p_depth), _mm_cmpgt_ps(p_depth, zeps)), w);
  p_intensity = _mm_mul_ps(w, p_intensity);

  __m128 iiww = _mm_hadd_ps(p_intensity, w);
  __m128 iw = _mm_hadd_ps(iiww, iiww);
  __m128 wi = _mm_shuffle_ps(iw, iw, _MM_SHUFFLE(1, 1, 1, 1));

  __m128 notzero = _mm_cmpgt_ps(wi, ZEROS);

  return _mm_or_ps(_mm_and_ps(notzero, _mm_div_ps(iw, wi)), _mm_andnot_ps(notzero, NANS));
}

template<int PointCloudOption>
void RgbdImage::warpIntensitySseImpl(const AffineTransform& transformationx, const PointCloud& reference_pointcloud, const IntrinsicMatrix& intrinsics, RgbdImage& result, PointCloud& transformed_pointcloud)
{
  // prepare transformation
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::RowMajor> transformation = transformationx.cast<float>();

  // prepare result images
  cv::Mat warped_image(intensity.size(), intensity.type());
  cv::Mat warped_depth(depth.size(), depth.type());

  float* warped_intensity_ptr = warped_image.ptr<IntensityType>();
  float* warped_depth_ptr = warped_depth.ptr<DepthType>();

  if(PointCloudOption == WithPointCloud)
  {
    transformed_pointcloud.resize(Eigen::NoChange, width * height);
  }

  unsigned int rnd_mode = _MM_GET_ROUNDING_MODE();

  if(rnd_mode != _MM_ROUND_TOWARD_ZERO) _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

  // transformation rows
  const __m128 t1 = _mm_load_ps(transformation.data());
  const __m128 t2 = _mm_load_ps(transformation.data() + 4);
  const __m128 t3 = _mm_load_ps(transformation.data() + 8);

  const __m128 fxyxy = _mm_setr_ps(intrinsics.fx(), intrinsics.fy(), intrinsics.fx(), intrinsics.fy()); // [fx, fy, fx, fy]
  const __m128 oxyxy = _mm_setr_ps(intrinsics.ox(), intrinsics.oy(), intrinsics.ox(), intrinsics.oy()); // [ox, oy, ox, oy]

  const __m128 img_lower_bound = _mm_setzero_ps();
  const __m128 img_upper_bound = _mm_set_ps(height, width, height, width);

  __m128 point1, point2, xy, uv, zw, xyuv, zwzw, z1w1, zzww, zzww_rcp, xyuv_proj, xyuv_inimage;

  const float* points = reference_pointcloud.data();
  float* tpoints = transformed_pointcloud.data();

  for(size_t y = 0; y < height; ++y)
  {
    for(size_t x = 0; x < width; x += 2, warped_intensity_ptr += 2, warped_depth_ptr += 2, points += 8, tpoints += 8)
    {
      point1 = _mm_load_ps(points + 0);
      point2 = _mm_load_ps(points + 4);

      // matrix multiply and first horizontal add
      xy = _mm_hadd_ps(_mm_mul_ps(t1, point1), _mm_mul_ps(t2, point1)); // [x0+1, x2+3, y0+1, y2+3]
      uv = _mm_hadd_ps(_mm_mul_ps(t1, point2), _mm_mul_ps(t2, point2)); // [u0+1, u2+3, v0+1, v2+3]
      zw = _mm_hadd_ps(_mm_mul_ps(t3, point1), _mm_mul_ps(t3, point2)); // [z0+1, z2+3, w0+1, w2+3]

      // second horizontal add
      xyuv = _mm_hadd_ps(xy, uv); // [x0+1+2+3, y0+1+2+3, u0+1+2+3, v0+1+2+3]
      zwzw = _mm_hadd_ps(zw, zw); // [z0+1+2+3, w0+1+2+3, z0+1+2+3, w0+1+2+3]

      if(PointCloudOption == WithPointCloud)
      {
        // interleave z and w with ones
        z1w1 = _mm_unpacklo_ps(zwzw, ONES); // [z0+1+2+3, 1, w0+1+2+3, 1]

        // reorder xyuv and z1w1 into two points
        point1 = _mm_movelh_ps(xyuv, z1w1); // [x0+1+2+3, y0+1+2+3, z0+1+2+3, 1]
        point2 = _mm_movehl_ps(z1w1, xyuv); // [u0+1+2+3, v0+1+2+3, w0+1+2+3, 1]

        // store transformed points
        _mm_stream_ps(tpoints + 0, point1);
        _mm_stream_ps(tpoints + 4, point2);
      }

      // reorder and invert z and w
      zzww = _mm_unpacklo_ps(zwzw, zwzw); // [z0+1+2+3, z0+1+2+3, w0+1+2+3, w0+1+2+3]
      zzww_rcp = _mm_rcp_ps(zzww); // 1 / zzww

      // projection
      xyuv_proj = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(xyuv, fxyxy), zzww_rcp), oxyxy);

      // check image bounds
      xyuv_inimage = _mm_and_ps(_mm_cmpge_ps(xyuv_proj, img_lower_bound), _mm_cmplt_ps(xyuv_proj, img_upper_bound));
      xyuv_inimage = _mm_and_ps(xyuv_inimage, _mm_shuffle_ps(xyuv_inimage, xyuv_inimage, _MM_SHUFFLE(2, 3, 0, 1))); // [x && y, y && x, u && v, v && u]
      xyuv_inimage = _mm_shuffle_ps(xyuv_inimage, xyuv_inimage, _MM_SHUFFLE(0, 0, 2, 0)); // [x && y, u && v, ?, ?]

      // store warped depth
      _mm_storel_pi((__m64*) warped_depth_ptr, _mm_or_ps(_mm_and_ps(xyuv_inimage, zwzw), _mm_andnot_ps(xyuv_inimage, NANS)));

      int xyuv_inimage_mask = _mm_movemask_ps(xyuv_inimage);

      // point1: only if bit 0 is set the point is in the image!
      if((xyuv_inimage_mask & 1) == 1)
      {
        __m128 zzzz = _mm_shuffle_ps(zzww, zzww, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 xyxy_proj = _mm_shuffle_ps(xyuv_proj, xyuv_proj, _MM_SHUFFLE(1, 0, 1, 0));

        _mm_store_ss(warped_intensity_ptr + 0, interpolateBilinearWithDepthBufferSse(intensity, depth, xyxy_proj, zzzz, img_upper_bound));
      }
      else
      {
        (*(warped_intensity_ptr + 0)) = Invalid;
      }

      // point2: only if bit 1 is set the point is in the image!
      if((xyuv_inimage_mask & 2) == 2)
      {
        __m128 wwww = _mm_shuffle_ps(zzww, zzww, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 uvuv_proj = _mm_shuffle_ps(xyuv_proj, xyuv_proj, _MM_SHUFFLE(3, 2, 3, 2));

        _mm_store_ss(warped_intensity_ptr + 1, interpolateBilinearWithDepthBufferSse(intensity, depth, uvuv_proj, wwww, img_upper_bound));
      }
      else
      {
        (*(warped_intensity_ptr + 1)) = Invalid;
      }
    }
  }

  if(rnd_mode != _MM_ROUND_TOWARD_ZERO) _MM_SET_ROUNDING_MODE(rnd_mode);

  result.intensity = warped_image;
  result.depth = warped_depth;
  result.initialize();
}

void RgbdImage::warpIntensitySse(const AffineTransform& transformation, const PointCloud& reference_pointcloud, const IntrinsicMatrix& intrinsics, RgbdImage& result, PointCloud& transformed_pointcloud)
{
  warpIntensitySseImpl<WithPointCloud>(transformation, reference_pointcloud, intrinsics, result, transformed_pointcloud);
}

void RgbdImage::warpIntensitySse(const AffineTransform& transformation, const PointCloud& reference_pointcloud, const IntrinsicMatrix& intrinsics, RgbdImage& result)
{
  PointCloud tmp;
  warpIntensitySseImpl<WithoutPointCloud>(transformation, reference_pointcloud, intrinsics, result, tmp);
}

void RgbdImage::calculateDerivativeYSseFloat(const cv::Mat& img, cv::Mat& result)
{
  result.create(img.size(), img.type());

  const float *prev_ptr, *next_ptr;
  float *result_ptr = result.ptr<float>();

  prev_ptr = img.ptr<float>(0); // point to row 0 (should point to -1)
  next_ptr = img.ptr<float>(1); // point to row 1

  const int inc = 4;

  __m128 scale = _mm_set1_ps(0.5f);

  // special loop for first row
  for(int x = 0; x < img.cols; x += inc, prev_ptr += inc, next_ptr += inc, result_ptr += inc)
  {
    _mm_store_ps(result_ptr, _mm_mul_ps(_mm_sub_ps(_mm_load_ps(next_ptr), _mm_load_ps(prev_ptr)), scale));
    //_mm_stream_ps(result_ptr + 4, _mm_sub_ps(_mm_load_ps(next_ptr + 4), _mm_load_ps(prev_ptr + 4)));
  }

  // prev_ptr points to row 1  (should point to 0)
  // next_ptr points to row 2

  prev_ptr -= img.cols; // go 1 row back

  for(int y = 1; y < img.rows - 1; y++)
  {
    for(int x = 0; x < img.cols; x += inc, prev_ptr += inc, next_ptr += inc, result_ptr += inc)
    {
      _mm_store_ps(result_ptr, _mm_mul_ps(_mm_sub_ps(_mm_load_ps(next_ptr), _mm_load_ps(prev_ptr)), scale));
      //_mm_stream_ps(result_ptr + 4, _mm_sub_ps(_mm_load_ps(next_ptr + 4), _mm_load_ps(prev_ptr + 4)));
    }
  }

  // special loop for last row
  next_ptr -= img.cols; // go 1 row back

  for(int x = 0; x < img.cols; x += inc, prev_ptr += inc, next_ptr += inc, result_ptr += inc)
  {
    _mm_store_ps(result_ptr, _mm_mul_ps(_mm_sub_ps(_mm_load_ps(next_ptr), _mm_load_ps(prev_ptr)), scale));
    //_mm_stream_ps(result_ptr + 4, _mm_sub_ps(_mm_load_ps(next_ptr + 4), _mm_load_ps(prev_ptr + 4)));
  }
}

} /* namespace core */
} /* namespace dvo */
