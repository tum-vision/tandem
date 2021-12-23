/**
 *  This file is part of dvo.
 *
 *  Copyright 2013 Christian Kerl <christian.kerl@in.tum.de> (Technical University of Munich)
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


#include <dvo/dense_tracking_impl.h>

#ifdef __CDT_PARSER__
  #define __SSE3__
#endif

#include <immintrin.h>
#include <pmmintrin.h>

namespace dvo
{
namespace core
{

void computeResiduals(const PointIterator& first_point, const PointIterator& last_point, const RgbdImage& current, const IntrinsicMatrix& intrinsics, const Eigen::Affine3f transform, const Vector8f& reference_weight, const Vector8f& current_weight, ComputeResidualsResult& result)
{
  result.last_point_error = result.first_point_error;
  result.last_residual = result.first_residual;

  Eigen::Matrix<float, 3, 3> K;
  K <<
      intrinsics.fx(), 0, intrinsics.ox(),
      0, intrinsics.fy(), intrinsics.oy(),
      0, 0, 1;

  //Sophus::Vector6d xi;
  //xi = Sophus::SE3(transform.rotation().cast<double>(), transform.translation().cast<double>()).log();


  Eigen::Matrix<float, 3, 4> KT = K * transform.matrix().block<3, 4>(0, 0);
  Eigen::Vector4f transformed_point;
  transformed_point.setConstant(1);
  //transformed_point1.setConstant(1);

  float t = std::numeric_limits<float>::quiet_NaN();

  for(PointIterator p_it = first_point; p_it != last_point; ++p_it)
  {
    // TODO: doesn't work yet
    //if(t != p_it->intensity_and_depth.time_interpolation)
    //{
    //  t = p_it->intensity_and_depth.time_interpolation;
    //  KT = K * Sophus::SE3::exp(xi + xi * t).matrix().block<3, 4>(0, 0).cast<float>();
    //}

    //transformed_point = transform * p_it->getPointVec4f();

    //float projected_x = transformed_point(0) * intrinsics.fx() / transformed_point(2) + intrinsics.ox();
    //float projected_y = transformed_point(1) * intrinsics.fy() / transformed_point(2) + intrinsics.oy();

    transformed_point.head<3>() = KT * p_it->getPointVec4f();

    float projected_x = transformed_point(0) / transformed_point(2);
    float projected_y = transformed_point(1) / transformed_point(2);

    if(!current.inImage(projected_x, projected_y) || !current.inImage(projected_x + 1, projected_y + 1)) continue;

    float x0 = std::floor(projected_x);
    float y0 = std::floor(projected_y);

    float x0w, x1w, y0w, y1w;
    x1w = projected_x - x0;
    x0w = 1.0f - x1w;
    y1w = projected_y - y0;
    y0w = 1.0f - y1w;

    const float *x0y0_ptr = current.acceleration.ptr<float>(int(y0), int(x0));
    const float *x0y1_ptr = current.acceleration.ptr<float>(int(y0 + 1), int(x0));

    Vector8f::ConstAlignedMapType x0y0(x0y0_ptr);
    Vector8f::ConstAlignedMapType x1y0(x0y0_ptr + 8);
    Vector8f::ConstAlignedMapType x0y1(x0y1_ptr);
    Vector8f::ConstAlignedMapType x1y1(x0y1_ptr + 8);
    /*
    Vector8f interpolated =
        x0y0 * x0w + x1y0 * x1w +
        x0y1 * x0w + x1y1 * x1w +
        x0y0 * y0w + x0y1 * y1w +
        x1y0 * y0w + x1y1 * y1w;
    interpolated *= 0.25f;
    */
    Vector8f interpolated = (x0y0 * x0w + x1y0 * x1w) * y0w + (x0y1 * x0w + x1y1 * x1w) * y1w;

    if(!std::isfinite(interpolated(1)) || !std::isfinite(interpolated(4)) || !std::isfinite(interpolated(5))) continue;

    // funny part: puzzling together depends on fwd. comp. / inverse comp. / esm / additiv / ...
    // fwd. comp. for now
    result.last_point_error->getPointVec4f() = p_it->getPointVec4f(); // transformed_point;

    Vector8f reference = p_it->getIntensityAndDepthWithDerivativesVec8f();
    reference(1) = transformed_point(2);

    result.last_point_error->getIntensityAndDepthWithDerivativesVec8f() = current_weight.cwiseProduct(interpolated) + reference_weight.cwiseProduct(reference);
    *result.last_residual = result.last_point_error->getIntensityAndDepthVec2f();

    ++result.last_point_error;
    ++result.last_residual;
  }
}

static inline float depthStdDevZ(float depth)
{
  float sigma_z = depth - 0.4f;
  sigma_z = 0.0012f + 0.0019f * sigma_z * sigma_z;

  return sigma_z;
}

static const __m128 ONES = _mm_set1_ps(1.0f);
static const __m128 BLEND_MASK = _mm_cmpgt_ps(_mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f), _mm_set1_ps(0.5f));
static const __m128 PHOTO_MASK = _mm_cmpgt_ps(_mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f), _mm_set1_ps(0.5f));
template<bool Debug>
void computeResidualsSse(const PointIterator& first_point, const PointIterator& last_point, const RgbdImage& current, const IntrinsicMatrix& intrinsics, const Eigen::Affine3f transform, const Vector8f& reference_weight, const Vector8f& current_weight, ComputeResidualsResult& result, const float affine_a, const float affine_b)
{
    static const __m128 AFFINE_AS = _mm_set1_ps(affine_a);
    static const __m128 AFFINE_BS = _mm_set1_ps(affine_b);
  result.last_point_error = result.first_point_error;
  result.last_residual = result.first_residual;

  if(Debug)
    result.last_valid_flag = result.first_valid_flag;

  Eigen::Matrix<float, 3, 3> K;
  K <<
      intrinsics.fx(), 0, intrinsics.ox(),
      0, intrinsics.fy(), intrinsics.oy(),
      0, 0, 1;

  Eigen::Matrix<float, 3, 4> KT = K * transform.matrix().block<3, 4>(0, 0);

  __m128 kt_r1 = _mm_setr_ps(KT(0, 0), KT(0, 1), KT(0, 2), KT(0, 3));
  __m128 kt_r2 = _mm_setr_ps(KT(1, 0), KT(1, 1), KT(1, 2), KT(1, 3));
  __m128 kt_r3 = _mm_setr_ps(KT(2, 0), KT(2, 1), KT(2, 2), KT(2, 3));

  __m128 current_weight_a = _mm_load_ps(current_weight.data());
  __m128 current_weight_b = _mm_load_ps(current_weight.data() + 4);

  __m128 reference_weight_a = _mm_load_ps(reference_weight.data());
  __m128 reference_weight_b = _mm_load_ps(reference_weight.data() + 4);

  __m128 lower_bound = _mm_set1_ps(0.0f);
  __m128 upper_bound = _mm_setr_ps(current.width - 2, current.height - 2, current.width - 2, current.height - 2);

  EIGEN_ALIGN16 int address[4];

  unsigned int rnd_mode = _MM_GET_ROUNDING_MODE();

  if(rnd_mode != _MM_ROUND_TOWARD_ZERO) _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);

  const PointIterator lp = ((last_point - first_point) % 2) != 0 ? last_point - 1 : last_point;

  for(PointIterator p_it = first_point; p_it != lp; p_it += 2)
  {
    // load points
    __m128 p1 = _mm_load_ps((p_it + 0)->point.data);
    __m128 p2 = _mm_load_ps((p_it + 1)->point.data);

    // transform
    __m128 pt1_x = _mm_mul_ps(kt_r1, p1);
    __m128 pt2_x = _mm_mul_ps(kt_r1, p2);

    __m128 pt1_y = _mm_mul_ps(kt_r2, p1);
    __m128 pt2_y = _mm_mul_ps(kt_r2, p2);

    __m128 pt1_z = _mm_mul_ps(kt_r3, p1);
    __m128 pt2_z = _mm_mul_ps(kt_r3, p2);

    __m128 pt1_xy_pt2_xy = _mm_hadd_ps(_mm_hadd_ps(pt1_x, pt1_y), _mm_hadd_ps(pt2_x, pt2_y));
    __m128 pt1_zz_pt2_zz = _mm_hadd_ps(_mm_hadd_ps(pt1_z, pt1_z), _mm_hadd_ps(pt2_z, pt2_z));

    // project
    //__m128 pt1_uv_pt2_uv = _mm_div_ps(pt1_xy_pt2_xy, pt1_zz_pt2_zz);
    __m128 pt1_uv_pt2_uv = _mm_mul_ps(pt1_xy_pt2_xy, _mm_rcp_ps(pt1_zz_pt2_zz));

    // floor
    __m128i pt1_uv_pt2_uv_int = _mm_cvtps_epi32(pt1_uv_pt2_uv);
    __m128 pt1_u0v0_pt2_u0v0 = _mm_cvtepi32_ps(pt1_uv_pt2_uv_int);

    // compute weights
    __m128 pt1w1_uv_pt2w1_uv = _mm_sub_ps(pt1_uv_pt2_uv, pt1_u0v0_pt2_u0v0);
    __m128 pt1w0_uv_pt2w0_uv = _mm_sub_ps(ONES, pt1w1_uv_pt2w1_uv);

    // check image bounds
    int bounds_mask = _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(pt1_uv_pt2_uv, lower_bound), _mm_cmple_ps(pt1_uv_pt2_uv, upper_bound)));

    _mm_store_si128((__m128i*) address, pt1_uv_pt2_uv_int);

    if(Debug)
      *result.last_valid_flag = uint8_t((bounds_mask & 3) == 3);

    if((bounds_mask & 3) == 3)
    {
      const float *x0y0_ptr = current.acceleration.ptr<float>(address[1], address[0]);
      const float *x0y1_ptr = current.acceleration.ptr<float>(address[1] + 1, address[0]);

      _mm_prefetch(x0y1_ptr, _MM_HINT_NTA);

      // shuffle weights
      __m128 w0_uuvv = _mm_unpacklo_ps(pt1w0_uv_pt2w0_uv, pt1w0_uv_pt2w0_uv);
      __m128 w0_uuuu = _mm_unpacklo_ps(w0_uuvv, w0_uuvv);
      __m128 w0_vvvv = _mm_unpackhi_ps(w0_uuvv, w0_uuvv);

      __m128 w1_uuvv = _mm_unpacklo_ps(pt1w1_uv_pt2w1_uv, pt1w1_uv_pt2w1_uv);
      __m128 w1_uuuu = _mm_unpacklo_ps(w1_uuvv, w1_uuvv);
      __m128 w1_vvvv = _mm_unpackhi_ps(w1_uuvv, w1_uuvv);

      // interpolate
      __m128 a1 = _mm_mul_ps(w0_vvvv,
          _mm_add_ps(
              _mm_mul_ps(w0_uuuu, _mm_load_ps(x0y0_ptr + 0)),
              _mm_mul_ps(w1_uuuu, _mm_load_ps(x0y0_ptr + 8))
          )
      );

      __m128 b1 = _mm_mul_ps(w0_vvvv,
          _mm_add_ps(
              _mm_mul_ps(w0_uuuu, _mm_load_ps(x0y0_ptr + 4)),
              _mm_mul_ps(w1_uuuu, _mm_load_ps(x0y0_ptr + 12))
          )
      );

      __m128 a2 = _mm_mul_ps(w1_vvvv,
          _mm_add_ps(
              _mm_mul_ps(w0_uuuu, _mm_load_ps(x0y1_ptr + 0)),
              _mm_mul_ps(w1_uuuu, _mm_load_ps(x0y1_ptr + 8))
          )
      );

      __m128 b2 = _mm_mul_ps(w1_vvvv,
          _mm_add_ps(
              _mm_mul_ps(w0_uuuu, _mm_load_ps(x0y1_ptr + 4)),
              _mm_mul_ps(w1_uuuu, _mm_load_ps(x0y1_ptr + 12))
          )
      );

      // first 4 values in interpolated Vec8f
      __m128 a = _mm_add_ps(a1, a2);
      // last 4 values in interpolated Vec8f
      __m128 b = _mm_add_ps(b1, b2);

      // check for NaNs in interpolated Vec8f
      int nans_mask = _mm_movemask_ps(_mm_cmpunord_ps(a, b));

      if(nans_mask == 0)
      {
        _mm_store_ps(result.last_point_error->point.data, p1);

        __m128 reference_a = _mm_load_ps((p_it + 0)->intensity_and_depth.data);
        // replace the reference image depth value with the transformed one
        reference_a = _mm_or_ps(_mm_and_ps(BLEND_MASK, pt1_zz_pt2_zz), _mm_andnot_ps(BLEND_MASK, reference_a));
        __m128 aff_reference_a = _mm_add_ps(_mm_mul_ps(reference_a, AFFINE_AS), AFFINE_BS);
        reference_a = _mm_or_ps(_mm_and_ps(PHOTO_MASK, aff_reference_a), _mm_andnot_ps(PHOTO_MASK, reference_a));
        __m128 residual_a = _mm_add_ps(_mm_mul_ps(current_weight_a, a), _mm_mul_ps(reference_weight_a, reference_a));
        _mm_store_ps(result.last_point_error->intensity_and_depth.data + 0, residual_a);

        // occlusion test
        if(result.last_point_error->intensity_and_depth.z > -20.0f * depthStdDevZ((p_it + 0)->intensity_and_depth.z))
        {
          _mm_storel_pi((__m64 *)result.last_residual->data(), residual_a);

          __m128 reference_b = _mm_load_ps((p_it + 0)->intensity_and_depth.data + 4);
          __m128 residual_b = _mm_add_ps(_mm_mul_ps(current_weight_b, b), _mm_mul_ps(reference_weight_b, reference_b));
          _mm_store_ps(result.last_point_error->intensity_and_depth.data + 4, residual_b);

          ++result.last_point_error;
          ++result.last_residual;
        }
        else
        {
          nans_mask = 1;
        }
      }

      if(Debug)
        *result.last_valid_flag = uint8_t(nans_mask == 0);
    }

    if(Debug)
      ++result.last_valid_flag;

    if(Debug)
      *result.last_valid_flag = uint8_t((bounds_mask & 12) == 12);

    if((bounds_mask & 12) == 12)
    {
      const float *x0y0_ptr = current.acceleration.ptr<float>(address[3], address[2]);
      const float *x0y1_ptr = current.acceleration.ptr<float>(address[3] + 1, address[2]);

      _mm_prefetch(x0y1_ptr, _MM_HINT_NTA);

      // shuffle weights
      __m128 w0_uuvv = _mm_unpackhi_ps(pt1w0_uv_pt2w0_uv, pt1w0_uv_pt2w0_uv);
      __m128 w0_uuuu = _mm_unpacklo_ps(w0_uuvv, w0_uuvv);
      __m128 w0_vvvv = _mm_unpackhi_ps(w0_uuvv, w0_uuvv);

      __m128 w1_uuvv = _mm_unpackhi_ps(pt1w1_uv_pt2w1_uv, pt1w1_uv_pt2w1_uv);
      __m128 w1_uuuu = _mm_unpacklo_ps(w1_uuvv, w1_uuvv);
      __m128 w1_vvvv = _mm_unpackhi_ps(w1_uuvv, w1_uuvv);

      // interpolate
      __m128 a1 = _mm_mul_ps(w0_vvvv,
          _mm_add_ps(
              _mm_mul_ps(w0_uuuu, _mm_load_ps(x0y0_ptr + 0)),
              _mm_mul_ps(w1_uuuu, _mm_load_ps(x0y0_ptr + 8))
          )
      );

      __m128 b1 = _mm_mul_ps(w0_vvvv,
          _mm_add_ps(
              _mm_mul_ps(w0_uuuu, _mm_load_ps(x0y0_ptr + 4)),
              _mm_mul_ps(w1_uuuu, _mm_load_ps(x0y0_ptr + 12))
          )
      );

      __m128 a2 = _mm_mul_ps(w1_vvvv,
          _mm_add_ps(
              _mm_mul_ps(w0_uuuu, _mm_load_ps(x0y1_ptr + 0)),
              _mm_mul_ps(w1_uuuu, _mm_load_ps(x0y1_ptr + 8))
          )
      );

      __m128 b2 = _mm_mul_ps(w1_vvvv,
          _mm_add_ps(
              _mm_mul_ps(w0_uuuu, _mm_load_ps(x0y1_ptr + 4)),
              _mm_mul_ps(w1_uuuu, _mm_load_ps(x0y1_ptr + 12))
          )
      );

      // first 4 values in interpolated Vec8f
      __m128 a = _mm_add_ps(a1, a2);
      // last 4 values in interpolated Vec8f
      __m128 b = _mm_add_ps(b1, b2);

      // check for NaNs in interpolated Vec8f
      int nans_mask = _mm_movemask_ps(_mm_cmpunord_ps(a, b));

      if(nans_mask == 0)
      {
        _mm_store_ps(result.last_point_error->point.data, p2);

        __m128 reference_a = _mm_load_ps((p_it + 1)->intensity_and_depth.data);
        // replace the reference image depth value with the transformed one
        reference_a = _mm_or_ps(_mm_and_ps(BLEND_MASK, _mm_unpackhi_ps(pt1_zz_pt2_zz, pt1_zz_pt2_zz)), _mm_andnot_ps(BLEND_MASK, reference_a));

        __m128 residual_a = _mm_add_ps(_mm_mul_ps(current_weight_a, a), _mm_mul_ps(reference_weight_a, reference_a));
        _mm_store_ps(result.last_point_error->intensity_and_depth.data + 0, residual_a);

        // occlusion test
        if(result.last_point_error->intensity_and_depth.z > -20.0f * depthStdDevZ((p_it + 1)->intensity_and_depth.z))
        {
          _mm_storel_pi((__m64 *)result.last_residual->data(), residual_a);

          __m128 reference_b = _mm_load_ps((p_it + 1)->intensity_and_depth.data + 4);
          __m128 residual_b = _mm_add_ps(_mm_mul_ps(current_weight_b, b), _mm_mul_ps(reference_weight_b, reference_b));
          _mm_store_ps(result.last_point_error->intensity_and_depth.data + 4, residual_b);

          ++result.last_point_error;
          ++result.last_residual;
        }
        else
        {
          nans_mask = 1;
        }
      }

      if(Debug)
        *result.last_valid_flag = uint8_t(nans_mask == 0);
    }

    if(Debug)
      ++result.last_valid_flag;
  }

  if(rnd_mode != _MM_ROUND_TOWARD_ZERO) _MM_SET_ROUNDING_MODE(rnd_mode);
}

void computeResidualsSse(const PointIterator& first_point, const PointIterator& last_point, const RgbdImage& current, const IntrinsicMatrix& intrinsics, const Eigen::Affine3f transform, const Vector8f& reference_weight, const Vector8f& current_weight, ComputeResidualsResult& result, const float affine_a, const float affine_b)
{
  computeResidualsSse<false>(first_point, last_point, current, intrinsics, transform, reference_weight, current_weight, result, affine_a, affine_b);
}

void computeResidualsAndValidFlagsSse(const PointIterator& first_point, const PointIterator& last_point, const RgbdImage& current, const IntrinsicMatrix& intrinsics, const Eigen::Affine3f transform, const Vector8f& reference_weight, const Vector8f& current_weight, ComputeResidualsResult& result, const float affine_a, const float affine_b)
{
  computeResidualsSse<true>(first_point, last_point, current, intrinsics, transform, reference_weight, current_weight, result, affine_a, affine_b);
}


float computeCompleteDataLogLikelihood(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const  Eigen::Vector2f& mean, const  Eigen::Matrix2f& precision)
{
  size_t n = (last_residual - first_residual);
  size_t c = 1;
  double error_sum = 0.0;
  double error_acc = 1.0;

  for(ResidualIterator err_it = first_residual; err_it != last_residual; ++err_it, ++c)
  {
    error_acc *= (1.0 + 0.2 * ((*err_it).transpose() * precision * (*err_it))(0,0));

    if((c % 50) == 0)
    {
      error_sum += std::log(error_acc);
      error_acc = 1.0;
    }
  }

  return 0.5 * n * std::log(precision.determinant()) - 0.5 * (5.0 + 2.0) * error_sum;
}

static inline float computeWeightedErrorPart(const float& weight, const Eigen::Vector2f& r, const Eigen::Matrix2f& precision)
{
  return weight * r.transpose() * precision * r;
}

float computeWeightedError(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const  Eigen::Matrix2f& precision)
{
  double weighted_error = 0.0f;
  WeightIterator w_it = first_weight;
  size_t n = (last_residual - first_residual);

  for(ResidualIterator err_it = first_residual; err_it != last_residual; ++err_it, ++w_it)
  {
    weighted_error += computeWeightedErrorPart(*w_it, *err_it, precision);
  }

  return float(weighted_error / n);
}

float computeWeightedErrorSse(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const  Eigen::Matrix2f& precision)
{
  const ResidualIterator lr = last_residual - ((last_residual - first_residual) % 4);
  WeightIterator w_it = first_weight;

  __m128 error_acc = _mm_setzero_ps();
  __m128 prec = _mm_load_ps(precision.data());

  for(ResidualIterator err_it = first_residual; err_it != lr; err_it += 4, w_it += 4)
  {
    __m128 r1r2 = _mm_load_ps((err_it + 0)->data());

    __m128 diff_r1r2 = r1r2;
    __m128 diff_r1r1 = _mm_movelh_ps(diff_r1r2, diff_r1r2);
    __m128 diff_r2r2 = _mm_movehl_ps(diff_r1r2, diff_r1r2);

    // mahalanobis distance parts, one hadd missing!
    __m128 dist_r1r2 = _mm_mul_ps(
        _mm_hadd_ps(
            _mm_mul_ps(diff_r1r1, prec),
            _mm_mul_ps(diff_r2r2, prec)
        ),
        diff_r1r2
    );

    __m128 r3r4 = _mm_load_ps((err_it + 2)->data());
    __m128 diff_r3r4 = r3r4;
    __m128 diff_r3r3 = _mm_movelh_ps(diff_r3r4, diff_r3r4);
    __m128 diff_r4r4 = _mm_movehl_ps(diff_r3r4, diff_r3r4);

    // mahalanobis distance parts, one hadd missing!
    __m128 dist_r3r4 = _mm_mul_ps(
        _mm_hadd_ps(
            _mm_mul_ps(diff_r3r3, prec),
            _mm_mul_ps(diff_r4r4, prec)
        ),
        diff_r3r4
    );

    __m128 dist_r1r2r3r4 = _mm_hadd_ps(dist_r1r2, dist_r3r4);
    __m128 weights1234 = _mm_load_ps(&(*w_it));

    error_acc = _mm_add_ps(error_acc, _mm_mul_ps(weights1234, dist_r1r2r3r4));
  }

  EIGEN_ALIGN16 float tmp[4];

  _mm_store_ps(tmp, error_acc);

  double weighted_error = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);

  for(ResidualIterator err_it = lr; err_it != last_residual; ++err_it, ++w_it)
  {
    weighted_error += computeWeightedErrorPart(*w_it, *err_it, precision);
  }

  size_t n = (last_residual - first_residual);
  return float(weighted_error / n);
}

Eigen::Vector2f computeMean(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight)
{
  WeightIterator w_it = first_weight;
  Eigen::Vector2d weighted_residual_sum;
  weighted_residual_sum.setZero();
  double weight_sum = 0.0f;

  for(ResidualIterator err_it = first_residual;err_it != last_residual; ++err_it, ++w_it)
  {
    double w = (*w_it);

    weighted_residual_sum += w * err_it->cast<double>();
    weight_sum += w;
  }

  return (weighted_residual_sum / weight_sum).cast<float>();
}

Eigen::Vector2f computeMeanSse(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight)
{
  const float *weights_ptr = &(*first_weight);

  __m128 residual_acc = _mm_setzero_ps();
  __m128 weight_acc = _mm_setzero_ps();

  const ResidualIterator lr = last_residual - ((last_residual - first_residual) % 4);

  for(ResidualIterator err_it = first_residual; err_it != lr; err_it += 4, weights_ptr += 4)
  {
    __m128 w1234 = _mm_load_ps(weights_ptr);
    weight_acc = _mm_add_ps(weight_acc, w1234);

    __m128 r12 = _mm_load_ps((err_it + 0)->data());
    __m128 r34 = _mm_load_ps((err_it + 2)->data());

    residual_acc = _mm_add_ps(residual_acc, _mm_add_ps(_mm_mul_ps(_mm_unpacklo_ps(w1234, w1234), r12), _mm_mul_ps(_mm_unpackhi_ps(w1234, w1234), r34)));
  }

  EIGEN_ALIGN16 float tmp[4];

  _mm_store_ps(tmp, residual_acc);

  Eigen::Vector2f weighted_residual_sum;
  weighted_residual_sum(0) = tmp[0] + tmp[2];
  weighted_residual_sum(1) = tmp[1] + tmp[3];

  _mm_store_ps(tmp, weight_acc);

  float weights = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for(ResidualIterator err_it = lr;err_it != last_residual; ++err_it, ++weights_ptr)
  {
    weighted_residual_sum += (*weights_ptr) * (*err_it);
    weights += (*weights_ptr);
  }

  return weighted_residual_sum / weights;
}


static inline Eigen::Matrix2f computeScalePart(const float& weight, const Eigen::Vector2f& r, const Eigen::Vector2f& mean)
{
  Eigen::Vector2f diff;
  diff = r - mean;

  return weight * diff * diff.transpose();
}

Eigen::Matrix2f computeScale(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const Eigen::Vector2f& mean)
{
  Eigen::Matrix2f covariance;
  covariance.setZero();
  WeightIterator w_it = first_weight;
  size_t n = (last_residual - first_residual);
  float scale = 1.0f / (n - 2 -1);

  for(ResidualIterator err_it = first_residual;err_it != last_residual; ++err_it, ++w_it)
  {
    covariance += scale * computeScalePart(*w_it, *err_it, mean);
  }

  return covariance;
}

Eigen::Matrix2f computeScaleSse(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const Eigen::Vector2f& mean)
{
  const ResidualIterator lr = last_residual - ((last_residual - first_residual) % 2);

  WeightIterator w_it = first_weight;
  size_t n = (last_residual - first_residual);
  float scale = 1.0f / (n - 2 -1);

  __m128 cov_acc = _mm_setzero_ps();
  __m128 s = _mm_set1_ps(scale);
  __m128 mean2 = _mm_setr_ps(mean(0), mean(1), mean(0), mean(1));

  __m128 fac1, fac2, w;
  for(ResidualIterator err_it = first_residual;err_it != lr; err_it += 2, w_it += 2)
  {
    __m128 r1r2 = _mm_load_ps(err_it->data());
    __m128 diff_r1r2 = _mm_sub_ps(r1r2, mean2); // [x1, y1, x2, y2]

    fac1 = _mm_movelh_ps(diff_r1r2, diff_r1r2); // [x1, y1, x1, y1]
    fac2 = _mm_unpacklo_ps(diff_r1r2, diff_r1r2); // [x1, x1, y1, y1]
    w = _mm_set1_ps(*(w_it + 0));

    __m128 p1 = _mm_mul_ps(s, _mm_mul_ps(w, _mm_mul_ps(fac1, fac2)));

    fac1 = _mm_movelh_ps(diff_r1r2, diff_r1r2); // [x2, y2, x2, y2]
    fac2 = _mm_unpacklo_ps(diff_r1r2, diff_r1r2); // [x2, x2, y2, y2]
    w = _mm_set1_ps(*(w_it + 1));

    __m128 p2 = _mm_mul_ps(s, _mm_mul_ps(w, _mm_mul_ps(fac1, fac2)));

    cov_acc = _mm_add_ps(cov_acc, _mm_add_ps(p1, p2));
  }

  EIGEN_ALIGN16 float tmp[4];
  _mm_store_ps(tmp, cov_acc);

  Eigen::Matrix2f covariance;
  covariance(0, 0) = tmp[0];
  covariance(0, 1) = tmp[1];
  covariance(1, 0) = tmp[1];
  covariance(1, 1) = tmp[3];

  for(ResidualIterator err_it = lr;err_it != last_residual; ++err_it, ++w_it)
  {
    covariance += scale * computeScalePart(*w_it, *err_it, mean);
  }

  return covariance;
}

static inline float computeWeight(const Eigen::Vector2f& r, const Eigen::Vector2f& mean, const Eigen::Matrix2f& precision)
{
  Eigen::Vector2f diff = r - mean;
  return (2.0 + 5.0f) / (5.0f + diff.transpose() * precision * diff);
}

void computeWeights(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const Eigen::Vector2f& mean, const Eigen::Matrix2f& precision)
{
  Eigen::Vector2f diff;
  WeightIterator w_it = first_weight;

  for(ResidualIterator err_it = first_residual; err_it != last_residual; ++err_it, ++w_it)
  {
    *w_it = computeWeight(*err_it, mean, precision);
  }
}

void computeWeightsSse(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const Eigen::Vector2f& mean, const Eigen::Matrix2f& precision)
{
  float *w_ptr = &(*first_weight);
  const ResidualIterator lr = last_residual - ((last_residual - first_residual) % 4);

  __m128 prec = _mm_load_ps(precision.data());
  __m128 mean2 = _mm_setr_ps(mean(0), mean(1), mean(0), mean(1));
  __m128 five = _mm_set1_ps(5.0f);
  // Nan: unlike on the paper: 1 + 5, but 2 + 5 here, why?
  __m128 six = _mm_set1_ps(7.0f);

  for(ResidualIterator err_it = first_residual; err_it != lr; err_it += 4, w_ptr += 4)
  {
    __m128 r1r2 = _mm_load_ps((err_it + 0)->data());

    __m128 diff_r1r2 = _mm_sub_ps(r1r2, mean2);
    __m128 diff_r1r1 = _mm_movelh_ps(diff_r1r2, diff_r1r2);
    __m128 diff_r2r2 = _mm_movehl_ps(diff_r1r2, diff_r1r2);

    // mahalanobis distance parts, one hadd missing!
    __m128 dist_r1r2 = _mm_mul_ps(
        _mm_hadd_ps(
            _mm_mul_ps(diff_r1r1, prec),
            _mm_mul_ps(diff_r2r2, prec)
        ),
        diff_r1r2
    );

    __m128 r3r4 = _mm_load_ps((err_it + 2)->data());
    __m128 diff_r3r4 = _mm_sub_ps(r3r4, mean2);
    __m128 diff_r3r3 = _mm_movelh_ps(diff_r3r4, diff_r3r4);
    __m128 diff_r4r4 = _mm_movehl_ps(diff_r3r4, diff_r3r4);

    // mahalanobis distance parts, one hadd missing!
    __m128 dist_r3r4 = _mm_mul_ps(
        _mm_hadd_ps(
            _mm_mul_ps(diff_r3r3, prec),
            _mm_mul_ps(diff_r4r4, prec)
        ),
        diff_r3r4
    );

    __m128 dist_r1r2r3r4 = _mm_hadd_ps(dist_r1r2, dist_r3r4);

    _mm_store_ps(w_ptr, _mm_mul_ps(six, _mm_rcp_ps(_mm_add_ps(five, dist_r1r2r3r4))));
  }

  for(ResidualIterator err_it = lr; err_it != last_residual; ++err_it, ++w_ptr)
  {
    *w_ptr = computeWeight(*err_it, mean, precision);
  }
}

void computeMeanScaleAndWeights(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, Eigen::Vector2f& mean, Eigen::Matrix2f& precision)
{
  bool converged = false;
  Eigen::Matrix2f last_precision;
  size_t n = (last_residual - first_residual);
  std::fill(first_weight, first_weight + n, 1.0f);

  int iterations = 0;

  float convergence_criterion = 1.0f;

  do
  {
    last_precision = precision;

    Eigen::Matrix2f S2 = computeScaleSse(first_residual, last_residual, first_weight, mean);
    precision = S2.inverse();

    computeWeightsSse(first_residual, last_residual, first_weight, mean, precision);

    //mean = computeMeanSse(first_residual, last_residual, first_weight);

    convergence_criterion = (last_precision - precision).lpNorm<Eigen::Infinity>();
    iterations += 1;
  }
  while(convergence_criterion >= 1e-3 && iterations < 50);
}

} /* namespace core */
} /* namespace dvo */
