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

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <dvo/core/weight_calculation.h>

#include <dvo/util/histogram.h>
//#include <dvo/visualization/visualizer.h>

namespace dvo
{
namespace core
{

const float TDistributionScaleEstimator::INITIAL_SIGMA = 5.0f;
const float TDistributionScaleEstimator::DEFAULT_DOF = 5.0f;

TDistributionScaleEstimator::TDistributionScaleEstimator(const float dof) :
    initial_sigma(INITIAL_SIGMA)
{
  configure(dof);
}

void TDistributionScaleEstimator::configure(const float& param)
{
  dof = param;
}

float TDistributionScaleEstimator::compute(const cv::Mat& errors) const
{
  float initial_lamda = 1.0f / (initial_sigma * initial_sigma);

  float num = 0.0f;
  float lambda = initial_lamda;

  int iterations = 0;

  do
  {
    iterations += 1;
    num = 0.0f;
    initial_lamda = lambda;
    lambda = 0.0f;

    const float* data_ptr = errors.ptr<float>();

    for(size_t idx = 0; idx < errors.size().area(); ++idx, ++data_ptr)
    {
      const float& data = *data_ptr;

      if(std::isfinite(data))
      {
        num += 1.0f;
        lambda += data * data * ( (dof + 1.0f) / (dof + initial_lamda * data * data) );
      }
    }

    lambda /= num;
    lambda = 1.0f / lambda;
  } while(std::abs(lambda - initial_lamda) > 1e-3);

  return std::sqrt(1.0f / lambda);
}

struct TDistributionScaleReduction
{
  TDistributionScaleReduction(const float *data, const float initial_lambda, const float dof) :
    data(data),
    initial_lambda(initial_lambda),
    dof(dof),
    lambda(0.0f),
    num(0)
  {

  }

  TDistributionScaleReduction(TDistributionScaleReduction& other, tbb::split) :
    data(other.data),
    initial_lambda(other.initial_lambda),
    dof(other.dof),
    lambda(0.0f),
    num(0)
  {

  }

  void operator()(const tbb::blocked_range<size_t>& r)
  {
    //size_t vectorizable = r.size() / 4;
    //size_t remaining = r.size() - (vectorizable * 4);

    const float* data_ptr = data + r.begin();

    float tmpnum = num, tmplambda = lambda;

    //float tmpnum4[4], tmplambda4[4];
    //
    //__m128 lambda4 = _mm_setzero_ps();
    //__m128 num4 = _mm_setzero_ps();
    //__m128 ini4 = _mm_set1_ps(initial_lambda);
    //__m128 dof4 = _mm_set1_ps(dof);
    //__m128 ones = _mm_set1_ps(0.0f);
    //__m128 isnan, data4, data4square;
    //
    //for(size_t idx = r.begin(); idx < r.end(); idx += 4, data_ptr += 4)
    //{
    //  data4 = _mm_load_ps(data_ptr);
    //  data4square = _mm_mul_ps(data4, data4);
    //  isnan = _mm_cmpord_ps(data4, data4);
    //
    //  num4 = _mm_add_ps(num4, _mm_and_ps(isnan, ones));
    //  lambda4 = _mm_add_ps(lambda4, _mm_and_ps(isnan, _mm_mul_ps(data4square, _mm_div_ps(_mm_add_ps(dof4, ones),  _mm_add_ps(dof4, _mm_mul_ps(ini4, data4square))))));
    //
    //  //const float& data = *data_ptr;
    //  //
    //  //if(std::isfinite(data))
    //  //{
    //  //  tmp_num += 1;
    //  //  tmp_lambda += data * data * ( (dof + 1.0f) / (dof + initial_lambda * data * data) );
    //  //}
    //}
    //_mm_store_ps(tmpnum4, num4);
    //_mm_store_ps(tmplambda4, lambda4);
    //
    //tmpnum += tmpnum4[0] + tmpnum4[1] + tmpnum4[2]+ tmpnum4[3];
    //tmplambda += tmplambda4[0] + tmplambda4[1] + tmplambda4[2]+ tmplambda4[3];

    for(size_t idx = r.begin(); idx != r.end(); ++idx, ++data_ptr)
    {
      const float& data = *data_ptr;

      if(std::isfinite(data))
      {
        tmpnum += 1.0f;
        tmplambda += data * data * ( (dof + 1.0f) / (dof + initial_lambda * data * data));
      }
    }

    num = tmpnum;
    lambda = tmplambda;
  }

  void join(TDistributionScaleReduction& other)
  {
    num += other.num;
    lambda += other.lambda;
  }

  const float *data, initial_lambda, dof;
  float lambda, num;
};

OptimizedTDistributionScaleEstimator::OptimizedTDistributionScaleEstimator(const float dof) :
    TDistributionScaleEstimator(dof)
{
}

float OptimizedTDistributionScaleEstimator::compute(const cv::Mat& errors) const
{
  float initial_lamda, lambda = 1.0f / (initial_sigma * initial_sigma);

  do
  {
    initial_lamda = lambda;
    TDistributionScaleReduction body(errors.ptr<float>(), initial_lamda, dof);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, errors.size().area()), body);

    lambda = body.lambda / body.num;
    lambda = 1.0f / lambda;
  }
  while(std::abs(lambda - initial_lamda) > 1e-3);

  return std::sqrt(1.0f / lambda);
}

ApproximateTDistributionScaleEstimator::ApproximateTDistributionScaleEstimator(const float dof) :
    TDistributionScaleEstimator(INITIAL_SIGMA)
{
}

float ApproximateTDistributionScaleEstimator::compute(const cv::Mat& errors) const
{
  cv::Mat mask = errors != errors;
  cv::Mat error_copy = errors.clone();
  error_copy.setTo(1, mask);
  cv::Mat log_square_error;
  cv::log(error_copy.mul(error_copy), log_square_error);
  cv::Scalar sum = cv::sum(log_square_error);
  int invalid = cv::countNonZero(mask);

  double z = sum(0) / double(errors.size().area() - invalid);
  //std::log(dof) + 0.70315664064 + 1.96351
  return std::sqrt(std::exp(z + 1.05722875)); // this sucks :(
}

MADScaleEstimator::MADScaleEstimator()
{
}

float MADScaleEstimator::compute(const cv::Mat& errors) const
{
  cv::Mat error_hist, error_median_absdiff, abs_error;
  float median;

  //dvo::util::compute1DHistogram(errors, error_hist, -255, 255, 1);
  //median = dvo::util::computeMedianFromHistogram(error_hist, -255, 255);

  //cv::absdiff(errors, median, error_median_absdiff);

  abs_error = cv::abs(errors);

  // recycle error_hist
  //dvo::util::compute1DHistogram(error_median_absdiff, error_hist, 0, 255, 1);
  dvo::util::compute1DHistogram(abs_error, error_hist, 0, 255, 1);
  median = dvo::util::computeMedianFromHistogram(error_hist, 0, 255);

  return NORMALIZER * median;
}

const float MADScaleEstimator::NORMALIZER = 1.48f;

NormalDistributionScaleEstimator::NormalDistributionScaleEstimator()
{
}

float NormalDistributionScaleEstimator::compute(const cv::Mat& errors) const
{
  cv::Mat mean, stddev, mask;
  mask = (errors == errors); // mask nans

  cv::meanStdDev(errors, mean, stddev, mask);

  return (float) stddev.at<double>(0, 0);
}

const char* ScaleEstimators::str(enum_t type)
{
  switch(type)
  {
    case ScaleEstimators::Unit:
      return "Unit";
    case ScaleEstimators::TDistribution:
      return "TDistribution";
    case ScaleEstimators::MAD:
      return "MAD";
    case ScaleEstimators::NormalDistribution:
      return "NormalDistribution";
    default:
      break;
  }
  assert(false && "Unknown scale estimator type!");

  return "";
}

ScaleEstimator* ScaleEstimators::get(ScaleEstimators::enum_t type)
{
  static OptimizedTDistributionScaleEstimator tdistribution;
  static MADScaleEstimator mad;
  static NormalDistributionScaleEstimator normaldistribution;
  static UnitScaleEstimator unit;

  switch(type)
  {
    case ScaleEstimators::Unit:
      return (ScaleEstimator*)&unit;
    case ScaleEstimators::TDistribution:
      return (ScaleEstimator*)&tdistribution;
    case ScaleEstimators::MAD:
      return (ScaleEstimator*)&mad;
    case ScaleEstimators::NormalDistribution:
      return (ScaleEstimator*)&normaldistribution;
    default:
      break;
  }
  assert(false && "Unknown scale estimator type!");

  return 0;
}

const float TukeyInfluenceFunction::DEFAULT_B = 4.6851f;

TukeyInfluenceFunction::TukeyInfluenceFunction(const float b)
{
  configure(b);
}

inline float TukeyInfluenceFunction::value(const float& x) const
{
  const float x_square = x * x;

  if(x_square <= b_square)
  {
    const float tmp = 1.0f - x_square / b_square;

    return tmp * tmp;
  }
  else
  {
    return 0.0f;
  }
}

void TukeyInfluenceFunction::configure(const float& param)
{
  b_square = param * param;
}

const float TDistributionInfluenceFunction::DEFAULT_DOF = 5.0f;

TDistributionInfluenceFunction::TDistributionInfluenceFunction(const float dof)
{
  configure(dof);
}

inline float TDistributionInfluenceFunction::value(const float & x) const
{

  return ((dof + 1.0f) / (dof + (x * x)));
}

void TDistributionInfluenceFunction::configure(const float& param)
{
  dof = param;
  normalizer = dof / (dof + 1.0f);
}

const float HuberInfluenceFunction::DEFAULT_K = 1.345f;

HuberInfluenceFunction::HuberInfluenceFunction(const float k)
{
  configure(k);
}

inline float HuberInfluenceFunction::value(const float& x) const
{
  const float x_abs = std::abs(x);

  if(x_abs < k)
  {
    return 1.0f;
  }
  else
  {
    return k / x_abs;
  }
}

void HuberInfluenceFunction::configure(const float& param)
{
  k = param;
}

const char* InfluenceFunctions::str(enum_t type)
{
  switch(type)
  {
    case InfluenceFunctions::Unit:
      return "Unit";
    case InfluenceFunctions::TDistribution:
      return "TDistribution";
    case InfluenceFunctions::Tukey:
      return "Tukey";
    case InfluenceFunctions::Huber:
      return "Huber";
    default:
      break;
  }
  assert(false && "Unknown influence function type!");

  return "";
}

InfluenceFunction* InfluenceFunctions::get(InfluenceFunctions::enum_t type)
{
  static TDistributionInfluenceFunction tdistribution;
  static TukeyInfluenceFunction tukey;
  static HuberInfluenceFunction huber;
  static UnitInfluenceFunction unit;

  switch(type)
  {
    case InfluenceFunctions::Unit:
      return (InfluenceFunction*)&unit;
    case InfluenceFunctions::TDistribution:
      return (InfluenceFunction*)&tdistribution;
    case InfluenceFunctions::Tukey:
      return (InfluenceFunction*)&tukey;
    case InfluenceFunctions::Huber:
      return (InfluenceFunction*)&huber;
    default:
      break;
  }
  assert(false && "Unknown influence function type!");

  return 0;
}

WeightCalculation::WeightCalculation() :
    scale_(1.0f)
{
}

void WeightCalculation::calculateScale(const cv::Mat& errors)
{
  // some scale estimators might return 0
  scale_ = std::max(scaleEstimator_->compute(errors), 0.001f);
}

float WeightCalculation::calculateWeight(const float error) const
{
  //if(std::isfinite(error))
  {
    return influenceFunction_->value(error / scale_);
  }
  //else
  {
  //  return 0.0f;
  }
}

void WeightCalculation::calculateWeights(const cv::Mat& errors, cv::Mat& weights)
{
  weights.create(errors.size(), errors.type());

  cv::Mat scaled_errors = errors / scale_;
  const float* err_ptr = scaled_errors.ptr<float>();
  float* weight_ptr = weights.ptr<float>();

  for(size_t idx = 0; idx < errors.size().area(); ++idx, ++err_ptr, ++weight_ptr)
  {
    if(std::isfinite(*err_ptr))
    {
      *weight_ptr = influenceFunction_->value(*err_ptr);
    }
    else
    {
      *weight_ptr = 0.0f;
    }
  }

  //dvo::visualization::Visualizer::instance()
    //.show("residuals", cv::abs(errors))
    //.showHistogram("residuals_histogram", errors, 5.0f, -255.0f, 255.0f)
    //.show("weights", weights)
    //.show("weighted_residuals", cv::abs(weights.mul(errors)))
  //;
}

} /* namespace core */
} /* namespace dvo */
