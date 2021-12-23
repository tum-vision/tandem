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

#include <iomanip>

#include <dvo/dense_tracking.h>
#include <dvo/dense_tracking_impl.h>

#include <assert.h>
#include <sophus/se3.hpp>

#include <Eigen/Core>

#include <dvo/core/datatypes.h>
#include <dvo/core/point_selection_predicates.h>
#include <dvo/util/revertable.h>
#include <dvo/util/stopwatch.h>
#include <dvo/util/id_generator.h>
#include <dvo/util/histogram.h>
#include <FullSystem/HessianBlocks.h>
//#include <dvo/visualization/visualizer.h>

namespace dvo
{

using namespace dvo::core;
using namespace dvo::util;

const DenseTracker::Config& DenseTracker::getDefaultConfig()
{
  static Config defaultConfig;

  return defaultConfig;
}

static const Eigen::IOFormat YamlArrayFmt(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");

DenseTracker::DenseTracker(const Config& config) :
    itctx_(cfg),
    weight_calculation_(),
    selection_predicate_(),
    reference_selection_(selection_predicate_)
{
  configure(config);
}

DenseTracker::DenseTracker(const DenseTracker& other) :
  itctx_(cfg),
  weight_calculation_(),
  selection_predicate_(),
  reference_selection_(selection_predicate_)
{
  configure(other.configuration());
}

void DenseTracker::configure(const Config& config)
{
  assert(config.IsSane());

  cfg = config;

  selection_predicate_.intensity_threshold = cfg.IntensityDerivativeThreshold;
  selection_predicate_.depth_threshold = cfg.DepthDerivativeThreshold;

  if(cfg.UseWeighting)
  {
    weight_calculation_
      .scaleEstimator(ScaleEstimators::get(cfg.ScaleEstimatorType))
      .scaleEstimator()->configure(cfg.ScaleEstimatorParam);

    weight_calculation_
      .influenceFunction(InfluenceFunctions::get(cfg.InfluenceFuntionType))
      .influenceFunction()->configure(cfg.InfluenceFunctionParam);
  }
  else
  {
    weight_calculation_
      .scaleEstimator(ScaleEstimators::get(ScaleEstimators::Unit))
      .influenceFunction(InfluenceFunctions::get(InfluenceFunctions::Unit));
  }
}

bool DenseTracker::match(RgbdImagePyramid& reference, RgbdImagePyramid& current, Eigen::Affine3d& transformation, const int on_level, float& lambda, float affine_a, float affine_b)
{
  Result result;
  result.Transformation = transformation;

  bool success = match(reference, current, result, on_level, lambda, affine_a, affine_b);

  transformation = result.Transformation;

  return success;
}

bool DenseTracker::match(dvo::core::PointSelection& reference, RgbdImagePyramid& current, Eigen::Affine3d& transformation)
{
  Result result;
  result.Transformation = transformation;
  float lambda = 0.01;
  bool success = match(reference, current, result, 1, lambda);

  transformation = result.Transformation;

  return success;
}

bool DenseTracker::match(dvo::core::RgbdImagePyramid& reference, dvo::core::RgbdImagePyramid& current, dvo::DenseTracker::Result& result, const int on_level, float& lambda, float affine_a, float affine_b)
{
  reference.compute(cfg.getNumLevels());
  reference_selection_.setRgbdImagePyramid(reference);

  return match(reference_selection_, current, result, on_level, lambda, affine_a, affine_b);
}

bool DenseTracker::match(dvo::core::PointSelection& reference, dvo::core::RgbdImagePyramid& current, dvo::DenseTracker::Result& result, const int on_level, float& lambda, float affine_a, float affine_b)
{
  current.compute(cfg.getNumLevels());

  bool success = true;

  if(cfg.UseInitialEstimate)
  {
    assert(!result.isNaN() && "Provided initialization is NaN!");
  }
  else
  {
    result.setIdentity();
  }

  // our first increment is the given guess
  Sophus::SE3d inc(result.Transformation.rotation(), result.Transformation.translation());

  Revertable<Sophus::SE3d> initial(inc);
  Revertable<Sophus::SE3d> estimate;

  bool accept = true;

  //static stopwatch_collection sw_level(5, "l", 100);
  //static stopwatch_collection sw_it(5, "it@l", 500);
  //static stopwatch_collection sw_error(5, "err@l", 500);
  //static stopwatch_collection sw_linsys(5, "linsys@l", 500);
  //static stopwatch_collection sw_prep(5, "prep@l", 100);

  if(points_error.size() < reference.getMaximumNumberOfPoints(cfg.LastLevel))
    points_error.resize(reference.getMaximumNumberOfPoints(cfg.LastLevel));
  if(residuals.size() < reference.getMaximumNumberOfPoints(cfg.LastLevel))
    residuals.resize(reference.getMaximumNumberOfPoints(cfg.LastLevel));
  if(weights.size() < reference.getMaximumNumberOfPoints(cfg.LastLevel))
    weights.resize(reference.getMaximumNumberOfPoints(cfg.LastLevel));

  std::vector<uint8_t> valid_residuals;

  bool debug = false;
  if(debug)
  {
    reference.debug(true);
    valid_residuals.resize(reference.getMaximumNumberOfPoints(cfg.LastLevel));
  }
  /*
  std::stringstream name;
  name << std::setiosflags(std::ios::fixed) << std::setprecision(2) << current.timestamp() << "_error.avi";

  cv::Size s = reference.getRgbdImagePyramid().level(size_t(config.LastLevel)).intensity.size();
  cv::Mat video_frame(s.height, s.width * 2, CV_32FC1), video_frame_u8;
  cv::VideoWriter vw(name.str(), CV_FOURCC('P','I','M','1'), 30, video_frame.size(), false);
  float rgb_max = 0.0;
  float depth_max = 0.0;

  std::stringstream name1;
  name1 << std::setiosflags(std::ios::fixed) << std::setprecision(2) << current.timestamp() << "_ref.png";

  cv::imwrite(name1.str(), current.level(0).rgb);

  std::stringstream name2;
  name2 << std::setiosflags(std::ios::fixed) << std::setprecision(2) << current.timestamp() << "_cur.png";

  cv::imwrite(name2.str(), reference.getRgbdImagePyramid().level(0).rgb);
  */
  Eigen::Vector2f mean;
  mean.setZero();
  Eigen::Matrix2f /*first_precision,*/ precision;
  precision.setZero();

  int start_level = cfg.FirstLevel;
  int end_level = cfg.LastLevel;

  if(on_level >= 0){
      start_level = on_level;
      end_level = on_level;
  }
//  float lambdaExtrapolationLimit = 0.001;
  // coarse to fine
  for(itctx_.Level = start_level; itctx_.Level >= end_level; --itctx_.Level)
  {
    result.Statistics.Levels.push_back(LevelStats());
    LevelStats& level_stats = result.Statistics.Levels.back();

    mean.setZero();
    precision.setZero();

    // reset error after every pyramid level? yes because errors from different levels are not comparable
    itctx_.Iteration = 0;
    itctx_.Error = std::numeric_limits<double>::max();

    RgbdImage& cur = current.level(itctx_.Level);
    const IntrinsicMatrix& K = cur.camera().intrinsics();

    Vector8f wcur, wref;
    // i z idx idy zdx zdy
    float wcur_id = 0.5f, wref_id = 0.5f, wcur_zd = 1.0f, wref_zd = 0.0f;

    wcur <<  1.0f / 255.0f,  1.0f, wcur_id * K.fx() / 255.0f, wcur_id * K.fy() / 255.0f, wcur_zd * K.fx(), wcur_zd * K.fy(), 0.0f, 0.0f;
    wref << -1.0f / 255.0f, -1.0f, wref_id * K.fx() / 255.0f, wref_id * K.fy() / 255.0f, wref_zd * K.fx(), wref_zd * K.fy(), 0.0f, 0.0f;

//    sw_prep[itctx_.Level].start();


    PointSelection::PointIterator first_point, last_point;
    reference.select(itctx_.Level, first_point, last_point);
    cur.buildAccelerationStructure();

    level_stats.Id = itctx_.Level;
    level_stats.MaxValidPixels = reference.getMaximumNumberOfPoints(itctx_.Level);
    level_stats.ValidPixels = last_point - first_point;

//    sw_prep[itctx_.Level].stopAndPrint();

    NormalEquationsLeastSquares ls;
    Matrix6d A;
    Vector6d x, b;
    x = inc.log();

    ComputeResidualsResult compute_residuals_result;
    compute_residuals_result.first_point_error = points_error.begin();
    compute_residuals_result.first_residual = residuals.begin();
    compute_residuals_result.first_valid_flag = valid_residuals.begin();


//    sw_level[itctx_.Level].start();

    do {
        level_stats.Iterations.push_back(IterationStats());
        IterationStats &iteration_stats = level_stats.Iterations.back();
        iteration_stats.Id = itctx_.Iteration;

//      sw_it[itctx_.Level].start();

        double total_error = 0.0f;
//      sw_error[itctx_.Level].start();
        Eigen::Affine3f transformf;

        inc = Sophus::SE3d::exp(x);
        initial.update() = inc.inverse() * initial();
        estimate.update() = inc * estimate();

        transformf = estimate().matrix().cast<float>();

        if (debug) {
            dvo::core::computeResidualsAndValidFlagsSse(first_point,
                                                        last_point, cur, K,
                                                        transformf, wref, wcur,
                                                        compute_residuals_result, affine_a, affine_b);
        } else {
            dvo::core::computeResidualsSse(first_point, last_point, cur, K,
                                           transformf, wref, wcur,
                                           compute_residuals_result, affine_a, affine_b);
        }
        size_t n = (compute_residuals_result.last_residual -
                    compute_residuals_result.first_residual);
        iteration_stats.ValidConstraints = n;

        if (n < 6) {
            initial.revert();
            estimate.revert();

            level_stats.TerminationCriterion = TerminationCriteria::TooFewConstraints;

            break;
        }

        if (itctx_.IsFirstIterationOnLevel()) {
            std::fill(weights.begin(), weights.begin() + n, 1.0f);
        } else {
            // weights is the weigting function for each residual term
            dvo::core::computeWeightsSse(
                    compute_residuals_result.first_residual,
                    compute_residuals_result.last_residual, weights.begin(),
                    mean, precision);
        }

        // precision(scale) is the covariance matrix which balances the photometric and depth residuals.
        precision = dvo::core::computeScaleSse(
                compute_residuals_result.first_residual,
                compute_residuals_result.last_residual, weights.begin(),
                mean).inverse();

        
        float ll = computeCompleteDataLogLikelihood(
                compute_residuals_result.first_residual,
                compute_residuals_result.last_residual, weights.begin(), mean,
                precision);

        iteration_stats.TDistributionLogLikelihood = -ll;
        iteration_stats.TDistributionMean = mean.cast<double>();
        iteration_stats.TDistributionPrecision = precision.cast<double>();
        iteration_stats.PriorLogLikelihood =
                cfg.Mu * initial().log().squaredNorm();

        total_error = -ll;//iteration_stats.TDistributionLogLikelihood + iteration_stats.PriorLogLikelihood;

        itctx_.LastError = itctx_.Error;
        itctx_.Error = total_error;

//      sw_error[itctx_.Level].stopAndPrint();

        // accept the last increment?
        accept = itctx_.Error < itctx_.LastError;

        if (!accept) {
            initial.revert();
            estimate.revert();

            level_stats.TerminationCriterion = TerminationCriteria::LogLikelihoodDecreased;

            break;
        }

        // now build equation system
//      sw_linsys[itctx_.Level].start();

        WeightVectorType::iterator w_it = weights.begin();

        Matrix2x6 J, Jw;
        Eigen::Vector2f Ji;
        Vector6 Jz;
        ls.initialize(1);
        for (PointIterator e_it = compute_residuals_result.first_point_error;
             e_it !=
             compute_residuals_result.last_point_error; ++e_it, ++w_it) {
            computeJacobianOfProjectionAndTransformation(e_it->getPointVec4f(),
                                                         Jw);
            compute3rdRowOfJacobianOfTransformation(e_it->getPointVec4f(), Jz);

            J.row(0) = e_it->getIntensityDerivativeVec2f().transpose() * Jw;
            J.row(1) = e_it->getDepthDerivativeVec2f().transpose() * Jw -
                       Jz.transpose();

            ls.update(J, e_it->getIntensityAndDepthVec2f(),
                      (*w_it) * precision);
        }
        ls.finish();

        A = ls.A.cast<double>() + cfg.Mu * Matrix6d::Identity();
        b = ls.b.cast<double>() + cfg.Mu * initial().log();
//        A.block<6, 3>(0, 0) *= SCALE_XI_ROT;
//        A.block<3, 6>(0, 0) *= SCALE_XI_ROT;
//        b.segment<3>(0) *= SCALE_XI_ROT;
//        A.block<6, 3>(0, 3) *= SCALE_XI_TRANS;
//        A.block<3, 6>(3, 0) *= SCALE_XI_TRANS;
//        b.segment<3>(3) *= SCALE_XI_TRANS;
        for (int i = 0; i < 6; i++) A(i, i) *= (1 + lambda);
        x = A.ldlt().solve(b);
//        x.segment<3>(0) *= SCALE_XI_ROT;
//        x.segment<3>(3) *= SCALE_XI_TRANS;

//        float extrapFac = 1;
//        if (lambda < lambdaExtrapolationLimit)
//            extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
//        x *= extrapFac;
//
        if(accept){
            lambda *= 0.5;
        }

        //      sw_linsys[itctx_.Level].stopAndPrint();

        iteration_stats.EstimateIncrement = x;
        iteration_stats.EstimateInformation = A;

        itctx_.Iteration++;
//      sw_it[itctx_.Level].stopAndPrint();
    } while(accept && x.lpNorm<Eigen::Infinity>() > cfg.Precision && !itctx_.IterationsExceeded());

    if(x.lpNorm<Eigen::Infinity>() <= cfg.Precision)
        level_stats.TerminationCriterion = TerminationCriteria::IncrementTooSmall;

    if(itctx_.IterationsExceeded())
      level_stats.TerminationCriterion = TerminationCriteria::IterationsExceeded;

//    sw_level[itctx_.Level].stopAndPrint();
  }

  LevelStats& last_level = result.Statistics.Levels.back();
  IterationStats& last_iteration = last_level.TerminationCriterion != TerminationCriteria::LogLikelihoodDecreased ? last_level.Iterations[last_level.Iterations.size() - 1] : last_level.Iterations[last_level.Iterations.size() - 2];

  result.Transformation = estimate().inverse().matrix();
  result.Information = last_iteration.EstimateInformation * 0.008 * 0.008;
  result.LogLikelihood = last_iteration.TDistributionLogLikelihood + last_iteration.PriorLogLikelihood;

  return success;
}

cv::Mat DenseTracker::computeIntensityErrorImage(dvo::core::RgbdImagePyramid& reference, dvo::core::RgbdImagePyramid& current, const dvo::core::AffineTransformd& transformation, size_t level)
{
  reference.compute(level + 1);
  current.compute(level + 1);
  reference_selection_.setRgbdImagePyramid(reference);
  reference_selection_.debug(true);

  std::vector<uint8_t> valid_residuals;

  if(points_error.size() < reference_selection_.getMaximumNumberOfPoints(level))
    points_error.resize(reference_selection_.getMaximumNumberOfPoints(level));
  if(residuals.size() < reference_selection_.getMaximumNumberOfPoints(level))
    residuals.resize(reference_selection_.getMaximumNumberOfPoints(level));

  valid_residuals.resize(reference_selection_.getMaximumNumberOfPoints(level));

  PointSelection::PointIterator first_point, last_point;
  reference_selection_.select(level, first_point, last_point);

  RgbdImage& cur = current.level(level);
  cur.buildAccelerationStructure();
  const IntrinsicMatrix& K = cur.camera().intrinsics();

  Vector8f wcur, wref;
  // i z idx idy zdx zdy
  float wcur_id = 0.5f, wref_id = 0.5f, wcur_zd = 1.0f, wref_zd = 0.0f;

  wcur <<  1.0f / 255.0f,  1.0f, wcur_id * K.fx() / 255.0f, wcur_id * K.fy() / 255.0f, wcur_zd * K.fx(), wcur_zd * K.fy(), 0.0f, 0.0f;
  wref << -1.0f / 255.0f, -1.0f, wref_id * K.fx() / 255.0f, wref_id * K.fy() / 255.0f, wref_zd * K.fx(), wref_zd * K.fy(), 0.0f, 0.0f;

  ComputeResidualsResult compute_residuals_result;
  compute_residuals_result.first_point_error = points_error.begin();
  compute_residuals_result.first_residual = residuals.begin();
  compute_residuals_result.first_valid_flag = valid_residuals.begin();

  dvo::core::computeResidualsAndValidFlagsSse(first_point, last_point, cur, K, transformation.cast<float>(), wref, wcur, compute_residuals_result, 1.0, 0.0);

  cv::Mat result = cv::Mat::zeros(reference.level(level).intensity.size(), CV_32FC1), debug_idx;

  reference_selection_.getDebugIndex(level, debug_idx);

  uint8_t *valid_pixel_it = debug_idx.ptr<uint8_t>();
  ValidFlagIterator valid_residual_it = compute_residuals_result.first_valid_flag;
  ResidualIterator residual_it = compute_residuals_result.first_residual;

  float *result_it = result.ptr<float>();
  float *result_end = result_it + result.total();

  for(; result_it != result_end; ++result_it)
  {
    if(*valid_pixel_it == 1)
    {
      if(*valid_residual_it == 1)
      {
        *result_it = std::abs(residual_it->coeff(0));

        ++residual_it;
      }
      ++valid_residual_it;
    }
    ++valid_pixel_it;
  }

  reference_selection_.debug(false);

  return result;
}


// jacobian computation
inline void DenseTracker::computeJacobianOfProjectionAndTransformation(const Vector4& p, Matrix2x6& j)
{
  NumType z = 1.0f / p(2);
  NumType z_sqr = 1.0f / (p(2) * p(2));

  j(0, 0) =  z;
  j(0, 1) =  0.0f;
  j(0, 2) = -p(0) * z_sqr;
  j(0, 3) = j(0, 2) * p(1);//j(0, 3) = -p(0) * p(1) * z_sqr;
  j(0, 4) = 1.0f - j(0, 2) * p(0);//j(0, 4) =  (1.0 + p(0) * p(0) * z_sqr);
  j(0, 5) = -p(1) * z;

  j(1, 0) =  0.0f;
  j(1, 1) =  z;
  j(1, 2) = -p(1) * z_sqr;
  j(1, 3) = -1.0f + j(1, 2) * p(1); //j(1, 3) = -(1.0 + p(1) * p(1) * z_sqr);
  j(1, 4) = -j(0, 3); //j(1, 4) =  p(0) * p(1) * z_sqr;
  j(1, 5) =  p(0) * z;
}

inline void DenseTracker::compute3rdRowOfJacobianOfTransformation(const Vector4& p, Vector6& j)
{
  j(0) = 0.0;
  j(1) = 0.0;
  j(2) = 1.0;
  j(3) = p(1);
  j(4) = -p(0);
  j(5) = 0.0;
}

} /* namespace dvo */
