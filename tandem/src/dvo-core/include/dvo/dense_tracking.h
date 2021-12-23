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

#ifndef DENSE_TRACKER_H_
#define DENSE_TRACKER_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <dvo/core/datatypes.h>
#include <dvo/core/intrinsic_matrix.h>
#include <dvo/core/rgbd_image.h>
#include <dvo/core/point_selection.h>
#include <dvo/core/point_selection_predicates.h>
#include <dvo/core/least_squares.h>
#include <dvo/core/weight_calculation.h>

namespace dvo
{

class DenseTracker
{
public:
  struct Config
  {
    int FirstLevel, LastLevel;
    int MaxIterationsPerLevel;
    double Precision;
    double Mu; // precision (1/sigma^2) of prior

    bool UseInitialEstimate;
    bool UseWeighting;

    bool UseParallel;

    dvo::core::InfluenceFunctions::enum_t InfluenceFuntionType;
    float InfluenceFunctionParam;

    dvo::core::ScaleEstimators::enum_t ScaleEstimatorType;
    float ScaleEstimatorParam;

    float IntensityDerivativeThreshold;
    float DepthDerivativeThreshold;

    Config();
    size_t getNumLevels() const;

    bool UseEstimateSmoothing() const;

    bool IsSane() const;
  };

  struct TerminationCriteria
  {
    enum Enum
    {
      IterationsExceeded,
      IncrementTooSmall,
      LogLikelihoodDecreased,
      TooFewConstraints,
      NumCriteria
    };
  };

  struct IterationStats
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    size_t Id, ValidConstraints;

    double TDistributionLogLikelihood;
    Eigen::Vector2d TDistributionMean;
    Eigen::Matrix2d TDistributionPrecision;

    double PriorLogLikelihood;

    dvo::core::Vector6d EstimateIncrement;
    dvo::core::Matrix6d EstimateInformation;

    void InformationEigenValues(dvo::core::Vector6d& eigenvalues) const;

    double InformationConditionNumber() const;
  };
  typedef std::vector<IterationStats, Eigen::aligned_allocator<IterationStats> > IterationStatsVector;

  struct LevelStats
  {
    size_t Id, MaxValidPixels, ValidPixels;
    TerminationCriteria::Enum TerminationCriterion;
    IterationStatsVector Iterations;

    bool HasIterationWithIncrement() const;

    IterationStats& LastIterationWithIncrement();
    IterationStats& LastIteration();

    const IterationStats& LastIterationWithIncrement() const;
    const IterationStats& LastIteration() const;
  };
  typedef std::vector<LevelStats> LevelStatsVector;

  struct Stats
  {
    LevelStatsVector Levels;
  };

  struct Result
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    dvo::core::AffineTransformd Transformation;
    dvo::core::Matrix6d Information;
    double LogLikelihood;

    Stats Statistics;

    Result();

    bool isNaN() const;
    void setIdentity();
    void clearStatistics();
  };

  static const Config& getDefaultConfig();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DenseTracker(const Config& cfg = getDefaultConfig());
  DenseTracker(const dvo::DenseTracker& other);

  const Config& configuration() const
  {
    return cfg;
  }

  void configure(const Config& cfg);

  bool match(dvo::core::RgbdImagePyramid& reference, dvo::core::RgbdImagePyramid& current, dvo::core::AffineTransformd& transformation, const int on_level, float& lambda, float affine_a = 1.0f, float affine_b = 0.0f);
  bool match(dvo::core::PointSelection& reference, dvo::core::RgbdImagePyramid& current, dvo::core::AffineTransformd& transformation);

  bool match(dvo::core::RgbdImagePyramid& reference, dvo::core::RgbdImagePyramid& current, dvo::DenseTracker::Result& result, const int on_level, float& lambda, float affine_a = 1.0f, float affine_b = 0.0f);
  bool match(dvo::core::PointSelection& reference, dvo::core::RgbdImagePyramid& current, dvo::DenseTracker::Result& result, const int on_level, float& lambda, float affine_a = 1.0f, float affine_b = 0.0f);

  cv::Mat computeIntensityErrorImage(dvo::core::RgbdImagePyramid& reference, dvo::core::RgbdImagePyramid& current, const dvo::core::AffineTransformd& transformation, size_t level = 0);


  static inline void computeJacobianOfProjectionAndTransformation(const dvo::core::Vector4& p, dvo::core::Matrix2x6& jacobian);

  static inline void compute3rdRowOfJacobianOfTransformation(const dvo::core::Vector4& p, dvo::core::Vector6& j);

  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > ResidualVectorType;
  typedef std::vector<float> WeightVectorType;
private:
  struct IterationContext
  {
    const Config& cfg;

    int Level;
    int Iteration;

    double Error, LastError;

    IterationContext(const Config& cfg);

    // returns true if this is the first iteration
    bool IsFirstIteration() const;

    // returns true if this is the first iteration on the current level
    bool IsFirstIterationOnLevel() const;

    // returns true if this is the first level
    bool IsFirstLevel() const;

    // returns true if this is the last level
    bool IsLastLevel() const;

    bool IterationsExceeded() const;

    // returns LastError - Error
    double ErrorDiff() const;
  };

  Config cfg;

  IterationContext itctx_;

  dvo::core::WeightCalculation weight_calculation_;
  dvo::core::PointSelection reference_selection_;
  dvo::core::ValidPointAndGradientThresholdPredicate selection_predicate_;

  dvo::core::PointWithIntensityAndDepth::VectorType points, points_error;

  ResidualVectorType residuals;
  WeightVectorType weights;
};

} /* namespace dvo */

template<typename CharT, typename Traits>
std::ostream& operator<< (std::basic_ostream<CharT, Traits> &out, const dvo::DenseTracker::Config &config)
{
  out
  << "First Level = " << config.FirstLevel
  << ", Last Level = " << config.LastLevel
  << ", Max Iterations per Level = " << config.MaxIterationsPerLevel
  << ", Precision = " << config.Precision
  << ", Mu = " << config.Mu
  << ", Use Initial Estimate = " << (config.UseInitialEstimate ? "true" : "false")
  << ", Use Weighting = " << (config.UseWeighting ? "true" : "false")
  << ", Scale Estimator = " << dvo::core::ScaleEstimators::str(config.ScaleEstimatorType)
  << ", Scale Estimator Param = " << config.ScaleEstimatorParam
  << ", Influence Function = " << dvo::core::InfluenceFunctions::str(config.InfluenceFuntionType)
  << ", Influence Function Param = " << config.InfluenceFunctionParam
  << ", Intensity Derivative Threshold = " << config.IntensityDerivativeThreshold
  << ", Depth Derivative Threshold = " << config.DepthDerivativeThreshold
  ;

  return out;
}

template<typename CharT, typename Traits>
std::ostream& operator<< (std::basic_ostream<CharT, Traits> &o, const dvo::DenseTracker::IterationStats &s)
{
  o << "Iteration: " << s.Id << " ValidConstraints: " << s.ValidConstraints << " DataLogLikelihood: " << s.TDistributionLogLikelihood << " PriorLogLikelihood: " << s.PriorLogLikelihood << std::endl;

  return o;
}

template<typename CharT, typename Traits>
std::ostream& operator<< (std::basic_ostream<CharT, Traits> &o, const dvo::DenseTracker::LevelStats &s)
{
  std::string termination;

  switch(s.TerminationCriterion)
  {
  case dvo::DenseTracker::TerminationCriteria::IterationsExceeded:
    termination = "IterationsExceeded";
    break;
  case dvo::DenseTracker::TerminationCriteria::IncrementTooSmall:
    termination = "IncrementTooSmall";
    break;
  case dvo::DenseTracker::TerminationCriteria::LogLikelihoodDecreased:
    termination = "LogLikelihoodDecreased";
    break;
  case dvo::DenseTracker::TerminationCriteria::TooFewConstraints:
    termination = "TooFewConstraints";
    break;
  default:
    break;
  }

  o << "Level: " << s.Id << " Pixel: " << s.ValidPixels << "/" << s.MaxValidPixels << " Termination: " << termination << " Iterations: " << s.Iterations.size() << std::endl;

  for(dvo::DenseTracker::IterationStatsVector::const_iterator it = s.Iterations.begin(); it != s.Iterations.end(); ++it)
  {
    o << *it;
  }

  return o;
}

template<typename CharT, typename Traits>
std::ostream& operator<< (std::basic_ostream<CharT, Traits> &o, const dvo::DenseTracker::Stats &s)
{
  o << s.Levels.size() << " levels" << std::endl;

  for(dvo::DenseTracker::LevelStatsVector::const_iterator it = s.Levels.begin(); it != s.Levels.end(); ++it)
  {
    o << *it;
  }

  return o;
}

#endif /* DENSE_TRACKER_H_ */
