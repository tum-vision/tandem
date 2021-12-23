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

#ifndef WEIGHT_CALCULATION_H_
#define WEIGHT_CALCULATION_H_

#include <opencv2/opencv.hpp>

#include <dvo/core/datatypes.h>
#include <dvo/util/fluent_interface.h>

namespace dvo
{
namespace core
{

// interface for scale estimators
class ScaleEstimator
{
public:
  virtual ~ScaleEstimator() {};
  virtual float compute(const cv::Mat& errors) const = 0;
  virtual void configure(const float& param) {};
};

class UnitScaleEstimator : public ScaleEstimator
{
public:
  UnitScaleEstimator() {}
  virtual ~UnitScaleEstimator() {}
  virtual float compute(const cv::Mat& errors) const { return 1.0f; };
};

// estimates scale by fitting a t-distribution to the data with the given degrees of freedom
class TDistributionScaleEstimator : public ScaleEstimator
{
public:
  TDistributionScaleEstimator(const float dof = DEFAULT_DOF);
  virtual ~TDistributionScaleEstimator() {};
  virtual float compute(const cv::Mat& errors) const;
  virtual void configure(const float& param);

  static const float DEFAULT_DOF;
  static const float INITIAL_SIGMA;
protected:
  float dof;
  float initial_sigma;
};

class OptimizedTDistributionScaleEstimator : public TDistributionScaleEstimator
{
public:
  OptimizedTDistributionScaleEstimator(const float dof = DEFAULT_DOF);
  virtual ~OptimizedTDistributionScaleEstimator() {};

  virtual float compute(const cv::Mat& errors) const;
};

class ApproximateTDistributionScaleEstimator : public TDistributionScaleEstimator
{
public:
  ApproximateTDistributionScaleEstimator(const float dof = DEFAULT_DOF);
  virtual ~ApproximateTDistributionScaleEstimator() {};
  virtual float compute(const cv::Mat& errors) const;
};

// estimates scale by computing the median absolute deviation
class MADScaleEstimator : public ScaleEstimator
{
public:
  MADScaleEstimator();
  virtual ~MADScaleEstimator() {};
  virtual float compute(const cv::Mat& errors) const;

private:
  // 1 / 0.6745
  static const float NORMALIZER;// = 1.48f;
};

// estimates scale by computing the standard deviation
class NormalDistributionScaleEstimator : public ScaleEstimator
{
public:
  NormalDistributionScaleEstimator();
  virtual ~NormalDistributionScaleEstimator() {};
  virtual float compute(const cv::Mat& errors) const;
private:
};

struct ScaleEstimators {
  typedef enum {
    Unit,
    NormalDistribution,
    TDistribution,
    MAD
    // don't forget to add to dynamic reconfigure!
  } enum_t;

  static const char* str(enum_t type);

  static ScaleEstimator* get(enum_t type);
};

/**
 * Interface for influence functions. An influence function is the first derivative of a symmetric robust function p(sqrt(x)).
 * The errors are assumed to be normalized to unit variance.
 *
 * See:
 *   "Lucas-Kanade 20 Years On: A Unifying Framework: Part 2"
 */
// TODO: rename to WeightFunction, this is what it really is!!!
class InfluenceFunction
{
public:
  virtual ~InfluenceFunction() {};
  virtual float value(const float& x) const = 0;
  virtual void configure(const float& param) {};
};

class UnitInfluenceFunction : public InfluenceFunction
{
public:
  UnitInfluenceFunction() {};
  virtual ~UnitInfluenceFunction() {};
  virtual inline float value(const float& x) const { return 1.0f; };
};

/**
 * Tukey's hard re-descending function.
 *
 * See:
 *   http://en.wikipedia.org/wiki/Redescending_M-estimator
 */
class TukeyInfluenceFunction : public InfluenceFunction
{
public:
  TukeyInfluenceFunction(const float b = DEFAULT_B);
  virtual ~TukeyInfluenceFunction() {};
  virtual inline float value(const float& x) const;
  virtual void configure(const float& param);

  static const float DEFAULT_B;
private:
  float b_square;
};

class TDistributionInfluenceFunction : public InfluenceFunction
{
public:
  TDistributionInfluenceFunction(const float dof = DEFAULT_DOF);
  virtual ~TDistributionInfluenceFunction() {};
  virtual inline float value(const float& x) const;
  virtual void configure(const float& param);

  static const float DEFAULT_DOF;
private:
  float dof;
  float normalizer;
};

class HuberInfluenceFunction : public InfluenceFunction
{
public:
  HuberInfluenceFunction(const float k = DEFAULT_K);
  virtual ~HuberInfluenceFunction() {};
  virtual inline float value(const float& x) const;
  virtual void configure(const float& param);

  static const float DEFAULT_K;
private:
  float k;
};

struct InfluenceFunctions {
  typedef enum {
    Unit,
    Tukey,
    TDistribution,
    Huber,
    // don't forget to add to dynamic reconfigure!
  } enum_t;

  static const char* str(enum_t type);

  static InfluenceFunction* get(enum_t type);
};

class WeightCalculation
{
public:
  WeightCalculation();
  FI_ATTRIBUTE(WeightCalculation, ScaleEstimator*, scaleEstimator);
  FI_ATTRIBUTE(WeightCalculation, InfluenceFunction*, influenceFunction);

  void calculateScale(const cv::Mat& errors);

  float calculateWeight(const float error) const;

  void calculateWeights(const cv::Mat& errors, cv::Mat& weights);

  FI_ATTRIBUTE(WeightCalculation, float, scale);
private:
};

} /* namespace core */
} /* namespace dvo */
#endif /* WEIGHT_CALCULATION_H_ */
