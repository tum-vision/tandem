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

#ifndef DENSE_TRACKING_IMPL_H_
#define DENSE_TRACKING_IMPL_H_

#include <dvo/dense_tracking.h>

namespace dvo
{
namespace core
{

typedef PointWithIntensityAndDepth::VectorType::iterator PointIterator;
typedef DenseTracker::ResidualVectorType::iterator ResidualIterator;
typedef DenseTracker::WeightVectorType::iterator WeightIterator;
typedef std::vector<uint8_t>::iterator ValidFlagIterator;

struct ComputeResidualsResult
{
  PointIterator first_point_error;
  PointIterator last_point_error;

  ResidualIterator first_residual;
  ResidualIterator last_residual;

  ValidFlagIterator first_valid_flag;
  ValidFlagIterator last_valid_flag;
};

void computeResiduals(const PointIterator& first_point, const PointIterator& last_point, const RgbdImage& current, const IntrinsicMatrix& intrinsics, const Eigen::Affine3f transform, const Vector8f& reference_weight, const Vector8f& current_weight, ComputeResidualsResult& result);

void computeResidualsSse(const PointIterator& first_point, const PointIterator& last_point, const RgbdImage& current, const IntrinsicMatrix& intrinsics, const Eigen::Affine3f transform, const Vector8f& reference_weight, const Vector8f& current_weight, ComputeResidualsResult& result, const float affine_a, const float affine_b);
void computeResidualsAndValidFlagsSse(const PointIterator& first_point, const PointIterator& last_point, const RgbdImage& current, const IntrinsicMatrix& intrinsics, const Eigen::Affine3f transform, const Vector8f& reference_weight, const Vector8f& current_weight, ComputeResidualsResult& result, const float affine_a, const float affine_b);

float computeCompleteDataLogLikelihood(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const  Eigen::Vector2f& mean, const  Eigen::Matrix2f& precision);

float computeWeightedError(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const  Eigen::Matrix2f& precision);
float computeWeightedErrorSse(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const  Eigen::Matrix2f& precision);

//Eigen::Vector2f computeMean(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight);

Eigen::Matrix2f computeScale(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const Eigen::Vector2f& mean);
Eigen::Matrix2f computeScaleSse(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const Eigen::Vector2f& mean);

void computeWeights(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const Eigen::Vector2f& mean, const Eigen::Matrix2f& precision);
void computeWeightsSse(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, const Eigen::Vector2f& mean, const Eigen::Matrix2f& precision);

void computeMeanScaleAndWeights(const ResidualIterator& first_residual, const ResidualIterator& last_residual, const WeightIterator& first_weight, Eigen::Vector2f& mean, Eigen::Matrix2f& precision);

} /* namespace core */
} /* namespace dvo */
#endif /* DENSE_TRACKING_IMPL_H_ */
