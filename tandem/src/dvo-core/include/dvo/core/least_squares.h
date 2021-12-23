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

#ifndef LEAST_SQUARES_H_
#define LEAST_SQUARES_H_

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include <dvo/core/datatypes.h>
#include <dvo/core/math_sse.h>

namespace dvo
{
namespace core
{

/**
 * Basic interface for algorithms solving 1 step of non-linear least squares.
 */
class LeastSquaresInterface
{
public:
  virtual ~LeastSquaresInterface() {};
  virtual void initialize(const size_t maxnum_constraints) = 0;
  virtual void update(const Vector6& J, const NumType& res, const NumType& weight = 1.0f) = 0;
  virtual void update(const Eigen::Matrix<NumType, 2, 6>& J, const Eigen::Matrix<NumType, 2, 1>& res, const Eigen::Matrix<NumType, 2, 2>& weight) {};
  virtual void finish() = 0;
  virtual void solve(Vector6& x) = 0;
};

/**
 * Basic interface for algorithms solving 1 step of non-linear least squares where jacobians can be precomputed and don't vary between iterations.
 */
class PrecomputedLeastSquaresInterface
{
public:
  virtual ~PrecomputedLeastSquaresInterface() {};
  virtual void initialize(const size_t maxnum_constraints) = 0;
  // add a jacobian to the cache
  virtual void addConstraint(const size_t& idx, const Vector6& J) = 0;

  // resets internal state created for one iteration
  virtual void reset() = 0;

  virtual void next() = 0;

  virtual void ignoreConstraint(const size_t& idx) = 0;
  virtual bool setResidualForConstraint(const size_t& idx, const NumType& res, const NumType& weight = 1.0f) = 0;
  virtual void finish() = 0;
  virtual void solve(Vector6& x) = 0;
};

/**
 * Builds normal equations and solves them with Cholesky decomposition.
 */
class NormalEquationsLeastSquares : public LeastSquaresInterface
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  OptimizedSelfAdjointMatrix6x6f A_opt;
  Matrix6x6 A;
  Vector6 b;

  double error;
  size_t maxnum_constraints, num_constraints;

  virtual ~NormalEquationsLeastSquares();

  virtual void initialize(const size_t maxnum_constraints);
  virtual void update(const Vector6& J, const NumType& res, const NumType& weight = 1.0f);
  virtual void update(const Eigen::Matrix<NumType, 2, 6>& J, const Eigen::Matrix<NumType, 2, 1>& res, const Eigen::Matrix<NumType, 2, 2>& weight);
  virtual void finish();
  virtual void solve(Vector6& x);

  void combine(const NormalEquationsLeastSquares& other);
};

/**
 * Builds normal equations and solves them with Cholesky decomposition.
 */
class PrecomputedNormalEquationsLeastSquares : public PrecomputedLeastSquaresInterface
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Matrix6x6 A;
  Vector6 b;

  double error;

  size_t maxnum_constraints, num_constraints;

  virtual ~PrecomputedNormalEquationsLeastSquares();

  virtual void initialize(const size_t maxnum_constraints);
  virtual void addConstraint(const size_t& idx, const Vector6& J);

  virtual void reset();
  virtual void next();

  virtual void ignoreConstraint(const size_t& idx);

  virtual bool setResidualForConstraint(const size_t& idx, const NumType& res, const NumType& weight = 1.0f);
  virtual void finish();
  virtual void solve(Vector6& x);

  const uchar* mask_ptr_;
private:

  Matrix6x6 hessian_;
  Matrix6x6 hessian_error_;
  Eigen::Matrix<NumType, 6, Eigen::Dynamic, Eigen::ColMajor> jacobian_cache_;
  cv::Mat1b mask_;
};

/**
 * Same as NormalEquationsLeastSquares, but solves normal equations with EigenValueDecomposition.
 */
class EvdLeastSquares : public NormalEquationsLeastSquares
{
public:
  virtual ~EvdLeastSquares();

  virtual void solve(Vector6& x);
};

class SvdLeastSquares : public LeastSquaresInterface
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Eigen::Matrix<NumType, Eigen::Dynamic, 6> J;
  Eigen::Matrix<NumType, Eigen::Dynamic, 1> residuals;

  virtual ~SvdLeastSquares();

  virtual void initialize(const size_t maxnum_constraints);
  virtual void update(const Vector6& J, const NumType& res, const NumType& weight = 1.0f);
  virtual void finish();
  virtual void solve(Vector6& x);
private:
  int current;
};

} /* namespace core */
} /* namespace dvo */
#endif /* LEAST_SQUARES_H_ */
