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

#include <dvo/core/least_squares.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <dvo/core/math_sse.h>

static const dvo::core::NumType normalizer = 1.0 / (255.0 * 255.0);
static const dvo::core::NumType normalizer_inverse = 255.0 * 255.0;

// ------ Normal Equations Cholesky ------

dvo::core::NormalEquationsLeastSquares::~NormalEquationsLeastSquares() { }

void dvo::core::NormalEquationsLeastSquares::initialize(const size_t maxnum_constraints)
{
  A.setZero();
  A_opt.setZero();
  b.setZero();
  error = 0;
  this->num_constraints = 0;
  this->maxnum_constraints = maxnum_constraints;
}

void dvo::core::NormalEquationsLeastSquares::update(const dvo::core::Vector6& J, const NumType& res, const NumType& weight)
{
  NumType factor = weight;//weight * normalizer; // what happens without the normalizer? nothing!
  A_opt.rankUpdate(J, factor);
  //MathSse<Sse::Enabled, NumType>::addOuterProduct(A, J, factor);
  //A += J * J.transpose() * factor;
  MathSse<Sse::Enabled, NumType>::add(b, J, -res * factor); // not much difference :(
  //b -= J * res * factor;

  //error += res * res * factor;
  num_constraints += 1;
}

void dvo::core::NormalEquationsLeastSquares::update(const Eigen::Matrix<NumType, 2, 6>& J, const Eigen::Matrix<NumType, 2, 1>& res, const Eigen::Matrix<NumType, 2, 2>& weight)
{
  A_opt.rankUpdate(J, weight);
  b -= J.transpose() * weight * res;

  num_constraints += 1;
}

void dvo::core::NormalEquationsLeastSquares::combine(const dvo::core::NormalEquationsLeastSquares& other)
{
  A_opt += other.A_opt;
  b += other.b;
  //error += other.error;
  num_constraints += other.num_constraints;
}

void dvo::core::NormalEquationsLeastSquares::finish()
{
  A_opt.toEigen(A);
  //A /= (NumType) num_constraints;
  //b /= (NumType) num_constraints;
  //error /= (NumType) num_constraints;
}

void dvo::core::NormalEquationsLeastSquares::solve(dvo::core::Vector6& x)
{
  x = A.ldlt().solve(b);
}

// ------ Normal Equations EVD ------

dvo::core::EvdLeastSquares::~EvdLeastSquares() { }

void dvo::core::EvdLeastSquares::solve(dvo::core::Vector6& x)
{
  // eigen value decomposition seems to be equivalent to SVD for our matrix A
  Eigen::SelfAdjointEigenSolver<dvo::core::Matrix6x6> eigensolver(A);
  dvo::core::Vector6 eigenvalues = eigensolver.eigenvalues();
  dvo::core::Matrix6x6 eigenvectors = eigensolver.eigenvectors();

  bool singular = false;

  for(int i = 0; i < 6; ++i)
  {
    if(eigenvalues(i) < 0.05)
    {
      singular = true;
      throw std::exception();
    }
    else
    {
      eigenvalues(i) = 1.0 / eigenvalues(i);
    }
  }

  x = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose() * b;
}

// ------ SVD ------

dvo::core::SvdLeastSquares::~SvdLeastSquares() { }

void dvo::core::SvdLeastSquares::initialize(const size_t maxnum_constraints)
{
  J.resize(maxnum_constraints, Eigen::NoChange);
  residuals.resize(maxnum_constraints, Eigen::NoChange);

  current = 0;
}

void dvo::core::SvdLeastSquares::update(const dvo::core::Vector6& J, const NumType& res, const NumType& weight)
{
  this->J.row(current) = J;
  this->residuals(current) = res;

  current += 1;
}

void dvo::core::SvdLeastSquares::finish()
{
  J.conservativeResize(current, Eigen::NoChange);
  residuals.conservativeResize(current);
}

void dvo::core::SvdLeastSquares::solve(dvo::core::Vector6& x)
{
  x = J.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(residuals);
}

// ------ Precomputed Normal Equations Cholesky ------

dvo::core::PrecomputedNormalEquationsLeastSquares::~PrecomputedNormalEquationsLeastSquares() { }

void dvo::core::PrecomputedNormalEquationsLeastSquares::initialize(const size_t maxnum_constraints)
{
  hessian_.setZero();
  jacobian_cache_.resize(Eigen::NoChange, maxnum_constraints);
  mask_ = cv::Mat1b::zeros(maxnum_constraints, 1);

  this->num_constraints = 0;
  this->maxnum_constraints = maxnum_constraints;
}

void dvo::core::PrecomputedNormalEquationsLeastSquares::reset()
{
  A = hessian_;
  hessian_error_.setZero();
  b.setZero();
  error = 0;

  mask_ptr_ = mask_.ptr();
  num_constraints = 0;
}
void dvo::core::PrecomputedNormalEquationsLeastSquares::next()
{
  ++mask_ptr_;
}

void dvo::core::PrecomputedNormalEquationsLeastSquares::addConstraint(const size_t& idx, const dvo::core::Vector6& J)
{
  //hessian_cache_.block<6, 6>(idx * 6, 0) = J * J.transpose() * normalizer;
  //hessian_cache_[idx] = J * J.transpose() * normalizer;

  //hessian_ += J * J.transpose() * normalizer;
  MathSse<Sse::Enabled, NumType>::addOuterProduct(hessian_, J, normalizer);

  jacobian_cache_.col(idx) = J * normalizer;
  mask_.at<uchar>(idx) = 1;
}

void dvo::core::PrecomputedNormalEquationsLeastSquares::ignoreConstraint(const size_t& idx)
{
  if((*mask_ptr_) == 0) return;

  const Vector6& J = jacobian_cache_.col(idx);

  /**
   *  J is already multiplied with normalizer, so it is:
   *
   *    J' * J'.transpose() * normalizer * normalizer
   *
   *  multiplying with normalizer_inverse gives:
   *
   *    J' * J'.transpose() * normalizer * normalizer * normalizer_inverse
   *
   *  resulting in:
   *
   *    J' * J'.transpose() * normalizer
   */
  hessian_error_ -= J * J.transpose() * normalizer_inverse;
}

bool dvo::core::PrecomputedNormalEquationsLeastSquares::setResidualForConstraint(const size_t& idx, const NumType& res, const NumType& weight)
{
  if(*mask_ptr_ == 0) return false;

  //A += *hessian_cache_it_;
  b -= jacobian_cache_.col(idx) * res * weight;

  error += res * res * weight * normalizer;
  this->num_constraints +=1;

  return true;
}

void dvo::core::PrecomputedNormalEquationsLeastSquares::finish()
{
  A += hessian_error_;
  A /= (double) num_constraints;
  b /= (double) num_constraints;
  error /= (double) num_constraints;
}

void dvo::core::PrecomputedNormalEquationsLeastSquares::solve(dvo::core::Vector6 & x)
{
  x = A.ldlt().solve(b);
}
