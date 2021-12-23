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

#include <dvo/dense_tracking.h>
#include <Eigen/Eigenvalues>

namespace dvo
{

DenseTracker::Config::Config() :
  FirstLevel(3),
  LastLevel(1),
  MaxIterationsPerLevel(5),
  Precision(0.0001),
  UseInitialEstimate(true),
  UseWeighting(false),
  Mu(0),
  InfluenceFuntionType(dvo::core::InfluenceFunctions::TDistribution),
  InfluenceFunctionParam(dvo::core::TDistributionInfluenceFunction::DEFAULT_DOF),
  ScaleEstimatorType(dvo::core::ScaleEstimators::TDistribution),
  ScaleEstimatorParam(dvo::core::TDistributionScaleEstimator::DEFAULT_DOF),
  IntensityDerivativeThreshold(0.0f),
  DepthDerivativeThreshold(0.0f)
{
}

size_t DenseTracker::Config::getNumLevels() const
{
  return FirstLevel + 1;
}

bool DenseTracker::Config::UseEstimateSmoothing() const
{
  return Mu > 1e-6;
}

bool DenseTracker::Config::IsSane() const
{
  return FirstLevel >= LastLevel;
}

DenseTracker::IterationContext::IterationContext(const Config& cfg) :
    cfg(cfg)
{
}

bool DenseTracker::IterationContext::IsFirstIteration() const
{
  return IsFirstLevel() && IsFirstIterationOnLevel();
}

bool DenseTracker::IterationContext::IsFirstIterationOnLevel() const
{
  return Iteration == 0;
}

bool DenseTracker::IterationContext::IsFirstLevel() const
{
  return cfg.FirstLevel == Level;
}

bool DenseTracker::IterationContext::IsLastLevel() const
{
  return cfg.LastLevel == Level;
}

double DenseTracker::IterationContext::ErrorDiff() const
{
  return LastError - Error;
}

bool DenseTracker::IterationContext::IterationsExceeded() const
{
  int max_iterations = cfg.MaxIterationsPerLevel;

  return Iteration >= max_iterations;
}

bool DenseTracker::Result::isNaN() const
{
  return !std::isfinite(Transformation.matrix().sum()) || !std::isfinite(Information.sum());
}

DenseTracker::Result::Result() :
    LogLikelihood(std::numeric_limits<double>::max())
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  Transformation.linear().setConstant(nan);
  Transformation.translation().setConstant(nan);
  Information.setIdentity();
}

void DenseTracker::Result::setIdentity()
{
  Transformation.setIdentity();
  Information.setIdentity();
  LogLikelihood = 0.0;
}

void DenseTracker::Result::clearStatistics()
{
  Statistics.Levels.clear();
}

void DenseTracker::IterationStats::InformationEigenValues(dvo::core::Vector6d& eigenvalues) const
{
  Eigen::EigenSolver<dvo::core::Matrix6d> evd(EstimateInformation);
  eigenvalues = evd.eigenvalues().real();
  std::sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.rows());
}

double DenseTracker::IterationStats::InformationConditionNumber() const
{
  dvo::core::Vector6d ev;
  InformationEigenValues(ev);

  return std::abs(ev(5) / ev(0));
}


bool DenseTracker::LevelStats::HasIterationWithIncrement() const
{
  int min = TerminationCriterion == DenseTracker::TerminationCriteria::LogLikelihoodDecreased || TerminationCriterion == DenseTracker::TerminationCriteria::TooFewConstraints ? 2 : 1;

  return Iterations.size() >= min;
}

DenseTracker::IterationStats& DenseTracker::LevelStats::LastIterationWithIncrement()
{
  if(!HasIterationWithIncrement())
  {
    std::cerr << "awkward " << *this << std::endl;

    assert(false);
  }

  return TerminationCriterion == DenseTracker::TerminationCriteria::LogLikelihoodDecreased ? Iterations[Iterations.size() - 2] : Iterations[Iterations.size() - 1];
}

DenseTracker::IterationStats& DenseTracker::LevelStats::LastIteration()
{
  return Iterations.back();
}

const DenseTracker::IterationStats& DenseTracker::LevelStats::LastIterationWithIncrement() const
{
  if(!HasIterationWithIncrement())
  {
    std::cerr << "awkward " << *this << std::endl;

    assert(false);
  }
  return TerminationCriterion == DenseTracker::TerminationCriteria::LogLikelihoodDecreased ? Iterations[Iterations.size() - 2] : Iterations[Iterations.size() - 1];
}

const DenseTracker::IterationStats& DenseTracker::LevelStats::LastIteration() const
{
  return Iterations.back();
}

} /* namespace dvo */

