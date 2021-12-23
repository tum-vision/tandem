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

#ifndef MATH_SSE_H_
#define MATH_SSE_H_

#include <Eigen/Core>

namespace dvo
{
namespace core
{

#define HIDE_CTOR(type) \
    private: \
      type() {}; \
      type(const type& other) {}; \
      ~type() {}; \

struct Sse
{
  enum {
    Enabled,
    Disabled
  };
};

/**
 * A 6x6 self adjoint matrix with optimized "rankUpdate(u, scale)" (10x faster than Eigen impl, 1.8x faster than MathSse::addOuterProduct(...)).
 */
class OptimizedSelfAdjointMatrix6x6f
{
public:
  OptimizedSelfAdjointMatrix6x6f();

  void rankUpdate(const Eigen::Matrix<float, 6, 1>& u, const float& alpha);

  void rankUpdate(const Eigen::Matrix<float, 2, 6>& u, const Eigen::Matrix2f& alpha);

  void operator +=(const OptimizedSelfAdjointMatrix6x6f& other);

  void setZero();

  void toEigen(Eigen::Matrix<float, 6, 6>& m) const;
private:
  enum {
    Size = 24
  };
  EIGEN_ALIGN16 float data[Size];
};

template<int Enabled, typename NumType>
class MathSse
{
public:
  static void addOuterProduct(Eigen::Matrix<NumType, 6, 6>& mat, const Eigen::Matrix<NumType, 6, 1>& vec, const NumType& scale);

  static void add(Eigen::Matrix<NumType, 6, 1>& vec, const Eigen::Matrix<NumType, 6, 1>& other, const NumType& scale);

  HIDE_CTOR(MathSse)
};

template<typename NumType>
class MathSse<Sse::Disabled, NumType>
{
public:
  static void addOuterProduct(Eigen::Matrix<NumType, 6, 6>& mat, const Eigen::Matrix<NumType, 6, 1>& vec, const NumType& scale)
  {
    mat += vec * vec.transpose() * scale;
  }

  static void add(Eigen::Matrix<NumType, 6, 1>& vec, const Eigen::Matrix<NumType, 6, 1>& other, const NumType& scale)
  {
    vec += other * scale;
  }

  HIDE_CTOR(MathSse)
};

template<>
void MathSse<Sse::Enabled, float>::addOuterProduct(Eigen::Matrix<float, 6, 6>& mat, const Eigen::Matrix<float, 6, 1>& vec, const float& scale);

template<>
void MathSse<Sse::Enabled, double>::addOuterProduct(Eigen::Matrix<double, 6, 6>& mat, const Eigen::Matrix<double, 6, 1>& vec, const double& scale);

template<>
void MathSse<Sse::Enabled, float>::add(Eigen::Matrix<float, 6, 1>& vec, const Eigen::Matrix<float, 6, 1>& other, const float& scale);

template<>
void MathSse<Sse::Enabled, double>::add(Eigen::Matrix<double, 6, 1>& vec, const Eigen::Matrix<double, 6, 1>& other, const double& scale);

} /* namespace core */
} /* namespace dvo */
#endif /* MATH_SSE_H_ */
