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

#ifndef INTRINSIC_MATRIX_H_
#define INTRINSIC_MATRIX_H_

#include <Eigen/Core>

namespace dvo
{
namespace core
{

struct IntrinsicMatrix
{
public:
  struct Hash : std::unary_function<IntrinsicMatrix, std::size_t>
  {
      std::size_t operator()(IntrinsicMatrix const& value) const;
  };

  struct Equal : std::binary_function<IntrinsicMatrix, IntrinsicMatrix, bool>
  {
      bool operator()(IntrinsicMatrix const& left, IntrinsicMatrix const& right) const;
  };

  static IntrinsicMatrix create(float fx, float fy, float ox, float oy);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IntrinsicMatrix() {}

  IntrinsicMatrix(const IntrinsicMatrix& other);

  float fx() const;
  float fy() const;

  float ox() const;
  float oy() const;

  void invertOffset();
  void scale(float factor);

  Eigen::Matrix3f data;
};

} /* namespace core */
} /* namespace dvo */
#endif /* INTRINSIC_MATRIX_H_ */
