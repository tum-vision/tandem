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

#include <dvo/core/intrinsic_matrix.h>

#include <boost/unordered_map.hpp>

namespace dvo
{
namespace core
{

std::size_t IntrinsicMatrix::Hash::operator ()(IntrinsicMatrix const& value) const
{
    std::size_t seed = 0;

    boost::hash_combine(seed, value.fx());
    boost::hash_combine(seed, value.fy());
    boost::hash_combine(seed, value.ox());
    boost::hash_combine(seed, value.oy());

    return seed;
}

bool IntrinsicMatrix::Equal::operator()(IntrinsicMatrix const& left, IntrinsicMatrix const& right) const
{
    return left.fx() == right.fx() && left.fy() == right.fy() && left.ox() == right.ox() && left.oy() == right.oy();
}

IntrinsicMatrix IntrinsicMatrix::create(float fx, float fy, float ox, float oy)
{
  IntrinsicMatrix result;
  result.data.setZero();
  result.data(0, 0) = fx;
  result.data(1, 1) = fy;
  result.data(2, 2) = 1.0f;
  result.data(0, 2) = ox;
  result.data(1, 2) = oy;

  return result;
}

IntrinsicMatrix::IntrinsicMatrix(const IntrinsicMatrix & other) : data(other.data)
{
}

float IntrinsicMatrix::fx() const
{
  return data(0, 0);
}

float IntrinsicMatrix::fy() const
{
  return data(1, 1);
}

float IntrinsicMatrix::ox() const
{
  return data(0, 2);
}

float IntrinsicMatrix::oy() const
{
  return data(1, 2);
}

void IntrinsicMatrix::invertOffset()
{
  data(0, 2) *= -1;
  data(1, 2) *= -1;
}

void IntrinsicMatrix::scale(float factor)
{
  data *= factor;
}

} /* namespace core */
} /* namespace dvo */
