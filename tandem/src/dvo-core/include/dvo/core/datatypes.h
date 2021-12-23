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

#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace dvo
{
namespace core
{

typedef float IntensityType;
static const IntensityType Invalid = std::numeric_limits<IntensityType>::quiet_NaN();

typedef float DepthType;
static const DepthType InvalidDepth = std::numeric_limits<DepthType>::quiet_NaN();

// float/double, determines numeric precision
typedef float NumType;

typedef Eigen::Matrix<NumType, 6, 6> Matrix6x6;
typedef Eigen::Matrix<NumType, 1, 2> Matrix1x2;
typedef Eigen::Matrix<NumType, 2, 6> Matrix2x6;

typedef Eigen::Matrix<NumType, 6, 1> Vector6;
typedef Eigen::Matrix<NumType, 4, 1> Vector4;

typedef Eigen::Transform<NumType,3, Eigen::Affine> AffineTransform;

typedef Eigen::Affine3d AffineTransformd;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

} /* namespace core */
} /* namespace dvo */

#endif /* DATATYPES_H_ */
