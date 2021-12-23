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

#ifndef POINT_SELECTION_H_
#define POINT_SELECTION_H_

#include <dvo/core/intrinsic_matrix.h>
#include <dvo/core/rgbd_image.h>

namespace dvo
{
namespace core
{

class PointSelectionPredicate
{
public:
  virtual ~PointSelectionPredicate() {}
  virtual bool isPointOk(const size_t& x, const size_t& y, const float& z, const float& idx, const float& idy, const float& zdx, const float& zdy) const = 0;
};

class ValidPointPredicate : public PointSelectionPredicate
{
public:
  virtual ~ValidPointPredicate() {}
  virtual bool isPointOk(const size_t& x, const size_t& y, const float& z, const float& idx, const float& idy, const float& zdx, const float& zdy) const
  {
    return z == z && zdx == zdx && zdy == zdy;
  }
};

class ValidPointAndGradientThresholdPredicate : public PointSelectionPredicate
{
public:
  float intensity_threshold;
  float depth_threshold;

  ValidPointAndGradientThresholdPredicate() :
    intensity_threshold(0.0f),
    depth_threshold(0.0f)
  {
  }

  virtual ~ValidPointAndGradientThresholdPredicate() {}

  virtual bool isPointOk(const size_t& x, const size_t& y, const float& z, const float& idx, const float& idy, const float& zdx, const float& zdy) const
  {
    return z == z && zdx == zdx && zdy == zdy && (std::abs(idx) > intensity_threshold || std::abs(idy) > intensity_threshold || std::abs(zdx) > depth_threshold ||  std::abs(zdy) > depth_threshold);
  }
};

class PointSelection
{
public:
  typedef PointWithIntensityAndDepth::VectorType PointVector;
  typedef PointVector::iterator PointIterator;

  PointSelection(const PointSelectionPredicate& predicate);
  PointSelection(dvo::core::RgbdImagePyramid& pyramid, const PointSelectionPredicate& predicate);
  virtual ~PointSelection();

  dvo::core::RgbdImagePyramid& getRgbdImagePyramid();

  void setRgbdImagePyramid(dvo::core::RgbdImagePyramid& pyramid);

  size_t getMaximumNumberOfPoints(const size_t& level);

  void select(const size_t& level, PointIterator& first_point, PointIterator& last_point);

  void recycle(dvo::core::RgbdImagePyramid& pyramid);

  bool getDebugIndex(const size_t& level, cv::Mat& dbg_idx);

  void debug(bool v)
  {
    debug_ = v;
  }

  bool debug() const
  {
    return debug_;
  }

private:
  struct Storage
  {
  public:
    PointVector points;
    PointIterator points_end;
    bool is_cached;

    cv::Mat debug_idx;

    Storage();
    void allocate(size_t max_points);
  };

  dvo::core::RgbdImagePyramid *pyramid_;
  std::vector<Storage> storage_;
  const PointSelectionPredicate& predicate_;

  bool debug_;

  PointIterator selectPointsFromImage(const dvo::core::RgbdImage& img, const PointIterator& first_point, const PointIterator& last_point, cv::Mat& debug_idx);
};

} /* namespace core */
} /* namespace dvo */

#endif /* POINT_SELECTION_H_ */
