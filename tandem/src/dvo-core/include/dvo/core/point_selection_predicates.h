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

#ifndef POINT_SELECTION_PREDICATES_H_
#define POINT_SELECTION_PREDICATES_H_

namespace dvo
{
namespace core
{

/*
struct PointSelectPredicate
{
  bool operator() (const size_t& x, const size_t& y, const float& z, const float& idx, const float& idy, const float& zdx, const float& zdy) const
  {
    return false;
  }
};

struct ValidPointPredicate
{
  bool operator() (const size_t& x, const size_t& y, const float& z, const float& idx, const float& idy, const float& zdx, const float& zdy) const
  {
    return z == z && zdx == zdx && zdy == zdy;
  }
};

struct ValidPointAndGradientThresholdPredicate
{
  float intensity_threshold;
  float depth_threshold;

  ValidPointAndGradientThresholdPredicate() :
    intensity_threshold(0.0f),
    depth_threshold(0.0f)
  {
  }

  bool operator() (const size_t& x, const size_t& y, const float& z, const float& idx, const float& idy, const float& zdx, const float& zdy) const
  {
    //&& std::abs(zdx) < 0.5 &&  std::abs(zdy) < 0.5
    return z == z && z < 3.0 && zdx == zdx && zdy == zdy && (std::abs(idx) > intensity_threshold || std::abs(idy) > intensity_threshold || std::abs(zdx) > depth_threshold ||  std::abs(zdy) > depth_threshold);
  }
};
template<typename TPredicate>
struct PredicateDebugDecorator
{
public:
  PredicateDebugDecorator() :
    max_x_(0),
    max_y_(0),
    current_rl_(0)
  {
  }

  bool operator() (const size_t& x, const size_t& y, const float& z, const float& idx, const float& idy, const float& zdx, const float& zdy) const
  {
    PredicateDebugDecorator<TPredicate> *me = const_cast<PredicateDebugDecorator<TPredicate>* >(this);

    if(x == 0 && y == 0)
    {
      me->selected_.clear();
      me->max_x_ = 0;me->max_y_ = 0; me->current_rl_ = 0;
    }

    bool success = delegate(x, y, z, idx, idy, zdx, zdy);

    me->max_x_ = std::max(me->max_x_, x);
    me->max_y_ = std::max(me->max_y_, y);

    if(success)
    {
      me->selected_.push_back(me->current_rl_);
      me->current_rl_ = 1;
    }
    else
    {
      me->current_rl_ += 1;
    }

    return success;
  }

  template<typename TMask>
  void toMask(cv::Mat& mask) const
  {
    mask = cv::Mat::zeros(max_y_ + 1, max_x_ + 1, cv::DataType<TMask>::type);

    TMask *mask_ptr = mask.ptr<TMask>();
    TMask one(1);

    for(std::vector<size_t>::const_iterator it = selected_.begin(); it != selected_.end(); ++it)
    {
      mask_ptr += (*it);
      *mask_ptr = one;
    }
  }

  template<typename TImage, typename TIter, typename TFunctor>
  void toImage(const TIter& begin, const TIter& end, cv::Mat& img, const TFunctor& transform) const
  {
    size_t n = std::min(selected_.size(), size_t(end - begin));

    //std::cerr << max_y_ << " " << max_x_ << std::endl;

    img = cv::Mat::zeros(max_y_ + 1, max_x_ + 1, cv::DataType<TImage>::type);

    TImage *img_ptr = img.ptr<TImage>();

    TIter value_it = begin;

    for(std::vector<size_t>::const_iterator it = selected_.begin(); it != selected_.begin() + n; ++it, ++value_it)
    {
      img_ptr += (*it);
      *img_ptr = transform(*value_it);
    }
  }

  TPredicate delegate;
private:
  size_t max_x_, max_y_, current_rl_;
  std::vector<size_t> selected_;

};
*/

} /* namespace core */
} /* namespace dvo */

#endif /* POINT_SELECTION_PREDICATES_H_ */
