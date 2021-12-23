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

#include <dvo/core/rgbd_image.h>

#include <assert.h>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <dvo/core/interpolation.h>

//#include "../util.h"
//#include "../stopwatch.h"

namespace dvo
{
namespace core
{

template<typename T>
static void pyrDownMeanSmooth(const cv::Mat& in, cv::Mat& out)
{
  out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

  for(int y = 0; y < out.rows; ++y)
  {
    for(int x = 0; x < out.cols; ++x)
    {
      int x0 = x * 2;
      int x1 = x0 + 1;
      int y0 = y * 2;
      int y1 = y0 + 1;

      out.at<T>(y, x) = (T) ( (in.at<T>(y0, x0) + in.at<T>(y0, x1) + in.at<T>(y1, x0) + in.at<T>(y1, x1)) / 4.0f );
    }
  }
}

template<typename T>
static void pyrDownMeanSmoothIgnoreInvalid(const cv::Mat& in, cv::Mat& out)
{
  out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

  for(int y = 0; y < out.rows; ++y)
  {
    for(int x = 0; x < out.cols; ++x)
    {
      int x0 = x * 2;
      int x1 = x0 + 1;
      int y0 = y * 2;
      int y1 = y0 + 1;

      T total = 0;
      int cnt = 0;

      if(std::isfinite(in.at<T>(y0, x0)))
      {
        total += in.at<T>(y0, x0);
        cnt++;
      }

      if(std::isfinite(in.at<T>(y0, x1)))
      {
        total += in.at<T>(y0, x1);
        cnt++;
      }

      if(std::isfinite(in.at<T>(y1, x0)))
      {
        total += in.at<T>(y1, x0);
        cnt++;
      }

      if(std::isfinite(in.at<T>(y1, x1)))
      {
        total += in.at<T>(y1, x1);
        cnt++;
      }

      if(cnt > 0)
      {
        out.at<T>(y, x) = (T) ( total / cnt );
      }
      else
      {
        out.at<T>(y, x) = InvalidDepth;
      }
    }
  }
}

template<typename T>
static void pyrDownMedianSmooth(const cv::Mat& in, cv::Mat& out)
{
  out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

  cv::Mat in_smoothed;
  cv::medianBlur(in, in_smoothed, 3);

  for(int y = 0; y < out.rows; ++y)
  {
    for(int x = 0; x < out.cols; ++x)
    {
      out.at<T>(y, x) = in_smoothed.at<T>(y * 2, x * 2);
    }
  }
}

template<typename T>
static void pyrDownSubsample(const cv::Mat& in, cv::Mat& out)
{
  out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

  for(int y = 0; y < out.rows; ++y)
  {
    for(int x = 0; x < out.cols; ++x)
    {
      out.at<T>(y, x) = in.at<T>(y * 2, x * 2);
    }
  }
}

RgbdImagePyramid::RgbdImagePyramid(RgbdCameraPyramid& camera, const cv::Mat& intensity, const cv::Mat& depth) :
    camera_(camera)
{
  levels_.push_back(camera_.level(0).create(intensity, depth));
}

RgbdImagePyramid::~RgbdImagePyramid()
{
}

void RgbdImagePyramid::compute(const size_t num_levels)
{
  build(num_levels);
}

void RgbdImagePyramid::build(const size_t num_levels)
{
  if(levels_.size() >= num_levels) return;

  // if we already have some levels, we just need to compute the coarser levels
  size_t first = levels_.size();

  for(size_t idx = first; idx < num_levels; ++idx)
  {
    levels_.push_back(camera_.level(idx).create());

    pyrDownMeanSmooth<IntensityType>(levels_[idx - 1]->intensity, levels_[idx]->intensity);
    //pyrDownMeanSmoothIgnoreInvalid<float>(levels[idx - 1].depth, levels[idx].depth);
    pyrDownSubsample<float>(levels_[idx - 1]->depth, levels_[idx]->depth);
    levels_[idx]->initialize();
  }
}

RgbdImage& RgbdImagePyramid::level(size_t idx)
{
  assert(idx < levels_.size());

  return *levels_[idx];
}

double RgbdImagePyramid::timestamp() const
{
  return !levels_.empty() ? levels_[0]->timestamp: 0.0;
}

RgbdCamera::RgbdCamera(size_t width, size_t height, const IntrinsicMatrix& intrinsics) :
    width_(width),
    height_(height),
    intrinsics_(intrinsics)
{
  pointcloud_template_.resize(Eigen::NoChange, width_ * height_);
  int idx = 0;

  for(size_t y = 0; y < height_; ++y)
  {
    for(size_t x = 0; x < width_; ++x, ++idx)
    {
      pointcloud_template_(0, idx) = (x - intrinsics_.ox()) / intrinsics_.fx();
      pointcloud_template_(1, idx) = (y - intrinsics_.oy()) / intrinsics_.fy();
      pointcloud_template_(2, idx) = 1.0;
      pointcloud_template_(3, idx) = 0.0;
    }
  }
}

RgbdCamera::~RgbdCamera()
{
}

size_t RgbdCamera::width() const
{
  return width_;
}

size_t RgbdCamera::height() const
{
  return height_;
}

const dvo::core::IntrinsicMatrix& RgbdCamera::intrinsics() const
{
  return intrinsics_;
}

RgbdImagePtr RgbdCamera::create(const cv::Mat& intensity, const cv::Mat& depth) const
{
  RgbdImagePtr result(new RgbdImage(*this));
  result->intensity = intensity;
  result->depth = depth;
  result->initialize();

  return result;
}

RgbdImagePtr RgbdCamera::create() const
{
  return boost::make_shared<RgbdImage>(*this);
}

bool RgbdCamera::hasSameSize(const cv::Mat& img) const
{
  return img.cols == width_ && img.rows == height_;
}

void RgbdCamera::buildPointCloud(const cv::Mat &depth, PointCloud& pointcloud) const
{
  assert(hasSameSize(depth));

  pointcloud.resize(Eigen::NoChange, width_ * height_);

  const float* depth_ptr = depth.ptr<float>();
  int idx = 0;

  for(size_t y = 0; y < height_; ++y)
  {
    for(size_t x = 0; x < width_; ++x, ++depth_ptr, ++idx)
    {
      pointcloud.col(idx) = pointcloud_template_.col(idx) * (*depth_ptr);
      pointcloud(3, idx) = 1.0;
    }
  }
}

RgbdCameraPyramid::RgbdCameraPyramid(const RgbdCamera& base)
{
  levels_.push_back(boost::make_shared<RgbdCamera>(base));
}

RgbdCameraPyramid::RgbdCameraPyramid(size_t base_width, size_t base_height, const dvo::core::IntrinsicMatrix& base_intrinsics)
{
  levels_.push_back(boost::make_shared<RgbdCamera>(base_width, base_height, base_intrinsics));
}

RgbdCameraPyramid::~RgbdCameraPyramid()
{
}

RgbdImagePyramidPtr RgbdCameraPyramid::create(const cv::Mat& base_intensity, const cv::Mat& base_depth)
{
  return RgbdImagePyramidPtr(new RgbdImagePyramid(*this, base_intensity, base_depth));
}

void RgbdCameraPyramid::build(size_t levels)
{
  size_t start = levels_.size();

  for(size_t idx = start; idx < levels; ++idx)
  {
    RgbdCameraPtr& previous = levels_[idx - 1];

    dvo::core::IntrinsicMatrix intrinsics(previous->intrinsics());
    intrinsics.scale(0.5f);

    levels_.push_back(boost::make_shared<RgbdCamera>(previous->width() / 2, previous->height() / 2, intrinsics));
  }
}

const RgbdCamera& RgbdCameraPyramid::level(size_t level)
{
  build(level + 1);

  return *levels_[level];
}

const RgbdCamera& RgbdCameraPyramid::level(size_t level) const
{
  return *levels_[level];
}
/*
RgbdImage::RgbdImage() :
  camera_(0),
  intensity_requires_calculation_(true),
  depth_requires_calculation_(true),
  pointcloud_requires_build_(true),
  width(0),
  height(0)
{
}
*/
RgbdImage::RgbdImage(const RgbdCamera& camera) :
  camera_(camera),
  intensity_requires_calculation_(true),
  depth_requires_calculation_(true),
  pointcloud_requires_build_(true),
  width(0),
  height(0)
{
}

RgbdImage::~RgbdImage()
{
}

const RgbdCamera& RgbdImage::camera() const
{
  return camera_;
}

void RgbdImage::initialize()
{
  assert(hasIntensity() || hasDepth());

  if(hasIntensity() && hasDepth())
  {
    assert(intensity.size() == depth.size());
  }

  if(hasIntensity())
  {
    assert(intensity.type() == cv::DataType<IntensityType>::type && intensity.channels() == 1);
    width = intensity.cols;
    height = intensity.rows;
  }
  if(hasDepth())
  {
    assert(depth.type() == cv::DataType<DepthType>::type && depth.channels() == 1);
    width = depth.cols;
    height = depth.rows;
  }

  intensity_requires_calculation_ = true;
  depth_requires_calculation_ = true;
  pointcloud_requires_build_ = true;
}

bool RgbdImage::hasIntensity() const
{
  return !intensity.empty();
}

bool RgbdImage::hasRgb() const
{
  return !rgb.empty();
}

bool RgbdImage::hasDepth() const
{
  return !depth.empty();
}

void RgbdImage::calculateDerivatives()
{
  calculateIntensityDerivatives();
  calculateDepthDerivatives();
}

bool RgbdImage::calculateIntensityDerivatives()
{
  if(!intensity_requires_calculation_) return false;

  assert(hasIntensity());

  calculateDerivativeX<IntensityType>(intensity, intensity_dx);
  //calculateDerivativeY<IntensityType>(intensity, intensity_dy);
  calculateDerivativeYSseFloat(intensity, intensity_dy);
  /*
  cv::Mat dy_ref, diff;
  calculateDerivativeY<IntensityType>(intensity, dy_ref);
  cv::absdiff(dy_ref, intensity_dy, diff);
  tracker::util::show("diff", diff);
  cv::waitKey(0);
   */
  intensity_requires_calculation_ = false;
  return true;
}

void RgbdImage::calculateDepthDerivatives()
{
  if(!depth_requires_calculation_) return;

  assert(hasDepth());

  calculateDerivativeX<DepthType>(depth, depth_dx);
  calculateDerivativeY<DepthType>(depth, depth_dy);

  depth_requires_calculation_ = false;
}

template<typename T>
void RgbdImage::calculateDerivativeX(const cv::Mat& img, cv::Mat& result)
{
  result.create(img.size(), img.type());

  for(int y = 0; y < img.rows; ++y)
  {
    for(int x = 0; x < img.cols; ++x)
    {
      int prev = std::max(x - 1, 0);
      int next = std::min(x + 1, img.cols - 1);

      result.at<T>(y, x) = (T) (img.at<T>(y, next) - img.at<T>(y, prev)) * 0.5f;
    }
  }

  //cv::Sobel(img, result, -1, 1, 0, 3, 1.0f / 4.0f, 0, cv::BORDER_REPLICATE);

  // compiler auto-vectorization
  /*
  const float* img_ptr = img.ptr<float>();
  float* result_ptr = result.ptr<float>();

  for(int y = 0; y < img.rows; ++y)
  {
    *result_ptr++ = img_ptr[1] - img_ptr[0];

    for(int x = 1; x < img.cols - 1; ++x, ++img_ptr)
    {
      *result_ptr++ = img_ptr[2] - img_ptr[0];
    }

    *result_ptr++ = img_ptr[1] - img_ptr[0];

    img_ptr++;
  }
   */
}

template<typename T>
void RgbdImage::calculateDerivativeY(const cv::Mat& img, cv::Mat& result)
{
  result.create(img.size(), img.type());

  for(int y = 0; y < img.rows; ++y)
  {
    for(int x = 0; x < img.cols; ++x)
    {
      int prev = std::max(y - 1, 0);
      int next = std::min(y + 1, img.rows - 1);

      result.at<T>(y, x) = (T) (img.at<T>(next, x) - img.at<T>(prev, x)) * 0.5f;
    }
  }
  //cv::Sobel(img, result, -1, 0, 1, 3, 1.0f / 4.0f, 0, cv::BORDER_REPLICATE);

  // compiler auto-vectorization
  /*
  for(int y = 0; y < img.rows; ++y)
  {
    const float* prev_row = img.ptr<float>(std::max(y - 1, 0), 0);
    const float* next_row = img.ptr<float>(std::min(y + 1, img.rows - 1), 0);
    float* cur_row = result.ptr<float>(y, 0);

    for(int x = 0; x < img.cols; ++x)
    {
      *cur_row++ = *next_row++ - *prev_row++;
    }
  }
   */
}

void RgbdImage::buildPointCloud()
{
  if(!pointcloud_requires_build_) return;

  assert(hasDepth());

  camera_.buildPointCloud(depth, pointcloud);

  pointcloud_requires_build_ = false;
}

void RgbdImage::calculateNormals()
{
  if(angles.total() == 0)
  {
    normals = cv::Mat::zeros(depth.size(), CV_32FC4);
    angles.create(depth.size(), CV_32FC1);

    float *angle_ptr = angles.ptr<float>();
    cv::Vec4f *normal_ptr = normals.ptr<cv::Vec4f>();

    int x_max = depth.cols - 1;
    int y_max = depth.rows - 1;

    for(int y = 0; y < depth.rows; ++y)
    {
      for(int x = 0; x < depth.cols; ++x, ++angle_ptr, ++normal_ptr)
      {
        int idx1 = y * depth.cols + std::max(x-1, 0);
        int idx2 = y * depth.cols + std::min(x+1, x_max);
        int idx3 = std::max(y-1, 0) * depth.cols + x;
        int idx4 = std::min(y+1, y_max) * depth.cols + x;

        Eigen::Vector4f::AlignedMapType n(normal_ptr->val);
        n = (pointcloud.col(idx2) - pointcloud.col(idx1)).cross3(pointcloud.col(idx4) - pointcloud.col(idx3));
        n.normalize();

        *angle_ptr = std::abs(n(2));
      }
    }
  }
}

void RgbdImage::buildAccelerationStructure()
{
  if(acceleration.total() == 0)
  {
    calculateDerivatives();
    cv::Mat zeros = cv::Mat::zeros(intensity.size(), intensity.type());
    cv::Mat channels[8] = { intensity, depth, intensity_dx, intensity_dy, depth_dx, depth_dy, zeros, zeros};
    cv::merge(channels, 8, acceleration);
  }
}

void RgbdImage::warpIntensity(const AffineTransform& transformationd, const PointCloud& reference_pointcloud, const IntrinsicMatrix& intrinsics, RgbdImage& result, PointCloud& transformed_pointcloud)
{
  Eigen::Affine3f transformation = transformationd.cast<float>();

  cv::Mat warped_image(intensity.size(), intensity.type());
  cv::Mat warped_depth(depth.size(), depth.type());

  float ox = intrinsics.ox();
  float oy = intrinsics.oy();

  float* warped_intensity_ptr = warped_image.ptr<IntensityType>();
  float* warped_depth_ptr = warped_depth.ptr<DepthType>();

  int outliers = 0;
  int total = 0;
  int idx = 0;

  transformed_pointcloud = transformation * reference_pointcloud;

  for(size_t y = 0; y < height; ++y)
  {
    for(size_t x = 0; x < width; ++x, ++idx, ++warped_intensity_ptr, ++warped_depth_ptr)
    {

      const Eigen::Vector4f& p3d = transformed_pointcloud.col(idx);

      if(!std::isfinite(p3d(2)))
      {
        *warped_intensity_ptr = Invalid;
        *warped_depth_ptr = InvalidDepth;
        continue;
      }

      float x_projected = (float) (p3d(0) * intrinsics.fx() / p3d(2) + ox);
      float y_projected = (float) (p3d(1) * intrinsics.fy() / p3d(2) + oy);

      if(inImage(x_projected, y_projected))
      {
        float z = (float) p3d(2);

        *warped_intensity_ptr = Interpolation::bilinearWithDepthBuffer(this->intensity, this->depth, x_projected, y_projected, z);
        *warped_depth_ptr = z;
      }
      else
      {
        *warped_intensity_ptr = Invalid;
        *warped_depth_ptr = InvalidDepth;
        //outliers++;
      }

      //total++;
    }
  }

  result.intensity = warped_image;
  result.depth = warped_depth;
  result.initialize();
}

void RgbdImage::warpDepthForward(const AffineTransform& transformationx, const IntrinsicMatrix& intrinsics, RgbdImage& result, cv::Mat_<cv::Vec3d>& cloud)
{
  Eigen::Affine3d transformation = transformationx.cast<double>();

  cloud = cv::Mat_<cv::Vec3d>(depth.size(), cv::Vec3d(0, 0, 0));
  cv::Mat warped_depth = cv::Mat::zeros(depth.size(), depth.type());
  warped_depth.setTo(InvalidDepth);

  float ox = intrinsics.ox();
  float oy = intrinsics.oy();

  const float* depth_ptr = depth.ptr<float>();
  int outliers = 0;
  int total = 0;

  for(size_t y = 0; y < height; ++y)
  {
    for(size_t x = 0; x < width; ++x, ++depth_ptr)
    {
      if(!std::isfinite(*depth_ptr))
      {
        continue;
      }

      float depth = *depth_ptr;
      Eigen::Vector3d p3d((x - ox) * depth / intrinsics.fx(), (y - oy) * depth / intrinsics.fy(), depth);
      Eigen::Vector3d p3d_transformed = transformation * p3d;

      float x_projected = (float) (p3d_transformed(0) * intrinsics.fx() / p3d_transformed(2) + ox);
      float y_projected = (float) (p3d_transformed(1) * intrinsics.fy() / p3d_transformed(2) + oy);

      if(inImage(x_projected, y_projected))
      {
        int yi = (int) y_projected, xi = (int) x_projected;

        if(!std::isfinite(warped_depth.at<DepthType>(yi, xi)) || (warped_depth.at<DepthType>(yi, xi) - 0.05) > depth)
          warped_depth.at<DepthType>(yi, xi) = depth;
      }

      p3d = p3d_transformed;

      total++;
      cloud(y, x) = cv::Vec3d(p3d(0), p3d(1), p3d(2));
    }
  }

  result.depth = warped_depth;
  result.initialize();
}

void RgbdImage::warpIntensityForward(const AffineTransform& transformationx, const IntrinsicMatrix& intrinsics, RgbdImage& result, cv::Mat_<cv::Vec3d>& cloud)
{
  Eigen::Affine3d transformation = transformationx.cast<double>();

  bool identity = transformation.affine().isIdentity(1e-6);

  cloud = cv::Mat_<cv::Vec3d>(intensity.size(), cv::Vec3d(0, 0, 0));
  cv::Mat warped_image = cv::Mat::zeros(intensity.size(), intensity.type());

  float ox = intrinsics.ox();
  float oy = intrinsics.oy();

  const float* depth_ptr = depth.ptr<float>();
  int outliers = 0;
  int total = 0;

  for(size_t y = 0; y < height; ++y)
  {
    for(size_t x = 0; x < width; ++x, ++depth_ptr)
    {
      if(*depth_ptr <= 1e-6f) continue;

      float depth = *depth_ptr;
      Eigen::Vector3d p3d((x - ox) * depth / intrinsics.fx(), (y - oy) * depth / intrinsics.fy(), depth);

      if(!identity)
      {
        Eigen::Vector3d p3d_transformed = transformation * p3d;

        float x_projected = (float) (p3d_transformed(0) * intrinsics.fx() / p3d_transformed(2) + ox);
        float y_projected = (float) (p3d_transformed(1) * intrinsics.fy() / p3d_transformed(2) + oy);

        if(inImage(x_projected, y_projected))
        {
          int xp, yp;
          xp = (int) std::floor(x_projected);
          yp = (int) std::floor(y_projected);

          warped_image.at<IntensityType>(yp, xp) = intensity.at<IntensityType>(y, x);
        }
        else
        {
          outliers++;
        }

        p3d = p3d_transformed;
      }

      total++;
      cloud(y, x) = cv::Vec3d(p3d(0), p3d(1), p3d(2));
    }
  }

  //std::cerr << "warp out: " << outliers << " total: " << total << std::endl;

  if(identity)
  {
    warped_image = intensity;
  }
  else
  {
    //std::cerr << "did warp" << std::endl;
  }

  result.intensity = warped_image;
  result.depth = depth;
  result.initialize();
}

void RgbdImage::warpDepthForwardAdvanced(const AffineTransform& transformation, const IntrinsicMatrix& intrinsics, RgbdImage& result)
{
  assert(hasDepth());

  this->buildPointCloud();

  PointCloud transformed_pointcloud = transformation.cast<float>() * pointcloud;

  cv::Mat warped_depth(depth.size(), depth.type());
  warped_depth.setTo(InvalidDepth);

  float z_factor1 = transformation.rotation()(0, 0) + transformation.rotation()(0, 1) * (intrinsics.fx() / intrinsics.fy());
  float x_factor1 = -transformation.rotation()(2, 0) - transformation.rotation()(2, 1) * (intrinsics.fx() / intrinsics.fy());

  float z_factor2 = transformation.rotation()(1, 1) + transformation.rotation()(1, 0) * (intrinsics.fy() / intrinsics.fx());
  float y_factor2 = -transformation.rotation()(2, 1) - transformation.rotation()(2, 0) * (intrinsics.fy() / intrinsics.fx());

  for(int idx = 0; idx < height * width; ++idx)
  {
    Vector4 p3d = pointcloud.col(idx);
    NumType z = p3d(2);

    if(!std::isfinite(z)) continue;

    int x_length = (int) std::ceil(z_factor1 + x_factor1 * p3d(0) / z) + 1; // magic +1
    int y_length = (int) std::ceil(z_factor2 + y_factor2 * p3d(1) / z) + 1; // magic +1

    Vector4 p3d_transformed = transformed_pointcloud.col(idx);
    NumType z_transformed = p3d_transformed(2);

    int x_projected = (int) std::floor(p3d_transformed(0) * intrinsics.fx() / z_transformed + intrinsics.ox());
    int y_projected = (int) std::floor(p3d_transformed(1) * intrinsics.fy() / z_transformed + intrinsics.oy());

    // TODO: replace inImage(...) checks, with max(..., 0) on initial value of x_, y_ and  min(..., width/height) for their respective upper bound
    //for (int y_ = y_projected; y_ < y_projected + y_length; y_++)
    //  for (int x_ = x_projected; x_ < x_projected + x_length; x_++)

    int x_begin = std::max(x_projected, 0);
    int y_begin = std::max(y_projected, 0);
    int x_end = std::min(x_projected + x_length, (int) width);
    int y_end = std::min(y_projected + y_length, (int) height);

    for (int y = y_begin; y < y_end; ++y)
    {
      DepthType* v = warped_depth.ptr<DepthType>(y, x_begin);

      for (int x = x_begin; x < x_end; ++x, ++v)
      {
        if(!std::isfinite(*v) || (*v) > z_transformed)
        {
          (*v) = (DepthType) z_transformed;
        }
      }
    }
  }

  result.depth = warped_depth;
  result.initialize();
}

bool RgbdImage::inImage(const float& x, const float& y) const
{
  return x >= 0 && x < width && y >= 0 && y < height;
}

} /* namespace core */
} /* namespace dvo */
