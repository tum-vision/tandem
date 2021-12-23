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

#ifndef RGBDIMAGE_H_
#define RGBDIMAGE_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
//#include <Eigen/StdVector>
#include <boost/smart_ptr.hpp>

#include <dvo/core/datatypes.h>
#include <dvo/core/intrinsic_matrix.h>

namespace dvo
{
namespace core
{

typedef Eigen::Matrix<float, 8, 1> Vector8f;

struct EIGEN_ALIGN16 PointWithIntensityAndDepth
{
  typedef EIGEN_ALIGN16 union
  {
    float data[4];
    struct
    {
      float x, y, z;
    };
  } Point;

  typedef EIGEN_ALIGN16 union
  {
    float data[8];
    struct
    {
      float i, z, idx, idy, zdx, zdy, time_interpolation;
    };
  } IntensityAndDepth;

  typedef std::vector<PointWithIntensityAndDepth, Eigen::aligned_allocator<PointWithIntensityAndDepth> > VectorType;

  Point point;
  IntensityAndDepth intensity_and_depth;

  Eigen::Vector4f::AlignedMapType getPointVec4f()
  {
    return Eigen::Vector4f::AlignedMapType(point.data);
  }

  Eigen::Vector2f::AlignedMapType getIntensityAndDepthVec2f()
  {
    return Eigen::Vector2f::AlignedMapType(intensity_and_depth.data);
  }

  Eigen::Vector2f::MapType getIntensityDerivativeVec2f()
  {
    return Eigen::Vector2f::MapType(intensity_and_depth.data + 2);
  }

  Eigen::Vector2f::MapType getDepthDerivativeVec2f()
  {
    return Eigen::Vector2f::MapType(intensity_and_depth.data + 4);

  }

  Vector8f::AlignedMapType getIntensityAndDepthWithDerivativesVec8f()
  {
    return Vector8f::AlignedMapType(intensity_and_depth.data);
  }
};

typedef Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> PointCloud;

class RgbdImage;
typedef boost::shared_ptr<RgbdImage> RgbdImagePtr;

class RgbdImagePyramid;
typedef boost::shared_ptr<RgbdImagePyramid> RgbdImagePyramidPtr;

class RgbdCamera
{
public:
  RgbdCamera(size_t width, size_t height, const dvo::core::IntrinsicMatrix& intrinsics);
  ~RgbdCamera();

  size_t width() const;

  size_t height() const;

  const dvo::core::IntrinsicMatrix& intrinsics() const;

  RgbdImagePtr create(const cv::Mat& intensity, const cv::Mat& depth) const;
  RgbdImagePtr create() const;

  void buildPointCloud(const cv::Mat &depth, PointCloud& pointcloud) const;
private:
  size_t width_, height_;

  bool hasSameSize(const cv::Mat& img) const;

  dvo::core::IntrinsicMatrix intrinsics_;
  PointCloud pointcloud_template_;
};

typedef boost::shared_ptr<RgbdCamera> RgbdCameraPtr;
typedef boost::shared_ptr<const RgbdCamera> RgbdCameraConstPtr;

class RgbdCameraPyramid
{
public:
  RgbdCameraPyramid(const RgbdCamera& base);
  RgbdCameraPyramid(size_t base_width, size_t base_height, const dvo::core::IntrinsicMatrix& base_intrinsics);

  ~RgbdCameraPyramid();

  RgbdImagePyramidPtr create(const cv::Mat& base_intensity, const cv::Mat& base_depth);

  void build(size_t levels);

  const RgbdCamera& level(size_t level);

  const RgbdCamera& level(size_t level) const;
private:
  std::vector<RgbdCameraPtr> levels_;
};

typedef boost::shared_ptr<RgbdCameraPyramid> RgbdCameraPyramidPtr;
typedef boost::shared_ptr<const RgbdCameraPyramid> RgbdCameraPyramidConstPtr;


class RgbdImage
{
public:
  //RgbdImage();
  RgbdImage(const RgbdCamera& camera);
  virtual ~RgbdImage();

  typedef dvo::core::PointCloud PointCloud;

  const RgbdCamera& camera() const;

  cv::Mat intensity;
  cv::Mat intensity_dx;
  cv::Mat intensity_dy;

  cv::Mat depth;
  cv::Mat depth_dx;
  cv::Mat depth_dy;

  cv::Mat normals, angles;

  cv::Mat rgb;

  PointCloud pointcloud;

  typedef cv::Vec<float, 8> Vec8f;
  cv::Mat_<Vec8f> acceleration;

  size_t width, height;
  double timestamp;

  bool hasIntensity() const;
  bool hasDepth() const;
  bool hasRgb() const;

  void initialize();

  void calculateDerivatives();
  bool calculateIntensityDerivatives();
  void calculateDepthDerivatives();

  void calculateNormals();

  void buildPointCloud();

  //void buildPointCloud(const IntrinsicMatrix& intrinsics);

  void buildAccelerationStructure();

  // inverse warping
  // transformation is the transformation from reference to this image
  void warpIntensity(const AffineTransform& transformation, const PointCloud& reference_pointcloud, const IntrinsicMatrix& intrinsics, RgbdImage& result, PointCloud& transformed_pointcloud);

  // SSE version
  void warpIntensitySse(const AffineTransform& transformation, const PointCloud& reference_pointcloud, const IntrinsicMatrix& intrinsics, RgbdImage& result, PointCloud& transformed_pointcloud);
  // SSE version without warped pointcloud
  void warpIntensitySse(const AffineTransform& transformation, const PointCloud& reference_pointcloud, const IntrinsicMatrix& intrinsics, RgbdImage& result);

  // forward warping
  // transformation is the transformation from this image to the reference image
  void warpIntensityForward(const AffineTransform& transformation, const IntrinsicMatrix& intrinsics, RgbdImage& result, cv::Mat_<cv::Vec3d>& cloud);
  void warpDepthForward(const AffineTransform& transformation, const IntrinsicMatrix& intrinsics, RgbdImage& result, cv::Mat_<cv::Vec3d>& cloud);

  void warpDepthForwardAdvanced(const AffineTransform& transformation, const IntrinsicMatrix& intrinsics, RgbdImage& result);

  bool inImage(const float& x, const float& y) const;
private:
  bool intensity_requires_calculation_, depth_requires_calculation_, pointcloud_requires_build_;

  const RgbdCamera& camera_;

  template<typename T>
  void calculateDerivativeX(const cv::Mat& img, cv::Mat& result);

  //template<typename T>
  //void calculateDerivativeXSse(const cv::Mat& img, cv::Mat& result);

  template<typename T>
  void calculateDerivativeY(const cv::Mat& img, cv::Mat& result);

  void calculateDerivativeYSseFloat(const cv::Mat& img, cv::Mat& result);

  enum WarpIntensityOptions
  {
    WithPointCloud,
    WithoutPointCloud,
  };

  template<int PointCloudOption>
  void warpIntensitySseImpl(const AffineTransform& transformation, const PointCloud& reference_pointcloud, const IntrinsicMatrix& intrinsics, RgbdImage& result, PointCloud& transformed_pointcloud);
};

class RgbdImagePyramid
{
public:
  typedef boost::shared_ptr<dvo::core::RgbdImagePyramid> Ptr;

  RgbdImagePyramid(RgbdCameraPyramid& camera, const cv::Mat& intensity, const cv::Mat& depth);
  RgbdImagePyramid(RgbdImagePyramid& _rgbd);

  virtual ~RgbdImagePyramid();

  // deprecated
  void compute(const size_t num_levels);

  void build(const size_t num_levels);

  RgbdImage& level(size_t idx);

  double timestamp() const;

  RgbdCameraPyramid camera_;
private:
  std::vector<RgbdImagePtr> levels_;
};

} /* namespace core */
} /* namespace dvo */
#endif /* RGBDIMAGE_H_ */
