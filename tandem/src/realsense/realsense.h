// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef PBA_REALSENSE_H
#define PBA_REALSENSE_H

#include <opencv2/opencv.hpp>

class D455Impl;

class D455 {
public:

  D455(bool debugPrint = false, int width = 640, int height = 400);

  ~D455();

  void StartPipeline();

  bool GetFrameBlocking(double &timestamp, cv::Mat &img, int& frame_id);

  // only available after StartPipeline
  // fx fy cx cy
  cv::Vec<float, 4> GetIntrinsics();

  // only available after StartPipeline
  // RadTan model with 5 coefficients (like OpenCV)
  // k1 k2 p1 p2 k3
  cv::Vec<float, 5> GetDistortion();

  // only available after StartPipeline
  int GetHeight();

  // only available after StartPipeline
  int GetWidth();

  // only available after StartPipeline
  void WriteDsoIntrinsicsFile(const std::string &filename, const std::string &l3 = "crop", int w_out = 640, int h_out = 480) const;

private:
  D455Impl *impl;
};

#endif //PBA_REALSENSE_H
