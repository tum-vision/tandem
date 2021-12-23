// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

namespace refusion {

/**
 * @brief      Structure storing the intrinsic parameters of an RGB-D sensor
 */
struct RgbdSensor {
 public:
  /** Horizontal focal lenght in pixel */
  float fx;

  /** Vertical focal lenght in pixel */
  float fy;

  /** Horizontal coordinate of the principal point */
  float cx;

  /** Vertical coordinate of the principal point */
  float cy;

  /** Scale factor to convert the depth in meters */
  float depth_factor;

  /** Number of rows in the image */
  unsigned int rows;

  /** Number of columns in the image */
  unsigned int cols;
};

}  // namespace refusion
