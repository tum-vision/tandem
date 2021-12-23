// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef PBA_TANDEM_BACKEND_H
#define PBA_TANDEM_BACKEND_H

#include "util/Timer.h"
#include "IOWrapper/Output3DWrapper.h"
#include "dr_fusion.h"
#include "dr_mvsnet.h"

#include <memory>
#include <utility>
#include <vector>
#include <boost/thread/mutex.hpp>
#include <opencv2/opencv.hpp>

#include <boost/thread/thread.hpp>
#include <algorithm>


class TandemCoarseTrackingDepthMap {
public:
  bool is_valid = false;
  float cam_to_world[16] = {};
  float *depth;

  TandemCoarseTrackingDepthMap(int width, int height);

  ~TandemCoarseTrackingDepthMap();
};


float get_idepth_quantile(int n, float const *const idepth, float fraction = 0.2f);

class TandemBackendImpl;

class TandemBackend {
public:
  explicit TandemBackend(
      int width, int height, bool dense_tracking,
      DrMvsnet *mvsnet, DrFusion *fusion,
      float mvsnet_discard_percentage,
      Timer *dr_timer,
      std::vector<dso::IOWrap::Output3DWrapper *> const &outputWrapper,
      int mesh_extraction_freq);

  ~TandemBackend();

  // Must check Ready() before!
  void CallAsync(
      int view_num_in,
      int index_offset_in,
      int corrected_ref_index_in,
      std::vector<cv::Mat> const &bgrs_in,
      cv::Mat const &intrinsic_matrix_in,
      std::vector<cv::Mat> const &cam_to_worlds_in,
      float depth_min_in,
      float depth_max_in,
      cv::Mat const &coarse_tracker_pose_in
  );

  boost::mutex &GetTrackingDepthMapMutex();

  TandemCoarseTrackingDepthMap const *GetTrackingDepthMap();

  // Non-blocking
  bool Ready();

  // Blocking
  void Wait();

public:
  TandemBackendImpl *impl;
};

#endif //PBA_TANDEM_BACKEND_H
