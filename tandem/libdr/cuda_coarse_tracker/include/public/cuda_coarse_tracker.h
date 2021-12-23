// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef PBA_CUDA_COARSE_TRACKER_H
#define PBA_CUDA_COARSE_TRACKER_H

#include <Eigen/Dense>
#include <cuda_runtime_api.h>

class CudaCoarseTracker {
public:
  CudaCoarseTracker(int w, int h, float setting_huberTH, float setting_coarseCutoffTH);

  void setK(int w, int h, float fx, float fy, float cx, float cy);

  ~CudaCoarseTracker();

  void init(int n_max_in = 0);

  void free();

  void setReference(int n_in, float const *pc_u_in, float const *pc_v_in, float const *pc_idepth_in, float const *pc_color_in, float ref_exposure_in, Eigen::Vector2d const &ref_aff_g2l_in);

  void setNew(float const *dInew_in);

  Eigen::Matrix<double, 6, 1> calcRes(Eigen::Matrix<double, 4, 4> const &refToNew, float new_exposure, Eigen::Vector2d const &aff_g2l, float cutoffTH);

  void calcG(Eigen::Matrix<double, 8, 8> &H_out, Eigen::Matrix<double, 8, 1> &b_out, const float new_exposure, const Eigen::Vector2d &aff_g2l);


  void synchronize();

  void startTiming();

  float endTimingMilliseconds();

private:

  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;

  const int w = 0, h = 0;
  float fx = 0, fy = 0, cx = 0, cy = 0;
  Eigen::Matrix<double, 3, 3> Ki;
  const float setting_huberTH, setting_coarseCutoffTH;

  int n_max = 0, n = 0;
  float *pc_u = nullptr;
  float *pc_v = nullptr;
  float *pc_idepth = nullptr;
  float *pc_color = nullptr;

  float *pc_u_host = nullptr;
  float *pc_v_host = nullptr;
  float *pc_idepth_host = nullptr;
  float *pc_color_host = nullptr;

  float *warped_u = nullptr;
  float *warped_v = nullptr;
  float *warped_dx = nullptr;
  float *warped_dy = nullptr;
  float *warped_idepth = nullptr;
  float *warped_residual = nullptr;
  float *warped_weight = nullptr;
  int num_terms_in_warped = 0;

  float ref_exposure;
  Eigen::Vector2d ref_aff_g2l;

  float *dInew = nullptr;
  float *dInew_host = nullptr;

  float *refToNew_Ki = nullptr;
  float *refToNew_Ki_host = nullptr;

  float *outputs_calcRes = nullptr;
  float *outputs_calcRes_host = nullptr;

  float *outputs_calcG = nullptr;
  float *outputs_calcG_host = nullptr;

  cudaStream_t stream;
};

#endif //PBA_CUDA_COARSE_TRACKER_H
