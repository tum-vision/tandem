// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#include "cuda_coarse_tracker.h"
#include "runtime_utils.h"
#include <stdexcept>
#include <sophus/se3.hpp>
#include "cuda_coarse_tracker_private.h"
#include <nvToolsExtCudaRt.h>
#include "cnpy.h"


#define SCALE_IDEPTH 1.0f    // scales internal value to idepth.
#define SCALE_XI_ROT 1.0f
#define SCALE_XI_TRANS 0.5f
#define SCALE_F 50.0f
#define SCALE_C 50.0f
#define SCALE_W 1.0f
#define SCALE_A 10.0f
#define SCALE_B 1000.0f

#define MIN(x, y) (x<=y ? x : y)
#define MAX(x, y) (x>=y ? x : y)

typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef Eigen::Matrix<double, 8, 8> Mat88;
typedef Sophus::SE3<double> SE3;


// transforms points from one frame to another.
struct AffLight {
  AffLight(Vec2 const &v) : a(v(0)), b(v(1)) {};

  AffLight(double a_, double b_) : a(a_), b(b_) {};

  AffLight() : a(0), b(0) {};

  // Affine Parameters:
  double a, b;  // I_frame = exp(a)*I_global + b. // I_global = exp(-a)*(I_frame - b).

  static Vec2 fromToVecExposure(float exposureF, float exposureT, AffLight g2F, AffLight g2T) {
    if (exposureF == 0 || exposureT == 0) {
      exposureT = exposureF = 1;
      //printf("got exposure value of 0! please choose the correct model.\n");
      //assert(setting_brightnessTransferFunc < 2);
    }

    double a = exp(g2T.a - g2F.a) * exposureT / exposureF;
    double b = g2T.b - a * g2F.b;
    return Vec2(a, b);
  }

  Vec2 vec() {
    return Vec2(a, b);
  }
};

inline int symmetricUpperTriangularLinearIndex(int n, int row_in, int col_in) {
  // It must hold: row <= col, otherwise flip because of symmetry
  const int row = MIN(row_in, col_in);
  const int col = MAX(row_in, col_in);

  const int full_idx = row * n + col;
  const int lower_left = row * (row + 1) / 2;

  return full_idx - lower_left;
}

CudaCoarseTracker::CudaCoarseTracker(int w, int h, float setting_huberTH, float setting_coarseCutoffTH) : w(w), h(h), setting_huberTH(setting_huberTH), setting_coarseCutoffTH(setting_coarseCutoffTH) {
  // TODO(perf): Set priority correctly
  int least_prio, greatest_prio;
  gpuErrchk(cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio));
  gpuErrchk(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_prio))

//  char name[200];
//  sprintf(name, "CudaCoarseTracker(w=%d,h=%d) at this=%p", w, h, this);
  nvtxNameCudaStreamA(stream, "CudaCoarseTracker");
}

void CudaCoarseTracker::setReference(int n_in, const float *const pc_u_in, const float *const pc_v_in, const float *const pc_idepth_in, const float *const pc_color_in, float ref_exposure_in, Eigen::Vector2d const &ref_aff_g2l_in) {
  if (n_in > n_max) throw std::runtime_error("\"Called CudaCoarseTracker::setReference with n > n_max points.");
  n = n_in;

  memcpy(pc_u_host, pc_u_in, sizeof(float) * n);
  memcpy(pc_v_host, pc_v_in, sizeof(float) * n);
  memcpy(pc_idepth_host, pc_idepth_in, sizeof(float) * n);
  memcpy(pc_color_host, pc_color_in, sizeof(float) * n);

  gpuErrchk(cudaMemcpyAsync(pc_u, pc_u_host, sizeof(float) * n, cudaMemcpyHostToDevice, stream))
  gpuErrchk(cudaMemcpyAsync(pc_v, pc_v_host, sizeof(float) * n, cudaMemcpyHostToDevice, stream))
  gpuErrchk(cudaMemcpyAsync(pc_idepth, pc_idepth_host, sizeof(float) * n, cudaMemcpyHostToDevice, stream))
  gpuErrchk(cudaMemcpyAsync(pc_color, pc_color_host, sizeof(float) * n, cudaMemcpyHostToDevice, stream))

  ref_exposure = ref_exposure_in;
  ref_aff_g2l = ref_aff_g2l_in;
}

void CudaCoarseTracker::setNew(const float *const dInew_in) {
  memcpy(dInew_host, dInew_in, sizeof(float) * 3 * w * h);
  gpuErrchk(cudaMemcpyAsync(dInew, dInew_host, sizeof(float) * 3 * w * h, cudaMemcpyHostToDevice, stream))
}

void CudaCoarseTracker::init(int n_max_in) {
  if (w * h == 0) throw std::runtime_error("\"CudaCoarseTracker::init has w*h==0.");
  if (n_max != 0) throw std::runtime_error("\"Cannot call CudaCoarseTracker::init more than once.");
  n_max = n_max_in > 0 ? n_max_in : w * h;

  // TODO(perf): Make this contiguous memory with one alloc and memcpy
  gpuErrchk(cudaMalloc((void **) &pc_u, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &pc_v, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &pc_idepth, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &pc_color, sizeof(float) * n_max))

  gpuErrchk(cudaMalloc((void **) &dInew, sizeof(float) * 3 * w * h))

  gpuErrchk(cudaMalloc((void **) &warped_u, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &warped_v, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &warped_dx, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &warped_dy, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &warped_idepth, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &warped_residual, sizeof(float) * n_max))
  gpuErrchk(cudaMalloc((void **) &warped_weight, sizeof(float) * n_max))

  gpuErrchk(cudaMalloc((void **) &refToNew_Ki, sizeof(float) * (16 + 9)))
  gpuErrchk(cudaMalloc((void **) &outputs_calcRes, sizeof(float) * 7))
  gpuErrchk(cudaMalloc((void **) &outputs_calcG, sizeof(float) * 45))

  gpuErrchk(cudaMallocHost((void **) &pc_u_host, sizeof(float) * n_max))
  gpuErrchk(cudaMallocHost((void **) &pc_v_host, sizeof(float) * n_max))
  gpuErrchk(cudaMallocHost((void **) &pc_idepth_host, sizeof(float) * n_max))
  gpuErrchk(cudaMallocHost((void **) &pc_color_host, sizeof(float) * n_max))

  gpuErrchk(cudaMallocHost((void **) &dInew_host, sizeof(float) * 3 * w * h))

  gpuErrchk(cudaMallocHost((void **) &refToNew_Ki_host, sizeof(float) * (16 + 9)))
  gpuErrchk(cudaMallocHost((void **) &outputs_calcRes_host, sizeof(float) * 7))
  gpuErrchk(cudaMallocHost((void **) &outputs_calcG_host, sizeof(float) * 45))
}

void CudaCoarseTracker::free() {
  if (n_max == 0) return;

  gpuErrchk(cudaFree((void *) pc_u))
  gpuErrchk(cudaFree((void *) pc_v))
  gpuErrchk(cudaFree((void *) pc_idepth))
  gpuErrchk(cudaFree((void *) pc_color))

  gpuErrchk(cudaFree((void *) warped_u))
  gpuErrchk(cudaFree((void *) warped_v))
  gpuErrchk(cudaFree((void *) warped_dx))
  gpuErrchk(cudaFree((void *) warped_dy))
  gpuErrchk(cudaFree((void *) warped_idepth))
  gpuErrchk(cudaFree((void *) warped_residual))
  gpuErrchk(cudaFree((void *) warped_weight))

  gpuErrchk(cudaFree((void *) dInew))

  gpuErrchk(cudaFree((void *) refToNew_Ki))
  gpuErrchk(cudaFree((void *) outputs_calcRes))
  gpuErrchk(cudaFree((void *) outputs_calcG))

  gpuErrchk(cudaFreeHost((void *) pc_u_host))
  gpuErrchk(cudaFreeHost((void *) pc_v_host))
  gpuErrchk(cudaFreeHost((void *) pc_idepth_host))
  gpuErrchk(cudaFreeHost((void *) pc_color_host))

  gpuErrchk(cudaFreeHost((void *) dInew_host))

  gpuErrchk(cudaFreeHost((void *) refToNew_Ki_host))
  gpuErrchk(cudaFreeHost((void *) outputs_calcRes_host))
  gpuErrchk(cudaFreeHost((void *) outputs_calcG_host))

  gpuErrchk(cudaStreamSynchronize(stream))
  gpuErrchk(cudaStreamDestroy(stream))

  n_max = 0;
  pc_u = pc_v = pc_idepth = pc_color = nullptr;
  warped_u = warped_v = warped_dx = warped_dy = warped_idepth = warped_residual = warped_weight = nullptr;
  dInew = nullptr;
  dInew_host = nullptr;
  refToNew_Ki = outputs_calcRes = nullptr;
  pc_u_host = pc_v_host = pc_idepth_host = pc_color_host = nullptr;
  refToNew_Ki_host = outputs_calcRes_host = nullptr;
}

CudaCoarseTracker::~CudaCoarseTracker() {
  free();
}

void CudaCoarseTracker::synchronize() {
  cudaStreamSynchronize(stream);
}

Vec6 CudaCoarseTracker::calcRes(Eigen::Matrix<double, 4, 4> const &refToNew, const float new_exposure, Eigen::Vector2d const &aff_g2l_in, float cutoffTH) {
  gpuErrchk(cudaMemsetAsync(outputs_calcRes, 0, sizeof(float) * 7, stream))
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++)
      refToNew_Ki_host[4 * r + c] = (float) refToNew(r, c);
  }
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++)
      refToNew_Ki_host[16 + 3 * r + c] = (float) Ki(r, c);
  }
  gpuErrchk(cudaMemcpyAsync(refToNew_Ki, refToNew_Ki_host, sizeof(float) * (16 + 9), cudaMemcpyHostToDevice, stream))

  AffLight aff_g2l(aff_g2l_in);
  Vec2 affLL_double = AffLight::fromToVecExposure(ref_exposure, new_exposure, ref_aff_g2l, aff_g2l);
  float2 affLL;
  affLL.x = (float) affLL_double.x();
  affLL.y = (float) affLL_double.y();

  float maxEnergy = 2 * setting_huberTH * cutoffTH - setting_huberTH * setting_huberTH;  // energy for r=setting_coarseCutoffTH.

  bool save = false;
  if (save) {
    float nfxfyaffLLb[6] = {(float) n, fx, fy, affLL.x, affLL.y, (float) ref_aff_g2l.y()};
    cnpy::npy_save("cct_data/nfxfyaffLLb.npy", nfxfyaffLLb, {6});

    float setting_huberTH_maxEnergy_cutoffTH[3] = {setting_huberTH, maxEnergy, cutoffTH};
    cnpy::npy_save("cct_data/setting_huberTH_maxEnergy_cutoffTH.npy", nfxfyaffLLb, {3});

    cnpy::npy_save("cct_data/refToNew_Ki_host.npy", refToNew_Ki_host, {16 + 9});

    float *buf = (float *) malloc(sizeof(float) * n);
    cudaMemcpy(buf, pc_u, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/pc_u.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, pc_v, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/pc_v.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, pc_idepth, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/pc_idepth.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, pc_color, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/pc_color.npy", buf, {(unsigned long) n});

    ::free(buf);
    buf = (float *) malloc(sizeof(float) * 3 * w * h);
    cudaMemcpy(buf, dInew, sizeof(float) * 3 * w * h, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/dInew.npy", buf, {(unsigned long) 3 * w * h});

    if (n > 0.5 * 640 * 480)
      exit(EXIT_SUCCESS);
  }

  callCalcResKernel(
      128, stream,
      setting_huberTH,
      w, h, fx, fy, cx, cy,
      refToNew_Ki,
      refToNew_Ki + 16,
      affLL,
      maxEnergy,
      cutoffTH,
      n, pc_u, pc_v, pc_idepth, pc_color,
      dInew,
      warped_u, warped_v, warped_dx, warped_dy, warped_idepth, warped_residual, warped_weight,
      outputs_calcRes
  );

  gpuErrchk(cudaMemcpyAsync(outputs_calcRes_host, outputs_calcRes, sizeof(float) * 7, cudaMemcpyDeviceToHost, stream))
  gpuErrchk(cudaStreamSynchronize(stream))
  Vec6 res = Vec6::Zero();
  res(0) = outputs_calcRes_host[IDX_E];
  res(1) = outputs_calcRes_host[IDX_numTermsInE];
  res(2) = outputs_calcRes_host[IDX_sumSquaredShiftT] / outputs_calcRes_host[IDX_sumSquaredShiftNum];
  res(3) = 0;
  res(4) = outputs_calcRes_host[IDX_sumSquaredShiftRT] / outputs_calcRes_host[IDX_sumSquaredShiftNum];
  res(5) = outputs_calcRes_host[IDX_numSaturated] / outputs_calcRes_host[IDX_numTermsInE];

  num_terms_in_warped = (int) outputs_calcRes_host[IDX_numTermsInWarped];

  return res;
}

void CudaCoarseTracker::calcG(Eigen::Matrix<double, 8, 8> &H_out, Eigen::Matrix<double, 8, 1> &b_out, const float new_exposure, const Eigen::Vector2d &aff_g2l) {
  gpuErrchk(cudaMemsetAsync(outputs_calcG, 0, sizeof(float) * 45, stream))

  const double factor = 1.0 / num_terms_in_warped;

  Vec2 affLL_double = AffLight::fromToVecExposure(ref_exposure, new_exposure, ref_aff_g2l, aff_g2l);
  float2 affLL;
  affLL.x = (float) affLL_double.x();
  affLL.y = (float) affLL_double.y();

  bool save = false;
  if (save) {
    float nfxfyaffLLb[6] = {(float) n, fx, fy, affLL.x, affLL.y, (float) ref_aff_g2l.y()};
    cnpy::npy_save("cct_data/nfxfyaffLLb.npy", nfxfyaffLLb, {6});
    float *buf = (float *) malloc(sizeof(float) * n);

    cudaMemcpy(buf, pc_color, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/pc_color.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, warped_u, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/warped_u.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, warped_v, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/warped_v.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, warped_dx, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/warped_dx.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, warped_dy, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/warped_dy.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, warped_idepth, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/warped_idepth.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, warped_residual, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/warped_residual.npy", buf, {(unsigned long) n});

    cudaMemcpy(buf, warped_weight, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cnpy::npy_save("cct_data/warped_weight.npy", buf, {(unsigned long) n});

    if (n > 0.5 * 640 * 480)
      exit(EXIT_SUCCESS);
  }

  callCalcGKernel(
      128, stream,
      fx, fy,
      affLL,
      (float) ref_aff_g2l.y(),
      n, 16,
      pc_color,
      warped_u, warped_v, warped_dx, warped_dy, warped_idepth, warped_residual, warped_weight,
      outputs_calcG
  );

  gpuErrchk(cudaMemcpyAsync(outputs_calcG_host, outputs_calcG, sizeof(float) * 45, cudaMemcpyDeviceToHost, stream))
  gpuErrchk(cudaStreamSynchronize(stream))

  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      const int idx_H = symmetricUpperTriangularLinearIndex(9, r, c);
      H_out(r, c) = (double) outputs_calcG_host[idx_H] * factor;
    }
    const int idx_b = symmetricUpperTriangularLinearIndex(9, r, 8);
    b_out(r) = (double) outputs_calcG_host[idx_b] * factor;
  }

  H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT;
  H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
  H_out.block<8, 1>(0, 6) *= SCALE_A;
  H_out.block<8, 1>(0, 7) *= SCALE_B;
  H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
  H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
  H_out.block<1, 8>(6, 0) *= SCALE_A;
  H_out.block<1, 8>(7, 0) *= SCALE_B;
  b_out.segment<3>(0) *= SCALE_XI_ROT;
  b_out.segment<3>(3) *= SCALE_XI_TRANS;
  b_out.segment<1>(6) *= SCALE_A;
  b_out.segment<1>(7) *= SCALE_B;
}

void CudaCoarseTracker::setK(int w_in, int h_in, float fx_in, float fy_in, float cx_in, float cy_in) {
  if (w != w_in || h != h_in) throw std::runtime_error("\"CudaCoarseTracker::setK wrong h,w.");
  fx = fx_in;
  fy = fy_in;
  cx = cx_in;
  cy = cy_in;

  Eigen::Matrix<double, 3, 3> K = Eigen::Matrix<double, 3, 3>::Zero();
  K(0, 0) = fx;
  K(1, 1) = fy;
  K(0, 2) = cx;
  K(1, 2) = cy;
  K(2, 2) = 1;
  Ki = K.inverse();
}

void CudaCoarseTracker::startTiming() {
  if (ev_start != nullptr || ev_stop != nullptr) throw std::runtime_error("CudaCoarseTracker::startTiming. Did not destroy events before correctly.");

  gpuErrchk(cudaEventCreate(&ev_start))
  gpuErrchk(cudaEventCreate(&ev_stop))

  gpuErrchk(cudaEventRecordWithFlags(ev_start, stream))
}

float CudaCoarseTracker::endTimingMilliseconds() {
  if (ev_start == nullptr || ev_stop == nullptr) throw std::runtime_error("CudaCoarseTracker::endTimingMilliseconds. Did not start before.");

  gpuErrchk(cudaEventRecordWithFlags(ev_stop, stream))
  gpuErrchk(cudaEventSynchronize(ev_stop))

  float milliseconds = -1.0f;
  gpuErrchk(cudaEventElapsedTime(&milliseconds, ev_start, ev_stop))
  gpuErrchk(cudaEventDestroy(ev_start))
  gpuErrchk(cudaEventDestroy(ev_stop))
  ev_start = ev_stop = nullptr;

  return milliseconds;
}
