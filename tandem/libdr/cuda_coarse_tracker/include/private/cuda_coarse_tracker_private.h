// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef PBA_CUDA_COARSE_TRACKER_PRIVATE_H
#define PBA_CUDA_COARSE_TRACKER_PRIVATE_H



// Calc Res outputs
#define IDX_E 0
#define IDX_numTermsInE 1
#define IDX_numTermsInWarped 2
#define IDX_numSaturated 3
#define IDX_sumSquaredShiftT 4
#define IDX_sumSquaredShiftRT 5
#define IDX_sumSquaredShiftNum 6

#include <cuda_runtime_api.h>

void callCalcResKernel(
    int TPB, cudaStream_t stream,
    float setting_huberTH,
    int w, int h, float fx, float fy, float cx, float cy,
    float const *refToNew,
    float const *Ki_in,
    float2 affLL,
    float maxEnergy,
    float cutoffTH,
    int n,
    float const *pc_u,
    float const *pc_v,
    float const *pc_idepth,
    float const *pc_color,
    float const *dInew,
    float *warped_u,
    float *warped_v,
    float *warped_dx,
    float *warped_dy,
    float *warped_idepth,
    float *warped_residual,
    float *warped_weight,
    float *outputs
);

template<typename Accum>
void callCalcGKernel(
    int TPB, cudaStream_t stream,
    float fx, float fy,
    float2 affLL,
    float lastRef_aff_g2l_b,
    int n, int loops,
    float const *pc_color,
    float const *warped_u,
    float const *warped_v,
    float const *warped_dx,
    float const *warped_dy,
    float const *warped_idepth,
    float const *warped_residual,
    float const *warped_weight,
    Accum *outputs);

#endif //PBA_CUDA_COARSE_TRACKER_PRIVATE_H
