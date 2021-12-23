// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#include "cuda_coarse_tracker_private.h"
#include "numeric_cuda.h"
#include <cub/block/block_reduce.cuh>
#include <iostream>


__forceinline__ __device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}


__forceinline__ __device__ float3 getInterpolatedElement33(float const *const mat, const float x, const float y, const int width) {
  int ix = (int) x;
  int iy = (int) y;
  float dx_s = x - ix;
  float dy_s = y - iy;
  const float3 dx = make_float3(dx_s, dx_s, dx_s);
  const float3 dy = make_float3(dy_s, dy_s, dy_s);
  const float3 dxdy = dx * dy;
  const float3 one = make_float3(1.0f, 1.0f, 1.0f);
  float3 const *bp = (float3 * )(mat + 3 * (ix + iy * width));


  return dxdy * bp[1 + width]
         + (dy - dxdy) * bp[width]
         + (dx - dxdy) * bp[1]
         + (one - dx - dy + dxdy) * bp[0];
}

template<int TPB>
__global__ void calcResKernelNew(
    const float setting_huberTH,
    int w, int h, float fx, float fy, float cx, float cy,
    float const *const refToNew,
    float const *const Ki_in,
    const float2 affLL,
    const float maxEnergy,
    const float cutoffTH,
    int n,
    float const *const pc_u,
    float const *const pc_v,
    float const *const pc_idepth,
    float const *const pc_color,
    float const *const dInew,
    float *const warped_u,
    float *const warped_v,
    float *const warped_dx,
    float *const warped_dy,
    float *const warped_idepth,
    float *const warped_residual,
    float *const warped_weight,
    float *const outputs
) {
  typedef cub::BlockReduce<float, TPB, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  float private_output[7] = {};

  using namespace pba;
  namespace nc = pba::numeric_cuda;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    // TODO(minor): Maybe use other default values
    warped_u[i] = 0;
    warped_v[i] = 0;
    warped_dx[i] = 0;
    warped_dy[i] = 0;
    warped_idepth[i] = 0;
    warped_residual[i] = 0;
    warped_weight[i] = 0;

    float id = pc_idepth[i];
    float x = pc_u[i];
    float y = pc_v[i];

    // Prep data
    MATRIX_LOCAL(R, float, 3, 3)
    MATRIX_LOCAL(t, float, 3, 1)
    MATRIX_LOCAL(Ki, float, 3, 3)
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        R.at(row, col) = refToNew[4 * row + col];
        Ki.at(row, col) = Ki_in[row * 3 + col];
      }
      t.at(row) = refToNew[4 * row + 3];
    }

    MATRIX_LOCAL(RKi, float, 3, 3)
    nc::Matmul(R, Ki, RKi);


    MATRIX_LOCAL(xy1, float, 3, 1)
    xy1.at(0) = x;
    xy1.at(1) = y;
    xy1.at(2) = 1;
    MATRIX_LOCAL(pt, float, 3, 1)
    nc::Matmul(RKi, xy1, pt);
#pragma unroll
    for (int row = 0; row < 3; row++)
      pt.at(row) += t.at(row) * id;

    float u = pt.at(0) / pt.at(2);
    float v = pt.at(1) / pt.at(2);
    float Ku = fx * u + cx;
    float Kv = fy * v + cy;
    float new_idepth = id / pt.at(2);

    if (i % 32 == 0) {
      // translation only (positive)
      MATRIX_LOCAL(ptT, float, 3, 1)
      nc::Matmul(Ki, xy1, ptT);
#pragma unroll
      for (int row = 0; row < 3; row++) {
        ptT.at(row) += t.at(row) * id;
      }
      float uT = ptT.at(0) / ptT.at(2);
      float vT = ptT.at(1) / ptT.at(2);
      float KuT = fx * uT + cx;
      float KvT = fy * vT + cy;

      // translation only (negative)
      MATRIX_LOCAL(ptT2, float, 3, 1)
      nc::Matmul(Ki, xy1, ptT2);
#pragma unroll
      for (int row = 0; row < 3; row++)
        ptT2.at(row) -= t.at(row) * id;
      float uT2 = ptT2.at(0) / ptT2.at(2);
      float vT2 = ptT2.at(1) / ptT2.at(2);
      float KuT2 = fx * uT2 + cx;
      float KvT2 = fy * vT2 + cy;

      //translation and rotation (negative)
      MATRIX_LOCAL(pt3, float, 3, 1)
      nc::Matmul(RKi, xy1, pt3);
#pragma unroll
      for (int row = 0; row < 3; row++)
        pt3.at(row) -= t.at(row) * id;
      float u3 = pt3.at(0) / pt3.at(2);
      float v3 = pt3.at(1) / pt3.at(2);
      float Ku3 = fx * u3 + cx;
      float Kv3 = fy * v3 + cy;

      //translation and rotation (positive)
      //already have it.

      float inc_sumSquaredShiftT = (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
      inc_sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
      private_output[IDX_sumSquaredShiftT] = inc_sumSquaredShiftT;

      float inc_sumSquaredShiftRT = (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
      inc_sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
      private_output[IDX_sumSquaredShiftRT] = inc_sumSquaredShiftRT;

      private_output[IDX_sumSquaredShiftNum] = 2.0;
    }

    if (Ku > 2 && Kv > 2 && Ku < w - 3 && Kv < h - 3 && new_idepth > 0) {


      const float refColor = pc_color[i];
      const float3 hitColor = getInterpolatedElement33(dInew, Ku, Kv, w);

      if (isfinite((float) hitColor.x)) {
        const float residual = hitColor.x - (affLL.x * refColor + affLL.y);
        const float hw = abs(residual) < setting_huberTH ? 1 : setting_huberTH / abs(residual);

//        if (i == 0) printf("[calcResGPU] i = %d, residual=%f, Ku=%f, Kv=%f, hitColor.x=%f\n", i, residual, Ku, Kv, hitColor.x);


        if (abs(residual) > cutoffTH) {
          private_output[IDX_E] = maxEnergy;
          private_output[IDX_numTermsInE] = 1;
          private_output[IDX_numSaturated] = 1;

        } else {
          private_output[IDX_E] = hw * residual * residual * (2 - hw);
          private_output[IDX_numTermsInE] = 1;
          private_output[IDX_numTermsInWarped] = 1;

          warped_idepth[i] = new_idepth;
          warped_u[i] = u;
          warped_v[i] = v;
          warped_dx[i] = hitColor.y;
          warped_dy[i] = hitColor.z;
          warped_residual[i] = residual;
          warped_weight[i] = hw;
        }
      }
    }
  }

  float aggregates[7];
  for (int idx = 0; idx < 7; idx++) {
    aggregates[idx] = BlockReduce(temp_storage).Sum(private_output[idx]);
    __syncthreads(); // Needed due to temp_storage reuse
  }

  if (threadIdx.x == 0) {
    for (int idx = 0; idx < 7; idx++) {
      atomicAdd(outputs + idx, aggregates[idx]);
    }
  }
};

void
callCalcResKernel(int TPB, cudaStream_t stream, float setting_huberTH, int w, int h, float fx, float fy, float cx, float cy, const float *refToNew, const float *Ki_in, float2 affLL, float maxEnergy, float cutoffTH, int n, const float *pc_u,
                  const float *pc_v,
                  const float *pc_idepth, const float *pc_color, const float *dInew, float *warped_u, float *warped_v, float *warped_dx, float *warped_dy, float *warped_idepth, float *warped_residual, float *warped_weight, float *outputs) {
  if (TPB != 128) std::cerr << "callCalcResKernel only supports TPB=128. Will use TPB=128." << std::endl;

  calcResKernelNew<128><<<DIV_UP(n, 128), 128, 0, stream>>>(
      setting_huberTH,
      w, h, fx, fy, cx, cy,
      refToNew,
      Ki_in,
      affLL,
      maxEnergy,
      cutoffTH,
      n, pc_u, pc_v, pc_idepth, pc_color,
      dInew,
      warped_u, warped_v, warped_dx, warped_dy, warped_idepth, warped_residual, warped_weight,
      outputs
  );

}

template<int n>
class SimpleVec {
public:
  float data[n] = {};
};

template<int n>
class MySum {
public:
  __host__ __device__
  __forceinline__ SimpleVec<n> operator()(const SimpleVec<n> &a, const SimpleVec<n> &b) const {

    SimpleVec<n> out;
#pragma unroll
    for (int i = 0; i < n; i++)
      out.data[i] = a.data[i] + b.data[i];

    return out;
  }
};


template<int TPB, typename Accum>
__global__ void calcGKernel(
    const float fx, const float fy,
    const float2 affLL,
    const float lastRef_aff_g2l_b,
    const int n,
    const int loops,
    float const *const pc_color,
    float const *const warped_u,
    float const *const warped_v,
    float const *const warped_dx,
    float const *const warped_dy,
    float const *const warped_idepth,
    float const *const warped_residual,
    float const *const warped_weight,
    Accum *const outputs) {
//  assert(TPB >= 45 && "Need 45 threads at least");

  // TODO(perf): Could maybe optimize by using Scalar[45] as type and not a loop
  typedef cub::BlockReduce <Accum, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
//  __shared__ float result_storage[45];

//  typedef cub::BlockReduce <SimpleVec<45>, TPB> BlockReduce;
//  __shared__ typename BlockReduce::TempStorage temp_storage;




  float result_private[45] = {0};

  const float a = affLL.x;
  const float b0 = lastRef_aff_g2l_b;

  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  for (int loop = 0; loop < loops; loop++) {
    const int i = bidx * (blockDim.x * loops) + loop * blockDim.x + tidx;

    float J[9] = {0};
    float w = 0;

    if (i < n) {
//      printf("i=%d, loop=%d\n", i, loop);

      const float dx = warped_dx[i] * fx;
      const float dy = warped_dy[i] * fy;
      const float u = warped_u[i];
      const float v = warped_v[i];
      const float id = warped_idepth[i];


      J[0] = id * dx;
      J[1] = id * dy;
      J[2] = -id * (u * dx + v * dy);
      J[3] = -(u * v * dx + dy + dy * v * v);
      J[4] = u * v * dy + dx + dx * u * u;
      J[5] = u * dy - v * dx;


      J[6] = a * (b0 - pc_color[i]);
      J[7] = -1;
      J[8] = warped_residual[i];
      w = warped_weight[i];

//    if (w*(J[0]+J[1]+J[2]+J[3]+J[4]+J[5]+J[6]+J[7]+J[8]) > 1e9)
//      printf("hi");

//    if (i == 0)printf("[calcGsGPU] i=%d, u=%f, v=%f, res=%f, refCol=%f, a=%f, b0=%f, n=%d, \n", i, u, v, warped_residual[i], pc_color[i], a, b0, n);


//    DO_DEBUG(printf("[calcGs] i=%d, w=%f, J0 = %f\n", i, w, J[0]))
//    if (i == 0) {
//      DO_DEBUG(printf("[calcGs] CCT-2: i=%d, dx=%f, dy=%f, u=%f, v=%f, id=%f\n", i, dx, dy, u, v, id))
//      for (int j = 0; j < 9; j++)
//        printf("[calcGs] i=%d, J[%d]=%f\n", i, j, J[j]);
//    }
      int idx_j = 0;
#pragma unroll
      for (int j1 = 0; j1 < 9; j1++) {
        const float Jj1w = J[j1] * w;
#pragma unroll
        for (int j2 = j1; j2 < 9; j2++) {
          result_private[idx_j] += (Jj1w * J[j2]);
          idx_j++;
        }
      }

    }
  }

  /* --- Reduction --- */
#pragma unroll
  for (int j = 0; j < 45; j++) {
    const Accum sum_block = BlockReduce(temp_storage).Sum((Accum) result_private[j]);
    if (tidx == 0) atomicAdd(outputs + j, sum_block);
    __syncthreads();
//    if (tidx == 0) result_storage[j] = sum_block;

  }

//  int idx_j = 0;
//  for (int j1 = 0; j1 < 9; j1++) {
//    const float Jj1w = J[j1] * w;
//    for (int j2 = j1; j2 < 9; j2++) {
//      const float sum_block = BlockReduce(temp_storage).Sum(Jj1w * J[j2]);
//      __syncthreads();
//      if (tidx == 0) result_storage[idx_j] = sum_block;
//      idx_j++;
//    }
//  }
//  __syncthreads();

//  if (tidx < 45) {
////    if (tidx == 44) printf("[calcGs] i=%d, idx=%d, inc=%f\n", i, tidx, result_storage[tidx]);
//    atomicAdd(outputs + tidx, result_storage[tidx]);
//  }


//  /* --- Reduction --- */
//  SimpleVec<45> in;
//  int idx_j = 0;
//  for (int j1 = 0; j1 < 9; j1++) {
//    const float Jj1w = J[j1] * w;
//    for (int j2 = j1; j2 < 9; j2++) {
//      in.data[idx_j] = Jj1w * J[j2];
//      idx_j++;
//    }
//  }
//  SimpleVec<45> out = BlockReduce(temp_storage).template Reduce(in, MySum<45>());
//  if (tidx == 0){
//    for(int j = 0; j < 45; j++)
//      atomicAdd(outputs + j, out.data[j]);
//  }
}

template
__global__ void calcGKernel<128, float>(
    const float fx, const float fy,
    const float2 affLL,
    const float lastRef_aff_g2l_b,
    const int n, const int loops,
    float const *const pc_color,
    float const *const warped_u,
    float const *const warped_v,
    float const *const warped_dx,
    float const *const warped_dy,
    float const *const warped_idepth,
    float const *const warped_residual,
    float const *const warped_weight,
    float *const outputs);

template
__global__ void calcGKernel<4, float>(
    const float fx, const float fy,
    const float2 affLL,
    const float lastRef_aff_g2l_b,
    const int n, const int loops,
    float const *const pc_color,
    float const *const warped_u,
    float const *const warped_v,
    float const *const warped_dx,
    float const *const warped_dy,
    float const *const warped_idepth,
    float const *const warped_residual,
    float const *const warped_weight,
    float *const outputs);

template
__global__ void calcGKernel<128, double>(
    const float fx, const float fy,
    const float2 affLL,
    const float lastRef_aff_g2l_b,
    const int n, const int loops,
    float const *const pc_color,
    float const *const warped_u,
    float const *const warped_v,
    float const *const warped_dx,
    float const *const warped_dy,
    float const *const warped_idepth,
    float const *const warped_residual,
    float const *const warped_weight,
    double *const outputs);

template
__global__ void calcGKernel<4, double>(
    const float fx, const float fy,
    const float2 affLL,
    const float lastRef_aff_g2l_b,
    const int n, const int loops,
    float const *const pc_color,
    float const *const warped_u,
    float const *const warped_v,
    float const *const warped_dx,
    float const *const warped_dy,
    float const *const warped_idepth,
    float const *const warped_residual,
    float const *const warped_weight,
    double *const outputs);

template<typename Accum>
void
callCalcGKernel(int TPB, cudaStream_t stream, float fx, float fy, float2 affLL, float lastRef_aff_g2l_b, int n, int loops, const float *pc_color, const float *warped_u, const float *warped_v, const float *warped_dx, const float *warped_dy,
                const float *warped_idepth,
                const float *warped_residual, const float *warped_weight, Accum *outputs) {
  if (TPB != 128 && TPB != 4) std::cerr << "callCalcGKernel only supports TPB=128. Will use TPB=128." << std::endl;

  if (TPB == 128) {
    calcGKernel<128, Accum><<<DIV_UP(n, (TPB * loops)), TPB, 0, stream>>>(
        fx, fy,
        affLL,
        lastRef_aff_g2l_b,
        n, loops,
        pc_color,
        warped_u, warped_v, warped_dx, warped_dy, warped_idepth, warped_residual, warped_weight,
        outputs
    );
  }

  if (TPB == 4) {
    calcGKernel<4, Accum><<<DIV_UP(n, (TPB * loops)), TPB, 0, stream>>>(
        fx, fy,
        affLL,
        lastRef_aff_g2l_b,
        n, loops,
        pc_color,
        warped_u, warped_v, warped_dx, warped_dy, warped_idepth, warped_residual, warped_weight,
        outputs
    );
  }
}

template
void
callCalcGKernel<float>(int TPB, cudaStream_t stream, float fx, float fy, float2 affLL, float lastRef_aff_g2l_b, int n, int loops, const float *pc_color, const float *warped_u, const float *warped_v, const float *warped_dx,
                       const float *warped_dy,
                       const float *warped_idepth,
                       const float *warped_residual, const float *warped_weight, float *outputs);

template
void
callCalcGKernel<double>(int TPB, cudaStream_t stream, float fx, float fy, float2 affLL, float lastRef_aff_g2l_b, int n, int loops, const float *pc_color, const float *warped_u, const float *warped_v, const float *warped_dx,
                        const float *warped_dy,
                        const float *warped_idepth,
                        const float *warped_residual, const float *warped_weight, double *outputs);
