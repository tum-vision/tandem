// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef PBA_UTILS_CUH
#define PBA_UTILS_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// For type_name
#include <type_traits>
#include <typeinfo>

#ifndef _MSC_VER

#   include <cxxabi.h>

#endif

#include <memory>
#include <string>
#include <cstdlib>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
#define DIV_UP(n, div) (((n) + (div) - 1) / (div))

#define cucheck_dev(call) \
  {\
  cudaError_t res = (call);\
  if(res != cudaSuccess) {\
  const char* err_str = cudaGetErrorString(res);\
  printf("%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);  \
  assert(0);                                                        \
  }\
  }

template<bool abort = true>
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

const char *cublasGetErrorEnum(cublasStatus_t error);

inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort = true) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasAssert: %s %s %d\n", cublasGetErrorEnum(code), file, line);
    if (abort) exit(code);
  }
}

template<class T>
std::string type_name() {
  typedef typename std::remove_reference<T>::type TR;
  std::unique_ptr<char, void (*)(void *)> own
      (
#ifndef _MSC_VER
      abi::__cxa_demangle(typeid(TR).name(), nullptr,
                          nullptr, nullptr),
#else
      nullptr,
#endif
      std::free
  );
  std::string r = own != nullptr ? own.get() : typeid(TR).name();
  if (std::is_const<TR>::value)
    r += " const";
  if (std::is_volatile<TR>::value)
    r += " volatile";
  if (std::is_lvalue_reference<T>::value)
    r += "&";
  else if (std::is_rvalue_reference<T>::value)
    r += "&&";
  return r;
}

struct CudaTimer {
public:
  cudaEvent_t start, stop;

  __host__ CudaTimer() {
    gpuErrchk(cudaEventCreate(&start))
    gpuErrchk(cudaEventCreate(&stop))
    gpuErrchk(cudaEventRecord(start))
  }

  __host__ float Synchronize() {
    gpuErrchk(cudaEventRecord(stop))
    gpuErrchk(cudaEventSynchronize(stop))
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop))
    cudaDeviceSynchronize();
    return milliseconds;
  }
};

namespace collective {
template<typename Scalar, int N_VALUES, int N_THREADS, int OFFSET = 0>
__device__ __always_inline void set_memory(Scalar *ptr, Scalar value, const int tidx) {
  if (tidx >= OFFSET && tidx < N_THREADS + OFFSET) {
#pragma unroll
    for (int l = 0; l < DIV_UP(N_VALUES, N_THREADS); l++) {
      const int idx = tidx - OFFSET + l * N_THREADS;
      if (idx < N_VALUES)
        ptr[idx] = value;
    }
  }
}

template<typename Scalar, int N_VALUES, int N_THREADS, int OFFSET = 0>
__device__ __always_inline void load_shared(Scalar *const dst, Scalar const *const src, const int tidx) {
  if (tidx >= OFFSET && tidx < N_THREADS + OFFSET) {
#pragma unroll
    for (int l = 0; l < DIV_UP(N_VALUES, N_THREADS); l++) {
      const int idx = tidx - OFFSET + l * N_THREADS;
      if (idx < N_VALUES)
        dst[idx] = src[idx];
    }
  }
}

template<typename Scalar, int N_VALUES, int N_THREADS, int OFFSET = 0>
__device__ __always_inline void memcpy_shared(Scalar *const dst, Scalar const *const src, const int tidx) {
  if (tidx >= OFFSET && tidx < N_THREADS + OFFSET) {
#pragma unroll
    for (int l = 0; l < DIV_UP(N_VALUES, N_THREADS); l++) {
      const int idx = tidx - OFFSET + l * N_THREADS;
      if (idx < N_VALUES)
        dst[idx] = src[idx];
    }
  }
}

//template<int N_TILES, int N_ROWS, int N_COLS, int N_THREADS>
//__device__ __always_inline void load_tiles_shared(char *const dst, char const *const src, const int tidx) {
//  if (tidx >= OFFSET && tidx < N_THREADS + OFFSET) {
//#pragma unroll
//    for (int l = 0; l < DIV_UP(N_VALUES, N_THREADS); l++) {
//      const int idx = tidx - OFFSET + l * N_THREADS;
//      if (idx < N_VALUES)
//        dst[idx] = src[idx];
//    }
//  }
//}

//template<typename U, typename F, U F2U>
//__device__ __always_inline void LoadImageTiles(F *const shared_tiles, htl2target)

} // namespace collective

#endif //PBA_UTILS_CUH
