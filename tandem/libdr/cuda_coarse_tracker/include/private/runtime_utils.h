// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef PBA_RUNTIME_UTILS_H
#define PBA_RUNTIME_UTILS_H

#include <cuda_runtime_api.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define DIV_UP(n, div) (((n) + (div) - 1) / (div))


template<bool abort = true>
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#endif //PBA_RUNTIME_UTILS_H
