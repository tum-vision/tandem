// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef PBA_NUMERIC_CUDA_CUH
#define PBA_NUMERIC_CUDA_CUH

#define CONCATENATE(e1, e2) e1 ## e2
#define MATRIX_LOCAL(name, T, N, M) T CONCATENATE(name, _s)[N*M] = {}; numeric_cuda::Matrix<T, N, M> name(CONCATENATE(name, _s));

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <boost/format.hpp>

#include "utils.h"

namespace pba::numeric_cuda {


template<typename Scalar, typename S=std::size_t>
struct VectorX {
public:
  Scalar *data = nullptr;
  S size = 0;

  __host__ VectorX() = default;

  __host__ explicit VectorX(S s, bool zero = true) { Malloc(s, zero); }

  __host__ void Malloc(S s, bool zero = true) {
    assert(size == 0 && data == nullptr && "Cannot Malloc vector that is already malloced.");
    assert(s > 0 && "Must Malloc more than 0 elements.");
    size = s;
    gpuErrchk(cudaMalloc(&data, Bytes()))
    if (zero)
      Zero();
  }

  __host__ void Free() {
//    assert(size > 0);
    if (size > 0) gpuErrchk(cudaFree(data));
  }

  __device__ __always_inline Scalar &at(int lin) { return data[lin]; }

  __device__ __always_inline Scalar const &at(int lin) const { return data[lin]; }

  __host__ __device__ S Bytes() const { return sizeof(Scalar) * size; }

  __host__ void CopyFrom(Scalar const *const data_in) {
    gpuErrchk(cudaMemcpy(data, data_in, Bytes(), cudaMemcpyHostToDevice))
  }

  __host__ void CopyTo(Scalar *const data_out) {
    gpuErrchk(cudaMemcpy(data_out, data, Bytes(), cudaMemcpyDeviceToHost))
  }

  __host__ std::vector<Scalar> ToVector() {
    std::vector<Scalar> vec(size);
    CopyTo(vec.data());
    return vec;
  }

  __host__ void Print(std::string const &prefix = "") {
    auto const &vec = ToVector();
    std::cout << prefix;
    for (auto const &elem: vec)
      std::cout << elem << ", ";
    std::cout << "\b\b" << "  " << std::endl;
  }

  __host__ void Zero() {
    gpuErrchk(cudaMemset(data, 0, Bytes()))
  }

};

template<typename Scalar, typename S=std::size_t>
struct SparseIndex {
public:
  numeric_cuda::VectorX<Scalar, S> start;
  numeric_cuda::VectorX<Scalar, S> size;

  __host__ void Free() {
    start.Free();
    size.Free();
  }

  __host__ void Print(std::string const &prefix = "") {
    start.Print(prefix + "Start: ");
    size.Print(prefix + "Size : ");
  }
};


template<typename Scalar, int N, int M>
struct Matrix {
public:
  Scalar *data = nullptr;

  __host__ __device__ __always_inline explicit Matrix() : data(nullptr) {}

  __host__ __device__ __always_inline explicit Matrix(Scalar *_data) : data(_data) {}

  __host__ void Print(std::string const &prefix = "") {
    VectorX<Scalar, int> vec;
    vec.data = data;
    vec.size = N * M;
    vec.Print(prefix + "flat(" + std::to_string(N) + "," + std::to_string(M) + ")=");
  }

  __device__ __forceinline__ void PrintDevice() {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++)
        printf("%f ", at(i, j));
      printf("\n");
    }
  }

  __host__ __device__ __always_inline Scalar &at(int row, int col) { return data[row * M + col]; }

  __host__ __device__ __always_inline Scalar const &at(int row, int col) const { return data[row * M + col]; }

  __host__ __device__ __always_inline Scalar &at(int lin) { return data[lin]; }

  __host__ __device__ __always_inline Scalar const &at(int lin) const { return data[lin]; }

  __host__ __device__ __always_inline void Zero() {
#pragma unroll
    for (int i = 0; i < N * M; i++)
      data[i] = Scalar(0);
  }
};

template<typename Scalar, int N, int M>
__device__  inline void PrintDevice(Matrix<Scalar, N, M> mat, char const *prefix = "") {
  assert(false && "General template not implemented due to printf.\n");
}

template<>
__device__  inline void PrintDevice(Matrix<float, 1, 1> mat, char const *prefix) {
  printf("%s(%f)\n", prefix, mat.at(0));
}

template<>
__device__  inline void PrintDevice(Matrix<float, 2, 1> mat, char const *prefix) {
  printf("%s(%f,%f)\n", prefix, mat.at(0), mat.at(1));
}

template<>
__device__  inline void PrintDevice(Matrix<float, 3, 1> mat, char const *prefix) {
  printf("%s(%f,%f,%f)\n", prefix, mat.at(0), mat.at(1), mat.at(2));
}

template<>
__device__  inline void PrintDevice(Matrix<float, 4, 1> mat, char const *prefix) {
  printf("%s(%f,%f,%f,%f)\n", prefix, mat.at(0), mat.at(1), mat.at(2), mat.at(3));
}

template<>
__device__  inline void PrintDevice(Matrix<double, 1, 1> mat, char const *prefix) {
  printf("%s(%f)\n", prefix, mat.at(0));
}

template<>
__device__  inline void PrintDevice(Matrix<double, 2, 1> mat, char const *prefix) {
  printf("%s(%f,%f)\n", prefix, mat.at(0), mat.at(1));
}

template<>
__device__  inline void PrintDevice(Matrix<double, 3, 1> mat, char const *prefix) {
  printf("%s(%f,%f,%f)\n", prefix, mat.at(0), mat.at(1), mat.at(2));
}

template<>
__device__  inline void PrintDevice(Matrix<double, 4, 1> mat, char const *prefix) {
  printf("%s(%f,%f,%f,%f)\n", prefix, mat.at(0), mat.at(1), mat.at(2), mat.at(3));
}

//template<typename Scalar, int N, int M>
//struct __align__(16) MatrixStorage : public Matrix<Scalar, N, M>  {
//public:
//
////  struct {} __align__(16);
//  Scalar storage[N * M] = {0};
//
//  __device__ __always_inline explicit MatrixStorage() : Matrix<Scalar, N, M>(storage) {}
//};
//
//template<typename Scalar, int N, int M>
//struct __align__(16) Storage{
//  Scalar storage[N * M] = {0};
//};


template<typename Scalar, int N, int M>
struct BatchedMatrix {
public:
  typedef size_t S;
  VectorX<Scalar, S> vec;

  __host__ BatchedMatrix() = default;

  __host__ explicit BatchedMatrix(S s, bool zero = true) : vec(s * N * M, zero) {}

  __host__ __device__ __always_inline S Size() const { return vec.size / (N * M); };

  __host__ void Free() { vec.Free(); }

  __host__ __device__ Matrix<Scalar, N, M> at(S idx) { return Matrix<Scalar, N, M>(vec.data + (idx * N * M)); }

  // TODO(minor): The returned Matrix is not really const which is bad.
  __host__ __device__ Matrix<Scalar, N, M> at(S idx) const { return Matrix<Scalar, N, M>(vec.data + (idx * N * M)); }

  __device__ __always_inline Scalar &at(S idx, int row, int col) { return vec.data[idx * N * M + row * M + col]; }

  __device__ __always_inline Scalar const &at(S idx, int row, int col) const { return vec.data[idx * N * M + row * M + col]; }

  __device__ __always_inline Scalar &at(S idx, int lin) { return vec.data[idx * N * M + lin]; }

  __device__ __always_inline Scalar const &at(S idx, int lin) const { return vec.data[idx * N * M + lin]; }

  __host__ void Print(std::string const &prefix = "") {
    for (S b = 0; b < Size(); b++)
      at(b).Print((boost::format("%s bmat[%3d]-") % prefix % b).str());
  }

  __host__ void Zero() { vec.Zero(); }
};

template<typename Scalar, int N, int K, int M>
__inline__ __device__ __host__ void Matmul(const Matrix<Scalar, N, K> A, const Matrix<Scalar, K, M> B, Matrix<Scalar, N, M> &out) {
  /*
   * A: N*K
   * B: K*M
   * out: N*M
   */

#pragma unroll
  for (int row = 0; row < N; row++) {

#pragma unroll
    for (int col = 0; col < M; col++) {

#pragma unroll
      for (int k = 0; k < K; k++) {
        out.at(row, col) += A.at(row, k) * B.at(k, col);
      }
    }
  }
}

namespace collective {
template<typename Scalar, int N, int K, int M, int N_THREADS, int OFFSET = 0>
__inline__ __device__ void Matmul(const Matrix<Scalar, N, K> A, const Matrix<Scalar, K, M> B, Matrix<Scalar, N, M> &out, const int tidx) {
  if (tidx >= OFFSET && tidx < N_THREADS + OFFSET) {
#pragma unroll
    for (int l = 0; l < DIV_UP(N * M, N_THREADS); l++) {
      const int idx = tidx - OFFSET + l * N_THREADS;
      const int row = idx / M;
      const int col = idx % M;

#pragma unroll
      for (int k = 0; k < K; k++) {
        out.at(row, col) += A.at(row, k) * B.at(k, col);
//        printf("out(%d, %d) += A(%d, %d) * B(%d, %d)\n", row, col, row, k, k, col);
      }

    }
  }
}

template<typename Scalar, int N, int N_THREADS, int OFFSET = 0>
__inline__ __device__ void ScalarProduct(const Matrix<Scalar, N, 1> x, const Matrix<Scalar, N, 1> y, Scalar *out, const int tidx) {
  if (tidx >= OFFSET && tidx < N_THREADS + OFFSET) {
    Scalar val = 0;

#pragma unroll
    for (int l = 0; l < DIV_UP(N, N_THREADS); l++) {
      const int idx = tidx - OFFSET + l * N_THREADS;
      val += x.at(idx) * y.at(idx);
    }

    atomicAdd(out, val);
  }
}

} // namespace collective


template<typename Scalar, int N, int K, int M>
__inline__ __device__ void MatmulTransposeA(const Matrix<Scalar, K, N> A, const Matrix<Scalar, K, M> B, Matrix<Scalar, N, M> out) {
  /*
   * A: N*K
   * B: K*M
   * out: N*M
   * TODO(perf): Maybe switch col and k loop for better performance
   */

#pragma unroll
  for (int row = 0; row < N; row++) {

#pragma unroll
    for (int col = 0; col < M; col++) {

#pragma unroll
      for (int k = 0; k < K; k++) {
        out.at(row, col) += A.at(k, row) * B.at(k, col);
      }
    }
  }
}

template<typename Scalar, int N, int K, int M>
__inline__ __device__ void MatmulTransposeB(const Matrix<Scalar, N, K> A, const Matrix<Scalar, M, K> B, Matrix<Scalar, N, M> out) {
  /*
   * A: N*K
   * B: K*M
   * out: N*M
   * TODO(perf): Maybe switch col and k loop for better performance
   */

#pragma unroll
  for (int row = 0; row < N; row++) {

#pragma unroll
    for (int col = 0; col < M; col++) {

#pragma unroll
      for (int k = 0; k < K; k++) {
        out.at(row, col) += A.at(row, k) * B.at(col, k);
      }
    }
  }
}


} // namespace pba::numeric_cuda

#endif //PBA_NUMERIC_CUDA_CUH
