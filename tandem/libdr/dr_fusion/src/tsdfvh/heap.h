// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Class handling the indices of the voxel blocks.
 */
class Heap {
 public:
  /**
   * @brief      Allocates the memory necessary for the heap
   *
   * @param[in]  heap_size  The maximum number of indices that can be assigned
   */
  void Init(int heap_size);

  /**
   * @brief      Function to request an index to be assigned to a voxel block.
   *
   * @return     The index to be assigned to a voxel block (i.e., consumed from
   *             the heap).
   */
  __device__ unsigned int Consume();

  /**
   * @brief      Frees the given index.
   *
   * @param[in]  ptr   The index to be freed (i.e., appended to the heap)
   */
  __device__ void Append(unsigned int ptr);

  /** Vector of the indices currently assigned */
  unsigned int *heap_;

  /** Index of the element of heap_ that contains the next available index */
  int heap_counter_;
};

}  // namespace tsdfvh

}  // namespace refusion
