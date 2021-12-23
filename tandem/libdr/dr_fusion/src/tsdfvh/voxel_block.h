// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>
#include "tsdfvh/voxel.h"

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Class that represents a voxel block.
 */
class VoxelBlock {
 public:
  /**
   * @brief      Initialize the object.
   *
   * @param      first_voxel  The address of the first voxel in the voxel array
   *                          of the hash table
   * @param[in]  block_size   The size in voxels of a side of the block
   */
  void Init(Voxel* first_voxel, int block_size) {
    voxels_ = first_voxel;
    block_size_ = block_size;
  }

  /**
   * @brief      Returns the voxel at a given 3D local (within the block)
   *             position.
   *
   * @param[in]  position  The 3D local position of the desired voxel
   *
   * @return     The voxel at the 3D position.
   */
  __host__ __device__
  Voxel &at(int3 position) {
    return voxels_[position.x * block_size_ * block_size_ +
                   position.y * block_size_ + position.z];
  }

  /**
   * @brief      Returns the voxel at a given index (linear position).
   *
   * @param[in]  idx   The index of the desired voxel
   *
   * @return     The voxel at the index.
   */
  __host__ __device__
  Voxel &at(int idx) {
    return voxels_[idx];
  }

 private:
  /** Address of the first voxel contained in the block */
  Voxel* voxels_;
  /** Size in voxels of a side of the block */
  int block_size_;
};

}  // namespace tsdfvh

}  // namespace refusion
