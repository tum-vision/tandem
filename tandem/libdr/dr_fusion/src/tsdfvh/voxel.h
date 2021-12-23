// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Struct representing a voxel.
 */
struct Voxel {
  /** Signed distance function */
  float sdf;

  /** Color */
  uchar3 color;

  /** Accumulated SDF weight */
  unsigned char weight;

  /**
   * @brief      Combine the voxel with a given one
   *
   * @param[in]  voxel       The voxel to be combined with
   * @param[in]  max_weight  The maximum weight
   */
  __host__ __device__ void Combine(const Voxel& voxel,
                                   unsigned char max_weight) {
    color.x = static_cast<unsigned char>(
        (static_cast<float>(color.x) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.x) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) +
        static_cast<float>(voxel.weight)));
    color.y = static_cast<unsigned char>(
        (static_cast<float>(color.y) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.y) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) +
        static_cast<float>(voxel.weight)));
    color.z = static_cast<unsigned char>(
        (static_cast<float>(color.z) * static_cast<float>(weight) +
         static_cast<float>(voxel.color.z) * static_cast<float>(voxel.weight)) /
            (static_cast<float>(weight) +
        static_cast<float>(voxel.weight)));

    sdf = (sdf * static_cast<float>(weight) +
          voxel.sdf * static_cast<float>(voxel.weight)) /
              (static_cast<float>(weight) + static_cast<float>(voxel.weight));

    weight = weight + voxel.weight;
    if (weight > max_weight) weight = max_weight;
  }
};

}  // namespace tsdfvh

}  // namespace refusion
