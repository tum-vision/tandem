// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once
#include <cuda_runtime.h>
#include "marching_cubes/mesh.h"
#include "tsdfvh/tsdf_volume.h"

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Class for extracting the mesh from a TSDF volume.
 */
class MeshExtractor {
 public:
  /**
   * @brief      Initializes the class and allocate the memory for the mesh.
   *
   * @param[in]  max_triangles  The maximum number of triangles that the mesh
   *                            has
   * @param[in]  voxel_size     The voxel size
   */
  void Init(unsigned int max_triangles, float voxel_size);

  /**
   * @brief      Frees the memory allocated by the class.
   */
  void Free();

  /**
   * @brief      Extracts the mesh from the region of a given TSDF volume
   *             specified by a bounding box
   *
   * @param      volume        The TSDF volume
   * @param[in]  lower_corner  The lower corner of the bounding box
   * @param[in]  upper_corner  The upper corner of the bounding box
   */
  void ExtractMesh(TsdfVolume *volume, float3 lower_corner,
                   float3 upper_corner, cudaStream_t stream = nullptr);

  /**
   * @brief      Gets the mesh.
   *
   * @return     The mesh.
   */
  Mesh GetMesh();

 protected:
  /** The size of a side of a voxel (in meters) */
  float voxel_size_;

  /** The mesh */
  Mesh *mesh_;
};

}  // namespace tsdfvh

}  // namespace refusion
