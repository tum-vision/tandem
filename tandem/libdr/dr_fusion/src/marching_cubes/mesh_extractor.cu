// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
// Part of the code in this file was adapted from the original VoxelHashing
// implementation by Niessner et al.
// https://github.com/niessner/VoxelHashing/blob/master/DepthSensingCUDA/Source/cuda_SimpleMatrixUtil.h
#include "marching_cubes/mesh_extractor.h"
#include "utils/utils.h"
#include "marching_cubes/lookup_tables.h"

namespace refusion {

namespace tsdfvh {

void MeshExtractor::Init(unsigned int max_triangles, float voxel_size) {
  cudaMallocManaged(&mesh_, sizeof(Mesh));
  mesh_->Init(max_triangles);
  voxel_size_ = voxel_size;
}

void MeshExtractor::Free() {
  mesh_->Free();
  cudaFree(mesh_);
}

__device__ bool TrilinearInterpolation(TsdfVolume *volume, float voxel_size,
                                       const float3 &position, float &distance,
                                       uchar3 &color) {
  // TODO(enhancement): Here we could specify a weight that could be used for mesh extraction.
  const float3 pos_dual =
      position -
      make_float3(voxel_size / 2.0f, voxel_size / 2.0f, voxel_size / 2.0f);
  float3 voxel_position = position / voxel_size;
  float3 weight = make_float3(voxel_position.x - floor(voxel_position.x),
                              voxel_position.y - floor(voxel_position.y),
                              voxel_position.z - floor(voxel_position.z));

  distance = 0.0f;
  float3 color_float = make_float3(0.0f, 0.0f, 0.0f);

  Voxel v = volume->GetVoxel(pos_dual + make_float3(0.0f, 0.0f, 0.0f));
  if (v.weight == 0) return false;
  float3 vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
  color_float =
      color_float +
      (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, 0.0f));
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
  color_float =
      color_float + weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(0.0f, voxel_size, 0.0f));
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v.sdf;
  color_float =
      color_float + (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(0.0f, 0.0f, voxel_size));
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v.sdf;
  color_float =
      color_float + (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(voxel_size, voxel_size, 0.0f));
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += weight.x * weight.y * (1.0f - weight.z) * v.sdf;
  color_float = color_float + weight.x * weight.y * (1.0f - weight.z) * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(0.0f, voxel_size, voxel_size));
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += (1.0f - weight.x) * weight.y * weight.z * v.sdf;
  color_float = color_float + (1.0f - weight.x) * weight.y * weight.z * vColor;

  v = volume->GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, voxel_size));
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += weight.x * (1.0f - weight.y) * weight.z * v.sdf;
  color_float = color_float + weight.x * (1.0f - weight.y) * weight.z * vColor;

  v = volume->GetVoxel(pos_dual +
                       make_float3(voxel_size, voxel_size, voxel_size));
  if (v.weight == 0) return false;
  vColor = make_float3(v.color.x, v.color.y, v.color.z);
  distance += weight.x * weight.y * weight.z * v.sdf;
  color_float = color_float + weight.x * weight.y * weight.z * vColor;

  color = make_uchar3(color_float.x, color_float.y, color_float.z);

  return true;
}

__device__ Vertex VertexInterpolation(float isolevel, const float3 &p1,
                                      const float3 &p2, float d1, float d2,
                                      const uchar3 &c1, const uchar3 &c2) {
  Vertex r1; r1.position = p1; r1.color = make_float3(c1.x, c1.y, c1.z) / 255.f;
  Vertex r2; r2.position = p2; r2.color = make_float3(c2.x, c2.y, c2.z) / 255.f;

  if (fabs(isolevel - d1) < 0.00001f) return r1;
  if (fabs(isolevel - d2) < 0.00001f) return r2;
  if (fabs(d1 - d2) < 0.00001f) return r1;

  float mu = (isolevel - d1) / (d2 - d1);

  Vertex res;
  // Position
  res.position.x = p1.x + mu * (p2.x - p1.x);
  res.position.y = p1.y + mu * (p2.y - p1.y);
  res.position.z = p1.z + mu * (p2.z - p1.z);

  // Color
  res.color.x =
      static_cast<float>(c1.x + mu * static_cast<float>(c2.x - c1.x)) / 255.f;
  res.color.y =
      static_cast<float>(c1.y + mu * static_cast<float>(c2.y - c1.y)) / 255.f;
  res.color.z =
      static_cast<float>(c1.z + mu * static_cast<float>(c2.z - c1.z)) / 255.f;

  return res;
}

__device__ void ExtractMeshAtPosition(TsdfVolume *volume,
                                      const float3 &position, float voxel_size,
                                      Mesh *mesh) {
  const float isolevel = 0.0f;
  const float P = voxel_size/2.0f;
  const float M = -P;

  float3 p000 = position + make_float3(M, M, M);
  float dist000;
  uchar3 color000;
  if (!TrilinearInterpolation(volume, voxel_size, p000, dist000, color000))
    return;

  float3 p100 = position + make_float3(P, M, M);
  float dist100;
  uchar3 color100;
  if (!TrilinearInterpolation(volume, voxel_size, p100, dist100, color100))
    return;

  float3 p010 = position + make_float3(M, P, M);
  float dist010;
  uchar3 color010;
  if (!TrilinearInterpolation(volume, voxel_size, p010, dist010, color010))
    return;

  float3 p001 = position + make_float3(M, M, P);
  float dist001;
  uchar3 color001;
  if (!TrilinearInterpolation(volume, voxel_size, p001, dist001, color001))
    return;

  float3 p110 = position + make_float3(P, P, M);
  float dist110;
  uchar3 color110;
  if (!TrilinearInterpolation(volume, voxel_size, p110, dist110, color110))
    return;

  float3 p011 = position + make_float3(M, P, P);
  float dist011;
  uchar3 color011;
  if (!TrilinearInterpolation(volume, voxel_size, p011, dist011, color011))
    return;

  float3 p101 = position + make_float3(P, M, P);
  float dist101;
  uchar3 color101;
  if (!TrilinearInterpolation(volume, voxel_size, p101, dist101, color101))
    return;

  float3 p111 = position + make_float3(P, P, P);
  float dist111;
  uchar3 color111;
  if (!TrilinearInterpolation(volume, voxel_size, p111, dist111, color111))
    return;

  uint cubeindex = 0;
  if (dist010 < isolevel) cubeindex += 1;
  if (dist110 < isolevel) cubeindex += 2;
  if (dist100 < isolevel) cubeindex += 4;
  if (dist000 < isolevel) cubeindex += 8;
  if (dist011 < isolevel) cubeindex += 16;
  if (dist111 < isolevel) cubeindex += 32;
  if (dist101 < isolevel) cubeindex += 64;
  if (dist001 < isolevel) cubeindex += 128;

  if (edgeTable[cubeindex] == 0) return;

  Voxel v = volume->GetVoxel(position);

  Vertex vertlist[12];
  if (edgeTable[cubeindex] & 1)
    vertlist[0] = VertexInterpolation(isolevel, p010, p110, dist010, dist110,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 2)
    vertlist[1] = VertexInterpolation(isolevel, p110, p100, dist110, dist100,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 4)
    vertlist[2] = VertexInterpolation(isolevel, p100, p000, dist100, dist000,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 8)
    vertlist[3] = VertexInterpolation(isolevel, p000, p010, dist000, dist010,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 16)
    vertlist[4] = VertexInterpolation(isolevel, p011, p111, dist011, dist111,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 32)
    vertlist[5] = VertexInterpolation(isolevel, p111, p101, dist111, dist101,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 64)
    vertlist[6] = VertexInterpolation(isolevel, p101, p001, dist101, dist001,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 128)
    vertlist[7] = VertexInterpolation(isolevel, p001, p011, dist001, dist011,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 256)
    vertlist[8] = VertexInterpolation(isolevel, p010, p011, dist010, dist011,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 512)
    vertlist[9] = VertexInterpolation(isolevel, p110, p111, dist110, dist111,
                                      v.color, v.color);
  if (edgeTable[cubeindex] & 1024)
    vertlist[10] = VertexInterpolation(isolevel, p100, p101, dist100, dist101,
                                       v.color, v.color);
  if (edgeTable[cubeindex] & 2048)
    vertlist[11] = VertexInterpolation(isolevel, p000, p001, dist000, dist001,
                                       v.color, v.color);

  for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
    Triangle t;
    t.v0 = vertlist[triTable[cubeindex][i + 0]];
    t.v1 = vertlist[triTable[cubeindex][i + 1]];
    t.v2 = vertlist[triTable[cubeindex][i + 2]];

    mesh->AppendTriangle(t);
  }
}

__global__ void ExtractMeshKernel(TsdfVolume *volume, float3 lower_corner,
                                  float3 upper_corner, float voxel_size,
                                  Mesh *mesh) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float3 size = lower_corner - upper_corner;
  size = make_float3(fabs(size.x), fabs(size.y), fabs(size.z));
  int3 grid_size =
      make_int3(size.x / voxel_size, size.y / voxel_size, size.z / voxel_size);
  int grid_linear_size = grid_size.x * grid_size.y * grid_size.z;
  for (int i = index; i < grid_linear_size; i += stride) {
    // Delinearize index
    int3 grid_position =
        make_int3(i % grid_size.x, (i / grid_size.x) % grid_size.y,
                  i / (grid_size.x * grid_size.y));
    float3 world_position = make_float3(
        static_cast<float>(grid_position.x) * voxel_size + lower_corner.x,
        static_cast<float>(grid_position.y) * voxel_size + lower_corner.y,
        static_cast<float>(grid_position.z) * voxel_size + lower_corner.z);
    ExtractMeshAtPosition(volume, world_position, voxel_size, mesh);
  }
}

void MeshExtractor::ExtractMesh(TsdfVolume *volume, float3 lower_corner,
                                float3 upper_corner, cudaStream_t stream) {
  int threads_per_block = 256;
  int thread_blocks =
      (volume->GetOptions().num_blocks * volume->GetOptions().block_size +
       threads_per_block - 1) /
      threads_per_block;
  if (stream == nullptr) {
    ExtractMeshKernel<<<thread_blocks, threads_per_block>>>(
        volume, lower_corner, upper_corner, voxel_size_, mesh_);
  }else{
    ExtractMeshKernel<<<thread_blocks, threads_per_block, 0, stream>>>(
        volume, lower_corner, upper_corner, voxel_size_, mesh_);
  }
  }

Mesh MeshExtractor::GetMesh() {
    cudaDeviceSynchronize();
    return *mesh_;
}

}  // namespace tsdfvh

}  // namespace refusion
