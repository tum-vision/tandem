// Copyright (c) 2020 Lukas Koestler, Nan Yang. All rights reserved.
#include <cuda_runtime.h>
#include <chrono>
#include "dr_fusion.h"
#include "utils/rgbd_sensor.h"
#include "tsdfvh/tsdf_volume.h"

DrFusion::DrFusion(const struct DrFusionOptions &options) {
  sensor_ = std::make_unique<refusion::RgbdSensor>();
  sensor_->cx = options.cx;
  sensor_->cy = options.cy;
  sensor_->fx = options.fx;
  sensor_->fy = options.fy;
  sensor_->rows = options.height;
  sensor_->cols = options.width;
  sensor_->depth_factor = 5000;

  refusion::tsdfvh::TsdfVolumeOptions tsdf_options{
      .voxel_size = options.voxel_size,
      .num_buckets = options.num_buckets,
      .bucket_size = options.bucket_size,
      .num_blocks = options.num_blocks,
      .block_size = options.block_size,
      .max_sdf_weight = options.max_sdf_weight,
      .truncation_distance = options.truncation_distance,
      .max_sensor_depth = options.max_sensor_depth,
      .min_sensor_depth = options.min_sensor_depth,
      .num_render_streams = options.num_render_streams,
      .height = options.height,
      .width = options.width,
  };

  cudaMallocManaged(&volume_, sizeof(refusion::tsdfvh::TsdfVolume));
  volume_->Init(tsdf_options);

  dr_mesh_vert = (float *) malloc(sizeof(float) * dr_mesh_num_max * 3);
  dr_mesh_cols = (float *) malloc(sizeof(float) * dr_mesh_num_max * 3);
}

// https://stackoverflow.com/questions/9954518/stdunique-ptr-with-an-incomplete-type-wont-compile
DrFusion::~DrFusion() {
  cudaDeviceSynchronize();
  volume_->Free();
  cudaDeviceSynchronize();
  cudaFree(volume_);
}


void DrFusion::IntegrateScanAsync(unsigned char *bgr, float *depth, float const *pose) {
  refusion::float4x4 pose_cuda(pose);
  volume_->IntegrateScanAsync(*sensor_, bgr, depth, pose_cuda);
}

void DrFusion::RenderAsync(std::vector<float const *> camera_poses) {
  std::vector<refusion::float4x4> poses_cuda;
  for (auto const &p: camera_poses)
    poses_cuda.push_back(refusion::float4x4(p));

  volume_->RenderAsync(poses_cuda, *sensor_);
}

void DrFusion::GetRenderResult(std::vector<unsigned char *> &bgr, std::vector<float *> &depth) {
  volume_->GetRenderResult(bgr, depth);
}

void DrFusion::Synchronize() {
  cudaDeviceSynchronize();
}

float3 ToFloat3(float in[3]) {
  return float3{.x = in[0], .y = in[1], .z = in[2]};
}

void DrFusion::SaveMeshToFile(const std::string &filename, float *lower_corner, float *upper_corner) {
  // TODO: Let's hope it's ok I removed this
  // Moved the sync into the volume but it's hacky
//    Synchronize();

  auto start = std::chrono::high_resolution_clock::now();
  refusion::tsdfvh::Mesh mesh = volume_->ExtractMesh(ToFloat3(lower_corner), ToFloat3(upper_corner));
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  printf("DrFusion::SaveMeshToFile volume_->ExtractMesh (%3d ms)\n", (int) elapsed);

  start = std::chrono::high_resolution_clock::now();
  mesh.SaveToFile(filename, true);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  printf("DrFusion::SaveMeshToFile mesh.SaveToFile (%3d ms)\n", (int) elapsed);

  // TODO: Is this needed
//    cudaFree(&mesh);
}

struct DrMesh DrFusion::GetMesh(float lower_corner[3], float upper_corner[3]) {
  auto start = std::chrono::high_resolution_clock::now();
  refusion::tsdfvh::Mesh mesh = volume_->ExtractMesh(ToFloat3(lower_corner), ToFloat3(upper_corner));
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  printf("DrFusion::GetMesh volume_->ExtractMesh (%3d ms)\n", (int) elapsed);

  start = std::chrono::high_resolution_clock::now();
  struct DrMesh dr_mesh;
  dr_mesh.num = 3 * mesh.num_triangles_;  // 1 triangle = 3 vert
  dr_mesh.vert = (float *) malloc(dr_mesh.num * sizeof(float) * 3);
  dr_mesh.cols = (float *) malloc(dr_mesh.num * sizeof(float) * 3);

  for (size_t i_tri = 0; i_tri < mesh.num_triangles_; i_tri++) {
    size_t idx;
    // v0
    idx = 9 * i_tri + 3 * 0;
    dr_mesh.vert[idx + 0] = mesh.triangles_[i_tri].v0.position.x;
    dr_mesh.vert[idx + 1] = mesh.triangles_[i_tri].v0.position.y;
    dr_mesh.vert[idx + 2] = mesh.triangles_[i_tri].v0.position.z;

    dr_mesh.cols[idx + 0] = mesh.triangles_[i_tri].v0.color.z;
    dr_mesh.cols[idx + 1] = mesh.triangles_[i_tri].v0.color.y;
    dr_mesh.cols[idx + 2] = mesh.triangles_[i_tri].v0.color.x;

    // v1
    idx = 9 * i_tri + 3 * 1;
    dr_mesh.vert[idx + 0] = mesh.triangles_[i_tri].v1.position.x;
    dr_mesh.vert[idx + 1] = mesh.triangles_[i_tri].v1.position.y;
    dr_mesh.vert[idx + 2] = mesh.triangles_[i_tri].v1.position.z;

    dr_mesh.cols[idx + 0] = mesh.triangles_[i_tri].v1.color.z;
    dr_mesh.cols[idx + 1] = mesh.triangles_[i_tri].v1.color.y;
    dr_mesh.cols[idx + 2] = mesh.triangles_[i_tri].v1.color.x;

    // v2
    idx = 9 * i_tri + 3 * 2;
    dr_mesh.vert[idx + 0] = mesh.triangles_[i_tri].v2.position.x;
    dr_mesh.vert[idx + 1] = mesh.triangles_[i_tri].v2.position.y;
    dr_mesh.vert[idx + 2] = mesh.triangles_[i_tri].v2.position.z;

    dr_mesh.cols[idx + 0] = mesh.triangles_[i_tri].v2.color.z;
    dr_mesh.cols[idx + 1] = mesh.triangles_[i_tri].v2.color.y;
    dr_mesh.cols[idx + 2] = mesh.triangles_[i_tri].v2.color.x;
  }
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  printf("DrFusion::GetMesh copy data (%3d ms)\n", (int) elapsed);

  // TODO(correctness): Is this needed
//    cudaFree(&mesh);

  return dr_mesh;
}

void DrFusion::ExtractMeshAsync(float *lower_corner, float *upper_corner) {
#ifdef DR_FUSION_DEBUG
  auto start = std::chrono::high_resolution_clock::now();
#endif

  volume_->ExtractMeshAsync(ToFloat3(lower_corner), ToFloat3(upper_corner));

#ifdef DR_FUSION_DEBUG
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  printf("DrFusion::ExtractMeshAsync (%3d ms)\n", (int) elapsed);
#endif
}

void DrFusion::GetMeshSync() {
#ifdef DR_FUSION_DEBUG
  auto start = std::chrono::high_resolution_clock::now();
#endif

  volume_->GetMeshSync(dr_mesh_num_max, &dr_mesh_num, dr_mesh_vert, dr_mesh_cols);

#ifdef DR_FUSION_DEBUG
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start).count();
  printf("DrFusion::GetMeshSync (%3d ms)\n", (int) elapsed);
#endif
}
