// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "tsdfvh/tsdf_volume.h"
#include <cfloat>
#include <cmath>
#include "marching_cubes/mesh_extractor.h"

#define THREADS_PER_BLOCK2 64

namespace refusion {

    namespace tsdfvh {

        void eventCreate(Event *event) {
            cudaEventCreateWithFlags(&event->cpy_htd, eventFlags());
            cudaEventCreateWithFlags(&event->compute, eventFlags());
            cudaEventCreateWithFlags(&event->cpy_dth, eventFlags());
        }

        void eventDestroy(Event *event) {
            cudaEventDestroy(event->cpy_htd);
            cudaEventDestroy(event->compute);
            cudaEventDestroy(event->cpy_dth);
        }

        void eventSynchronize(Event *event) {
            cudaEventSynchronize(event->cpy_dth);
        }

        TsdfVolume::~TsdfVolume() {
            cudaDeviceSynchronize();

            gpuErrchk(cudaStreamDestroy(int_stream_))
            gpuErrchk(cudaStreamDestroy(mesh_stream_))

            for (auto x: render_streams_) gpuErrchk(cudaStreamDestroy(x))
            for (auto x: render_events_) { delete x; }

            delete int_event_;

            gpuErrchk(cudaFree(d_bgr_in_))
            gpuErrchk(cudaFree(d_depth_in_))

            gpuErrchk(cudaFreeHost(h_bgr_in_))
            gpuErrchk(cudaFreeHost(h_depth_in_))

            for (int i = 0; i < options_.num_render_streams; i++) {
                gpuErrchk(cudaFree(d_bgr_render_[i]))
                gpuErrchk(cudaFree(d_depth_render_[i]))

                gpuErrchk(cudaFreeHost(std::get<0>(h_bgr_render_)[i]))
                gpuErrchk(cudaFreeHost(std::get<1>(h_bgr_render_)[i]))
                gpuErrchk(cudaFreeHost(std::get<0>(h_depth_render_)[i]))
                gpuErrchk(cudaFreeHost(std::get<1>(h_depth_render_)[i]))
            }
        }

        void TsdfVolume::Init(const TsdfVolumeOptions &options) {
            options_ = options;
            HashTable::Init(options_.num_buckets, options_.bucket_size,
                            options_.num_blocks, options_.block_size);

            should_call_next_ = "IntegrateScanAsync";

            // TODO(perf): Using low priority for now but maybe higher is better
            int least_prio, greatest_prio;
            gpuErrchk(cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio));

            gpuErrchk(cudaStreamCreateWithPriority(&int_stream_, cudaStreamNonBlocking, least_prio))
            gpuErrchk(cudaStreamCreateWithPriority(&mesh_stream_, cudaStreamNonBlocking, least_prio))
            for (int i = 0; i < options_.num_render_streams; i++) {
                cudaStream_t render_stream;
                gpuErrchk(cudaStreamCreateWithPriority(&render_stream, cudaStreamNonBlocking, least_prio))
                render_streams_.push_back(render_stream);
                render_events_.push_back(nullptr);
            }

            gpuErrchk(cudaMalloc(&d_bgr_in_, sizeof(uchar3) * num_pixels()))
            gpuErrchk(cudaMalloc(&d_depth_in_, sizeof(float) * num_pixels()))

            gpuErrchk(cudaMallocHost(&h_bgr_in_, sizeof(uchar3) * num_pixels()))
            gpuErrchk(cudaMallocHost(&h_depth_in_, sizeof(float) * num_pixels()))

            uchar3 *h_bgr, *d_bgr;
            float *h_depth, *d_depth;
            for (int i = 0; i < options_.num_render_streams; i++) {
                gpuErrchk(cudaMalloc(&d_bgr, sizeof(uchar3) * num_pixels()))
                d_bgr_render_.push_back(d_bgr);
                gpuErrchk(cudaMalloc(&d_depth, sizeof(float) * num_pixels()))
                d_depth_render_.push_back(d_depth);

                gpuErrchk(cudaMallocHost(&h_bgr, sizeof(uchar3) * num_pixels()))
                std::get<0>(h_bgr_render_).push_back(h_bgr);
                gpuErrchk(cudaMallocHost(&h_bgr, sizeof(uchar3) * num_pixels()))
                std::get<1>(h_bgr_render_).push_back(h_bgr);

                gpuErrchk(cudaMallocHost(&h_depth, sizeof(float) * num_pixels()))
                std::get<0>(h_depth_render_).push_back(h_depth);
                gpuErrchk(cudaMallocHost(&h_depth, sizeof(float) * num_pixels()))
                std::get<1>(h_depth_render_).push_back(h_depth);
            }
        }

        __host__ __device__ float3 TsdfVolume::GlobalVoxelToWorld(int3 position) {
            return make_float3(position.x * options_.voxel_size,
                               position.y * options_.voxel_size,
                               position.z * options_.voxel_size);
        }

        __host__ __device__ int3 TsdfVolume::WorldToGlobalVoxel(float3 position) {
            return make_int3(position.x / options_.voxel_size + signf(position.x) * 0.5f,
                             position.y / options_.voxel_size + signf(position.y) * 0.5f,
                             position.z / options_.voxel_size + signf(position.z) * 0.5f);
        }

        __host__ __device__ int3 TsdfVolume::WorldToBlock(float3 position) {
            int3 voxel_position = WorldToGlobalVoxel(position);
            int3 block_position;
            if (voxel_position.x < 0)
                block_position.x = (voxel_position.x - block_size_ + 1) / block_size_;
            else
                block_position.x = voxel_position.x / block_size_;

            if (voxel_position.y < 0)
                block_position.y = (voxel_position.y - block_size_ + 1) / block_size_;
            else
                block_position.y = voxel_position.y / block_size_;

            if (voxel_position.z < 0)
                block_position.z = (voxel_position.z - block_size_ + 1) / block_size_;
            else
                block_position.z = voxel_position.z / block_size_;

            return block_position;
        }

        __host__ __device__ int3 TsdfVolume::WorldToLocalVoxel(float3 position) {
            int3 position_global = WorldToGlobalVoxel(position);
            int3 position_local = make_int3(position_global.x % block_size_,
                                            position_global.y % block_size_,
                                            position_global.z % block_size_);
            if (position_local.x < 0) position_local.x += block_size_;
            if (position_local.y < 0) position_local.y += block_size_;
            if (position_local.z < 0) position_local.z += block_size_;
            return position_local;
        }

        __host__ __device__ Voxel TsdfVolume::GetVoxel(float3 position) {
            int3 block_position = WorldToBlock(position);
            int3 local_voxel = WorldToLocalVoxel(position);
            HashEntry entry = HashTable::FindHashEntry(block_position);
            if (entry.pointer == kFreeEntry) {
                Voxel voxel;
                voxel.sdf = 0;
                voxel.color = make_uchar3(0, 0, 0);
                voxel.weight = 0;
                return voxel;
            }
            return HashTable::voxel_blocks_[entry.pointer].at(local_voxel);
        }

        __host__ __device__ Voxel TsdfVolume::GetInterpolatedVoxel(float3 position) {
            Voxel v0 = GetVoxel(position);
            if (v0.weight == 0) return v0;
            float voxel_size = options_.voxel_size;
            const float3 pos_dual =
                    position -
                    make_float3(voxel_size / 2.0f, voxel_size / 2.0f, voxel_size / 2.0f);
            float3 voxel_position = position / voxel_size;
            float3 weight = make_float3(voxel_position.x - floor(voxel_position.x),
                                        voxel_position.y - floor(voxel_position.y),
                                        voxel_position.z - floor(voxel_position.z));

            float distance = 0.0f;
            float3 color_float = make_float3(0.0f, 0.0f, 0.0f);
            float3 vColor;

            Voxel v = GetVoxel(pos_dual + make_float3(0.0f, 0.0f, 0.0f));
            if (v.weight == 0) {
                vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
                distance +=
                        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v0.sdf;
                color_float =
                        color_float +
                        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
            } else {
                vColor = make_float3(v.color.x, v.color.y, v.color.z);
                distance +=
                        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
                color_float =
                        color_float +
                        (1.0f - weight.x) * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
            }

            v = GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, 0.0f));
            if (v.weight == 0) {
                vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
                distance += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v0.sdf;
                color_float =
                        color_float + weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
            } else {
                vColor = make_float3(v.color.x, v.color.y, v.color.z);
                distance += weight.x * (1.0f - weight.y) * (1.0f - weight.z) * v.sdf;
                color_float =
                        color_float + weight.x * (1.0f - weight.y) * (1.0f - weight.z) * vColor;
            }

            v = GetVoxel(pos_dual + make_float3(0.0f, voxel_size, 0.0f));
            if (v.weight == 0) {
                vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
                distance += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v0.sdf;
                color_float =
                        color_float + (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;
            } else {
                vColor = make_float3(v.color.x, v.color.y, v.color.z);
                distance += (1.0f - weight.x) * weight.y * (1.0f - weight.z) * v.sdf;
                color_float =
                        color_float + (1.0f - weight.x) * weight.y * (1.0f - weight.z) * vColor;
            }

            v = GetVoxel(pos_dual + make_float3(0.0f, 0.0f, voxel_size));
            if (v.weight == 0) {
                vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
                distance += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v0.sdf;
                color_float =
                        color_float + (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;
            } else {
                vColor = make_float3(v.color.x, v.color.y, v.color.z);
                distance += (1.0f - weight.x) * (1.0f - weight.y) * weight.z * v.sdf;
                color_float =
                        color_float + (1.0f - weight.x) * (1.0f - weight.y) * weight.z * vColor;
            }

            v = GetVoxel(pos_dual + make_float3(voxel_size, voxel_size, 0.0f));
            if (v.weight == 0) {
                vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
                distance += weight.x * weight.y * (1.0f - weight.z) * v0.sdf;
                color_float =
                        color_float + weight.x * weight.y * (1.0f - weight.z) * vColor;
            } else {
                vColor = make_float3(v.color.x, v.color.y, v.color.z);
                distance += weight.x * weight.y * (1.0f - weight.z) * v.sdf;
                color_float =
                        color_float + weight.x * weight.y * (1.0f - weight.z) * vColor;
            }

            v = GetVoxel(pos_dual + make_float3(0.0f, voxel_size, voxel_size));
            if (v.weight == 0) {
                vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
                distance += (1.0f - weight.x) * weight.y * weight.z * v0.sdf;
                color_float =
                        color_float + (1.0f - weight.x) * weight.y * weight.z * vColor;
            } else {
                vColor = make_float3(v.color.x, v.color.y, v.color.z);
                distance += (1.0f - weight.x) * weight.y * weight.z * v.sdf;
                color_float =
                        color_float + (1.0f - weight.x) * weight.y * weight.z * vColor;
            }

            v = GetVoxel(pos_dual + make_float3(voxel_size, 0.0f, voxel_size));
            if (v.weight == 0) {
                vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
                distance += weight.x * (1.0f - weight.y) * weight.z * v0.sdf;
                color_float =
                        color_float + weight.x * (1.0f - weight.y) * weight.z * vColor;
            } else {
                vColor = make_float3(v.color.x, v.color.y, v.color.z);
                distance += weight.x * (1.0f - weight.y) * weight.z * v.sdf;
                color_float =
                        color_float + weight.x * (1.0f - weight.y) * weight.z * vColor;
            }

            v = GetVoxel(pos_dual + make_float3(voxel_size, voxel_size, voxel_size));
            if (v.weight == 0) {
                vColor = make_float3(v0.color.x, v0.color.y, v0.color.z);
                distance += weight.x * weight.y * weight.z * v0.sdf;
                color_float = color_float + weight.x * weight.y * weight.z * vColor;

            } else {
                vColor = make_float3(v.color.x, v.color.y, v.color.z);
                distance += weight.x * weight.y * weight.z * v.sdf;
                color_float = color_float + weight.x * weight.y * weight.z * vColor;
            }

            uchar3 color = make_uchar3(color_float.x, color_float.y, color_float.z);
            v.weight = v0.weight;
            v.sdf = distance;
            v.color = color;
            return v;
        }

        __host__ __device__ bool TsdfVolume::SetVoxel(float3 position,
                                                      const Voxel &voxel) {
            int3 block_position = WorldToBlock(position);
            int3 local_voxel = WorldToLocalVoxel(position);
            HashEntry entry = HashTable::FindHashEntry(block_position);
            if (entry.pointer == kFreeEntry) {
                return false;
            }
            HashTable::voxel_blocks_[entry.pointer].at(local_voxel) = voxel;
            return true;
        }

        __host__ __device__ bool TsdfVolume::UpdateVoxel(float3 position,
                                                         const Voxel &voxel) {
            int3 block_position = WorldToBlock(position);
            int3 local_voxel = WorldToLocalVoxel(position);
            HashEntry entry = HashTable::FindHashEntry(block_position);
            if (entry.pointer == kFreeEntry) {
                return false;
            }
            HashTable::voxel_blocks_[entry.pointer]
                    .at(local_voxel)
                    .Combine(voxel, options_.max_sdf_weight);
            return true;
        }

        __global__ void AllocateFromDepthKernel(TsdfVolume *volume, float *depth,
                                                RgbdSensor sensor, float4x4 transform) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int size = sensor.rows * sensor.cols;

//            DEBUG_PRINT("AllocateFromDepthKernel, index = %d\n", index)

            float truncation_distance = volume->GetOptions().truncation_distance;
            float block_size =
                    volume->GetOptions().block_size * volume->GetOptions().voxel_size;

            float3 start_pt = make_float3(transform.m14, transform.m24, transform.m34);
            for (int i = index; i < size; i += stride) {
                if (depth[i] < volume->GetOptions().min_sensor_depth ||
                    depth[i] > volume->GetOptions().max_sensor_depth)
                    continue;
                float3 point_unproject = GetPoint3d(i, depth[i], sensor);
                float3 point = transform * point_unproject;
                if (point.x == 0 && point.y == 0 && point.z == 0) continue;
                // compute start and end of the ray
                float3 ray_direction = normalize(point - start_pt);
                float surface_distance = distance(start_pt, point);
                float3 ray_start = start_pt;
                float3 ray_end =
                        start_pt + ray_direction * (surface_distance + truncation_distance);
                // traverse the ray discretely using the block size and allocate it
                // adapted from https://github.com/francisengelmann/fast_voxel_traversal/blob/master/main.cpp
                int3 block_start = make_int3(floor(ray_start.x / block_size),
                                             floor(ray_start.y / block_size),
                                             floor(ray_start.z / block_size));

                int3 block_end = make_int3(floor(ray_end.x / block_size),
                                           floor(ray_end.y / block_size),
                                           floor(ray_end.z / block_size));

                int3 block_position = block_start;
                int3 step = make_int3(sign(ray_direction.x),
                                      sign(ray_direction.y),
                                      sign(ray_direction.z));

                float3 delta_t;
                delta_t.x =
                        (ray_direction.x != 0) ? fabs(block_size / ray_direction.x) : FLT_MAX;
                delta_t.y =
                        (ray_direction.y != 0) ? fabs(block_size / ray_direction.y) : FLT_MAX;
                delta_t.z =
                        (ray_direction.z != 0) ? fabs(block_size / ray_direction.z) : FLT_MAX;

                float3 boundary = make_float3(
                        (block_position.x + static_cast<float>(step.x)) * block_size,
                        (block_position.y + static_cast<float>(step.y)) * block_size,
                        (block_position.z + static_cast<float>(step.z)) * block_size);

                float3 max_t;
                max_t.x = (ray_direction.x != 0)
                          ? (boundary.x - ray_start.x) / ray_direction.x
                          : FLT_MAX;
                max_t.y = (ray_direction.y != 0)
                          ? (boundary.y - ray_start.y) / ray_direction.y
                          : FLT_MAX;
                max_t.z = (ray_direction.z != 0)
                          ? (boundary.z - ray_start.z) / ray_direction.z
                          : FLT_MAX;

                int3 diff = make_int3(0, 0, 0);
                bool neg_ray = false;

                if (block_position.x != block_end.x && ray_direction.x < 0) {
                    diff.x--;
                    neg_ray = true;
                }
                if (block_position.y != block_end.y && ray_direction.y < 0) {
                    diff.y--;
                    neg_ray = true;
                }
                if (block_position.z != block_end.z && ray_direction.z < 0) {
                    diff.z--;
                    neg_ray = true;
                }
                volume->AllocateBlock(block_position);

                if (neg_ray) {
                    block_position = block_position + diff;
                    volume->AllocateBlock(block_position);
                }

                while (block_position.x != block_end.x || block_position.y != block_end.y ||
                       block_position.z != block_end.z) {
                    if (max_t.x < max_t.y) {
                        if (max_t.x < max_t.z) {
                            block_position.x += step.x;
                            max_t.x += delta_t.x;
                        } else {
                            block_position.z += step.z;
                            max_t.z += delta_t.z;
                        }
                    } else {
                        if (max_t.y < max_t.z) {
                            block_position.y += step.y;
                            max_t.y += delta_t.y;
                        } else {
                            block_position.z += step.z;
                            max_t.z += delta_t.z;
                        }
                    }
                    volume->AllocateBlock(block_position);

//                    z++;
//                    if (z > 10000){
//                        DEBUG_PRINT("Index = %d, i = %d, z = %d, pos = (%d, %d, %d), end = (%d, %d, %d), depth = %f, block_start = (%d, %d, %d), ray_start = (%f, %f, %f), block_end = (%d, %d, %d), ray_end = (%f, %f, %f)\n",index, i, z,block_position.x, block_position.y, block_position.z,block_end.x, block_end.y, block_end.z,depth[i],block_start.x,block_start.y,block_start.z,ray_start.x,ray_start.y,ray_start.z,block_end.x, block_end.y, block_end.z, ray_end.x, ray_end.y, ray_end.z)
//                        DEBUG_PRINT("point = (%f, %f, %f), point_unproj = (%f, %f, %f), start_point = (%f, %f, %f), transform=(%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f)\n",point.x,point.y,point.z,point_unproject.x,point_unproject.y,point_unproject.z,start_pt.x,start_pt.y,start_pt.z,transform.entries[0],transform.entries[1],transform.entries[2],transform.entries[3],transform.entries[4],transform.entries[5],transform.entries[6],transform.entries[7],transform.entries[8],transform.entries[9],transform.entries[10],transform.entries[11],transform.entries[12],transform.entries[13],transform.entries[14],transform.entries[15])
//                    }
                }
            }

//            DEBUG_PRINT("AllocateFromDepthKernel done, index = %d\n", index)
        }

        __global__ void IntegrateScanKernel(TsdfVolume *volume, uchar3 *color,
                                            float *depth, RgbdSensor sensor,
                                            float4x4 transform, float4x4 inv_transform) {
            //loop through ALL entries
            //  if entry is in camera frustum
            //    loop through voxels inside block
            //    if voxel is in truncation region
            //      update voxels
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            int block_size = volume->GetOptions().block_size;
            float voxel_size = volume->GetOptions().voxel_size;
            float truncation_distance = volume->GetOptions().truncation_distance;

            for (int i = index; i < volume->GetNumEntries(); i += stride) {
                float3 position = make_float3(
                        volume->GetHashEntry(i).position.x * voxel_size * block_size,
                        volume->GetHashEntry(i).position.y * voxel_size * block_size,
                        volume->GetHashEntry(i).position.z * voxel_size * block_size);
                // To camera coordinates
                float3 position_cam = inv_transform * position;
                // If behind camera plane discard
                if (position_cam.z < 0) continue;
                float3 block_center =
                        make_float3(position_cam.x + 0.5 * voxel_size * block_size,
                                    position_cam.y + 0.5 * voxel_size * block_size,
                                    position_cam.z + 0.5 * voxel_size * block_size);
                int2 image_position = Project(block_center, sensor);
                if (image_position.x >= 0 && image_position.y >= 0 &&
                    image_position.x < sensor.cols && image_position.y < sensor.rows) {
                    float3 start_pt = make_float3(0, 0, 0);

                    for (int bx = 0; bx < block_size; bx++) {
                        for (int by = 0; by < block_size; by++) {
                            for (int bz = 0; bz < block_size; bz++) {
                                float3 voxel_position = make_float3(position.x + bx * voxel_size,
                                                                    position.y + by * voxel_size,
                                                                    position.z + bz * voxel_size);
                                voxel_position = inv_transform * voxel_position;
                                image_position = Project(voxel_position, sensor);
                                // Check again inside the block
                                if (image_position.x >= 0 && image_position.y >= 0 &&
                                    image_position.x < sensor.cols &&
                                    image_position.y < sensor.rows) {
                                    int idx = image_position.y * sensor.cols + image_position.x;
                                    if (depth[idx] <= 0) continue;
                                    if (depth[idx] < volume->GetOptions().min_sensor_depth) continue;
                                    if (depth[idx] > volume->GetOptions().max_sensor_depth) continue;
                                    float3 point3d = GetPoint3d(idx, depth[idx], sensor);
                                    float surface_distance = distance(start_pt, point3d);
                                    float voxel_distance = distance(start_pt, voxel_position);
                                    if (voxel_distance > surface_distance - truncation_distance &&
                                        voxel_distance < surface_distance + truncation_distance &&
                                        (depth[idx] < volume->GetOptions().max_sensor_depth)) {
                                        Voxel voxel;
                                        voxel.sdf = surface_distance - voxel_distance;
                                        voxel.color = color[idx];
                                        voxel.weight = (unsigned char) 1;
                                        // To world coordinates
                                        voxel_position = transform * voxel_position;
                                        volume->UpdateVoxel(voxel_position, voxel);
                                    } else if (voxel_distance <
                                               surface_distance - truncation_distance) {
                                        voxel_position = transform * voxel_position;
                                        Voxel voxel;
                                        voxel.sdf = truncation_distance;
                                        voxel.color = color[idx];
                                        voxel.weight = (unsigned char) 1;
                                        volume->UpdateVoxel(voxel_position, voxel);
                                    }
                                }
                            }
                        }
                    }  // End single block update
                }
            }
        }

        void TsdfVolume::IntegrateScanAsync(
                const RgbdSensor &sensor,
                unsigned char *bgr,
                float *depth,
                float4x4 const &camera_pose) {
            if (should_call_next_ != "IntegrateScanAsync") {
                std::cerr << "Please call the functions like Integration -> RenderAsync -> GetRenderResults."
                          << " You should have called " << should_call_next_ << std::endl;
                exit(EXIT_FAILURE);
            }
            should_call_next_ = "RenderAsync";

            float4x4 inv_camera_pose = camera_pose.getInverse();
            int threads_per_block = THREADS_PER_BLOCK2;
            int thread_blocks =
                    (options_.num_buckets * options_.bucket_size + threads_per_block - 1) /
                    threads_per_block;

            if (int_event_) {
                gpuErrchk(cudaEventSynchronize(int_event_->cpy_dth))
                eventDestroy(int_event_);
            } else {
                int_event_ = new Event;
            }
            eventCreate(int_event_);

            // Copy inputs to page-locked memory
            memcpy((void *) h_bgr_in_, (void *) bgr, sizeof(uchar3) * num_pixels());
            memcpy(h_depth_in_, depth, sizeof(float) * num_pixels());

            // Copy mem to device
            gpuErrchk(cudaMemcpyAsync(d_bgr_in_, h_bgr_in_, sizeof(uchar3) * num_pixels(),
                            cudaMemcpyHostToDevice, int_stream_))
            gpuErrchk(cudaMemcpyAsync(d_depth_in_, h_depth_in_, sizeof(float) * num_pixels(),
                            cudaMemcpyHostToDevice, int_stream_))
            gpuErrchk(cudaEventRecord(int_event_->cpy_htd, int_stream_))

            // Potentially wait for last rendering to complete
            for (auto& x : render_events_) {
                if (x) {
                    gpuErrchk(cudaStreamWaitEvent(int_stream_, x->cpy_dth, 0));
                }
            }

            // Call Kernels
            int mem = 0;
            DEBUG_PRINT("Launching AllocateFromDepthKernel kernel\n")
            clock_t started = clock();

            if (camera_pose.entries2[0][0] > 10.0f || camera_pose.entries2[0][0] < -10.0f) {
                std::cout << "TRANSFORM!!! " << std::endl;
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++)
                        std::cout << camera_pose.entries2[r][c] << " ";
                    std::cout << std::endl;
                }
            }


            AllocateFromDepthKernel<<<thread_blocks, threads_per_block, mem, int_stream_>>>(
                    this, d_depth_in_, sensor, camera_pose);
#ifdef DR_FUSION_DEBUG_SYNC_LAUNCH
            // TODO: This one throws: an illegal memory access was encountered
            gpuErrchk( cudaPeekAtLastError() )

            // TODO(lukas)
//            gpuErrchk( cudaDeviceSynchronize() )
            gpuErrchk( cudaStreamSynchronize(int_stream_) )
            clock_t ended = clock();
            double MilliSecondsTaken = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
            DEBUG_PRINT("Sync After AllocateFromDepthKernel kernel: %f ms\n", MilliSecondsTaken)
#endif
            IntegrateScanKernel<<<thread_blocks, threads_per_block, mem, int_stream_>>>(
                    this, d_bgr_in_, d_depth_in_, sensor, camera_pose,
                    inv_camera_pose);
#ifdef DR_FUSION_DEBUG_SYNC_LAUNCH
            gpuErrchk( cudaPeekAtLastError() )
            // TODO(lukas)
//            gpuErrchk( cudaDeviceSynchronize() )
            gpuErrchk( cudaStreamSynchronize(int_stream_) )
#endif
            gpuErrchk(cudaEventRecord(int_event_->compute, int_stream_))
            gpuErrchk(cudaEventRecord(int_event_->cpy_dth, int_stream_))
        }

        __global__ void GenerateRgbDepthKernel(TsdfVolume *volume, RgbdSensor sensor,
                                               float4x4 camera_pose, uchar3 *virtual_rgb, float *virtual_depth) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int size = sensor.rows * sensor.cols;

            float3 start_pt =
                    make_float3(camera_pose.m14, camera_pose.m24, camera_pose.m34);
            for (int i = index; i < size; i += stride) {
                float current_depth = 0;
                while (current_depth < volume->GetOptions().max_sensor_depth) {
                    float3 point = GetPoint3d(i, current_depth, sensor);
                    point = camera_pose * point;
                    Voxel v = volume->GetInterpolatedVoxel(point);
                    if (v.weight == 0) {
                        current_depth += volume->GetOptions().truncation_distance;
                    } else {
                        current_depth += v.sdf;
                    }
                    if (v.weight != 0 && v.sdf < volume->GetOptions().voxel_size) break;
                }
                if (current_depth < volume->GetOptions().max_sensor_depth) {
                    float3 point = GetPoint3d(i, current_depth, sensor);
                    point = camera_pose * point;
                    Voxel v = volume->GetInterpolatedVoxel(point);
                    virtual_rgb[i] = v.color;
                    virtual_depth[i] = current_depth;
                } else {
                    virtual_rgb[i] = make_uchar3(0, 0, 0);
                    virtual_depth[i] = 0.0;
                }
            }
        }

        void TsdfVolume::RenderAsync(std::vector<float4x4> camera_poses, RgbdSensor sensor) {
            if (should_call_next_ != "RenderAsync") {
                std::cerr << "Please call the functions like IntegrateScanAsync -> RenderAsync -> GetRenderResult."
                          << " You should have called " << should_call_next_ << std::endl;
                exit(EXIT_FAILURE);
            }
            should_call_next_ = "GetRenderResult";

            // Check input
            if (render_streams_.size() != camera_poses.size()) {
                std::cerr << "Can only render exactly as many poses as streams."
                          << " Streams: " << std::to_string(render_streams_.size())
                          << ", Poses: " << std::to_string(camera_poses.size()) << "." << std::endl;
                exit(EXIT_FAILURE);
            }

            if ((sensor.rows != options_.height) || (sensor.cols != options_.width)) {
                std::cerr << "Image sizes don't match." << std::endl;
                exit(EXIT_FAILURE);
            }

            // Wait for all rendering to complete (because we reuse device buffers)
            for (auto& x: render_events_){
                if (x){
                    cudaEventSynchronize(render_events_[0]->cpy_dth);
                    eventDestroy(x);
                }else{
                    x = new Event;
                }

                eventCreate(x);
            }

            // Kernel options
            int threads_per_block = THREADS_PER_BLOCK2;
            int thread_blocks = DIV_UP(num_pixels(), threads_per_block);
            int mem = 0;

            // Start Kernel
            for (int i = 0; i < options_.num_render_streams; i++) {
                GenerateRgbDepthKernel<<<thread_blocks, threads_per_block, mem, render_streams_[i]>>>(
                        this, sensor, camera_poses[i], d_bgr_render_[i], d_depth_render_[i]);
                cudaEventRecord(render_events_[i]->cpy_htd, render_streams_[i]);
                cudaEventRecord(render_events_[i]->compute, render_streams_[i]);
            }

            // Copy memory back
            auto &h_bgr_free = h_bgr_render_free();
            auto &h_depth_free = h_depth_render_free();
            // TODO: a sync here seems to solve the problem (no it does not)
            gpuErrchk( cudaDeviceSynchronize() );

            for (int i = 0; i < options_.num_render_streams; i++) {
              gpuErrchk(cudaMemcpyAsync(h_bgr_free[i], d_bgr_render_[i], sizeof(uchar3) * sensor.rows * sensor.cols,
                                cudaMemcpyDeviceToHost, render_streams_[i]));
              gpuErrchk( cudaDeviceSynchronize() ); // This throws
              gpuErrchk(cudaMemcpyAsync(h_depth_free[i], d_depth_render_[i], sizeof(float) * sensor.rows * sensor.cols,
                                cudaMemcpyDeviceToHost, render_streams_[i]));
              gpuErrchk( cudaDeviceSynchronize() );
              gpuErrchk(cudaEventRecord(render_events_[i]->cpy_dth, render_streams_[i]) );
              gpuErrchk( cudaDeviceSynchronize() );
            }

            // TODO: remove for perf.
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() ); // This throws
        }

        void TsdfVolume::GetRenderResult(std::vector<unsigned char *> &bgr, std::vector<float *> &depth) {
            if (should_call_next_ != "GetRenderResult") {
                std::cerr << "Please call the functions in a loop: IntegrateScanAsync -> RenderAsync -> GetRenderResult"
                          << ". You should have called " << should_call_next_ << std::endl;
                exit(EXIT_FAILURE);
            }
            should_call_next_ = "IntegrateScanAsync";

            if ((!bgr.empty()) || (!depth.empty())) {
                std::cerr << "Input vectors must be empty." << std::endl;
                exit(EXIT_FAILURE);
            }

            // Wait for all renderings
            for (auto &x: render_events_)
                eventSynchronize(x);

            // Change blocked vs free state
            if (render_blocked_ == 0)
                render_blocked_ = 1;
            else
                render_blocked_ = 0;

            // Get results
            auto &h_bgr_blocked = h_bgr_render_blocked();
            auto &h_depth_blocked = h_depth_render_blocked();

            for (int i = 0; i < options_.num_render_streams; i++) {
                bgr.push_back((unsigned char *) h_bgr_blocked[i]);
                depth.push_back(h_depth_blocked[i]);
            }

            // TODO: remove for perf.
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }

        Mesh TsdfVolume::ExtractMesh(const float3 &lower_corner,
                                     const float3 &upper_corner) {
            // TODO: Dirty Hack
            // Wait for all renderings
            for (auto &x: render_events_)
                eventSynchronize(x);

            MeshExtractor *mesh_extractor;
            cudaMallocManaged(&mesh_extractor, sizeof(MeshExtractor));
            mesh_extractor->Init(20000000, options_.voxel_size);
            mesh_extractor->ExtractMesh(this, lower_corner, upper_corner);
            Mesh *mesh;
            cudaMallocManaged(&mesh, sizeof(Mesh));
            *mesh = mesh_extractor->GetMesh();

            // TODO: Is this needed
            cudaFree(mesh_extractor);
            return *mesh;
        }

        void TsdfVolume::ExtractMeshAsync(const float3 &lower_corner, const float3 &upper_corner) {
            if (should_call_next_ != "IntegrateScanAsync") {
                std::cerr << "Please call this functions after GetRenderResult" << std::endl;
                exit(EXIT_FAILURE);
            }
//            // TODO: Dirty Hack and should be unnecessary
//            // Wait for all renderings
//            for (auto &x: render_events_)
//                eventSynchronize(x);

            if (mesh_extractor != nullptr) {
                std::cerr << "mesh_extractor should be NULL" << std::endl;
                exit(EXIT_FAILURE);
            }

            cudaMallocManaged(&mesh_extractor, sizeof(MeshExtractor));

            mesh_extractor->Init(20000000, options_.voxel_size);

            mesh_extractor->ExtractMesh(this, lower_corner, upper_corner);
        }

        void TsdfVolume::GetMeshSync(size_t num_max, size_t *num, float *vert, float *cols) {
            if (should_call_next_ != "IntegrateScanAsync") {
                std::cerr << "Please call this functions after GetRenderResult" << std::endl;
                exit(EXIT_FAILURE);
            }

            if (mesh_extractor == nullptr){
                std::cerr << "mesh_extractor should not be NULL (did you call ExtractMeshAsync before)?" << std::endl;
                exit(EXIT_FAILURE);
            }

            // Synchronizes
            Mesh mesh = mesh_extractor->GetMesh();

            // Copy Data
            if (num_max < mesh.num_triangles_){
                std::cerr << "Did not provide enough storage for mesh." << std::endl;
                exit(EXIT_FAILURE);
            }
            *num = 3 * mesh.num_triangles_;  // 1 triangle = 3 vert

            for (size_t i_tri = 0; i_tri < mesh.num_triangles_; i_tri++){
                size_t idx;
                // v0
                idx = 9 * i_tri + 3*0;
                vert[idx + 0] = mesh.triangles_[i_tri].v0.position.x;
                vert[idx + 1] = mesh.triangles_[i_tri].v0.position.y;
                vert[idx + 2] = mesh.triangles_[i_tri].v0.position.z;

                cols[idx + 0] = mesh.triangles_[i_tri].v0.color.z;
                cols[idx + 1] = mesh.triangles_[i_tri].v0.color.y;
                cols[idx + 2] = mesh.triangles_[i_tri].v0.color.x;

                // v1
                idx = 9 * i_tri + 3*1;
                vert[idx + 0] = mesh.triangles_[i_tri].v1.position.x;
                vert[idx + 1] = mesh.triangles_[i_tri].v1.position.y;
                vert[idx + 2] = mesh.triangles_[i_tri].v1.position.z;

                cols[idx + 0] = mesh.triangles_[i_tri].v1.color.z;
                cols[idx + 1] = mesh.triangles_[i_tri].v1.color.y;
                cols[idx + 2] = mesh.triangles_[i_tri].v1.color.x;

                // v2
                idx = 9 * i_tri + 3*2;
                vert[idx + 0] = mesh.triangles_[i_tri].v2.position.x;
                vert[idx + 1] = mesh.triangles_[i_tri].v2.position.y;
                vert[idx + 2] = mesh.triangles_[i_tri].v2.position.z;

                cols[idx + 0] = mesh.triangles_[i_tri].v2.color.z;
                cols[idx + 1] = mesh.triangles_[i_tri].v2.color.y;
                cols[idx + 2] = mesh.triangles_[i_tri].v2.color.x;
            }

            // Free Memory
            mesh_extractor->Free();
            cudaFree(mesh_extractor);
            mesh_extractor = nullptr;
        }

        __host__ __device__ TsdfVolumeOptions TsdfVolume::GetOptions() {
            return options_;
        }


        __host__ std::vector<uchar3 *> &TsdfVolume::h_bgr_render_blocked() {
            if (render_blocked_ == 0)
                return std::get<0>(h_bgr_render_);
            else
                return std::get<1>(h_bgr_render_);
        }

        __host__ std::vector<float *> &TsdfVolume::h_depth_render_blocked() {
            if (render_blocked_ == 0)
                return std::get<0>(h_depth_render_);
            else
                return std::get<1>(h_depth_render_);
        }

        __host__ std::vector<uchar3 *> &TsdfVolume::h_bgr_render_free() {
            if (render_blocked_ == 0)
                return std::get<1>(h_bgr_render_);
            else
                return std::get<0>(h_bgr_render_);
        }

        __host__ std::vector<float *> &TsdfVolume::h_depth_render_free() {
            if (render_blocked_ == 0)
                return std::get<1>(h_depth_render_);
            else
                return std::get<0>(h_depth_render_);
        }

    }  // namespace tsdfvh

}  // namespace refusion
