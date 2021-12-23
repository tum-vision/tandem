// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include "tsdfvh/hash_table.h"
#include "utils/rgbd_image.h"
#include "utils/matrix_utils.h"
#include "marching_cubes/mesh.h"
#include "utils/utils.h"

#include <vector>
#include <map>
#include <utility>

namespace refusion {

    namespace tsdfvh {

        class MeshExtractor;

        typedef struct CombinedEvent {
            cudaEvent_t cpy_htd;
            cudaEvent_t compute;
            cudaEvent_t cpy_dth;
        } Event;

        void eventCreate(Event *event);

        void eventDestroy(Event *event);

        void eventSynchronize(Event *event);

/**
 * @brief      Options for the TSDF representation (see TsdfVolume).
 */
        struct TsdfVolumeOptions {
            /** Size of the side of a voxel (in meters) */
            float voxel_size;

            /** Total number of buckets in the table */
            int num_buckets;

            /** Maximum number of entries in a bucket */
            int bucket_size;

            /** Maximum number of blocks that can be allocated */
            int num_blocks;

            /** Size in voxels of the side of a voxel block */
            int block_size;

            /** Maximum weight that a voxel can have */
            int max_sdf_weight;

            /** Truncation distance of the TSDF */
            float truncation_distance;

            /**
             * Maximum sensor depth that is considered. Higher depth values are
             * discarded
             */
            float max_sensor_depth;

            /**
             * Minimum sensor depth that is considered. Lower depth values are discarded
             */
            float min_sensor_depth;

            int num_render_streams;

            int height;
            int width;
        };

/**
 * @brief      Class that represents a TSDF volume. It handles access and
 *             manipulation of voxels in world coordinates, scan integration
 *             from an RGB-D sensor, and mesh extraction.
 */
        class TsdfVolume : public HashTable {
        public:
            ~TsdfVolume();

            /**
             * @brief      Initializes the TSDF volume.
             *
             * @param[in]  options  The options for the TSDF volume representation
             */
            void Init(const TsdfVolumeOptions &options);

            /**
             * @brief      Gets the voxel at the specified world position.
             *
             * @param[in]  position  The 3D position in world coordinates
             *
             * @return     The voxel at the given position.
             */
            __host__ __device__ Voxel GetVoxel(float3 position);

            /**
             * @brief      Gets the voxel at the specified world position obtained using
             *             trilinear interpolation.
             *
             * @param[in]  position  The 3D position in world coordinates
             *
             * @return     The voxel containing the interpolated values.
             */
            __host__ __device__ Voxel GetInterpolatedVoxel(float3 position);

            /**
             * @brief      Sets the voxel at the specified position to the specified
             *             values.
             *
             * @param[in]  position  The 3D position in world coordinates
             * @param[in]  voxel     The voxel containing the values to be set
             *
             * @return     True if the voxel was successfully set. False if the voxel was
             *             not found (it is not allocated).
             */
            __host__ __device__ bool SetVoxel(float3 position, const Voxel &voxel);

            /**
             * @brief      Updates the voxel at the given position by computing a weighted
             *             average with the given voxel.
             *
             * @param[in]  position  The 3D position in world coordinates
             * @param[in]  voxel     The new voxel used to update the one in the volume
             *
             * @return     True if the voxel was successfully set. False if the voxel was
             *             not found (it is not allocated).
             */
            __host__ __device__ bool UpdateVoxel(float3 position, const Voxel &voxel);

            /**
             * @brief      Extracts a mesh from the portion of the volume within the
             *             specified bounding box.
             *
             * @param[in]  lower_corner  The lower corner of the bounding box
             * @param[in]  upper_corner  The upper corner of the bounding box
             *
             * @return     The mesh.
             */
            Mesh ExtractMesh(const float3 &lower_corner, const float3 &upper_corner);

            void ExtractMeshAsync(const float3 &lower_corner, const float3 &upper_corner);

            void GetMeshSync(size_t num_max, size_t* num, float* vert, float* cols);

            /**
             * @brief Integrates Scan into tsdf volume
             * @param sensor
             * @param bgr
             * @param depth
             * @param camera_pose
             */
            void IntegrateScanAsync(
                    RgbdSensor const &sensor,
                    unsigned char *bgr,
                    float *depth,
                    float4x4 const &camera_pose);

            /**
             * @brief Starts the rendering process.
             * @param camera_poses Vector of poses. Must have the same number of elements as options.num_render_streams
             * @param sensor
             */
            void RenderAsync(std::vector<float4x4> camera_poses, RgbdSensor sensor);

            /**
             * @brief Returns the results of the previous call to RenderAsync. This function is blocking.
             * @param bgr   An *empty* vector into which the pointers to the virtual bgr images will be pushed_back.
             *              These pointers will be valid until the next call to this function.
             * @param depth An *empty* vector into which the pointers to the virtual depth images will be pushed_back.
             *              These pointers will be valid until the next call to this function.
             */
            void GetRenderResult(std::vector<unsigned char *> &bgr, std::vector<float *> &depth);

            /**
             * @brief      Gets the options for the TSDF representation.
             *
             * @return     The options.
             */
            __host__ __device__ TsdfVolumeOptions GetOptions();

            /** Options for the TSDF representation */
            TsdfVolumeOptions options_;
        protected:
            /**
             * @brief      Converts coordinates from global voxel indices to world
             *             coordinates (in meters).
             *
             * @param[in]  position  The global voxel position
             *
             * @return     The world coordinates (in meters).
             */
            __host__ __device__ float3 GlobalVoxelToWorld(int3 position);

            /**
             * @brief      Converts coordinates from world coordinates (in meters) to
             *             global voxel indices.
             *
             * @param[in]  position  The position in world coordinates
             *
             * @return     The global voxel position.
             */
            __host__ __device__ int3 WorldToGlobalVoxel(float3 position);

            /**
             * @brief      Converts coordinates from world coordinates (in meters) to
             *             blocks coordinates (indices)
             *
             * @param[in]  position  The position in world coordinates
             *
             * @return     The block position.
             */
            __host__ __device__ int3 WorldToBlock(float3 position);

            /**
             * @brief      Converts coordinates from world coordinates (in meters) to
             *             local indices of the voxel within its block
             *
             * @param[in]  position  The position in world coordinates
             *
             * @return     The local indices of the voxel within its block.
             */
            __host__ __device__ int3 WorldToLocalVoxel(float3 position);


            __host__ std::vector<uchar3 *> &h_bgr_render_blocked();

            __host__ std::vector<float *> &h_depth_render_blocked();

            __host__ std::vector<uchar3 *> &h_bgr_render_free();

            __host__ std::vector<float *> &h_depth_render_free();

            __host__ inline int num_pixels() { return options_.height * options_.width; };

            cudaStream_t int_stream_, mesh_stream_;
            std::vector<cudaStream_t> render_streams_;

            Event *int_event_ = NULL;
            std::vector<Event *> render_events_;

            uchar3 *d_bgr_in_, *h_bgr_in_;
            float *d_depth_in_, *h_depth_in_;

            std::vector<uchar3 *> d_bgr_render_;
            std::vector<float *> d_depth_render_;

            std::pair<std::vector<uchar3 *>, std::vector<uchar3 *>> h_bgr_render_;
            std::pair<std::vector<float *>, std::vector<float *>> h_depth_render_;
            int render_blocked_ = 0;

            std::string should_call_next_;
            MeshExtractor* mesh_extractor = nullptr;
        };

    }  // namespace tsdfvh

}  // namespace refusion
