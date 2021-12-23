// Copyright (c) 2020 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef DR_FUSION_DR_FUSION_H
#define DR_FUSION_DR_FUSION_H

#include <memory>
#include <string>
#include <vector>

namespace refusion {
    struct RgbdSensor;

    namespace tsdfvh {
        class TsdfVolume;
    }
}

struct DrFusionOptions {
    float voxel_size;
    int num_buckets;
    int bucket_size;
    int num_blocks;
    int block_size;
    int max_sdf_weight;
    float truncation_distance;
    float max_sensor_depth;
    float min_sensor_depth;
    int num_render_streams;

    float fx;
    float fy;
    float cx;
    float cy;
    int height;
    int width;
};

struct DrMesh {
    size_t num = 0;
    float* vert = nullptr;
    float* cols = nullptr;
};

class DrFusion {
public:
    DrFusion(struct DrFusionOptions const &options);

    ~DrFusion();

    void IntegrateScanAsync(unsigned char *bgr, float *depth, float const *pose);

    void RenderAsync(std::vector<float const *> camera_poses);

    void GetRenderResult(std::vector<unsigned char *> &bgr, std::vector<float *> &depth);

    void SaveMeshToFile(std::string const& filename,  float lower_corner[3], float upper_corner[3]);

    struct DrMesh GetMesh(float lower_corner[3], float upper_corner[3]);

    void ExtractMeshAsync(float lower_corner[3], float upper_corner[3]);
    void GetMeshSync();

    void Synchronize();

    size_t dr_mesh_num = 0;
    const size_t dr_mesh_num_max = 60000000;
    float* dr_mesh_vert;
    float* dr_mesh_cols;

private:
    refusion::tsdfvh::TsdfVolume* volume_;
    std::unique_ptr<refusion::RgbdSensor> sensor_;
};


#endif //DR_FUSION_DR_FUSION_H