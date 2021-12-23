// Copyright (c) 2020 Lukas Koestler, Nan Yang. All rights reserved.

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "fr_parser_memory.h"
#include "dr_fusion.h"
#include "dr_mvsnet.h"
#include <chrono>
#include <iostream>

struct Options {
    bool display_rgb = true;
    int num_images = 1000;
    int skip_images = 1;
    int num_render_streams = 2;
};

struct Options options;

void parse_argument(char const *arg) {
    int tmp_int;
    if (1 == sscanf(arg, "display_rgb=%d", &tmp_int)) {
        options.display_rgb = tmp_int;
        return;
    }

    if (1 == sscanf(arg, "num_images=%d", &tmp_int)) {
        options.num_images = tmp_int;
        return;
    }

    if (1 == sscanf(arg, "skip_images=%d", &tmp_int)) {
        options.skip_images = tmp_int;
        return;
    }

    if (1 == sscanf(arg, "num_render_streams=%d", &tmp_int)) {
        options.num_render_streams = tmp_int;
        return;
    }
}

int main(int argc, char const *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: dr_example <dataset_path> <mvsnet_folder> options" << std::endl;
        return -1;
    }

    for (int i = 3; i < argc; i++)
        parse_argument(argv[i]);

    // Options for the Fusion
    DrFusionOptions dr_options{
            .voxel_size = 0.02,
            .num_buckets = 50000,
            .bucket_size = 10,
            .num_blocks = 2000,
            .block_size = 8,
            .max_sdf_weight = 64,
            .truncation_distance = 0.1,
            .max_sensor_depth = 10,
            .min_sensor_depth = 0.1,
            .num_render_streams = options.num_render_streams,
//            .fx = 640.0,
//            .fy = 640.0,
//            .cx = 319.5f,
//            .cy = 239.5f,
//            .height = 480,
//            .width = 640,
            .fx = 481.20,
            .fy = 480.0,
            .cx = 319.5f,
            .cy = 239.5f,
            .height = 480,
            .width = 640,
    };

    DrFusion fusion(dr_options);

    std::string mvsnet_folder(argv[2]);
    if (mvsnet_folder.back() != '/')
        mvsnet_folder = mvsnet_folder + std::string("/");
    DrMvsnet* mvsnet = new DrMvsnet((mvsnet_folder + "model.pt").c_str());
    if (!test_dr_mvsnet(*mvsnet, (mvsnet_folder + "sample_inputs.pt").c_str(), true)) {
        printf("Couldn't load MVSNet successfully.");
        exit(EXIT_FAILURE);
    }

    std::string filebase(argv[1]);
    filebase += '/';
    std::stringstream filepath_out, filepath_time;
    filepath_out << filebase << "/result.txt";
    std::ofstream result(filepath_out.str());

    int start = 0;
    int end = options.num_images;
    int frame_skip = options.skip_images;
    FrParserMemory parser(start, end, frame_skip);
    bool flag = true;
    flag &= parser.OpenFile(filebase);
    flag &= parser.ReadAll();

    std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
    double processed_count = 0;
    double cum_time_nonblocking = 0;

    if (flag) {
        while (parser.ReadNext()) {
            if ((processed_count > 0)) {
                if ((int)processed_count % 15 == 0) {
                    if (!test_dr_mvsnet(*mvsnet, (mvsnet_folder + "sample_inputs.pt").c_str(), false)) {
                        printf("MvsNet test failed.");
                        exit(EXIT_FAILURE);
                    }
                }

                std::vector<unsigned char *> bgr;
                std::vector<float *> depth;

                // Blocking
                fusion.GetRenderResult(bgr, depth);

                if (options.display_rgb) {
                    for (int i = 0; i < bgr.size(); i++) {
                        cv::Mat virtual_bgr(dr_options.height, dr_options.width, CV_8UC3, bgr[i]);
                        cv::Mat virtual_depth(dr_options.height, dr_options.width, CV_32F, depth[i]);

                        cv::imshow("RGB: " + std::to_string(i + 1), virtual_bgr);
                        int k1 = cv::waitKey(1);

                        cv::imshow("DEPTH " + std::to_string(i + 1), virtual_depth / 10);
                        int k2 = cv::waitKey(1);

                        if ((k1 == 27) || (k2 == 27)) {
                            break;
                        }
                    }
                }


            }

            auto start = std::chrono::high_resolution_clock::now();
            float* depth = (float *) parser.GetDepth().data;
            for (int i = 0; 2*i < dr_options.width * dr_options.height; i++)
                depth[i] = 0.0;
            fusion.IntegrateScanAsync(parser.GetBgr().data, depth, parser.GetPose());
            std::vector<float const *> poses;
            for (int i = 0; i < dr_options.num_render_streams; i++)
                poses.push_back(parser.GetPose());
            fusion.RenderAsync(poses);
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            cum_time_nonblocking += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

            processed_count += 1;
        }
    } else {
        std::cerr << "Error while opening file associated.txt" << std::endl;
        return -1;
    }

    fusion.Synchronize();
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    double time_elapsed =
            (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / 1000000.0;

    std::cout << "elapsed: " << time_elapsed << ", FPS: " << (processed_count / time_elapsed) << std::endl;
    std::cout << "Mean for non-blocking: " << (cum_time_nonblocking / (1000.0 * processed_count)) << " ms" << std::endl;

    result.close();

//    std::cout << "Creating mesh..." << std::endl;
//    float3 low_limits = make_float3(-10, -10, -10);
//    float3 high_limits = make_float3(10, 10, 10);
//    time_begin = std::chrono::steady_clock::now();
//    refusion::tsdfvh::Mesh *mesh;
//    cudaMallocManaged(&mesh, sizeof(refusion::tsdfvh::Mesh));
//    *mesh = fusion.ExtractMesh(low_limits, high_limits);
//    fusion.synchronize();
//    time_end = std::chrono::steady_clock::now();
//    time_elapsed =
//            (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / 1000000.0;
//    std::cout << "Mesh creation: " << time_elapsed << std::endl;
//
//    filepath_out.str("");
//    filepath_out.clear();
//    filepath_out << "mesh.obj";
//    mesh->SaveToFile(filepath_out.str());
    return 0;
}
