// Copyright (c) 2020 Lukas Koestler, Nan Yang. All rights reserved.

#include "fr_parser_memory.h"

// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <iostream>

FrParserMemory::FrParserMemory(int start_frame, int end_frame,
                               int frame_skip)
        : start_frame_(start_frame),
          end_frame_(end_frame),
          frame_skip_(frame_skip) {}

bool FrParserMemory::OpenFile(std::string filepath) {
    filebase_ = filepath;
    filebase_ += '/';
    std::stringstream assoc_path;
    assoc_path << filebase_ << "/associations.txt";
    fassociations_.open(assoc_path.str());

    std::stringstream poses_path;
    poses_path << filebase_ << "/poses_dso.txt";
    fposes_.open(poses_path.str());

    if (fassociations_.is_open() && fposes_.is_open()) {
        file_opened_ = true;
        return true;
    } else {
        return false;
    }
}

cv::Mat FrParserMemory::GetBgr() { return bgr_; }

cv::Mat FrParserMemory::GetDepth() { return depth_; }

float const *FrParserMemory::GetPose() { return pose_; }

bool FrParserMemory::ReadNext() {
    if (index_ >= files_rgb_.size())
        return false;

    bgr_ = cv::imdecode(
            cv::Mat(1, files_rgb_[index_].size(), CV_8UC1, files_rgb_[index_].data()),
            cv::ImreadModes::IMREAD_COLOR);

    depth_ = cv::imdecode(
            cv::Mat(1, files_depth_[index_].size(), CV_8UC1, files_depth_[index_].data()),
            cv::ImreadModes::IMREAD_ANYDEPTH);
    depth_.convertTo(depth_, CV_32FC1, 1.0f / 5000);

    for (int i = 0; i < 16; i++)
        pose_[i] = poses_[index_][i];

    index_ += 1;
}


std::vector<char> load_binary_file(std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Couldn't read data at " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return buffer;
}

bool FrParserMemory::ReadAll() {
    if (!file_opened_)
        return false;

    std::string filepath_depth, filepath_rgb;
    for (int index = 0; index < end_frame_; index += 1) {
        fassociations_ >> timestamp_depth_ >> filepath_depth >> timestamp_rgb_ >> filepath_rgb;
        fposes_ >> timestamp_pose_
                >> pose_[0] >> pose_[1] >> pose_[2] >> pose_[3]
                >> pose_[4] >> pose_[5] >> pose_[6] >> pose_[7]
                >> pose_[8] >> pose_[9] >> pose_[10] >> pose_[11]
                >> pose_[12] >> pose_[13] >> pose_[14] >> pose_[15];

        if (timestamp_depth_ != timestamp_pose_) {
            std::cerr << "Timestamps do not match " << timestamp_depth_ << ", " << timestamp_pose_ << std::endl;
            exit(EXIT_FAILURE);
        }

        if (index >= start_frame_ && index % frame_skip_ == 0) {
            {
                std::stringstream filepath;
                filepath << filebase_ << filepath_rgb;
                files_rgb_.push_back(load_binary_file(filepath.str()));
            }

            {
                std::stringstream filepath;
                filepath << filebase_ << filepath_depth;
                files_depth_.push_back(load_binary_file(filepath.str()));
            }

            std::vector<float> tmp_pose(16);
            for (int i = 0; i < 16; i++)
                tmp_pose[i] = pose_[i];
            poses_.push_back(tmp_pose);


        }
    }

    std::cout << "Read " << files_rgb_.size() << " files into memory" << std::endl;


    return true;
}
