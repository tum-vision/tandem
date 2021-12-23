// Copyright (c) 2020 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef DR_FUSION_FR_PARSER_MEMORY_H
#define DR_FUSION_FR_PARSER_MEMORY_H

// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

/**
 * @brief      Simple parser for data in the TUM RGB-D Benchmark format.
 */
class FrParserMemory {
public:
    /**
     * @brief      Constructs the class.
     *
     * @param[in]  start_frame  The start frame
     * @param[in]  end_frame    The end frame (-1 = until the end of the file)
     * @param[in]  frame_skip   The frame skip (1 = loads every image, 2 = loads
     *                          every other image, etc.)
     */
    FrParserMemory(int start_frame = 0, int end_frame = -1,
                   int frame_skip = 1);

    /**
     * @brief      Opens the associated.txt file.
     *
     * @param[in]  filepath  The path of the folder of the dataset
     *
     * @return     True if the file was successfully opened. False otherwise.
     */
    bool OpenFile(std::string filepath);

    /**
     * @brief      Gets the current RGB image.
     *
     * @return     The current RGB image.
     */
    cv::Mat GetBgr();

    /**
     * @brief      Gets the current depth image.
     *
     * @return     The current depth image.
     */
    cv::Mat GetDepth();

    /**
     * @brief      Gets the current pose (cam to world).
     *
     * @return     The current pose as 4x4 row-major pointer.
     */
    float const *GetPose();

    /**
     * @brief      Reads the next line of associated.txt and loads the respective
     *             RGB and depth images.
     *
     * @return     True if the images were successfully loaded. False if the file
     *             associated.txt has ended or was never loaded.
     */
    bool ReadNext();

    bool ReadAll();

    int GetIndex() {return index_;};

protected:
    /** Stores whether the file associated.txt is open or not */
    bool file_opened_ = false;

    /** The index of the current image */
    int index_ = 0;

    /** The file associated.txt */
    std::ifstream fassociations_;

    /** The file poses_dso.txt */
    std::ifstream fposes_;

    /** The path of the dataset */
    std::string filebase_;

    /** The desired initial frame */
    int start_frame_;

    /** The desired final frame */
    int end_frame_;

    /** The frame skip */
    int frame_skip_;

    /** The current RGB image */
    cv::Mat bgr_;

    /** The current depth image */
    cv::Mat depth_;

    std::vector<std::vector<char>> files_rgb_;
    std::vector<std::vector<char>> files_depth_;
    std::vector<std::vector<float>> poses_;

    /** The current pose */
    float pose_[16];

    /** The timestamp of the current RGB image */
    double timestamp_rgb_;

    /** The timestamp of the current depth image */
    double timestamp_depth_;

    /** The timestamp of the current pose */
    double timestamp_pose_;
};


#endif //DR_FUSION_FR_PARSER_MEMORY_H
