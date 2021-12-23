// Copyright (c) 2020 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef DR_MVSNET_H
#define DR_MVSNET_H

#include <memory>
#include <utility>


class DrMvsnetImpl;

class DrMvsnetOutput {
public:
  DrMvsnetOutput(int height, int width) : height(height), width(width) {
    depth = (float *) malloc(sizeof(float) * width * height);
    confidence = (float *) malloc(sizeof(float) * width * height);
    depth_dense = (float *) malloc(sizeof(float) * width * height);
    confidence_dense = (float *) malloc(sizeof(float) * width * height);
  };

  ~DrMvsnetOutput() {
    free(depth);
    free(confidence);
    free(depth_dense);
    free(confidence_dense);
  }

  float *depth;
  float *confidence;
  float *depth_dense;
  float *confidence_dense;
  const int height;
  const int width;
};

class DrMvsnet {
public:
  explicit DrMvsnet(char const *filename);

  ~DrMvsnet();

  // Blocking for last input. Non-blocking for this input.
  void CallAsync(int height,
                 int width,
                 int view_num,
                 int ref_index,
                 unsigned char **bgrs,
                 float const *intrinsic_matrix,
                 float **cam_to_worlds,
                 float depth_min,
                 float depth_max,
                 float discard_percentage,
                 bool debug_print = false);

  // Blocking
  DrMvsnetOutput *GetResult();

  // Blocking
  void Wait();

  // Non-blocking
  bool Ready();

private:
  DrMvsnetImpl *impl;
};

bool test_dr_mvsnet(DrMvsnet &model, char const *filename_inputs, bool print = false, int repetitions = 1, char const *out_folder = NULL);

#endif //DR_MVSNET_H
