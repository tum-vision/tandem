// Copyright (c) 2020 Lukas Koestler, Nan Yang. All rights reserved.

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <string>
#include <iostream>
#include <memory>
#include <vector>

#include <boost/thread/thread.hpp>
#include <chrono>

#include "dr_mvsnet.h"

class DrMvsnetImpl {
public:
  explicit DrMvsnetImpl(const char *filename) : stream(at::cuda::getStreamFromPool(false)) {
    // Try to fix CUDA errors: https://github.com/pytorch/pytorch/issues/35736
    if (torch::cuda::is_available())std::cout << "DrMvsnet torch::cuda::is_vailable == true --> seems good" << std::endl;
    else std::cerr << "DrMvsnet torch::cuda::is_vailable == false --> probably this will crash" << std::endl;
    module = torch::jit::load(filename);
    worker_thread = boost::thread(&DrMvsnetImpl::Loop, this);
  };

  ~DrMvsnetImpl() {
    {
      boost::unique_lock<boost::mutex> lock(mut);
      while (unprocessed_data) {
        dataProcessedSignal.wait(lock);
      }
      running = false;
      newInputSignal.notify_all();
    }
    worker_thread.join();
  }

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
                 bool debug_print);

  DrMvsnetOutput *GetResult();

  bool Ready() { return !unprocessed_data; };

  void Wait();

private:
  void CallSequential();

  void Loop();

  // Will run Loop.
  boost::thread worker_thread;

  // Protects all below variables
  boost::mutex mut;
  bool running = true;
  bool unprocessed_data = false;

  boost::condition_variable newInputSignal;
  boost::condition_variable dataProcessedSignal;

  std::vector<torch::jit::IValue> inputs;
  int width_, height_;
  DrMvsnetOutput *output = nullptr;

  torch::jit::script::Module module;

  at::cuda::CUDAStream stream;
};

void DrMvsnetImpl::Loop() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (running) {
    if (unprocessed_data) {
      CallSequential();
      unprocessed_data = false;
      dataProcessedSignal.notify_all();
    }
    newInputSignal.wait(lock);
  }
}

DrMvsnetOutput *DrMvsnetImpl::GetResult() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (unprocessed_data) {
    dataProcessedSignal.wait(lock);
  }
  if (!output) {
    std::cerr << "Output should be valid pointer. Maybe you called GetResult more than once?" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    DrMvsnetOutput *ret = output;
    output = nullptr;
    return ret;
  }
}

void DrMvsnetImpl::Wait() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (unprocessed_data) {
    dataProcessedSignal.wait(lock);
  }
  if (!Ready()) {
    std::cerr << "DrMvsnetImpl must be ready after Wait" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void DrMvsnet::Wait() {
  impl->Wait();
}

void DrMvsnetImpl::CallAsync(int height,
                             int width,
                             int view_num,
                             int ref_index,
                             unsigned char **bgrs,
                             float const *intrinsic_matrix,
                             float **cam_to_worlds,
                             float depth_min,
                             float depth_max,
                             float discard_percentage,
                             bool debug_print) {

  using std::cout;
  using std::endl;

  boost::unique_lock<boost::mutex> lock(mut);
  // Now we have the lock
  {
    at::cuda::CUDAStreamGuard stream_guard(stream);

    inputs.clear();
    height_ = height;
    width_ = width;

    constexpr int batch_size = 1;
    constexpr int channels = 3;

    // Check inputs
    for (int i = 0; i < view_num - 1; i++) {
      for (int j = i + 1; j < view_num; j++) {
        if (bgrs[i] == bgrs[j] || cam_to_worlds[i] == cam_to_worlds[j]) {
          std::cerr << "ERROR: In Call Async passing the same data for index " << i << " and " << j << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    }

    if (debug_print) {
      printf("--- DrMvsnetImpl::CallAsync ---\n");
      printf("W=%d, H=%d, view_num=%d, ref_index=%d, depth_min=%f, depth_max=%f, discard_percentage=%f\n", width, height, view_num, ref_index, depth_min, depth_max, discard_percentage);
      printf("Intrinsics:\n");
      for (int r = 0; r < 3; r++) printf("%f %f %f\n", intrinsic_matrix[3 * r], intrinsic_matrix[3 * r + 1], intrinsic_matrix[3 * r + 2]);

      for (int i = 0; i < view_num; i++) {
        printf("C2W[%d]:\n", i);
        float const *const c2w = cam_to_worlds[i];
        for (int r = 0; r < 4; r++) printf("%f %f %f %f\n", c2w[4 * r], c2w[4 * r + 1], c2w[4 * r + 2], c2w[4 * r + 3]);
      }

      for (int i = 0; i < view_num; i++) printf("bgrs[%d] = %p\n", i, bgrs[i]);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);

    // image: (B, V, C, H, W)
    auto image = torch::empty({batch_size, view_num, channels, height, width}, options);
    auto image_a = image.accessor<float, 5>();

    auto cam_to_world = torch::empty({batch_size, view_num, 4, 4}, options);
    auto cam_to_world_a = cam_to_world.accessor<float, 4>();

    for (int _view_index = 0; _view_index < view_num; _view_index++) {
      int view;
      if (_view_index == 0)
        view = ref_index;
      else if (_view_index <= ref_index)
        view = _view_index - 1;
      else
        view = _view_index;

      float const *c2w = cam_to_worlds[view];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          cam_to_world_a[0][_view_index][i][j] = c2w[4 * i + j];
        }
      }

      unsigned char const *bgr = bgrs[view];
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          const int offset = channels * (width * h + w);

          // BGR -> RGB
          image_a[0][_view_index][0][h][w] = ((float) bgr[offset + 2]) / 255.0;
          image_a[0][_view_index][1][h][w] = ((float) bgr[offset + 1]) / 255.0;
          image_a[0][_view_index][2][h][w] = ((float) bgr[offset + 0]) / 255.0;
        }
      }
    }

    // intrinsic_matrix: tuple(stage) (B, 3, 3) per stage
    auto K_stage3 = torch::empty({batch_size, 3, 3}, options);
    auto K_stage3_a = K_stage3.accessor<float, 3>();
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        K_stage3_a[0][i][j] = intrinsic_matrix[3 * i + j];

    // TODO: Correct scaling of intr mat (+0.5 then * 0.5 then -0.5)
    auto K_stage2 = torch::empty({batch_size, 3, 3}, options);
    auto K_stage2_a = K_stage2.accessor<float, 3>();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        if (i < 2)
          K_stage2_a[0][i][j] = 0.5 * intrinsic_matrix[3 * i + j];
        else
          K_stage2_a[0][i][j] = intrinsic_matrix[3 * i + j];
      }
    }

    auto K_stage1 = torch::empty({batch_size, 3, 3}, options);
    auto K_stage1_a = K_stage1.accessor<float, 3>();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        if (i < 2)
          K_stage1_a[0][i][j] = 0.25 * intrinsic_matrix[3 * i + j];
        else
          K_stage1_a[0][i][j] = intrinsic_matrix[3 * i + j];
      }
    }

    auto depth_min_tensor = torch::empty({batch_size}, options);
    depth_min_tensor.index_put_({0}, depth_min);

    auto depth_max_tensor = torch::empty({batch_size}, options);
    depth_max_tensor.index_put_({0}, depth_max);

    auto discard_percentage_tensor = torch::empty({batch_size}, options);
    discard_percentage_tensor.index_put_({0}, discard_percentage);

    // image: (B, V, C, H, W)
    // TODO: This throws strange errors sometimes, nvidia-smi says (Detected Critical Xid Error) but due to async this might be from somewhere else
    inputs.emplace_back(image.to(torch::kCUDA));

    // intrinsic_matrix: tuple(stage) (B, 3, 3) per stage
    inputs.emplace_back(torch::ivalue::Tuple::create(
        K_stage1.to(torch::kCUDA),
        K_stage2.to(torch::kCUDA),
        K_stage3.to(torch::kCUDA)
    ));

    // cam_to_world: (B, V, 4, 4)
    inputs.emplace_back(cam_to_world.to(torch::kCUDA));

    // depth_min: (B, )
    inputs.emplace_back(depth_min_tensor.to(torch::kCUDA));

    // depth_max: (B, )
    inputs.emplace_back(depth_max_tensor.to(torch::kCUDA));

    // discard_percentage: (B, )
    inputs.emplace_back(discard_percentage_tensor.to(torch::kCUDA));
  }
  unprocessed_data = true;
  newInputSignal.notify_all();
}

void DrMvsnetImpl::CallSequential() {
  // Inside here we are protected by a mutex //
  // inputs is already set correctly

  /* ---  Execute Model ---*/
  // The outputs are tuple(stage) tuple(depth, confidence)
  //  torch::NoGradGuard guard;
  c10::InferenceMode guard;
  at::cuda::CUDAStreamGuard stream_guard(stream);
  auto model_output = module.forward(inputs).toTuple();

  constexpr int stage = 2;
  auto depth_tensor = model_output->elements()[stage].toTuple()->elements()[0].toTensor().to(torch::kCPU);
  auto confidence_tensor = model_output->elements()[stage].toTuple()->elements()[1].toTensor().to(torch::kCPU);
  auto depth_a = depth_tensor.accessor<float, 3>();
  auto confidence_a = confidence_tensor.accessor<float, 3>();
  auto depth_dense_tensor = model_output->elements()[stage].toTuple()->elements()[3].toTensor().to(torch::kCPU);
  auto confidence_dense_tensor = model_output->elements()[stage].toTuple()->elements()[4].toTensor().to(torch::kCPU);
  auto depth_dense_a = depth_dense_tensor.accessor<float, 3>();
  auto confidence_dense_a = confidence_dense_tensor.accessor<float, 3>();

  /* --- Outputs --- */
  if (output) {
    std::cerr << "Output should internally be nullptr. Maybe you called CallAsync more than once?" << std::endl;
    exit(EXIT_FAILURE);
  }
  output = new DrMvsnetOutput(height_, width_);

  for (int h = 0; h < height_; h++) {
    for (int w = 0; w < width_; w++) {
      output->depth[width_ * h + w] = depth_a[0][h][w];
      output->confidence[width_ * h + w] = confidence_a[0][h][w];

      output->depth_dense[width_ * h + w] = depth_dense_a[0][h][w];
      output->confidence_dense[width_ * h + w] = confidence_dense_a[0][h][w];
    }
  }

}

DrMvsnet::DrMvsnet(const char *filename) {
  impl = new DrMvsnetImpl(filename);
}

DrMvsnet::~DrMvsnet() {
  delete impl;
}

void DrMvsnet::CallAsync(int height,
                         int width,
                         int view_num,
                         int ref_index,
                         unsigned char **bgrs,
                         float const *intrinsic_matrix,
                         float **cam_to_worlds,
                         float depth_min,
                         float depth_max,
                         float discard_percentage,
                         bool debug_print) {
  impl->CallAsync(
      height,
      width,
      view_num,
      ref_index,
      bgrs,
      intrinsic_matrix,
      cam_to_worlds,
      depth_min,
      depth_max,
      discard_percentage,
      debug_print
  );
}

DrMvsnetOutput *DrMvsnet::GetResult() {
  return impl->GetResult();
}

bool DrMvsnet::Ready() {
  return impl->Ready();
}


bool test_dr_mvsnet(DrMvsnet &model, char const *filename_inputs, bool print, int repetitions, char const *out_folder) {
  using std::cerr;
  using std::endl;
  using std::cout;

  constexpr int batch_size = 1;
  constexpr int channels = 3;

  using torch::kCPU;
  /* --- Convert Tensors to C data -- */

  /* ---  Load Input ---*/
  torch::jit::script::Module tensors = torch::jit::load(filename_inputs);

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;

  // image: (B, V, C, H, W)
  auto image = tensors.attr("image").toTensor().to(kCPU);
  if (image.size(0) != batch_size) {
    cerr << "Incorrect batch size." << endl;
    return false;
  }
  if (image.size(2) != channels) {
    cerr << "Incorrect channels." << endl;
    return false;
  }

  const int view_num = image.size(1);
  const int ref_index = view_num - 2;

  if (print)
    cout << "View Num: " << view_num << ", ref index: " << ref_index << endl;

  const int height = image.size(3);
  const int width = image.size(4);

  unsigned char *bgrs[view_num];
  for (int view = 0; view < view_num; view++)
    bgrs[view] = (unsigned char *) malloc(sizeof(unsigned char) * height * width * channels);

  auto image_a = image.accessor<float, 5>();
  for (int view = 0; view < view_num; view++) {
    unsigned char *bgr = bgrs[view];
    for (int h = 0; h < height; h++)
      for (int w = 0; w < width; w++) {
        // RGB -> BGR
        bgr[channels * (width * h + w) + 0] = (unsigned char) (255.0 * image_a[0][view][2][h][w]);
        bgr[channels * (width * h + w) + 1] = (unsigned char) (255.0 * image_a[0][view][1][h][w]);
        bgr[channels * (width * h + w) + 2] = (unsigned char) (255.0 * image_a[0][view][0][h][w]);
      }
  }

  auto intrinsic_matrix_tensor = tensors.attr("intrinsic_matrix.stage3").toTensor().to(kCPU);
  auto intrinsic_matrix_tensor_a = intrinsic_matrix_tensor.accessor<float, 3>();
  float *intrinsic_matrix = (float *) malloc(sizeof(float) * 3 * 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      intrinsic_matrix[i * 3 + j] = intrinsic_matrix_tensor_a[0][i][j];

  // cam_to_world: (B, V, 4, 4)
  auto c2w_tensor = tensors.attr("cam_to_world").toTensor().to(kCPU);
  auto c2w_tensor_a = c2w_tensor.accessor<float, 4>();
  float **c2ws = (float **) malloc(sizeof(float *) * view_num);
  for (int view = 0; view < view_num; view++) {
    c2ws[view] = (float *) malloc(sizeof(float) * 4 * 4);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        c2ws[view][i * 4 + j] = c2w_tensor_a[0][view][i][j];
  }

  float depth_min, depth_max, discard_percentage;

  auto depth_min_tensor = tensors.attr("depth_min").toTensor().to(kCPU);
  auto depth_max_tensor = tensors.attr("depth_max").toTensor().to(kCPU);
  depth_min = depth_min_tensor.accessor<float, 1>()[0];
  depth_max = depth_max_tensor.accessor<float, 1>()[0];

  auto discard_percentage_tensor = tensors.attr("discard_percentage").toTensor().to(kCPU);
  discard_percentage = discard_percentage_tensor.accessor<float, 1>()[0];

  constexpr int stage = 3;
  auto depth_ref = tensors.attr("outputs.stage" + std::to_string(stage) + ".depth").toTensor().to(kCPU);
  auto confidence_ref = tensors.attr(
      "outputs.stage" + std::to_string(stage) + ".confidence").toTensor().to(kCPU);

  double elapsed1 = 0.0;
  double elapsed2 = 0.0;
  double elapsed3 = 0.0;

  bool correct = true;

  int warmup = (repetitions == 1) ? 0 : 5;

  for (int rep = 0; rep < repetitions + warmup; rep++) {
    if (rep == warmup) {
      elapsed1 = 0.0;
      elapsed2 = 0.0;
      elapsed3 = 0.0;
    }
    auto start = std::chrono::high_resolution_clock::now();
    model.CallAsync(
        height,
        width,
        view_num,
        ref_index,
        bgrs,
        intrinsic_matrix,
        c2ws,
        depth_min,
        depth_max,
        discard_percentage
    );
    elapsed1 += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    start = std::chrono::high_resolution_clock::now();
    bool ready = model.Ready();
    elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    if (print && ready)
      std::cout << "Was ready directly. Quite unexpected. Debug. " << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto output = model.GetResult();
    elapsed3 += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    auto depth_out = torch::from_blob(output->depth, {height, width});
    auto confidence_out = torch::from_blob(output->confidence, {height, width});

    auto error_depth = torch::mean(torch::abs(depth_out - depth_ref)).item().toFloat();
    auto error_confidence = torch::mean(torch::abs(confidence_out - confidence_ref)).item().toFloat();

    const double atol = 1e-2;
    auto correct_depth = error_depth < atol;
    auto correct_confidence = error_confidence < atol;
    if (print) {
      cout << "Correctness:" << endl;
      cout << "\tDepth correct     : " << correct_depth << ", error: " << error_depth << endl;
      cout << "\tConfidence correct: " << correct_confidence << ", error: " << error_confidence << endl;
    }

    correct &= correct_depth;
    correct &= correct_confidence;

    if (out_folder && rep == 0) {
      std::string out_name = std::string(out_folder) + "pred_outputs.pt";
      cout << "Writing Result to: " << out_name << endl;

      //      Vesion 1.6
      //      torch::save(out_name, "x.pt"); // this is actually a zip file

      //      Version 1.5
      auto bytes = torch::jit::pickle_save(depth_out);
      std::ofstream fout(out_name, std::ios::out | std::ios::binary);
      fout.write(bytes.data(), bytes.size());
      fout.close();
    }

    delete output;
  }

  if (print) {
    cout << "Performance:" << endl;
    cout << "\tCallAsync     : " << (double) elapsed1 / (1000.0 * repetitions) << " ms" << endl;
    cout << "\tReady         : " << (double) elapsed2 / (1000.0 * repetitions) << " ms" << endl;
    cout << "\tGetResult     : " << (double) elapsed3 / (1000.0 * repetitions) << " ms" << endl;
  }

  if (correct) {
    if (print)
      cout << "All looks good!" << endl;
    return true;
  } else {
    if (print)
      cout << "There has been an error. Do not use the model." << endl;
    return false;
  }
}
