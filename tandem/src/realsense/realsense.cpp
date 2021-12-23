// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#include "realsense.h"
#include <librealsense2/rs.hpp>

#include <boost/thread/mutex.hpp>

#include <chrono> // std::chrono::microseconds
#include <thread> // std::this_thread::sleep_for

class D455Impl {
public:

  D455Impl(bool debugPrint, int width, int height);

  void StartPipeline();

  bool GetFrameBlocking(double &timestamp, cv::Mat &img, int &frame_id);

  // fx fy cx cy
  cv::Vec<float, 4> GetIntrinsics() const;

  // RadTan model with 5 coefficients (like OpenCV)
  // k1 k2 p1 p2 k3
  cv::Vec<float, 5> GetDistortion() const;

  int GetHeight() const;

  int GetWidth() const;

  void WriteDsoIntrinsicsFile(const std::string &filename, const std::string &l3, int w_out, int h_out) const;

  void FrameCallback(rs2::frame frame);

private:
  const int width;
  const int height;

  int delay_ms = 500;
  double first_timestamp = -1;
  double last_timestamp = -1;
  int frame_id_internal = -1;
  bool has_new_data = false;

  boost::mutex mut;

  rs2::device device;
  rs2::sensor sensor;
  rs2::config config;
  rs2::pipeline pipeline;
  rs2_intrinsics intrinsics;

  cv::Mat bgr_small;

  bool debugPrint = true;
};

D455Impl::D455Impl(bool debugPrint, int width, int height) : debugPrint(debugPrint), width(width), height(height) {
  using std::cout;
  using std::cerr;
  using std::endl;

  rs2::log_to_console(RS2_LOG_SEVERITY_FATAL);
  rs2::context context;

  if (context.query_devices().size() == 0) {
    cerr << "Couldn't find realsense device. Will just exit as this never works." << endl;
    exit(EXIT_FAILURE);
  }

  if (context.query_devices().size() > 1) {
    cerr << "Have multiple realsense devices which doesn't make sense. Will just exit." << endl;
    exit(EXIT_FAILURE);
  }

  device = context.query_devices()[0];
  device.hardware_reset();

  sensor = device.query_sensors()[0];

  std::this_thread::sleep_for(std::chrono::milliseconds{delay_ms});
}

void D455Impl::StartPipeline() {
//  std::this_thread::sleep_for(std::chrono::milliseconds{500});
  config.enable_stream(RS2_STREAM_COLOR, 1280, 800, RS2_FORMAT_BGR8, 30);
  bgr_small = cv::Mat(height, width, CV_8UC3);
  pipeline.start(config, std::bind(&D455Impl::FrameCallback, this, std::placeholders::_1));
  intrinsics = pipeline.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
}

bool D455Impl::GetFrameBlocking(double &timestamp, cv::Mat &img, int &frame_id) {
  using std::cout;
  using std::cerr;
  using std::endl;

  if (!has_new_data) return false;
  {
    boost::unique_lock<boost::mutex> lock(mut);
    timestamp = last_timestamp;
    img = bgr_small.clone();
    frame_id = frame_id_internal;
    has_new_data = false;
  }
  return true;
}

cv::Vec<float, 4> D455Impl::GetIntrinsics() const {
  const float scale_x = (float) width / (float) intrinsics.width;
  const float scale_y = (float) height / (float) intrinsics.height;

  return cv::Vec<float, 4>(scale_x * intrinsics.fx, scale_y * intrinsics.fy, scale_x * (intrinsics.ppx + 0.5f) - 0.5f, scale_y * (intrinsics.ppy + 0.5f) - 0.5f);
}

cv::Vec<float, 5> D455Impl::GetDistortion() const {
  if (debugPrint) printf("Returning negative intrinsic parameters from camera. This is only a coarse approximation and there is no guarantee.\n");
  return cv::Vec<float, 5>(-intrinsics.coeffs[0], -intrinsics.coeffs[1], -intrinsics.coeffs[2], -intrinsics.coeffs[3], -intrinsics.coeffs[4]);
}

int D455Impl::GetHeight() const {
  return height;
}

int D455Impl::GetWidth() const {
  return width;
}


void D455Impl::WriteDsoIntrinsicsFile(const std::string &filename, const std::string &l3, int w_out, int h_out) const {
  FILE *file = fopen(filename.c_str(), "w");
  if (!file) {
    std::cerr << "Couldn't open " << filename << ". Will exit now." << std::endl;
    exit(EXIT_FAILURE);
  }

  const auto intr = GetIntrinsics();
  const auto dist = GetDistortion();
  fprintf(file, "RadTanK3 %f %f %f %f %f %f %f %f %f\n",
          intr[0], intr[1], intr[2], intr[3],
          dist[0], dist[1], dist[2], dist[3], dist[4]);

  fprintf(file, "%d %d\n", GetWidth(), GetHeight());
  fprintf(file, "%s\n", l3.c_str());
  fprintf(file, "%d %d\n", w_out, h_out);
  fclose(file);

}

void D455Impl::FrameCallback(rs2::frame frame) {
  auto start = std::chrono::high_resolution_clock::now();
  using std::cout;
  using std::cerr;
  using std::endl;
  const double timestamp = 1e-3 * frame.get_timestamp();

  if (const auto &fs = frame.as<rs2::frameset>()) {
    rs2::video_frame bgr_frame = fs.get_color_frame();
    if (bgr_frame) {
      boost::unique_lock<boost::mutex> lock(mut);

      if (first_timestamp < 0) first_timestamp = timestamp;
      last_timestamp = timestamp - first_timestamp;
      frame_id_internal++;
      has_new_data = true;

      cv::Mat bgr_big(bgr_frame.get_height(), bgr_frame.get_width(), CV_8UC3, const_cast<void *>(bgr_frame.get_data()), (size_t) bgr_frame.get_stride_in_bytes());
      cv::resize(bgr_big, bgr_small, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    }
  }
}

void D455::StartPipeline() {
  impl->StartPipeline();
}

bool D455::GetFrameBlocking(double &timestamp, cv::Mat &img, int &frame_id) {
  return impl->GetFrameBlocking(timestamp, img, frame_id);
}


D455::D455(bool debugPrint, int width, int height) {
  impl = new D455Impl(debugPrint, width, height);
}

D455::~D455() {
  delete impl;
}

cv::Vec<float, 5> D455::GetDistortion() {
  return impl->GetDistortion();
}

cv::Vec<float, 4> D455::GetIntrinsics() {
  return impl->GetIntrinsics();
}

int D455::GetHeight() {
  return impl->GetHeight();
}

int D455::GetWidth() {
  return impl->GetWidth();
}

void D455::WriteDsoIntrinsicsFile(const std::string &filename, const std::string &l3, int w_out, int h_out) const {
  impl->WriteDsoIntrinsicsFile(filename, l3, w_out, h_out);
}


