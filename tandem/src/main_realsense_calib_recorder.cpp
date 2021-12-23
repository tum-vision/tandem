// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#include "realsense.h"

#include <iostream>
#include <chrono> // std::chrono::microseconds
#include <thread> // std::this_thread::sleep_for
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <experimental/filesystem>
#include <stdio.h>

// Options
int num_images;
int frame_skip;
std::string output_dir;


void parseArgument(char *arg) {
  int option;
  float foption;
  char buf[1000];

  if (1 == sscanf(arg, "num_images=%d", &option)) {
    num_images = option;
    printf("Recording %6d images\n", num_images);
    return;
  }

  if (1 == sscanf(arg, "frame_skip=%d", &option)) {
    frame_skip = option;
    printf("Using every nth frame: %2d\n", frame_skip);
    return;
  }

  if (1 == sscanf(arg, "output_dir=%s", buf)) {
    output_dir = std::string(buf);
    output_dir = output_dir.back() == '/' ? output_dir : output_dir + "/";
    printf("saving images to %s/images !\n", output_dir.c_str());
    return;
  }

  printf("could not parse argument \"%s\"!!!!\n", arg);
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  namespace fs = std::experimental::filesystem;

  using std::cout;
  using std::cerr;
  using std::endl;

  cout << "Realsense Calib Recorder: Call like ./bin num_images=100 output_dir=images frame_skip=5" << endl;

  if (argc != 4) {
    cerr << "Wrong Number of arguments" << endl;
    return EXIT_FAILURE;
  }

  for (int i = 1; i < argc; i++)
    parseArgument(argv[i]);

  std::string image_dir = output_dir + "images/";
  if (!fs::create_directories(image_dir)) {
    cerr << "Could not create " << image_dir << ". Will exit." << endl;
    return EXIT_FAILURE;
  }

  const int width = 1280;
  const int height = 800;

  D455 camera(true, width, height);
  camera.StartPipeline();
  camera.WriteDsoIntrinsicsFile(output_dir + "rs2_intrinsics.txt", "crop", width, height);

  cv::namedWindow("Camera Image", cv::WINDOW_AUTOSIZE);// Create a window for display.

  std::vector<cv::Mat> images(num_images);
  std::vector<double> timestamps(num_images, -1.0);

  cout << "WAITING 10 SECONDS FOR YOU TO GET READY!!!" << endl;
  std::this_thread::sleep_for(std::chrono::seconds{10});

  int frame_id = 0, frame_id_cam, last_frame_id_cam = -1;
  while (frame_id < num_images) {
    const bool success = camera.GetFrameBlocking(timestamps[frame_id], images[frame_id], frame_id_cam);

    if (success) {
      /* --- Frame ID Handling --- */
      if (last_frame_id_cam > 0) {
        if (frame_id_cam - last_frame_id_cam != frame_skip)
          continue;
      }
      last_frame_id_cam = frame_id_cam;

      /* --- Display --- */
      cv::imshow("Camera Image", images[frame_id]);
      cv::waitKey(5);

      frame_id++;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds{500});
    }
  }

  cout << "Recorded data. Write data to disk" << endl;
  FILE *fp = fopen((output_dir + "timestamps_sec.txt").c_str(), "w");

  if (!fp) {
    cerr << "Could not open timestamps file" << endl;
    return EXIT_FAILURE;
  }
  double last_timestamp = -1;
  for (frame_id = 0; frame_id < num_images; frame_id++) {
    const double timestamp = timestamps[frame_id];

    if (last_timestamp >= timestamp) {
      cerr << "Timestamps are not increasing" << endl;
      return EXIT_FAILURE;
    }

    fprintf(fp, "%.17g\n", timestamp);
    char buf[100];
    sprintf(buf, "%06d.png", frame_id);
    cv::imwrite(image_dir + buf, images[frame_id]);
  }

  cout << "All done. Bye :)" << endl;

  return EXIT_SUCCESS;
}
