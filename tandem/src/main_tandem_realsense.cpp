// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

//#include "main_tandem_realsense.h"
#include "realsense.h"
#include "util/globalCalib.h"
#include "util/Undistort.h"
#include "util/commandline.h"

#include "FullSystem/FullSystem.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

#include <iostream>
#include <chrono> // std::chrono::microseconds
#include <thread> // std::this_thread::sleep_for
#include <opencv2/imgproc.hpp>

#include <boost/circular_buffer.hpp>

#include <signal.h>

using namespace dso;

bool firstRosSpin = false;

void my_exit_handler(int s) {
  printf("Caught signal %d\n", s);
  exit(1);
}

void exitThread() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  firstRosSpin = true;
  while (true) pause();
}

cv::Mat getImageBGR_8UC3_undistort_internal(cv::Mat const &img_in, Undistort const *u) {

  if (u->wOrg != img_in.cols || u->hOrg != img_in.rows) {
    printf("u.size=(%d, %d) img_in.size=(%d, %d)\n", u->wOrg, u->hOrg, img_in.cols, img_in.rows);
    std::cerr << "Trying to undistort wrong size. Will just exit" << std::endl;
    exit(EXIT_FAILURE);
  }

  cv::Mat img_out(u->h, u->w, CV_8UC3);
  unsigned char const *in_data = img_in.data;
  unsigned char *out_data = img_out.data;

  const int widthOrg = u->w;

  for (int idx = u->w * u->h - 1; idx >= 0; idx--) {
    // get interp. values
    float xx = u->remapX[idx];
    float yy = u->remapY[idx];

    if (xx < 0)
      out_data[3 * idx] = out_data[3 * idx + 1] = out_data[3 * idx + 2] = 0;
    else {
      // get integer and rational parts
      int xxi = xx;
      int yyi = yy;
      xx -= xxi;
      yy -= yyi;
      float xxyy = xx * yy;

      // get array base pointer
      for (int bgr = 0; bgr < 3; ++bgr) {
        const unsigned char *src = in_data + (3 * xxi + bgr) + 3 * yyi * widthOrg;

        // interpolate (bilinear)
        out_data[3 * idx + bgr] = static_cast<unsigned char>(xxyy * src[3 + 3 * widthOrg]
                                                             + (yy - xxyy) * src[3 * widthOrg]
                                                             + (xx - xxyy) * src[3]
                                                             + (1 - xx - yy + xxyy) * src[0]);
      }
    }
  }

  return img_out;

}


int main(int argc, char **argv) {
  using std::cout;
  using std::cerr;
  using std::endl;
  using namespace std::chrono;

  cout << "TANDEM DEMO" << endl;

  CommandLineOptions opt;

  tandemDefaultSettings(opt, argv[1]);
  for (int i = 2; i < argc; i++) parseArgument(opt, argv[i]);

  // hook crtl+C.
  boost::thread exThread = boost::thread(exitThread);

  // Interface Realsense
  const int width = 512;
  const int height = 320;
  cout << "Connecting to D455. This takes a few seconds." << endl;
  D455 camera(true, width, height);
  camera.StartPipeline();


  // Read Undistorter
  Undistort *undistorter;
  if (opt.calib.empty()) {
    cerr << "No calibration file given. Will attempt to convert (== bad approximation) the realsense intrinsics from the device buffer. These might be inaccurate." << endl;
    camera.WriteDsoIntrinsicsFile(opt.result_folder + "rs2_intrinsics.txt", "crop", width, height);
    cout << "Width = " << camera.GetWidth() << ", Height = " << camera.GetHeight() << endl;
    cout << "Intrinsics = " << camera.GetIntrinsics() << endl;
    cout << "Distortion = " << camera.GetDistortion() << endl;
    undistorter = Undistort::getUndistorterForFile(opt.result_folder + "rs2_intrinsics.txt", "", "");
  } else {
    cerr << "Reading given calib from " << opt.calib << endl;
    undistorter = Undistort::getUndistorterForFile(opt.calib, "", "");
    if (undistorter->wOrg != width || undistorter->hOrg != height) {
      cerr << "Wrong width or height between calib and camera. Exit." << endl;
      return EXIT_FAILURE;
    }
  }
  setGlobalCalib((int) undistorter->getSize()[0], (int) undistorter->getSize()[1], undistorter->getK().cast<float>());

  /* --- Setup Full System --- */
  FullSystem *fullSystem = new FullSystem();
  fullSystem->setGammaFunction(nullptr);
  fullSystem->linearizeOperation = false;
  fullSystem->result_folder = std::string(opt.result_folder);

  IOWrap::PangolinDSOViewer *viewer = 0;
  viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
  fullSystem->outputWrapper.push_back(viewer);

  /* --- TANDEM STuff --- */
  fullSystem->dr_mvsnet_view_num = dr_mvsnet_view_num;
  fullSystem->dr_width = undistorter->w;
  fullSystem->dr_height = undistorter->h;
  fullSystem->initDr();

  RGBDepth *depth = nullptr;
  dvo::core::RgbdImagePyramid *dvo_img = nullptr;

  constexpr bool save_images = true;

  boost::circular_buffer<cv::Mat> imgs_undis_buffer(900);

  std::thread runthread([&]() {

    double first_timestamp = -1;
    double last_timestamp = 0;
    double current_timestamp = 0;

    int frame_drops = 0;
    int frame_id = 0, frame_id_cam, last_frame_id_cam = -1;
    for (; (current_timestamp - first_timestamp) < opt.demo_secs;) {
      int timing_key;
      if (dr_timing) timing_key = fullSystem->dr_timer.start_timing("camera.GetFrameBlocking");
      cv::Mat bgr;
      const bool success = camera.GetFrameBlocking(current_timestamp, bgr, frame_id_cam);
      if (dr_timing) fullSystem->dr_timer.end_timing("camera.GetFrameBlocking", timing_key);

      if (success) {
        frame_id++;
        if (first_timestamp < 0) first_timestamp = current_timestamp;


        if (last_frame_id_cam > 0) {
          if (frame_id_cam - last_frame_id_cam != 1) frame_drops += frame_id_cam - last_frame_id_cam - 1;
        }
        last_frame_id_cam = frame_id_cam;

        imgs_undis_buffer.push_back(getImageBGR_8UC3_undistort_internal(bgr, undistorter));
        cv::Mat img_gray;
        cv::cvtColor(imgs_undis_buffer.back(), img_gray, cv::COLOR_BGR2GRAY);
        ImageAndExposure img_dso(undistorter->w, undistorter->h, current_timestamp);
        cv::Mat img_dso_wrapper(undistorter->h, undistorter->w, CV_32F, img_dso.image);
        img_gray.convertTo(img_dso_wrapper, CV_32F, 1.0);


        /* -- Call into Full System --- */
        if (dr_timing) timing_key = fullSystem->dr_timer.start_timing("fullSystem->addActiveFrame");
        fullSystem->addActiveFrame(&img_dso, imgs_undis_buffer.back().data, depth, dvo_img, frame_id);
        if (dr_timing) fullSystem->dr_timer.end_timing("fullSystem->addActiveFrame", timing_key);
        if ((frame_id + 1) % 60 == 0)
          printf("NewFrame %6d (t=%10.5f sec, dt=%10.5f ms): lost=%d, init=%d, initFailed=%d, FPS=%5.2f, FD=%3d\n",
                 frame_id, current_timestamp, 1000 * (current_timestamp - last_timestamp), fullSystem->isLost, fullSystem->initialized, fullSystem->initFailed, 1 / (current_timestamp - last_timestamp), frame_drops);
        last_timestamp = current_timestamp;

//       Reset init
        if (fullSystem->initFailed || setting_fullResetRequested || fullSystem->isLost) {
          cerr << "INIT FAILED -> RESET" << endl;
          std::vector<IOWrap::Output3DWrapper *> wraps = fullSystem->outputWrapper;
          delete fullSystem;
          for (IOWrap::Output3DWrapper *ow : wraps) ow->reset();

          fullSystem = new FullSystem();
          fullSystem->setGammaFunction(nullptr);
          fullSystem->linearizeOperation = false;
          fullSystem->result_folder = std::string(opt.result_folder);
          fullSystem->outputWrapper = wraps;
          fullSystem->dr_mvsnet_view_num = dr_mvsnet_view_num;
          fullSystem->dr_width = undistorter->w;
          fullSystem->dr_height = undistorter->h;
          fullSystem->initDr();
        }

      } else {
        std::this_thread::sleep_for(std::chrono::microseconds{500});
      }
    }

    printf("Processed %06d frames in %10.5f seconds -> FPS=%5.2f\n", frame_id, (current_timestamp - first_timestamp), frame_id / (current_timestamp - first_timestamp));

    fullSystem->blockUntilMappingIsFinished();
    fullSystem->printResult(opt.result_folder + "result.txt");
    fullSystem->printAllResultSE3(opt.result_folder + "poses_dso.txt");
    fullSystem->printKeyframeIndex(opt.result_folder + "keyframes_dso.txt");
    fullSystem->printOptimizationWindowIndex(opt.result_folder + "dso_optimization_windows.txt");
    if(dr_timing) fullSystem->printDrStatistics(opt.result_folder);

    if (setting_tsdf_fusion) {
      float lower_corner[3] = {-5, -5, -5};
      float upper_corner[3] = {+5, +5, +5};
      fullSystem->fusion->SaveMeshToFile(opt.result_folder + "mesh.obj.incomplete", lower_corner, upper_corner);
      fullSystem->fusion->Synchronize();

      std::rename((opt.result_folder + "mesh.obj.incomplete").c_str(), (opt.result_folder + "mesh.obj").c_str());
      printf("Mesh Saving done!\n");
    }

  });

  if (viewer != 0)
    viewer->run();

  runthread.join();

  for (IOWrap::Output3DWrapper *ow : fullSystem->outputWrapper) {
    ow->join();
    delete ow;
  }


  printf("DELETE FULLSYSTEM!\n");
  delete fullSystem;

  printf("EXIT NOW!\n");
  return EXIT_SUCCESS;
}
