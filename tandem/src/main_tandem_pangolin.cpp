/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>


#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"
#include "util/Timer.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"


#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

#include "util/commandline.h"

#include <dvo/core/surface_pyramid.h>
#include <dvo/core/rgbd_image.h>


bool firstRosSpin = false;

using namespace dso;


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


int main(int argc, char **argv) {

  CommandLineOptions opt;

  tandemDefaultSettings(opt, argv[1]);
  for (int i = 2; i < argc; i++) parseArgument(opt, argv[i]);

  // Print Settings
  printSettings(opt);

  // hook crtl+C.
  boost::thread exThread = boost::thread(exitThread);


  ImageFolderReader *reader;
  if (!rgbd_flag)
    reader = new ImageFolderReader(opt.source, opt.calib, opt.gammaCalib, opt.vignette);
  else
    reader = new RGBDReader(opt.source, opt.rgbdepth_folder, opt.calib, opt.gammaCalib, opt.vignette);
  reader->setGlobalCalibration();


  if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0) {
    printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
    exit(1);
  }


  int lstart = opt.start;
  int lend = opt.end;
  int linc = 1;
  if (opt.reverse) {
    printf("REVERSE!!!!");
    lstart = opt.end - 1;
    if (lstart >= reader->getNumImages())
      lstart = reader->getNumImages() - 1;
    lend = opt.start;
    linc = -1;
  }


  FullSystem *fullSystem = new FullSystem();
  fullSystem->setGammaFunction(reader->getPhotometricGamma());
  fullSystem->linearizeOperation = (opt.playbackSpeed == 0);
  fullSystem->result_folder = std::string(opt.result_folder);

  if (save_depth_maps) {
    mkdir((opt.result_folder + "mvs_depth").c_str(), 0775);
  }


  IOWrap::PangolinDSOViewer *viewer = 0;
  if (!disableAllDisplay) {
    viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
    fullSystem->outputWrapper.push_back(viewer);
  }


  if (opt.useSampleOutput)
    fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());




  // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
  std::thread runthread([&]() {
    std::vector<int> idsToPlay;
    std::vector<double> timesToPlayAt;
    for (int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc) {
      idsToPlay.push_back(i);
      if (timesToPlayAt.size() == 0) {
        timesToPlayAt.push_back((double) 0);
      } else {
        double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
        double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
        timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / opt.playbackSpeed);
      }
    }


    std::vector<ImageAndExposure *> preloadedImages;
    std::vector<unsigned char *> preloadedImagesBGR;
    if (opt.preload) {
      printf("LOADING ALL IMAGES!\n");
      for (int ii = 0; ii < (int) idsToPlay.size(); ii++) {
        int i = idsToPlay[ii];
        preloadedImages.push_back(reader->getImage(i));
        preloadedImagesBGR.push_back(reader->getImageBGR_8UC3_undis(i, reader->undistort->get_remapX(), reader->undistort->get_remapY()));
      }
    }


    // DRMVSNET stuff
    fullSystem->dr_mvsnet_view_num = dr_mvsnet_view_num;
    fullSystem->dr_width = reader->getWidth();
    fullSystem->dr_height = reader->getHeight();
    fullSystem->initDr();


    struct timeval tv_start;
    gettimeofday(&tv_start, NULL);
    clock_t started = clock();
    const auto timer_start = Timer::start();
    double sInitializerOffset = 0;


    for (int ii = 0; ii < (int) idsToPlay.size(); ii++) {
      if (!fullSystem->initialized)  // if not initialized: reset start time.
      {
        gettimeofday(&tv_start, NULL);
        started = clock();
        sInitializerOffset = timesToPlayAt[ii];
      }

      int i = idsToPlay[ii];


      ImageAndExposure *img;
      RGBDepth *depth = nullptr;
      dvo::core::RgbdImagePyramid *dvo_img = nullptr;
      unsigned char *img_bgr = nullptr;
      if (opt.preload) {
        img = preloadedImages[ii];
        img_bgr = preloadedImagesBGR[ii];
      } else {
        img = reader->getImage(i);
//                img_bgr = reader->getImageBGR_8UC3(i);
        img_bgr = reader->getImageBGR_8UC3_undis(i, reader->undistort->get_remapX(), reader->undistort->get_remapY());
        if (rgbd_flag) {
          dvo_img = reader->getDVORGBDetph(i, 5000.0);
          depth = reader->getRGBDetph(i, 5000.0);
        }
      }

      bool skipFrame = false;
      if (opt.playbackSpeed != 0) {
        struct timeval tv_now;
        gettimeofday(&tv_now, NULL);
        double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) +
                                                   (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

        if (sSinceStart < timesToPlayAt[ii])
          usleep((int) ((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
        else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2)) {
          printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
          skipFrame = true;
        }
      }

      if (!skipFrame) fullSystem->addActiveFrame(img, img_bgr, depth, dvo_img, i);

      delete img;
      if (rgbd_flag) {
        delete depth;
      }

      if (fullSystem->initFailed || setting_fullResetRequested) {
        if (ii < 250 || setting_fullResetRequested) {
          printf("RESETTING!\n");

          std::vector<IOWrap::Output3DWrapper *> wraps = fullSystem->outputWrapper;
          delete fullSystem;

          for (IOWrap::Output3DWrapper *ow : wraps) ow->reset();

          fullSystem = new FullSystem();
          fullSystem->setGammaFunction(reader->getPhotometricGamma());
          fullSystem->linearizeOperation = (opt.playbackSpeed == 0);
          fullSystem->result_folder = std::string(opt.result_folder);

          fullSystem->outputWrapper = wraps;

          setting_fullResetRequested = false;
        }
      }

      if (fullSystem->isLost) {
        printf("LOST!!\n");
        break;
      }

    }
    fullSystem->blockUntilMappingIsFinished();
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);
    double MilliSecondsTakenHRC = Timer::end_ms(timer_start);

    fullSystem->printResult(opt.result_folder + "result.txt");
    fullSystem->printAllResultSE3(opt.result_folder + "poses_dso.txt");
    fullSystem->printKeyframeIndex(opt.result_folder + "keyframes_dso.txt");
    if (dr_timing) fullSystem->printDrStatistics(opt.result_folder);

    int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());

    printf("\nTANDEM TIMING: =================="
           "\n%d Frames (%.1f fps)"
           "\n%.2fms per frame; "
           "\n%.2fs total time; "
           "\n======================\n\n",
           numFramesProcessed, 1000 * numFramesProcessed / MilliSecondsTakenHRC,
           MilliSecondsTakenHRC / numFramesProcessed,
           MilliSecondsTakenHRC / 1000.0);


    //fullSystem->printFrameLifetimes();
    if (setting_logStuff) {
      std::ofstream tmlog;
      tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
      tmlog << 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC * reader->getNumImages()) << " "
            << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) / (float) reader->getNumImages() << "\n";
      tmlog.flush();
      tmlog.close();
    }

    if (setting_tsdf_fusion) {
      float lower_corner[3] = {-5, -5, -5};
      float upper_corner[3] = {+5, +5, +5};
      fullSystem->fusion->SaveMeshToFile(opt.result_folder + "mesh.obj.incomplete", lower_corner, upper_corner);
      fullSystem->fusion->Synchronize();

      std::rename((opt.result_folder + "mesh.obj.incomplete").c_str(), (opt.result_folder + "mesh.obj").c_str());
      printf("Mesh Saving done!\n");
    }


    if (exit_when_done)
      exit(EXIT_SUCCESS);

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

  printf("DELETE READER!\n");
  delete reader;

  printf("EXIT NOW!\n");
  return 0;
}
