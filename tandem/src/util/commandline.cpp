// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#include "commandline.h"
#include "settings.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <sstream>
#include <vector>
#include <iterator>

namespace dso {

void tandemDefaultSettings(CommandLineOptions &opt, char const *const buf) {
  char val[100];
  if (1 != sscanf(buf, "preset=%s", val)) {
    printf("The first argument must be 'preset=X', but it is '%s'\n", buf);
    exit(EXIT_FAILURE);
  }

  opt.preset = val;
  if (strcmp(val, "dataset") == 0) {
    // DSO
    opt.preload = false;
    opt.playbackSpeed = 0;
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    // log/output
    setting_logStuff = false;
    setting_debugout_runquiet = true;
    disableAllDisplay = true;

    // TANDEM
    setting_tsdf_fusion = true;
    setting_tracking_type = "dense";
    setting_tracking_device = "cpu";
    mesh_extraction_freq = 0;

    // Pangolin
    pangolin_fullscreen = false;
    pangolin_mesh = true;
    pangolin_reduced_images = true;
  } else if (strcmp(val, "gui") == 0) {
    // DSO
    opt.preload = false;
    opt.playbackSpeed = 0;
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    // log/output
    setting_logStuff = false;
    setting_debugout_runquiet = true;

    // TANDEM
    setting_tsdf_fusion = true;
    setting_tracking_type = "dense";
    setting_tracking_device = "cpu";
    mesh_extraction_freq = 5;

    // Pangolin
    pangolin_fullscreen = false;
    pangolin_mesh = true;
    pangolin_reduced_images = true;
  } else if (strcmp(val, "runtime") == 0) {
    // DSO
    opt.preload = true;
    opt.playbackSpeed = 0;
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    // log/output
    setting_logStuff = false;
    setting_debugout_runquiet = true;
    disableAllDisplay = true;

    // TANDEM
    setting_tsdf_fusion = true;
    setting_tracking_type = "dense";
    setting_tracking_device = "cuda";
    mesh_extraction_freq = 0;

    // Pangolin
    pangolin_fullscreen = false;
    pangolin_mesh = true;
    pangolin_reduced_images = true;
  } else if (strcmp(val, "demo") == 0) {
    // DSO
    opt.playbackSpeed = 1;
    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;

    // log/output
    setting_logStuff = false;
    setting_debugout_runquiet = true;

    // TANDEM
    setting_tsdf_fusion = true;
    setting_tracking_type = "dense";
    setting_tracking_device = "cpu";
    setting_tracking_step = 2;
    mesh_extraction_freq = 5;

    // Pangolin
    pangolin_fullscreen = false;
    pangolin_mesh = true;
    pangolin_reduced_images = true;
  } else {
    printf("The preset option '%s' is not implemented.\n", val);
    exit(EXIT_FAILURE);
  }
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
  std::istringstream iss(s);
  std::string item;
  while (std::getline(iss, item, delim)) {
    *result++ = item;
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}


void parseArgument(CommandLineOptions &opt, char *arg) {
  int option;
  float foption;
  char buf[1000];

  if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
    if (option == 1) {
      opt.useSampleOutput = true;
      printf("USING SAMPLE OUTPUT WRAPPER!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "quiet=%d", &option)) {
    if (option == 1) {
      setting_debugout_runquiet = true;
      printf("QUIET MODE, I'll shut up!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "rec=%d", &option)) {
    if (option == 0) {
      disableReconfigure = true;
      printf("DISABLE RECONFIGURE!\n");
    }
    return;
  }


  if (1 == sscanf(arg, "noros=%d", &option)) {
    if (option == 1) {
      opt.disableROS = true;
      disableReconfigure = true;
      printf("DISABLE ROS (AND RECONFIGURE)!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "nolog=%d", &option)) {
    if (option == 1) {
      setting_logStuff = false;
      printf("DISABLE LOGGING!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "reverse=%d", &option)) {
    if (option == 1) {
      opt.reverse = true;
      printf("REVERSE!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nogui=%d", &option)) {
    if (option == 1) {
      disableAllDisplay = true;
      printf("NO GUI!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "nomt=%d", &option)) {
    if (option == 1) {
      multiThreading = false;
      printf("NO MultiThreading!\n");
    }
    return;
  }
  if (1 == sscanf(arg, "start=%d", &option)) {
    opt.start = option;
    printf("START AT %d!\n", opt.start);
    return;
  }
  if (1 == sscanf(arg, "end=%d", &option)) {
    opt.end = option;
    printf("END AT %d!\n", opt.end);
    return;
  }

  if (1 == sscanf(arg, "files=%s", buf)) {
    opt.source = buf;
    printf("loading data from %s!\n", opt.source.c_str());
    return;
  }

  if (1 == sscanf(arg, "calib=%s", buf)) {
    opt.calib = buf;
    printf("loading calibration from %s!\n", opt.calib.c_str());
    return;
  }

  if (1 == sscanf(arg, "vignette=%s", buf)) {
    opt.vignette = buf;
    printf("loading vignette from %s!\n", opt.vignette.c_str());
    return;
  }

  if (1 == sscanf(arg, "gamma=%s", buf)) {
    opt.gammaCalib = buf;
    printf("loading gammaCalib from %s!\n", opt.gammaCalib.c_str());
    return;
  }

  if (1 == sscanf(arg, "rescale=%f", &foption)) {
    opt.rescale = foption;
    printf("RESCALE %f!\n", opt.rescale);
    return;
  }

  if (1 == sscanf(arg, "speed=%f", &foption)) {
    opt.playbackSpeed = foption;
    printf("PLAYBACK SPEED %f!\n", opt.playbackSpeed);
    return;
  }

  if (1 == sscanf(arg, "preload=%d", &option)) {
    opt.preload = option;
    return;
  }


  if (1 == sscanf(arg, "save=%d", &option)) {
    if (option == 1) {
      debugSaveImages = true;
      if (42 == system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
      if (42 == system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
      if (42 == system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
      if (42 == system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
      printf("SAVE IMAGES!\n");
    }
    return;
  }

  if (1 == sscanf(arg, "mode=%d", &option)) {

    opt.mode = option;
    if (option == 0) {
      printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
    }
    if (option == 1) {
      printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    }
    if (option == 2) {
      printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
      setting_photometricCalibration = 0;
      setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
      setting_minGradHistAdd = 3;
    }
    return;
  }

  if (1 == sscanf(arg, "depth_folder=%s", buf)) {
    depth_save_folder = buf;
    depth_save_folder = depth_save_folder.back() == '/' ? depth_save_folder : depth_save_folder + "/";
    debugSaveDepthImages = true;
    mkdir((depth_save_folder).c_str(), 0775);
    printf("Save depth maps to %s!\n", depth_save_folder.c_str());
    return;
  }

  if (1 == sscanf(arg, "tracking=%s", buf)) {
    std::vector<std::string> s = split(std::string(buf), ':');
    setting_tracking_type = s.at(0);
    if (s.size() >= 2) setting_tracking_device = s.at(1);
    if (s.size() >= 3) setting_tracking_step = std::atoi(s.at(2).c_str());
    printf("BLUB: Setting tracking\n");
    return;
  }

  if (1 == sscanf(arg, "result_folder=%s", buf)) {
    opt.result_folder = buf;
    opt.result_folder = opt.result_folder.back() == '/' ? opt.result_folder : opt.result_folder + "/";
    return;
  }

  if (1 == sscanf(arg, "mvsnet_folder=%s", buf)) {
    mvsnet_folder = buf;
    mvsnet_folder = mvsnet_folder.back() == '/' ? mvsnet_folder : mvsnet_folder + "/";
    mvsnet_flag = true;
    printf("Loading MVSNet from %s!\n", mvsnet_folder.c_str());
    return;
  }

  if (1 == sscanf(arg, "mvsnet_discard_percentage=%f", &foption)) {
    mvsnet_discard_percentage = foption;
    printf("MVSNET discarding %f %% of points!\n", mvsnet_discard_percentage);
    return;
  }

  if (1 == sscanf(arg, "rgbdepth_folder=%s", buf)) {
    opt.rgbdepth_folder = buf;
    opt.rgbdepth_folder = opt.rgbdepth_folder.back() == '/' ? opt.rgbdepth_folder : opt.rgbdepth_folder + "/";
    rgbd_flag = true;
    printf("Run RGBD VO!\n");
    printf("[RBGD] Load Depth Data from %s!\n", opt.rgbdepth_folder.c_str());
    return;
  }

  if (1 == sscanf(arg, "tsdf_fusion=%d", &option)) {
    setting_tsdf_fusion = option;
    return;
  }

  if (1 == sscanf(arg, "dense_tracking_with_dense_depth_only=%d", &option)) {
    dense_tracking_with_dense_depth_only = option;
    return;
  }

  if (1 == sscanf(arg, "exit_when_done=%d", &option)) {
    exit_when_done = option;
    return;
  }

  if (1 == sscanf(arg, "save_depth_maps=%d", &option)) {
    save_depth_maps = option;
    return;
  }

  if (1 == sscanf(arg, "use_int_sorting=%d", &option)) {
    use_int_sorting = option;
    return;
  }

  if (1 == sscanf(arg, "frame_save_index=%d", &option)) {
    frame_save_index = option;
    return;
  }

  if (1 == sscanf(arg, "kFweight=%f", &foption)) {
    setting_kfGlobalWeight = foption;
    printf("KF Global Weight %f!\n", setting_kfGlobalWeight);
    return;
  }

  if (1 == sscanf(arg, "dr_timing=%d", &option)) {
    dr_timing = option;
    return;
  }

  if (1 == sscanf(arg, "dr_mvsnet_view_num=%d", &option)) {
    dr_mvsnet_view_num = option;
    return;
  }

  if (1 == sscanf(arg, "mesh_extraction_freq=%d", &option)) {
    mesh_extraction_freq = option;
    return;
  }
  if (1 == sscanf(arg, "save_coarse_tracker=%d", &option)) {
    save_coarse_tracker = option;
    return;
  }
  if (1 == sscanf(arg, "demo_secs=%f", &foption)) {
    opt.demo_secs = foption;
    return;
  }

  printf("could not parse argument \"%s\"!!!!\n", arg);
}


void printSettings(CommandLineOptions &opt) {
  printf("\n=============== TANDEM Settings: ===============\n");
  printf("\tSetting '%s':\n", opt.preset.c_str());
  printf("\t- %s real-time enforcing\n", opt.playbackSpeed == 0 ? "no" : "yes");
  printf("\t- %4d active points\n", (int) setting_desiredPointDensity);
  printf("\t- %d-%d active frames\n", setting_minFrames, setting_maxFrames);
  printf("\t- %d-%d LM iteration each KF\n", setting_minOptIterations, setting_maxOptIterations);
  printf("\t- TSDF fusion: %s\n", setting_tsdf_fusion ? "yes" : "no");
  printf("\t- %s tracking on %s (step=%d)\n", setting_tracking_type.c_str(), setting_tracking_device.c_str(), setting_tracking_step);
  printf("\t- Pangolin\n");
  printf("\t  - Fullscreen: %d\n", pangolin_fullscreen);
  printf("\t  - Mesh: %d\n", pangolin_mesh);
  printf("\t  - Smaller Images: %d\n", pangolin_reduced_images);
  printf("\n");
}

} // namespace dso