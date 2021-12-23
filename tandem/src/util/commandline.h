// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef PBA_COMMANDLINE_H
#define PBA_COMMANDLINE_H

#include <string>

namespace dso {

class CommandLineOptions {
public:
  std::string vignette = "";
  std::string gammaCalib = "";
  std::string source = "";
  std::string calib = "";
  std::string result_folder = "";
  std::string rgbdepth_folder = "";
  std::string preset = "";

  double rescale = 1;
  bool reverse = false;
  bool disableROS = false;
  int start = 0;
  int end = 100000;
  float playbackSpeed = 0;  // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
  bool preload = false;
  bool useSampleOutput = false;


  int mode = 0;

  // Only for demo
  float demo_secs = 30.0f;
};

void tandemDefaultSettings(CommandLineOptions &opt, char const *buf);

void parseArgument(CommandLineOptions &opt, char *arg);

void printSettings(CommandLineOptions &opt);

} // namespace dso

#endif //PBA_COMMANDLINE_H
