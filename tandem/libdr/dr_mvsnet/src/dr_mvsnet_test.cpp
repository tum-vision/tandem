// Copyright (c) 2020 Lukas Koestler, Nan Yang. All rights reserved.

#include <iostream>
#include <string>
#include <chrono>
#include "dr_mvsnet.h"

int main(int argc, const char *argv[]) {
    using std::cout;
    using std::endl;

    std::string module_path;
    std::string sample_path;
    int repetitions = 1;
    char const* out_folder = NULL;

    if (argc == 3){
        module_path = std::string(argv[1]);
        sample_path = std::string(argv[2]);
    }else if (argc == 4) {
      module_path = std::string(argv[1]);
      sample_path = std::string(argv[2]);
      repetitions = std::atoi(argv[3]);
    }else if (argc == 5){
        module_path = std::string(argv[1]);
        sample_path = std::string(argv[2]);
        repetitions = std::atoi(argv[3]);
        out_folder = argv[4];
    }else{
        std::cerr << "usage: ./dr_mvsnet_test <path-to-exported-script-module> <path-to-sample-input> [repetitions] [out_folder]" << endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    DrMvsnet mvsnet(module_path.c_str());
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
    cout << "Loading Model: " << (double) elapsed / 1000000.0 << " s" << endl;

    bool correct = test_dr_mvsnet(mvsnet, sample_path.c_str(), true, repetitions, out_folder);

    if (!correct) {
        return -1;
    }

    return 0;
}
