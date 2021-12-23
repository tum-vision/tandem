// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include "cuda_runtime.h"

#define kFreeEntry -1
#define kLockEntry -2

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Struct that represents a hash entry
 */
struct HashEntry {
  /** Entry position (lower left corner of the voxel block) */
  int3 position;
  /** Pointer to the position in the heap of the voxel block */
  int pointer = kFreeEntry;      
};

}  // namespace tsdfvh

}  // namespace refusion
