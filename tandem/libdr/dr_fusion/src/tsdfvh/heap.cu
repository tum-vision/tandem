// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "tsdfvh/heap.h"
#include "utils/utils.h"
#include <stdio.h>
#include <assert.h>

namespace refusion {

    namespace tsdfvh {

        void Heap::Init(int heap_size) {
            cudaMallocManaged(&heap_, sizeof(unsigned int) * heap_size);
        }

        __device__ unsigned int Heap::Consume() {
            if (heap_counter_ <= 0){
                KERNEL_ABORT("heap_counter_ = %d\n", heap_counter_)
            }
//            if (heap_counter_ <= 1000) {
//              // TODO: Remove due to perf
//              printf("HEAP COUNTER = %u \n", heap_counter_);
//            }
            int idx = atomicSub(&heap_counter_, 1);
            return heap_[idx];
        }

        __device__ void Heap::Append(unsigned int ptr) {
            if (heap_counter_ <= 0){
                KERNEL_ABORT("abort in append\n")
            }
            unsigned int idx = atomicAdd(&heap_counter_, 1);
            heap_[idx + 1] = ptr;
        }


    }  // namespace tsdfvh

}  // namespace refusion
