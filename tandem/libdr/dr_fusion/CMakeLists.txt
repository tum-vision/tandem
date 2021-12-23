cmake_minimum_required(VERSION 3.0)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif ()

project(dr_fusion)

add_compile_options(-std=c++14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++14;-O3)

# See: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_60,code=sm_60)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_70,code=sm_70)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_75,code=sm_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_75,code=compute_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lineinfo)

cuda_add_library(dr-fusion STATIC
        src/utils/rgbd_image.cu
        src/marching_cubes/mesh.cu
        src/marching_cubes/mesh_extractor.cu
        src/tsdfvh/heap.cu
        src/tsdfvh/hash_table.cu
        src/tsdfvh/tsdf_volume.cu
        src/dr_fusion/dr_fusion.cpp
        )
target_include_directories(dr-fusion
        PRIVATE src ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
        INTERFACE src/dr_fusion
        )

# This has a cost of about 1 fps but it prohibits some hard to trace bug
target_compile_definitions(dr-fusion PUBLIC DR_FUSION_DEBUG_SYNC_LAUNCH)
