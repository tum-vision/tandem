cmake_minimum_required(VERSION 3.0)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif ()

project(cuda_coarse_tracker CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug mode. Adding -G to NVCC FLAGS")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G")
else ()
    message("-- Release mode (!= not Debug). Adding -O3;-lineinfo to NVCC FLAGS")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo")
endif ()
# https://cmake.org/pipermail/cmake/2018-July/067904.html
# TODO(lukas): Doesn't work if compiled with whole project. Probably because of not having CUDA in top level?
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_NVCC_FLAGS}")

add_library(cuda_coarse_tracker STATIC src/cuda_coarse_tracker.cpp src/cuda_coarse_tracker_private.cu)
#target_compile_options(cuda_coarse_tracker PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

target_include_directories(cuda_coarse_tracker PRIVATE include/private)
target_include_directories(cuda_coarse_tracker PUBLIC include/public ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

target_link_libraries(cuda_coarse_tracker ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY} CUDA::cublas cnpy)

#add_executable(main
#        src/main.cu
#        src/cuda_coarse_tracker.cpp src/cuda_coarse_tracker_private.cu
#        )
##target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
#
#target_include_directories(main PUBLIC include/public include/private ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
#target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY} CUDA::cublas cnpy)