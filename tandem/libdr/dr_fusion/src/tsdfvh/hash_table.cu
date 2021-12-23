// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "tsdfvh/hash_table.h"
//#include "utils/utils.h"
#include <iostream>

#define THREADS_PER_BLOCK 512

namespace refusion {

namespace tsdfvh {

__global__ void InitEntriesKernel(HashEntry *entries, int num_entries) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < num_entries; i += stride) {
    entries[i].pointer = kFreeEntry;
    entries[i].position = make_int3(0, 0, 0);
  }
}

__global__ void InitHeapKernel(Heap *heap, VoxelBlock *voxel_blocks,
                               int num_blocks, int block_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (index == 0) {
    heap->heap_counter_ = num_blocks - 1;
  }

  for (int i = index; i < num_blocks; i += stride) {
    heap->heap_[i] = num_blocks - i - 1;

    for (int j = 0; j < block_size * block_size * block_size; j++) {
      voxel_blocks[i].at(j).sdf = 0;
      voxel_blocks[i].at(j).color = make_uchar3(0, 0, 0);
      voxel_blocks[i].at(j).weight = 0;
    }
  }
}

void HashTable::Init(int num_buckets, int bucket_size, int num_blocks,
                     int block_size) {
  num_buckets_ = num_buckets;
  bucket_size_ = bucket_size;
  num_entries_ = num_buckets * bucket_size;
  num_blocks_ = num_blocks;
  block_size_ = block_size;
  num_allocated_blocks_ = 0;
  cudaMallocManaged(&entries_, sizeof(HashEntry) * num_entries_);
  cudaMallocManaged(&voxels_, sizeof(Voxel) * block_size * block_size *
                                  block_size * num_blocks);
  cudaMallocManaged(&voxel_blocks_,
                    sizeof(VoxelBlock) * num_blocks);
  cudaMallocManaged(&heap_, sizeof(Heap));
  cudaDeviceSynchronize();
  for (size_t i = 0; i < num_blocks; i++) {
    voxel_blocks_[i].Init(&(voxels_[i * block_size * block_size * block_size]),
                          block_size);
  }
  heap_->Init(num_blocks);
  int threads_per_block = THREADS_PER_BLOCK;
  int thread_blocks =
      (num_entries_ + threads_per_block - 1) / threads_per_block;
  InitEntriesKernel<<<thread_blocks, threads_per_block>>>(entries_,
                                                          num_entries_);
  cudaDeviceSynchronize();

  thread_blocks = (num_blocks + threads_per_block - 1) / threads_per_block;
  InitHeapKernel<<<thread_blocks, threads_per_block>>>(heap_, voxel_blocks_,
                                                       num_blocks, block_size);
  cudaDeviceSynchronize();
}

void HashTable::Free() {
  cudaFree(entries_);
  cudaFree(voxels_);
  cudaFree(voxel_blocks_);
  cudaFree(heap_);
}

__device__ int HashTable::AllocateBlock(
    const int3 &position) {
  // This should be ok
  int bucket_idx = Hash(position);

  int free_entry_idx = -1;
  for (int i = 0; i < bucket_size_; i++) {
//    // TODO: remove
//    if ((bucket_idx + i) >= num_entries_) {
//      KERNEL_ABORT("bucket_idx + i = %d, entries = %d\n", bucket_idx + i, num_entries_);
//    }
    if (entries_[bucket_idx + i].position.x == position.x &&
        entries_[bucket_idx + i].position.y == position.y &&
        entries_[bucket_idx + i].position.z == position.z &&
        entries_[bucket_idx + i].pointer != kFreeEntry) {
      return 0;
    }
    if (free_entry_idx == -1 &&
        entries_[bucket_idx + i].pointer == kFreeEntry) {
      free_entry_idx = bucket_idx + i;
    }
  }

  if (free_entry_idx != -1) {
    int mutex = 0;
    mutex =
        atomicCAS(&entries_[free_entry_idx].pointer, kFreeEntry, kLockEntry);
    if (mutex == kFreeEntry) {
      entries_[free_entry_idx].position = position;
      entries_[free_entry_idx].pointer = heap_->Consume();
      atomicAdd(&num_allocated_blocks_, 1);
      return 1;
    }
  }
  return -1;
}

__device__ bool HashTable::DeleteBlock(
    const int3 &position) {
  int bucket_idx = Hash(position);

  for (int i = 0; i < bucket_size_; i++) {
    if (entries_[bucket_idx + i].position.x == position.x &&
        entries_[bucket_idx + i].position.y == position.y &&
        entries_[bucket_idx + i].position.z == position.z &&
        entries_[bucket_idx + i].pointer != kFreeEntry) {
      int ptr = entries_[bucket_idx + i].pointer;
      for(int j=0;j<block_size_ * block_size_ * block_size_; j++) {
        voxel_blocks_[ptr].at(j).sdf = 0;
        voxel_blocks_[ptr].at(j).color = make_uchar3(0, 0, 0);
        voxel_blocks_[ptr].at(j).weight = 0;
      }
      heap_->Append(ptr);
      entries_[bucket_idx + i].pointer = kFreeEntry;
      entries_[bucket_idx + i].position = make_int3(0, 0, 0);
      return true;
    }
  }
  return false;
}

__host__ __device__ HashEntry HashTable::FindHashEntry(int3 position) {
  int bucket_idx = Hash(position);
  for (int i = 0; i < bucket_size_; i++) {
    if (entries_[bucket_idx + i].position.x == position.x &&
        entries_[bucket_idx + i].position.y == position.y &&
        entries_[bucket_idx + i].position.z == position.z &&
        entries_[bucket_idx + i].pointer != kFreeEntry) {
      return entries_[bucket_idx + i];
    }
  }
  HashEntry entry;
  entry.position = position;
  entry.pointer = kFreeEntry;
  return entry;
}

__host__ __device__ int HashTable::Hash(int3 position) {
  const int p1 = 73856093;
  const int p2 = 19349669;
  const int p3 = 83492791;

  int result = ((position.x * p1) ^ (position.y * p2) ^ (position.z * p3)) %
               num_buckets_;
  if (result < 0) {
    result += num_buckets_;
  }
  return result * bucket_size_;
}

int HashTable::GetNumAllocatedBlocks() {
  return num_allocated_blocks_;
}

__host__ __device__ int HashTable::GetNumEntries() {
  return num_entries_;
}

__host__ __device__ HashEntry HashTable::GetHashEntry(int i) {
  return entries_[i];
}

}  // namespace tsdfvh

}  // namespace refusion
