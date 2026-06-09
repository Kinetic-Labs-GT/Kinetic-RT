#pragma once

#include <vector>
#include <mutex>
#include <stdexcept>
#include <iostream>

#if defined(MOCK_HIP)
#include "../tests/mock_hip.h"
#else
#include <hip/hip_runtime.h>
#endif

#define CHECK_HIP_BM(cmd) \
do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        throw std::runtime_error("HIP memory error in BlockManager"); \
    } \
} while(0)

struct Block {
    int id;
    int max_tokens;
    int current_tokens;
    void* memory_ptr;
};

class BlockManager {
public:
    // PagedAttention BlockManager managing a massive contiguous VRAM allocation
    BlockManager(int num_blocks, int tokens_per_block, int token_byte_size)
        : tokens_per_block_(tokens_per_block), vram_pool_ptr_(nullptr) {

        size_t block_bytes = tokens_per_block * token_byte_size;
        size_t total_vram_pool_size = num_blocks * block_bytes;

        // Mandated Native VRAM Allocation: Allocate massive contiguous pool once
#if defined(MOCK_HIP)
        vram_pool_ptr_ = malloc(total_vram_pool_size); // malloc for mock CI
#else
        CHECK_HIP_BM(hipMalloc(&vram_pool_ptr_, total_vram_pool_size));
#endif

        char* pool_offset = reinterpret_cast<char*>(vram_pool_ptr_);
        for (int i = 0; i < num_blocks; ++i) {
            free_blocks_.push_back({i, tokens_per_block, 0, reinterpret_cast<void*>(pool_offset + (i * block_bytes))});
        }
    }

    ~BlockManager() {
        if (vram_pool_ptr_) {
#if defined(MOCK_HIP)
            free(vram_pool_ptr_);
#else
            hipFree(vram_pool_ptr_);
#endif
            vram_pool_ptr_ = nullptr;
        }
    }

    int allocate_block() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.empty()) {
            throw std::runtime_error("Out of KV Cache blocks! Massive pool exhausted.");
        }
        Block b = free_blocks_.back();
        free_blocks_.pop_back();
        return b.id;
    }

    void free_block(int id) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Recalculate block pointer safely mathematically avoiding fragmentation
        size_t block_bytes = tokens_per_block_ * 4096; // Example scalar
        char* pool_offset = reinterpret_cast<char*>(vram_pool_ptr_);
        free_blocks_.push_back({id, tokens_per_block_, 0, reinterpret_cast<void*>(pool_offset + (id * block_bytes))});
    }

    void* get_pool_ptr() const {
        return vram_pool_ptr_;
    }

private:
    std::vector<Block> free_blocks_;
    int tokens_per_block_;
    std::mutex mutex_;
    void* vram_pool_ptr_;
};
