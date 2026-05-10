#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <mutex>

#if defined(MOCK_HIP)
#include "../tests/mock_hip.h"
#else
#include <hip/hip_runtime.h>
#endif

// HardwareMismatch Exception
class HardwareMismatch : public std::runtime_error {
public:
    explicit HardwareMismatch(const std::string& msg) : std::runtime_error(msg) {}
};

// Represents the .kin file structured header
#pragma pack(push, 1)
struct KinHeader {
    uint32_t magic_number; // e.g., 0x4B494E00
    uint32_t version;
    char device_id[256];   // e.g., "gfx1100"
    uint64_t weights_hash;
    uint64_t op_graph_data_offset;
    uint64_t op_graph_data_size;
    uint64_t kernel_binaries_offset;
    uint64_t kernel_binaries_size;
};
#pragma pack(pop)

// Simple structured wrapper around kernel configurations (e.g. GEMM)
struct KernelVariant {
    std::string name;
    // other config properties...
};

class SmartAutotuner {
public:
    // Profiles 3 GEMM variants: High Occupancy, Balanced, High-LDS
    // Returns the winning variant's name
    std::string profile_gemm(uintptr_t stream_ptr);
};

class Serializer {
public:
    // Serializes the graph recipe and kernel binaries into a single .kin file
    void save_kin_file(const std::string& filepath, const std::string& device_id, uint64_t weights_hash, const std::vector<uint8_t>& op_graph_data, const std::vector<uint8_t>& kernel_binaries);

    // Reads the .kin file and validates hardware compatibility
    // Returns the kernel binaries
    std::vector<uint8_t> load_kin_file(const std::string& filepath);
};

class AOTEngine {
public:
    AOTEngine();
    ~AOTEngine();

    // Perform AOT compilation and tuning, save as .kin
    void compile_ahead_of_time(const std::string& output_filepath, uintptr_t stream_ptr);

    // Load from .kin and initialize the engine
    void load_model(const std::string& filepath);

private:
    SmartAutotuner autotuner_;
    Serializer serializer_;
    hipModule_t module_;
    std::recursive_mutex engine_mutex_;

    void load_kernel(const std::vector<uint8_t>& binary_data);
    void validate_elf_structure(const std::vector<uint8_t>& binary_data) const;
};
