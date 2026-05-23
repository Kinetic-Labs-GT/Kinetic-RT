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

#include <pybind11/pybind11.h>

// HardwareMismatch Exception
class HardwareMismatchError : public std::runtime_error {
public:
    explicit HardwareMismatchError(const std::string& msg) : std::runtime_error(msg) {}
};

// Represents the .kin file structured header
#pragma pack(push, 1)
struct KinHeader {
    uint32_t magic_number; // e.g., 0x4B494E00
    uint32_t version;
    char device_id[256];   // e.g., "gfx1100"
    char kinetic_target[256]; // e.g., "CUDA_sm75" or "ROCm_gfx942"
    uint64_t weights_hash;
    uint64_t op_graph_data_offset;
    uint64_t op_graph_data_size;
    uint64_t kernel_binaries_offset;
    uint64_t kernel_binaries_size;
    uint32_t tensor_parallel_degree;
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
    Serializer();

    // Serializes the graph recipe and kernel binaries into a single .kin file
    void save_kin_file(const std::string& filepath, const std::string& device_id, const std::string& kinetic_target, uint64_t weights_hash, const std::vector<uint8_t>& op_graph_data, const std::vector<uint8_t>& kernel_binaries);

    // Reads the .kin file and validates hardware compatibility
    // Returns the kernel binaries
    std::vector<uint8_t> load_kin_file(const std::string& filepath);

    // Reads metadata without fully loading the binary
    uint32_t get_tensor_parallel_degree(const std::string& filepath);

    const std::string& get_device_id() const { return device_id_; }
    const std::string& get_loaded_kinetic_target() const { return loaded_kinetic_target_; }

private:
    std::string device_id_;
    std::string loaded_kinetic_target_;
};

class AOTEngine {
public:
    AOTEngine();
    ~AOTEngine();

    // Perform AOT compilation and tuning, save as .kin
    void compile_ahead_of_time(const std::string& output_filepath, uintptr_t stream_ptr, const std::string& kinetic_target);

    // Load from .kin and initialize the engine
    void load_model(const std::string& filepath);

    void launch(pybind11::object py_input, uintptr_t stream_ptr);
    void synchronize_and_clear(uintptr_t stream_ptr);

private:
    SmartAutotuner autotuner_;
    Serializer serializer_;
    hipModule_t module_;
    std::recursive_mutex engine_mutex_;
    std::vector<pybind11::object> pinned_buffers_;

    void load_kernel(const std::vector<uint8_t>& binary_data, const std::string& kinetic_target);
    void validate_elf_structure(const std::vector<uint8_t>& binary_data, const std::string& kinetic_target) const;
};
