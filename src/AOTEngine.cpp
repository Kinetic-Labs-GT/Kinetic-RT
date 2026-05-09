#include "AOTEngine.h"
#include <fstream>
#include <cstring>

#define CHECK_HIP(cmd) \
do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("HIP error"); \
    } \
} while(0)

// --- SmartAutotuner ---

std::string SmartAutotuner::profile_gemm(uintptr_t stream_ptr) {
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

    std::vector<KernelVariant> variants = {
        {"High Occupancy"},
        {"Balanced"},
        {"High-LDS"}
    };

    std::string best_variant = "";
    float best_time = 1e9f;

    for (const auto& variant : variants) {
        hipEvent_t start, stop;
        CHECK_HIP(hipEventCreate(&start));
        CHECK_HIP(hipEventCreate(&stop));

        CHECK_HIP(hipEventRecord(start, stream));

        // In a real implementation, we would launch the specific kernel variant here
        // e.g., hipModuleLaunchKernel(...)

        CHECK_HIP(hipEventRecord(stop, stream));
        CHECK_HIP(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));

        if (elapsed_ms < best_time) {
            best_time = elapsed_ms;
            best_variant = variant.name;
        }

        CHECK_HIP(hipEventDestroy(start));
        CHECK_HIP(hipEventDestroy(stop));
    }

    return best_variant;
}

// --- Serializer ---

void Serializer::save_kin_file(const std::string& filepath, const std::string& device_id, uint64_t weights_hash, const std::vector<uint8_t>& op_graph_data, const std::vector<uint8_t>& kernel_binaries) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    KinHeader header;
    header.magic_number = 0x4B494E00; // "KIN\0"
    header.version = 1;
    std::strncpy(header.device_id, device_id.c_str(), sizeof(header.device_id) - 1);
    header.device_id[sizeof(header.device_id) - 1] = '\0';
    header.weights_hash = weights_hash;

    header.op_graph_data_offset = sizeof(KinHeader);
    header.op_graph_data_size = op_graph_data.size();

    header.kernel_binaries_offset = header.op_graph_data_offset + header.op_graph_data_size;
    header.kernel_binaries_size = kernel_binaries.size();

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(op_graph_data.data()), op_graph_data.size());
    out.write(reinterpret_cast<const char*>(kernel_binaries.data()), kernel_binaries.size());
}

std::vector<uint8_t> Serializer::load_kin_file(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    KinHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.magic_number != 0x4B494E00) {
        throw std::runtime_error("Invalid file format: bad magic number.");
    }

    // Verify Hardware Mismatch
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0)); // Assuming device 0
    std::string current_device_id(prop.gcnArchName);

    if (current_device_id != header.device_id) {
        throw HardwareMismatch("Hardware mismatch: expected " + std::string(header.device_id) + " but got " + current_device_id);
    }

    // Skip op graph data for now, just read kernel binaries
    in.seekg(header.kernel_binaries_offset, std::ios::beg);
    std::vector<uint8_t> kernel_binaries(header.kernel_binaries_size);
    in.read(reinterpret_cast<char*>(kernel_binaries.data()), header.kernel_binaries_size);

    return kernel_binaries;
}

// --- AOTEngine ---

AOTEngine::AOTEngine() : module_(nullptr) {
}

AOTEngine::~AOTEngine() {
    // If we had a module unload, we'd do it here
}

void AOTEngine::compile_ahead_of_time(const std::string& output_filepath, uintptr_t stream_ptr) {
    // 1. Profile and find best kernel
    std::string best_variant = autotuner_.profile_gemm(stream_ptr);
    std::cout << "Selected best GEMM variant: " << best_variant << std::endl;

    // 2. Fetch current device properties
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::string device_id(prop.gcnArchName);

    // 3. Serialize to .kin
    std::vector<uint8_t> dummy_op_graph_data = {0x01, 0x02, 0x03}; // Mock data
    std::vector<uint8_t> dummy_hsaco = {0x7F, 'E', 'L', 'F'};      // Mock hsaco binary

    uint64_t dummy_weights_hash = 123456789;

    serializer_.save_kin_file(output_filepath, device_id, dummy_weights_hash, dummy_op_graph_data, dummy_hsaco);
}

void AOTEngine::load_model(const std::string& filepath) {
    // 1. Load the .kin file and verify hardware
    std::vector<uint8_t> kernel_binaries = serializer_.load_kin_file(filepath);

    // 2. Load the kernel into the HIP module
    load_kernel(kernel_binaries);
}

void AOTEngine::load_kernel(const std::vector<uint8_t>& binary_data) {
    // hipModuleLoadData expects the binary image.
    CHECK_HIP(hipModuleLoadData(&module_, binary_data.data()));
}
