#include "AOTEngine.h"
#include <fstream>
#include <cstring>
#include <endian.h>

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

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    for (const auto& variant : variants) {
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
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    return best_variant;
}

// --- Serializer ---

void Serializer::save_kin_file(const std::string& filepath, const std::string& device_id, uint64_t weights_hash, const std::vector<uint8_t>& op_graph_data, const std::vector<uint8_t>& kernel_binaries) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    KinHeader header;
    header.magic_number = htole32(0x4B494E00); // "KIN\0"
    header.version = htole32(1);
    std::strncpy(header.device_id, device_id.c_str(), sizeof(header.device_id) - 1);
    header.device_id[sizeof(header.device_id) - 1] = '\0';
    header.weights_hash = htole64(weights_hash);

    header.op_graph_data_offset = htole64(sizeof(KinHeader));
    header.op_graph_data_size = htole64(op_graph_data.size());

    header.kernel_binaries_offset = htole64(sizeof(KinHeader) + op_graph_data.size());
    header.kernel_binaries_size = htole64(kernel_binaries.size());

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(op_graph_data.data()), op_graph_data.size());
    out.write(reinterpret_cast<const char*>(kernel_binaries.data()), kernel_binaries.size());
}

std::vector<uint8_t> Serializer::load_kin_file(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streamsize>(sizeof(KinHeader))) {
        throw std::runtime_error("Invalid file format: file too small for header.");
    }

    KinHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    header.device_id[sizeof(header.device_id) - 1] = '\0';

    uint32_t magic_number = le32toh(header.magic_number);
    uint32_t version = le32toh(header.version);
    uint64_t weights_hash = le64toh(header.weights_hash);
    uint64_t op_graph_data_offset = le64toh(header.op_graph_data_offset);
    uint64_t op_graph_data_size = le64toh(header.op_graph_data_size);
    uint64_t kernel_binaries_offset = le64toh(header.kernel_binaries_offset);
    uint64_t kernel_binaries_size = le64toh(header.kernel_binaries_size);

    if (magic_number != 0x4B494E00) {
        throw std::runtime_error("Invalid file format: bad magic number.");
    }

    // Offset/Size Validation (Serialization Hardening)
    // Check against arithmetic overflow and file boundaries
    if (op_graph_data_offset + op_graph_data_size < op_graph_data_offset ||
        kernel_binaries_offset + kernel_binaries_size < kernel_binaries_offset) {
        throw std::runtime_error("Invalid file format: offset overflow.");
    }

    if (op_graph_data_offset + op_graph_data_size > static_cast<uint64_t>(file_size) ||
        kernel_binaries_offset + kernel_binaries_size > static_cast<uint64_t>(file_size)) {
        throw std::runtime_error("Invalid file format: sizes exceed file bounds.");
    }

    // Additional sanity bounds for massive allocations
    if (kernel_binaries_size > 1024ULL * 1024ULL * 1024ULL * 2ULL) { // 2 GB max arbitrary limit
        throw std::runtime_error("Invalid file format: kernel binary excessively large.");
    }

    // Verify Hardware Mismatch
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0)); // Assuming device 0
    std::string current_device_id(prop.gcnArchName);

    if (current_device_id != header.device_id) {
        throw HardwareMismatch("Hardware mismatch: expected " + std::string(header.device_id) + " but got " + current_device_id);
    }

    // Skip op graph data for now, just read kernel binaries
    in.seekg(kernel_binaries_offset, std::ios::beg);
    std::vector<uint8_t> kernel_binaries(kernel_binaries_size);
    in.read(reinterpret_cast<char*>(kernel_binaries.data()), kernel_binaries_size);

    if (in.gcount() != static_cast<std::streamsize>(kernel_binaries_size)) {
        throw std::runtime_error("Invalid file format: short read on kernel binaries.");
    }

    return kernel_binaries;
}

// --- AOTEngine ---

AOTEngine::AOTEngine() : module_(nullptr) {
}

AOTEngine::~AOTEngine() {
    if (module_ != nullptr) {
        hipModuleUnload(module_);
        module_ = nullptr;
    }
}

void AOTEngine::compile_ahead_of_time(const std::string& output_filepath, uintptr_t stream_ptr) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    // 1. Profile and find best kernel
    std::string best_variant = autotuner_.profile_gemm(stream_ptr);
    std::cout << "Selected best GEMM variant: " << best_variant << std::endl;

    // 2. Fetch current device properties
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::string device_id(prop.gcnArchName);

    // 3. Serialize to .kin
    std::vector<uint8_t> dummy_op_graph_data = {0x01, 0x02, 0x03}; // Mock data

    // Mock 64-byte hsaco binary that passes deep validation
    std::vector<uint8_t> dummy_hsaco(64, 0);
    dummy_hsaco[0] = 0x7F; dummy_hsaco[1] = 'E'; dummy_hsaco[2] = 'L'; dummy_hsaco[3] = 'F';
    dummy_hsaco[4] = 2; // 64-bit class
    dummy_hsaco[18] = 0xE0; dummy_hsaco[19] = 0x00; // EM_AMDGPU

    uint64_t dummy_weights_hash = 123456789;

    serializer_.save_kin_file(output_filepath, device_id, dummy_weights_hash, dummy_op_graph_data, dummy_hsaco);
}

void AOTEngine::load_model(const std::string& filepath) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    // 1. Load the .kin file and verify hardware
    std::vector<uint8_t> kernel_binaries = serializer_.load_kin_file(filepath);

    // 2. Load the kernel into the HIP module
    load_kernel(kernel_binaries);
}

void AOTEngine::validate_elf_structure(const std::vector<uint8_t>& binary_data) const {
    // Deep Binary Validation
    if (binary_data.size() < 64) {
        throw std::runtime_error("Cannot load kernel: binary data too small for a valid ELF header.");
    }

    // Check for ELF magic number (\x7fELF)
    if (binary_data[0] != 0x7f || binary_data[1] != 'E' ||
        binary_data[2] != 'L'  || binary_data[3] != 'F') {
        throw std::runtime_error("Cannot load kernel: invalid or missing ELF header.");
    }

    // Check for 64-bit class
    if (binary_data[4] != 2) {
        throw std::runtime_error("Cannot load kernel: ELF binary is not 64-bit.");
    }

    // Check for EM_AMDGPU (0xE0) architecture
    // In an ELF header, e_machine is a 16-bit little-endian integer at offset 18
    uint16_t e_machine = binary_data[18] | (binary_data[19] << 8);
    if (e_machine != 0xE0) { // EM_AMDGPU
        throw std::runtime_error("Cannot load kernel: ELF binary is not for AMDGPU architecture.");
    }
}

void AOTEngine::load_kernel(const std::vector<uint8_t>& binary_data) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);

    validate_elf_structure(binary_data);

    if (module_ != nullptr) {
        CHECK_HIP(hipModuleUnload(module_));
        module_ = nullptr;
    }
    // hipModuleLoadData expects the binary image.
    CHECK_HIP(hipModuleLoadData(&module_, binary_data.data()));
}
