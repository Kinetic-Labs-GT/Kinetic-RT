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

Serializer::Serializer() {
    const char* forced_arch = std::getenv("KINETIC_FORCE_ARCH");
    if (forced_arch != nullptr && std::strlen(forced_arch) > 0) {
        device_id_ = std::string(forced_arch);
    } else {
        hipDeviceProp_t prop;
        CHECK_HIP(hipGetDeviceProperties(&prop, 0)); // Assuming device 0
        device_id_ = std::string(prop.gcnArchName);
    }
}

void Serializer::save_kin_file(const std::string& filepath, const std::string& device_id, const std::string& target_architecture, uint64_t weights_hash, const std::vector<uint8_t>& op_graph_data, const std::vector<uint8_t>& kernel_binaries) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    KinHeader header;
    std::memset(&header, 0, sizeof(KinHeader));
    header.magic_number = htole32(0x4B494E00); // "KIN\0"
    header.version = htole32(1);
    std::strncpy(header.device_id, device_id.c_str(), sizeof(header.device_id) - 1);
    header.device_id[sizeof(header.device_id) - 1] = '\0';
    std::strncpy(header.target_architecture, target_architecture.c_str(), sizeof(header.target_architecture) - 1);
    header.target_architecture[sizeof(header.target_architecture) - 1] = '\0';
    header.weights_hash = htole64(weights_hash);

    header.op_graph_data_offset = htole64(sizeof(KinHeader));
    header.op_graph_data_size = htole64(op_graph_data.size());

    header.kernel_binaries_offset = htole64(sizeof(KinHeader) + op_graph_data.size());
    header.kernel_binaries_size = htole64(kernel_binaries.size());
    header.tensor_parallel_degree = htole32(1);

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

    char safe_target[256];
    std::strncpy(safe_target, header.target_architecture, 255);
    safe_target[255] = '\0';
    loaded_target_architecture_ = std::string(safe_target);

    header.device_id[255] = '\0';
    // Verify Hardware Mismatch
    if (device_id_ != header.device_id) {
        throw HardwareMismatchError("Hardware mismatch: expected " + std::string(header.device_id) + " but got " + device_id_);
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

void AOTEngine::compile_ahead_of_time(const std::string& output_filepath, uintptr_t stream_ptr, const std::string& target_architecture) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    // 1. Profile and find best kernel
    std::string best_variant = autotuner_.profile_gemm(stream_ptr);
    std::cout << "Selected best GEMM variant: " << best_variant << std::endl;

    // 2. Fetch current device properties (cached)
    const std::string& device_id = serializer_.get_device_id();

    // 3. Serialize to .kin
    std::vector<uint8_t> dummy_op_graph_data = {0x01, 0x02, 0x03}; // Mock data

    // Mock 64-byte hsaco binary that passes deep validation
    std::vector<uint8_t> dummy_hsaco(64, 0);
    dummy_hsaco[0] = 0x7F; dummy_hsaco[1] = 'E'; dummy_hsaco[2] = 'L'; dummy_hsaco[3] = 'F';
    dummy_hsaco[4] = 2; // 64-bit class
    dummy_hsaco[18] = 0xE0; dummy_hsaco[19] = 0x00; // EM_AMDGPU

    uint64_t dummy_weights_hash = 123456789;

#ifdef __HIP_PLATFORM_NVIDIA__
    dummy_hsaco[18] = 0xBE; dummy_hsaco[19] = 0x00; // EM_CUDA
#endif

    serializer_.save_kin_file(output_filepath, device_id, target_architecture, dummy_weights_hash, dummy_op_graph_data, dummy_hsaco);
}

void AOTEngine::load_model(const std::string& filepath) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    // 1. Load the .kin file and verify hardware
    std::vector<uint8_t> kernel_binaries = serializer_.load_kin_file(filepath);

    // 2. Load the kernel into the HIP module
    load_kernel(kernel_binaries, serializer_.get_loaded_target_architecture());
}

void AOTEngine::validate_elf_structure(const std::vector<uint8_t>& binary_data, const std::string& target_architecture) const {
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

    uint16_t e_machine = binary_data[18] | (binary_data[19] << 8);

    // Cross-platform interlock check
#ifdef __HIP_PLATFORM_NVIDIA__
    std::string expected_prefix = "CUDA";
    uint16_t expected_em = 0xBE; // EM_CUDA (190)
#else
    std::string expected_prefix = "ROCm";
    uint16_t expected_em = 0xE0; // EM_AMDGPU (224)
#endif

    if (target_architecture.substr(0, 4) != expected_prefix) {
        throw HardwareMismatchError("Hardware mismatch: expected target architecture starting with " + expected_prefix + " but found " + target_architecture);
    }

    if (e_machine != expected_em) {
        throw std::runtime_error("Cannot load kernel: ELF binary architecture mismatch for current platform.");
    }
}

void AOTEngine::load_kernel(const std::vector<uint8_t>& binary_data, const std::string& target_architecture) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);

    validate_elf_structure(binary_data, target_architecture);

    if (module_ != nullptr) {
        CHECK_HIP(hipModuleUnload(module_));
        module_ = nullptr;
    }
    // hipModuleLoadData expects the binary image.
    CHECK_HIP(hipModuleLoadData(&module_, binary_data.data()));
}
