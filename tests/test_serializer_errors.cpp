#include "../include/AOTEngine.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <cstring>
#include <endian.h>
#include <chrono>

void create_test_file(const std::string& filepath, const void* data, size_t size) {
    std::ofstream out(filepath, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), size);
    out.close();
}

void test_file_not_found() {
    Serializer s;
    try {
        s.load_kin_file("non_existent_file.kin");
        assert(false && "Should have thrown runtime_error for file not found");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("Failed to open file for reading") != std::string::npos);
        std::cout << "test_file_not_found passed" << std::endl;
    }
}

void test_file_too_small() {
    Serializer s;
    std::string filepath = "too_small.kin";
    char data[10];
    create_test_file(filepath, data, sizeof(data));
    try {
        s.load_kin_file(filepath);
        assert(false && "Should have thrown runtime_error for file too small");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("file too small for header") != std::string::npos);
        std::cout << "test_file_too_small passed" << std::endl;
    }
    remove(filepath.c_str());
}

void test_bad_magic() {
    Serializer s;
    std::string filepath = "bad_magic.kin";
    KinHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic_number = htole32(0x12345678);
    create_test_file(filepath, &header, sizeof(header));
    try {
        s.load_kin_file(filepath);
        assert(false && "Should have thrown runtime_error for bad magic");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("bad magic number") != std::string::npos);
        std::cout << "test_bad_magic passed" << std::endl;
    }
    remove(filepath.c_str());
}

void test_bad_magic_number_in_load() {
    Serializer s;
    std::string filepath = "missing_bad_magic.kin";
    // write an invalid magic number to a file
    uint32_t invalid_magic = htole32(0xDEADBEEF);
    KinHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic_number = invalid_magic;
    header.version = htole32(1);
    header.op_graph_data_offset = htole64(sizeof(KinHeader));
    header.op_graph_data_size = htole64(0);
    header.kernel_binaries_offset = htole64(sizeof(KinHeader));
    header.kernel_binaries_size = htole64(0);

    std::ofstream out(filepath, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.close();

    try {
        // calls load_kin_file, checking for the runtime error
        s.load_kin_file(filepath);
        assert(false && "Should have thrown std::runtime_error for bad magic number");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("bad magic number") != std::string::npos);
        std::cout << "test_bad_magic_number_in_load passed" << std::endl;
    }
    remove(filepath.c_str());
}

void test_offset_overflow() {
    Serializer s;
    std::string filepath = "offset_overflow.kin";
    KinHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic_number = htole32(0x4B494E00);
    header.op_graph_data_offset = htole64(0xFFFFFFFFFFFFFFFF);
    header.op_graph_data_size = htole64(10);
    create_test_file(filepath, &header, sizeof(header));
    try {
        s.load_kin_file(filepath);
        assert(false && "Should have thrown runtime_error for offset overflow");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("offset overflow") != std::string::npos);
        std::cout << "test_offset_overflow passed" << std::endl;
    }
    remove(filepath.c_str());
}

void test_kernel_offset_overflow() {
    Serializer s;
    std::string filepath = "kernel_offset_overflow.kin";
    KinHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic_number = htole32(0x4B494E00);
    header.op_graph_data_offset = htole64(100);
    header.op_graph_data_size = htole64(10);
    header.kernel_binaries_offset = htole64(0xFFFFFFFFFFFFFFFF);
    header.kernel_binaries_size = htole64(10);
    create_test_file(filepath, &header, sizeof(header));
    try {
        s.load_kin_file(filepath);
        assert(false && "Should have thrown runtime_error for offset overflow");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("offset overflow") != std::string::npos);
        std::cout << "test_kernel_offset_overflow passed" << std::endl;
    }
    remove(filepath.c_str());
}

void test_exceed_bounds() {
    Serializer s;
    std::string filepath = "exceed_bounds.kin";
    KinHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic_number = htole32(0x4B494E00);
    header.op_graph_data_offset = htole64(sizeof(KinHeader));
    header.op_graph_data_size = htole64(100); // Exceeds file size which is just sizeof(KinHeader)
    create_test_file(filepath, &header, sizeof(header));
    try {
        s.load_kin_file(filepath);
        assert(false && "Should have thrown runtime_error for exceeding bounds");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("sizes exceed file bounds") != std::string::npos);
        std::cout << "test_exceed_bounds passed" << std::endl;
    }
    remove(filepath.c_str());
}

void test_payload_bounds_check() {
    Serializer s;
    std::string filepath = "payload_bounds.kin";
    KinHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic_number = htole32(0x4B494E00);
    header.kernel_binaries_offset = htole64(sizeof(KinHeader));
    header.kernel_binaries_size = htole64(10000);

    std::vector<char> file_data(sizeof(KinHeader) + 50, 0);
    std::memcpy(file_data.data(), &header, sizeof(KinHeader));

    create_test_file(filepath, file_data.data(), file_data.size());
    try {
        s.load_kin_file(filepath);
        assert(false && "Should have thrown runtime_error for exceeding bounds");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("sizes exceed file bounds") != std::string::npos);
        std::cout << "test_payload_bounds_check passed" << std::endl;
    }
    remove(filepath.c_str());
}

void test_hardware_mismatch() {
    Serializer s;
    std::string filepath = "hw_mismatch.kin";
    KinHeader header;
    std::memset(&header, 0, sizeof(header));
    header.magic_number = htole32(0x4B494E00);
    std::strncpy(header.device_id, "wrong_hw", sizeof(header.device_id) - 1);
    header.kernel_binaries_offset = htole64(sizeof(KinHeader));
    header.kernel_binaries_size = htole64(0);

    create_test_file(filepath, &header, sizeof(header));

    global_mock_hip_state.mock_gcn_arch_name = "correct_hw";

    try {
        s.load_kin_file(filepath);
        assert(false && "Should have thrown HardwareMismatchError");
    } catch (const HardwareMismatchError& e) {
        std::string msg = e.what();
        assert(msg.find("Hardware mismatch") != std::string::npos);
        std::cout << "test_hardware_mismatch passed" << std::endl;
    }
    remove(filepath.c_str());
}

#include <filesystem>

void test_write_failure() {
    Serializer s;
    std::vector<uint8_t> empty_vec;
    // Generate a pseudo-random path based on timestamp
    std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::string bad_path = std::filesystem::temp_directory_path().string() + "/invalid_dir_" + timestamp + "/test.kin";

    try {
        s.save_kin_file(bad_path, "gfx1100", "ROCm_gfx1100", 12345, empty_vec, empty_vec);
        assert(false && "Should have thrown runtime_error for write failure");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        assert(msg.find("Failed to open file for writing") != std::string::npos);
        std::cout << "test_write_failure passed" << std::endl;
    }
}

int main() {
    test_file_not_found();
    test_file_too_small();
    test_bad_magic();
    test_bad_magic_number_in_load();
    test_offset_overflow();
    test_kernel_offset_overflow();
    test_exceed_bounds();
    test_payload_bounds_check();
    test_hardware_mismatch();
    test_write_failure();

    std::cout << "All Serializer error tests passed!" << std::endl;
    return 0;
}
