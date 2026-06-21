#pragma once

#include "AOTEngine.h"
#include "TRTEngine.h"
#include <string>
#include <memory>
#include <iostream>

class HardwareRouter {
public:
    HardwareRouter() {
        const char* forced = std::getenv("KINETIC_TARGET");
        if (forced && std::string(forced).find("sm") != std::string::npos) {
            backend_ = Backend::NVIDIA;
            std::cout << "[HardwareRouter] Detected NVIDIA GPU. Routing to TensorRT execution path." << std::endl;
        } else if (forced && std::string(forced).find("gfx") != std::string::npos) {
            backend_ = Backend::AMD;

#if !defined(MOCK_HIP)
            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, 0);
            std::cout << "[HardwareRouter] Detected AMD GPU (CUs: " << prop.multiProcessorCount << ", SRAM: " << prop.sharedMemPerBlock << " bytes). Routing to Native HIP/Triton path." << std::endl;
#else
            std::cout << "[HardwareRouter] Detected AMD GPU (Mock). Routing to Native HIP/Triton path." << std::endl;
#endif
        } else {
            backend_ = Backend::AMD; // Fallback for headless tests
        }
    }

    void load_model(const std::string& filepath) {
        if (backend_ == Backend::NVIDIA) {
            trt_engine_.load_model(filepath);
        } else {
            aot_engine_.load_model(filepath);
        }
    }

    void launch(void* input_ptr, void* output_ptr, int seq_len, size_t byte_size = 0) {
        if (backend_ == Backend::NVIDIA) {
            // Mapping buffer pointer natively into TRT execution
            trt_engine_.launch(input_ptr, output_ptr, seq_len);
        } else {
            // AOT Engine launch natively accepts pybind objects for lifecycle pinning.
            // When falling back to pointer mapping via Router, we construct an explicit execution path.
            aot_engine_.launch_ptr(input_ptr, output_ptr, seq_len, byte_size);
        }
    }

    void launch_py(pybind11::object py_input, uintptr_t stream_ptr) {
        aot_engine_.launch(py_input, stream_ptr);
    }

private:
    enum class Backend { AMD, NVIDIA, CPU };
    Backend backend_;
    AOTEngine aot_engine_;
    TRTEngine trt_engine_;
};
