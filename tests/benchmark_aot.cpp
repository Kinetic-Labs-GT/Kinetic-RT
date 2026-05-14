#include "AOTEngine.h"
#include <iostream>
#include <cassert>

int main() {
    AOTEngine engine;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::string device_id(prop.gcnArchName);

#ifdef __HIP_PLATFORM_NVIDIA__
    std::string target_arch = "CUDA_" + device_id;
#else
    std::string target_arch = "ROCm_" + device_id;
#endif

#if defined(MOCK_HIP)
    int initial_calls = global_mock_hip_state.get_device_properties_calls;
    std::cout << "Initial hipGetDeviceProperties calls: " << initial_calls << std::endl;
#endif

    for (int i = 0; i < 100; ++i) {
        engine.compile_ahead_of_time("test_output.kin", 0, target_arch);
    }

#if defined(MOCK_HIP)
    int calls_after_compile = global_mock_hip_state.get_device_properties_calls;
    std::cout << "Calls after 100 compile_ahead_of_time: " << calls_after_compile << std::endl;
#endif

    for (int i = 0; i < 100; ++i) {
        // We need a valid .kin file to test load_model
        // compile_ahead_of_time already created one
        engine.load_model("test_output.kin");
    }

#if defined(MOCK_HIP)
    int final_calls = global_mock_hip_state.get_device_properties_calls;
    std::cout << "Final hipGetDeviceProperties calls: " << final_calls << std::endl;
#endif

    return 0;
}
