#include "AOTEngine.h"
#include <iostream>
#include <cassert>

int main() {
    AOTEngine engine;

    int initial_calls = global_mock_hip_state.get_device_properties_calls;

    std::cout << "Initial hipGetDeviceProperties calls: " << initial_calls << std::endl;

    for (int i = 0; i < 100; ++i) {
        engine.compile_ahead_of_time("test_output.kin", 0);
    }

    int calls_after_compile = global_mock_hip_state.get_device_properties_calls;
    std::cout << "Calls after 100 compile_ahead_of_time: " << calls_after_compile << std::endl;

    for (int i = 0; i < 100; ++i) {
        // We need a valid .kin file to test load_model
        // compile_ahead_of_time already created one
        engine.load_model("test_output.kin");
    }

    int final_calls = global_mock_hip_state.get_device_properties_calls;
    std::cout << "Final hipGetDeviceProperties calls: " << final_calls << std::endl;

    return 0;
}
