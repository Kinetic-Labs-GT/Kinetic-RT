#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

typedef void* hipStream_t;
typedef void* hipGraph_t;
typedef void* hipGraphExec_t;
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef void* hipEvent_t;
typedef int hipError_t;

#define hipSuccess 0
#define hipErrorInvalidValue 1
#define hipStreamCaptureModeGlobal 0

struct hipDeviceProp_t {
    char gcnArchName[256];
};

struct MockHipState {
    int stream_capture_calls = 0;
    int stream_end_capture_calls = 0;
    int graph_instantiate_calls = 0;
    int graph_launch_calls = 0;
    int graph_exec_destroy_calls = 0;
    int graph_destroy_calls = 0;
    int module_load_data_calls = 0;
    int module_get_function_calls = 0;
    int event_create_calls = 0;
    int event_record_calls = 0;
    int event_synchronize_calls = 0;
    int event_elapsed_time_calls = 0;
    int event_destroy_calls = 0;
    int get_device_properties_calls = 0;

    // For autotuner mock
    float mock_elapsed_time = 1.0f;
    std::string mock_gcn_arch_name = "gfx1100";

    void reset() {
        stream_capture_calls = 0;
        stream_end_capture_calls = 0;
        graph_instantiate_calls = 0;
        graph_launch_calls = 0;
        graph_exec_destroy_calls = 0;
        graph_destroy_calls = 0;
        module_load_data_calls = 0;
        module_get_function_calls = 0;
        event_create_calls = 0;
        event_record_calls = 0;
        event_synchronize_calls = 0;
        event_elapsed_time_calls = 0;
        event_destroy_calls = 0;
        get_device_properties_calls = 0;
        mock_elapsed_time = 1.0f;
        mock_gcn_arch_name = "gfx1100";
    }
};

extern MockHipState global_mock_hip_state;

inline const char* hipGetErrorString(hipError_t error) {
    if (error == hipSuccess) return "hipSuccess";
    return "hipErrorUnknown";
}

inline hipError_t hipStreamBeginCapture(hipStream_t stream, int mode) {
    global_mock_hip_state.stream_capture_calls++;
    return hipSuccess;
}

inline hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
    global_mock_hip_state.stream_end_capture_calls++;
    // allocate some dummy pointer so we can distinguish it later
    *pGraph = (hipGraph_t)0x12345678;
    return hipSuccess;
}

inline hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph, void* pNode, void* pLogBuffer, size_t bufferSize) {
    global_mock_hip_state.graph_instantiate_calls++;
    *pGraphExec = (hipGraphExec_t)0x87654321;
    return hipSuccess;
}

inline hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
    global_mock_hip_state.graph_launch_calls++;
    return hipSuccess;
}

inline hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
    global_mock_hip_state.graph_exec_destroy_calls++;
    return hipSuccess;
}

inline hipError_t hipGraphDestroy(hipGraph_t graph) {
    global_mock_hip_state.graph_destroy_calls++;
    return hipSuccess;
}

inline hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    global_mock_hip_state.module_load_data_calls++;
    *module = (hipModule_t)0x11111111;
    return hipSuccess;
}

inline hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
    global_mock_hip_state.module_get_function_calls++;
    *function = (hipFunction_t)0x22222222;
    return hipSuccess;
}

inline hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra) {
    return hipSuccess;
}

inline hipError_t hipEventCreate(hipEvent_t* event) {
    global_mock_hip_state.event_create_calls++;
    *event = (hipEvent_t)0x33333333;
    return hipSuccess;
}

inline hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    global_mock_hip_state.event_record_calls++;
    return hipSuccess;
}

inline hipError_t hipEventSynchronize(hipEvent_t event) {
    global_mock_hip_state.event_synchronize_calls++;
    return hipSuccess;
}

inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    global_mock_hip_state.event_elapsed_time_calls++;
    *ms = global_mock_hip_state.mock_elapsed_time;
    // slightly randomize or change elapsed time to simulate different variant timings
    global_mock_hip_state.mock_elapsed_time -= 0.1f;
    return hipSuccess;
}

inline hipError_t hipEventDestroy(hipEvent_t event) {
    global_mock_hip_state.event_destroy_calls++;
    return hipSuccess;
}

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
    global_mock_hip_state.get_device_properties_calls++;
    snprintf(prop->gcnArchName, 256, "%s", global_mock_hip_state.mock_gcn_arch_name.c_str());
    return hipSuccess;
}
