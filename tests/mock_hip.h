#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

typedef void* hipStream_t;
typedef void* hipGraph_t;
typedef void* hipGraphExec_t;
typedef int hipError_t;

#define hipSuccess 0
#define hipErrorInvalidValue 1
#define hipStreamCaptureModeGlobal 0

struct MockHipState {
    int stream_capture_calls = 0;
    int stream_end_capture_calls = 0;
    int graph_instantiate_calls = 0;
    int graph_launch_calls = 0;
    int graph_exec_destroy_calls = 0;
    int graph_destroy_calls = 0;

    void reset() {
        stream_capture_calls = 0;
        stream_end_capture_calls = 0;
        graph_instantiate_calls = 0;
        graph_launch_calls = 0;
        graph_exec_destroy_calls = 0;
        graph_destroy_calls = 0;
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
