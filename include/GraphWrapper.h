#include <cstdint>
#pragma once

// We include a mock HIP header if real HIP is not available
// but let's assume <hip/hip_runtime.h> is what we'd normally include.
// For the sake of testing in environments without ROCm, we can optionally define a mock.
#if defined(MOCK_HIP)
#include "../tests/mock_hip.h"
#else
#include <hip/hip_runtime.h>
#endif

#include <iostream>
#include <stdexcept>
#include <deque>
#include <vector>
#include <mutex>
#include <pybind11/pybind11.h>

struct InFlightState {
    hipEvent_t event;
    std::vector<pybind11::object> refs;
};

class GraphWrapper {
public:
    GraphWrapper();
    ~GraphWrapper();

    // Begin capturing the operations on the specified stream.
    // Also track the batch size and sequence length for dynamic shape handling.
    void begin_capture(uintptr_t stream_ptr, int batch_size, int seq_len);

    // End capturing and instantiate the graph
    void end_capture(uintptr_t stream_ptr);

    // Launch the instantiated graph
    void launch(uintptr_t stream_ptr, pybind11::object stream_obj, pybind11::list buffers);

    // Check if the graph is valid for the current shapes
    bool is_valid(int batch_size, int seq_len) const;

    // Manually invalidate the graph
    void invalidate();

private:
    void cleanup_in_flight_states();

    hipGraph_t graph_;
    hipGraphExec_t graph_exec_;
    bool is_instantiated_;
    hipEvent_t sync_event_;

    int current_batch_size_;
    int current_seq_len_;

    // Registry of states currently executing on the GPU
    std::deque<InFlightState> in_flight_states_;

    // Mutex for thread safety
    std::recursive_mutex engine_mutex_;
};
