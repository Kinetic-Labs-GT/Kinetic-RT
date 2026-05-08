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
    void launch(uintptr_t stream_ptr);

    // Check if the graph is valid for the current shapes
    bool is_valid(int batch_size, int seq_len) const;

    // Manually invalidate the graph
    void invalidate();

private:
    hipGraph_t graph_;
    hipGraphExec_t graph_exec_;
    bool is_instantiated_;

    int current_batch_size_;
    int current_seq_len_;
};
