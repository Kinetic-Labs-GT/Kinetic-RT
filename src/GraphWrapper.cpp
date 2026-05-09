#include "GraphWrapper.h"

#define CHECK_HIP(cmd) \
do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("HIP error"); \
    } \
} while(0)

GraphWrapper::GraphWrapper() : graph_(nullptr), graph_exec_(nullptr), is_instantiated_(false), current_batch_size_(-1), current_seq_len_(-1) {
    CHECK_HIP(hipEventCreate(&sync_event_));
}

GraphWrapper::~GraphWrapper() {
    try {
        invalidate();
        hipEventDestroy(sync_event_);
    } catch (...) {
        // Suppress exceptions in destructor
    }
}

void GraphWrapper::invalidate() {
    // Ensure no in-flight launches are using this graph before destroying
    CHECK_HIP(hipEventSynchronize(sync_event_));

    if (graph_exec_ != nullptr) {
        CHECK_HIP(hipGraphExecDestroy(graph_exec_));
        graph_exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        CHECK_HIP(hipGraphDestroy(graph_));
        graph_ = nullptr;
    }
    is_instantiated_ = false;
    current_batch_size_ = -1;
    current_seq_len_ = -1;
    pinned_buffers_ = pybind11::list(); // Release python references securely
}

bool GraphWrapper::is_valid(int batch_size, int seq_len) const {
    if (!is_instantiated_) {
        return false;
    }
    return (current_batch_size_ == batch_size && current_seq_len_ == seq_len);
}

void GraphWrapper::begin_capture(uintptr_t stream_ptr, int batch_size, int seq_len) {
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

    // If shapes change or graph is not instantiated, invalidate
    if (!is_valid(batch_size, seq_len)) {
        invalidate();
        current_batch_size_ = batch_size;
        current_seq_len_ = seq_len;
    } else {
        // If it's already valid and instantiated, we shouldn't be capturing again.
        // We could either ignore or throw an error. For simplicity, we assume the user checks is_valid().
        return;
    }

    CHECK_HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
}

void GraphWrapper::end_capture(uintptr_t stream_ptr) {
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

    CHECK_HIP(hipStreamEndCapture(stream, &graph_));

    // Instantiate the graph
    CHECK_HIP(hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    is_instantiated_ = true;
}

void GraphWrapper::set_pinned_buffers(pybind11::list buffers) {
    pinned_buffers_ = buffers;
}

void GraphWrapper::launch(uintptr_t stream_ptr) {
    if (!is_instantiated_) {
        throw std::runtime_error("Cannot launch graph: not instantiated.");
    }

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    CHECK_HIP(hipGraphLaunch(graph_exec_, stream));
    // Record event to know when this launch finishes, so invalidate() can wait
    CHECK_HIP(hipEventRecord(sync_event_, stream));
}
