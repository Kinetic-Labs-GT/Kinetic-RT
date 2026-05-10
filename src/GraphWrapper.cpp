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
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);

    // Ensure no in-flight launches are using this graph before destroying
    // Robust Multi-Stream Synchronization: Sync every in-flight event
    for (auto& state : in_flight_states_) {
        CHECK_HIP(hipEventSynchronize(state.event));
    }
    // Also sync the generic event just in case
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

    // Everything is synchronized, so we can clean it all up
    for (auto& state : in_flight_states_) {
        CHECK_HIP(hipEventDestroy(state.event));
    }
    in_flight_states_.clear();
}

void GraphWrapper::cleanup_in_flight_states() {
    while (!in_flight_states_.empty()) {
        hipError_t status = hipEventQuery(in_flight_states_.front().event);
        if (status == hipSuccess) {
            CHECK_HIP(hipEventDestroy(in_flight_states_.front().event));
            in_flight_states_.pop_front();
        } else if (status == hipErrorNotReady) {
            break; // Still executing
        } else {
            CHECK_HIP(status); // Throw on actual errors
        }
    }
}

bool GraphWrapper::is_valid(int batch_size, int seq_len) const {
    std::lock_guard<std::recursive_mutex> lock(const_cast<GraphWrapper*>(this)->engine_mutex_);
    if (!is_instantiated_) {
        return false;
    }
    return (current_batch_size_ == batch_size && current_seq_len_ == seq_len);
}

void GraphWrapper::begin_capture(uintptr_t stream_ptr, int batch_size, int seq_len) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
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
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

    CHECK_HIP(hipStreamEndCapture(stream, &graph_));

    // Instantiate the graph
    CHECK_HIP(hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    is_instantiated_ = true;
}

void GraphWrapper::launch(uintptr_t stream_ptr, pybind11::object stream_obj, pybind11::list buffers) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    cleanup_in_flight_states();

    if (!is_instantiated_) {
        throw std::runtime_error("Cannot launch graph: not instantiated.");
    }

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    CHECK_HIP(hipGraphLaunch(graph_exec_, stream));

    // Record a new event for this specific launch
    hipEvent_t event;
    hipError_t err_create = hipEventCreate(&event);
    hipError_t err_record = hipSuccess;
    if (err_create == hipSuccess) {
        err_record = hipEventRecord(event, stream);
    }

    if (err_create != hipSuccess || err_record != hipSuccess) {
        // Fail-Safe: Synchronize immediately if event creation fails
        // This ensures Python objects stay alive while GPU executes
        CHECK_HIP(hipStreamSynchronize(stream));
    } else {
        // Store references to prevent Python from GC'ing the stream or buffers
        InFlightState state;
        state.event = event;
        state.refs.reserve(buffers.size() + 1); // Micro-optimization
        state.refs.emplace_back(stream_obj);
        for (auto item : buffers) {
            state.refs.emplace_back(pybind11::reinterpret_borrow<pybind11::object>(item));
        }

        in_flight_states_.emplace_back(std::move(state));
    }

    // Also record the general sync event for invalidate() wait
    hipError_t sync_err = hipEventRecord(sync_event_, stream);
    if (sync_err != hipSuccess) {
        CHECK_HIP(hipStreamSynchronize(stream));
    }
}
