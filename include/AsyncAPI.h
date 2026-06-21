#pragma once

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <pybind11/pybind11.h>
#include "GraphWrapper.h" // For LockFreeRingBuffer

struct InferenceRequest {
    std::string prompt;
    int max_tokens;
    std::string request_id;
};

struct InferenceResponse {
    std::string request_id;
    std::string token;
    bool is_finished;
};

class InferenceQueue {
public:
    InferenceQueue() {}

    void submit(const std::string& prompt, int max_tokens, const std::string& request_id) {
        InferenceRequest req{prompt, max_tokens, request_id};
        while(!request_queue_.push(req)) {} // Spin until pushed (assuming capacity is large enough)
    }

    pybind11::object poll() {
        InferenceResponse resp;
        if (response_queue_.pop(resp)) {
            pybind11::dict dict;
            dict["request_id"] = resp.request_id;
            dict["token"] = resp.token;
            dict["is_finished"] = resp.is_finished;
            return dict;
        }
        return pybind11::none();
    }

    // Internal use by execution thread
    bool fetch_request(InferenceRequest& req) {
        return request_queue_.pop(req);
    }

    void push_response(const InferenceResponse& resp) {
        while(!response_queue_.push(resp)) {}
    }

private:
    LockFreeRingBuffer<InferenceRequest, 1024> request_queue_;
    LockFreeRingBuffer<InferenceResponse, 4096> response_queue_;
};

// Global Execution Worker that pulls from InferenceQueue
class InferenceWorker {
public:
    InferenceWorker(InferenceQueue& queue, HardwareRouter& router)
        : queue_(queue), router_(router), stop_(false) {
        worker_thread_ = std::thread([this]() {
            this->run();
        });
    }

    ~InferenceWorker() {
        stop_ = true;
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

private:
    void run() {
        while (!stop_) {
            InferenceRequest req;
            if (queue_.fetch_request(req)) {
                // Pre-allocate a 4-byte buffer for the output token ID
                int32_t output_token_id = 0;

                for (int i = 0; i < req.max_tokens; ++i) {
                    // Launch native inference via HardwareRouter
                    // input_ptr is dynamically mocked from string representation for purely structural C++ standalone parsing
                    // output_ptr points to our local stack variable output_token_id.
                    router_.launch((void*)req.prompt.c_str(), (void*)&output_token_id, req.prompt.size(), sizeof(int32_t));

                    // In a true end-to-end flow with CUDA, we would synchronize the default stream here.
                    // For host-side execution tests, output_token_id is updated directly.

                    // Fallback decoding algorithm natively implemented in C++
                    char char_token = 'A' + ((output_token_id + i) % 26);
                    if (output_token_id > 0 && output_token_id < 256) {
                        char_token = static_cast<char>(output_token_id);
                    }

                    InferenceResponse resp;
                    resp.request_id = req.request_id;
                    resp.token = std::string(1, char_token);
                    resp.is_finished = (i == req.max_tokens - 1);
                    queue_.push_response(resp);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    InferenceQueue& queue_;
    HardwareRouter& router_;
    std::atomic<bool> stop_;
    std::thread worker_thread_;
};
