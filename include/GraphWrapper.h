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
#include <atomic>
#include <thread>
#include <functional>
#include <condition_variable>
#include <queue>
#include <pybind11/pybind11.h>
#include "BlockManager.h"

struct InFlightState {
    hipEvent_t event;
    std::vector<pybind11::object> refs;
};

template <typename T, size_t Size>
class LockFreeRingBuffer {
private:
    struct Slot {
        T data;
        std::atomic<bool> readable{false};
    };
    Slot buffer_[Size];
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};

public:
    bool push(const T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        for (;;) {
            size_t head = head_.load(std::memory_order_acquire);
            size_t next_tail = (tail + 1) % Size;
            if (next_tail == head) {
                return false; // full
            }
            // Claim the slot
            if (tail_.compare_exchange_weak(tail, next_tail, std::memory_order_relaxed, std::memory_order_relaxed)) {
                // Safely write data to the claimed slot
                buffer_[tail].data = item;
                // Memory fence via release to ensure the data write is visible
                buffer_[tail].readable.store(true, std::memory_order_release);
                return true;
            }
        }
    }

    bool pop(T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        for (;;) {
            size_t tail = tail_.load(std::memory_order_acquire);
            if (head == tail) {
                return false; // empty
            }
            // If the slot is claimed but not yet fully written, we spin or fail.
            // In a ring buffer, if head != tail, the slot *will* become readable shortly.
            if (!buffer_[head].readable.load(std::memory_order_acquire)) {
                // Yield thread to avoid heavy spinning, let writer finish
                std::this_thread::yield();
                continue;
            }
            // Read data before advancing head to prevent fast producers from overwriting
            T extracted_item = buffer_[head].data;

            // Now attempt to advance the head index
            if (head_.compare_exchange_weak(head, (head + 1) % Size, std::memory_order_release, std::memory_order_relaxed)) {
                item = extracted_item;
                buffer_[head].readable.store(false, std::memory_order_release);
                return true;
            }
        }
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }
};

class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back(
                [this] {
                    for (;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock,
                                [this]{ return this->stop || !this->tasks.empty(); });
                            if (this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    }
                }
            );
    }

    template<class F, class... Args>
    void enqueue(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
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
    void launch(std::vector<pybind11::object> stream_objs, std::vector<pybind11::object> buffers);

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

    // Lock-free event pool for max throughput (pre-allocated)
    static constexpr size_t POOL_SIZE = 2048;
    LockFreeRingBuffer<hipEvent_t, POOL_SIZE> event_pool_;

    // Persistent thread pool for execution
    std::unique_ptr<ThreadPool> execution_pool_;

    // PagedAttention Block Manager
    std::unique_ptr<BlockManager> block_manager_;
    std::vector<int> current_block_table_;

    // Mutex for thread safety
    std::recursive_mutex engine_mutex_;
};
