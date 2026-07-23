#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <functional>
#include <cstdint>
#include <stdexcept>
#include <memory>

enum class RequestState : uint8_t {
    RECEIVED = 0,
    VALIDATED,
    TOKENIZED,
    ENQUEUED,
    PREFILL,
    DECODE,
    STREAMING,
    COMPLETE,
    CANCELLED,
    TIMED_OUT,
    FAILED
};

const char* to_string(RequestState state);

enum class CompletionReason : uint8_t {
    STREAMING = 0,
    EOS,
    STOP_TOKEN,
    LENGTH,
    CANCELLED,
    TIMEOUT,
    ERROR
};

const char* to_string(CompletionReason reason);

enum class Owner : uint8_t {
    Ingress = 0,
    Scheduler,
    Executor,
    Backend
};

const char* to_string(Owner owner);

// Typed pointer forward declarations replacing raw void*.
// Note: Tokenizer, LogicalDecodeState, and KVBlockTable are intentionally forward-declared
// opaque typed handles in Chapter 1. Full definitions are provided in subsequent chapter PRs.
class Tokenizer;
struct LogicalDecodeState;
struct KVBlockTable;

class RequestContextError : public std::runtime_error {
public:
    explicit RequestContextError(const std::string& msg) : std::runtime_error(msg) {}
};

class InvalidStateTransitionError : public RequestContextError {
public:
    explicit InvalidStateTransitionError(const std::string& msg) : RequestContextError(msg) {}
};

class RequestOwnershipError : public RequestContextError {
public:
    explicit RequestOwnershipError(const std::string& msg) : RequestContextError(msg) {}
};

class RequestContext {
public:
    using Clock = std::chrono::steady_clock;

    struct Config {
        std::string request_id;
        std::string prompt_text;
        std::vector<int32_t> prompt_tokens;
        int max_new_tokens = 0;
        float temperature = 1.0f;
        float top_p = 1.0f;
        int top_k = 0;
        std::vector<int32_t> stop_token_ids;
        std::string backend_hint;
        Clock::duration timeout{Clock::duration::zero()}; // zero == no deadline
        Owner initial_owner{Owner::Ingress};
    };

    explicit RequestContext(Config config);
    RequestContext(const RequestContext&) = delete;
    RequestContext& operator=(const RequestContext&) = delete;
    RequestContext(RequestContext&&) = delete;
    RequestContext& operator=(RequestContext&&) = delete;

    // Destructor performs defensive resource cleanup if uncleaned,
    // but never mutates observable lifecycle state (state_ remains unchanged).
    ~RequestContext();

    // Immutable configuration & metadata accessors
    const std::string& request_id() const noexcept { return request_id_; }
    const std::string& prompt_text() const noexcept { return prompt_text_; }
    const std::vector<int32_t>& prompt_tokens() const noexcept { return prompt_tokens_; }
    int max_new_tokens() const noexcept { return max_new_tokens_; }
    float temperature() const noexcept { return temperature_; }
    float top_p() const noexcept { return top_p_; }
    int top_k() const noexcept { return top_k_; }
    const std::vector<int32_t>& stop_token_ids() const noexcept { return stop_token_ids_; }
    const std::string& backend_hint() const noexcept { return backend_hint_; }

    // Request validity and executability
    bool is_valid() const noexcept;
    void ensure_executable() const;

    // State machine
    RequestState state() const;
    bool can_transition_to(RequestState to) const;
    void transition_to(RequestState to);
    static bool is_terminal(RequestState state) noexcept;
    bool is_terminal() const noexcept;

    // Two-phase atomic cancellation
    void request_cancellation() noexcept;
    bool is_cancellation_requested() const noexcept;
    void acknowledge_cancellation();

    // Timeout metadata & expiration tracking
    bool has_deadline() const noexcept;
    int64_t deadline_ns() const noexcept;
    bool is_expired(Clock::time_point now = Clock::now()) const noexcept;
    void acknowledge_timeout(Clock::time_point now = Clock::now());

    // Strongly-typed ownership transfer
    Owner current_owner() const;
    void transfer_ownership(Owner new_owner, Owner expected_current_owner);

    // Token tracking
    void append_generated_token(int32_t token_id);
    std::vector<int32_t> generated_tokens() const;

    // Device memory handles
    void set_input_device_ptr(uintptr_t ptr);
    void set_output_device_ptr(uintptr_t ptr);
    uintptr_t input_device_ptr() const noexcept;
    uintptr_t output_device_ptr() const noexcept;

    // KV metadata and block table handle
    void add_kv_block_id(int block_id);
    void set_kv_block_ids(std::vector<int> block_ids);
    std::vector<int> kv_block_ids() const;
    void set_kv_block_table_handle(KVBlockTable* handle);
    KVBlockTable* kv_block_table_handle() const noexcept;

    // Strongly-typed handles & execution state
    void set_tokenizer_ref(Tokenizer* tokenizer_ref);
    Tokenizer* tokenizer_ref() const noexcept;
    void set_logical_decode_state(LogicalDecodeState* state);
    LogicalDecodeState* logical_decode_state() const noexcept;
    void set_session_sequence(uint64_t sequence) noexcept;
    uint64_t session_sequence() const noexcept;

    // Terminal reason & error code slot
    void set_finish_reason(CompletionReason reason);
    CompletionReason finish_reason() const;
    void set_error_code(int code) noexcept;
    int error_code() const noexcept;

    // Teardown & explicit cleanup framework:
    // - Primary teardown path invoked after reaching a terminal state.
    // - Idempotent: second call returns false without re-executing callbacks.
    // - Executes registered cleanup callbacks outside the critical section in LIFO order.
    // - Terminal-gated to prevent racing with in-flight backend steps.
    void register_cleanup_callback(std::function<void()> callback);
    bool is_cleaned_up() const noexcept;
    bool cleanup();

private:
    static Config validate_config(Config config);
    RequestContext(Config config, int /* validated_tag */);
    void transition_to_locked(RequestState to, std::unique_lock<std::mutex>& lock);
    std::vector<std::function<void()>> extract_cleanup_callbacks_locked();

    // Immutable request-scoped fields
    const std::string request_id_;
    const std::string prompt_text_;
    const std::vector<int32_t> prompt_tokens_;
    const int max_new_tokens_;
    const float temperature_;
    const float top_p_;
    const int top_k_;
    const std::vector<int32_t> stop_token_ids_;
    const std::string backend_hint_;
    const bool has_deadline_;
    const Clock::time_point deadline_;

    // Lock-free atomic cancellation flag
    std::atomic<bool> cancel_flag_{false};

    // Mutex-protected dynamic state
    mutable std::mutex mutex_;
    RequestState state_{RequestState::RECEIVED};
    Owner owner_{Owner::Ingress};
    std::vector<int32_t> generated_tokens_;
    uintptr_t input_device_ptr_{0};
    uintptr_t output_device_ptr_{0};
    std::vector<int> kv_block_ids_;
    KVBlockTable* kv_block_table_handle_{nullptr};
    Tokenizer* tokenizer_ref_{nullptr};
    LogicalDecodeState* logical_decode_state_{nullptr};
    uint64_t session_sequence_{0};
    CompletionReason finish_reason_{CompletionReason::STREAMING};
    int error_code_{0};
    std::vector<std::function<void()>> cleanup_callbacks_;
    bool cleaned_up_{false};
};
