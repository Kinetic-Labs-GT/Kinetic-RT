#include "RequestContext.h"
#include <utility>
#include <algorithm>

const char* to_string(RequestState state) {
    switch (state) {
        case RequestState::RECEIVED:  return "RECEIVED";
        case RequestState::VALIDATED: return "VALIDATED";
        case RequestState::TOKENIZED: return "TOKENIZED";
        case RequestState::ENQUEUED:  return "ENQUEUED";
        case RequestState::PREFILL:   return "PREFILL";
        case RequestState::DECODE:    return "DECODE";
        case RequestState::STREAMING: return "STREAMING";
        case RequestState::COMPLETE:  return "COMPLETE";
        case RequestState::CANCELLED: return "CANCELLED";
        case RequestState::TIMED_OUT: return "TIMED_OUT";
        case RequestState::FAILED:    return "FAILED";
    }
    return "UNKNOWN";
}

const char* to_string(CompletionReason reason) {
    switch (reason) {
        case CompletionReason::STREAMING:  return "STREAMING";
        case CompletionReason::EOS:        return "EOS";
        case CompletionReason::STOP_TOKEN: return "STOP_TOKEN";
        case CompletionReason::LENGTH:     return "LENGTH";
        case CompletionReason::CANCELLED:  return "CANCELLED";
        case CompletionReason::TIMEOUT:    return "TIMEOUT";
        case CompletionReason::ERROR:      return "ERROR";
    }
    return "UNKNOWN";
}

const char* to_string(Owner owner) {
    switch (owner) {
        case Owner::Ingress:   return "Ingress";
        case Owner::Scheduler: return "Scheduler";
        case Owner::Executor:  return "Executor";
        case Owner::Backend:   return "Backend";
    }
    return "UNKNOWN";
}

static bool is_legal_transition(RequestState from, RequestState to) {
    if (RequestContext::is_terminal(from)) {
        return false;
    }
    if (to == RequestState::CANCELLED || to == RequestState::TIMED_OUT || to == RequestState::FAILED) {
        return true;
    }
    switch (from) {
        case RequestState::RECEIVED:  return to == RequestState::VALIDATED;
        case RequestState::VALIDATED: return to == RequestState::TOKENIZED;
        case RequestState::TOKENIZED: return to == RequestState::ENQUEUED;
        case RequestState::ENQUEUED:  return to == RequestState::PREFILL;
        case RequestState::PREFILL:   return to == RequestState::DECODE;
        case RequestState::DECODE:    return to == RequestState::STREAMING;
        case RequestState::STREAMING: return to == RequestState::DECODE || to == RequestState::COMPLETE;
        default: return false;
    }
}

RequestContext::Config RequestContext::validate_config(Config config) {
    if (config.request_id.empty()) {
        throw RequestContextError("RequestContext request_id cannot be empty");
    }
    if (config.max_new_tokens < 0) {
        throw RequestContextError("RequestContext max_new_tokens cannot be negative");
    }
    if (config.temperature < 0.0f) {
        throw RequestContextError("RequestContext temperature cannot be negative");
    }
    if (config.top_p < 0.0f || config.top_p > 1.0f) {
        throw RequestContextError("RequestContext top_p must be in range [0.0, 1.0]");
    }
    if (config.top_k < 0) {
        throw RequestContextError("RequestContext top_k cannot be negative");
    }
    if (config.timeout < Clock::duration::zero()) {
        throw RequestContextError("RequestContext timeout cannot be negative");
    }
    return config;
}

RequestContext::RequestContext(Config config)
    : RequestContext(validate_config(std::move(config)), 0) {}

RequestContext::RequestContext(Config config, int /* validated_tag */)
    : request_id_(std::move(config.request_id)),
      prompt_text_(std::move(config.prompt_text)),
      prompt_tokens_(std::move(config.prompt_tokens)),
      max_new_tokens_(config.max_new_tokens),
      temperature_(config.temperature),
      top_p_(config.top_p),
      top_k_(config.top_k),
      stop_token_ids_(std::move(config.stop_token_ids)),
      backend_hint_(std::move(config.backend_hint)),
      has_deadline_(config.timeout > Clock::duration::zero()),
      deadline_(has_deadline_ ? Clock::now() + config.timeout : Clock::time_point{}),
      state_(RequestState::RECEIVED),
      owner_(config.initial_owner) {}

RequestContext::~RequestContext() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cleaned_up_) {
        execute_cleanup_callbacks_locked();
        cleaned_up_ = true;
    }
}

bool RequestContext::is_valid() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return !request_id_.empty() && !is_terminal(state_) && !cleaned_up_;
}

void RequestContext::ensure_executable() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_terminal(state_)) {
        throw RequestContextError("RequestContext is in terminal state: " + std::string(to_string(state_)));
    }
    if (cleaned_up_) {
        throw RequestContextError("RequestContext has already been cleaned up");
    }
}

RequestState RequestContext::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

bool RequestContext::can_transition_to(RequestState to) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_legal_transition(state_, to);
}

void RequestContext::transition_to(RequestState to) {
    std::unique_lock<std::mutex> lock(mutex_);
    transition_to_locked(to, lock);
}

void RequestContext::transition_to_locked(RequestState to, std::unique_lock<std::mutex>& /* lock */) {
    if (!is_legal_transition(state_, to)) {
        throw InvalidStateTransitionError(
            "Invalid lifecycle transition from " + std::string(to_string(state_)) +
            " to " + std::string(to_string(to)) + " for request_id " + request_id_);
    }
    state_ = to;
}

bool RequestContext::is_terminal(RequestState state) noexcept {
    return state == RequestState::COMPLETE ||
           state == RequestState::CANCELLED ||
           state == RequestState::TIMED_OUT ||
           state == RequestState::FAILED;
}

bool RequestContext::is_terminal() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_terminal(state_);
}

void RequestContext::request_cancellation() noexcept {
    cancel_flag_.store(true, std::memory_order_release);
}

bool RequestContext::is_cancellation_requested() const noexcept {
    return cancel_flag_.load(std::memory_order_acquire);
}

void RequestContext::acknowledge_cancellation() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!is_cancellation_requested()) {
        throw RequestContextError("Cannot acknowledge cancellation: cancel flag is not set for request_id " + request_id_);
    }
    if (is_terminal(state_)) {
        throw InvalidStateTransitionError("Cannot acknowledge cancellation on terminal request_id " + request_id_);
    }
    finish_reason_ = CompletionReason::CANCELLED;
    transition_to_locked(RequestState::CANCELLED, lock);
}

bool RequestContext::has_deadline() const noexcept {
    return has_deadline_;
}

int64_t RequestContext::deadline_ns() const noexcept {
    if (!has_deadline_) {
        return 0;
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        deadline_.time_since_epoch()).count();
}

bool RequestContext::is_expired(Clock::time_point now) const noexcept {
    return has_deadline_ && (now >= deadline_);
}

void RequestContext::acknowledge_timeout(Clock::time_point now) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!has_deadline_) {
        throw RequestContextError("Cannot acknowledge timeout: no deadline configured for request_id " + request_id_);
    }
    if (!is_expired(now)) {
        throw RequestContextError("Cannot acknowledge timeout: deadline has not passed for request_id " + request_id_);
    }
    if (is_terminal(state_)) {
        throw InvalidStateTransitionError("Cannot acknowledge timeout on terminal request_id " + request_id_);
    }
    finish_reason_ = CompletionReason::TIMEOUT;
    transition_to_locked(RequestState::TIMED_OUT, lock);
}

Owner RequestContext::current_owner() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return owner_;
}

void RequestContext::transfer_ownership(Owner new_owner, Owner expected_current_owner) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_terminal(state_)) {
        throw RequestOwnershipError("Cannot transfer ownership of terminal request_id " + request_id_);
    }
    if (owner_ != expected_current_owner) {
        throw RequestOwnershipError(
            "Ownership transfer failed for request_id " + request_id_ +
            ": expected owner " + std::string(to_string(expected_current_owner)) +
            " but actual owner is " + std::string(to_string(owner_)));
    }
    owner_ = new_owner;
}

void RequestContext::append_generated_token(int32_t token_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    generated_tokens_.push_back(token_id);
}

std::vector<int32_t> RequestContext::generated_tokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return generated_tokens_;
}

void RequestContext::set_input_device_ptr(uintptr_t ptr) {
    if (ptr == 0) {
        throw RequestContextError("Null device pointer rejected for input_device_ptr");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    input_device_ptr_ = ptr;
}

void RequestContext::set_output_device_ptr(uintptr_t ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    output_device_ptr_ = ptr;
}

uintptr_t RequestContext::input_device_ptr() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return input_device_ptr_;
}

uintptr_t RequestContext::output_device_ptr() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return output_device_ptr_;
}

void RequestContext::add_kv_block_id(int block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    kv_block_ids_.push_back(block_id);
}

void RequestContext::set_kv_block_ids(std::vector<int> block_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    kv_block_ids_ = std::move(block_ids);
}

std::vector<int> RequestContext::kv_block_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kv_block_ids_;
}

void RequestContext::set_kv_block_table_handle(KVBlockTable* handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    kv_block_table_handle_ = handle;
}

KVBlockTable* RequestContext::kv_block_table_handle() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return kv_block_table_handle_;
}

void RequestContext::set_tokenizer_ref(Tokenizer* ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tokenizer_ref_ != nullptr && ref != tokenizer_ref_) {
        throw RequestContextError("Tokenizer reference already bound to a different instance");
    }
    tokenizer_ref_ = ref;
}

Tokenizer* RequestContext::tokenizer_ref() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return tokenizer_ref_;
}

void RequestContext::set_logical_decode_state(LogicalDecodeState* state) {
    std::lock_guard<std::mutex> lock(mutex_);
    logical_decode_state_ = state;
}

LogicalDecodeState* RequestContext::logical_decode_state() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return logical_decode_state_;
}

void RequestContext::set_session_sequence(uint64_t sequence) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    session_sequence_ = sequence;
}

uint64_t RequestContext::session_sequence() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return session_sequence_;
}

void RequestContext::set_finish_reason(CompletionReason reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    finish_reason_ = reason;
}

CompletionReason RequestContext::finish_reason() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_reason_;
}

void RequestContext::set_error_code(int code) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    error_code_ = code;
}

int RequestContext::error_code() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_code_;
}

void RequestContext::register_cleanup_callback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cleaned_up_) {
        throw RequestContextError("Cannot register cleanup callback on already cleaned up RequestContext");
    }
    cleanup_callbacks_.push_back(std::move(callback));
}

bool RequestContext::is_cleaned_up() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return cleaned_up_;
}

bool RequestContext::cleanup() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!is_terminal(state_)) {
        throw RequestContextError("Cannot cleanup RequestContext before reaching terminal state (current state: " + std::string(to_string(state_)) + ")");
    }
    return cleanup_locked(lock);
}

bool RequestContext::cleanup_locked(std::unique_lock<std::mutex>& /* lock */) {
    if (cleaned_up_) {
        return false;
    }
    execute_cleanup_callbacks_locked();
    cleaned_up_ = true;
    return true;
}

void RequestContext::execute_cleanup_callbacks_locked() {
    for (auto it = cleanup_callbacks_.rbegin(); it != cleanup_callbacks_.rend(); ++it) {
        if (*it) {
            try {
                (*it)();
            } catch (...) {
                // Suppress exceptions during cleanup callback execution
            }
        }
    }
    cleanup_callbacks_.clear();
}
