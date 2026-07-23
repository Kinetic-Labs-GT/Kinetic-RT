#include "RequestContext.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

void test_construction_valid() {
    RequestContext::Config config;
    config.request_id = "req_001";
    config.prompt_text = "Hello world";
    config.prompt_tokens = {101, 2054, 2088};
    config.max_new_tokens = 50;
    config.temperature = 0.7f;
    config.top_p = 0.9f;
    config.top_k = 40;
    config.stop_token_ids = {102};
    config.backend_hint = "AOTEngine";
    config.timeout = std::chrono::milliseconds(500);
    config.initial_owner = Owner::Ingress;

    RequestContext ctx(config);

    assert(ctx.request_id() == "req_001");
    assert(ctx.prompt_text() == "Hello world");
    assert(ctx.prompt_tokens().size() == 3);
    assert(ctx.max_new_tokens() == 50);
    assert(ctx.temperature() == 0.7f);
    assert(ctx.top_p() == 0.9f);
    assert(ctx.top_k() == 40);
    assert(ctx.stop_token_ids().size() == 1);
    assert(ctx.backend_hint() == "AOTEngine");
    assert(ctx.state() == RequestState::RECEIVED);
    assert(ctx.current_owner() == Owner::Ingress);
    assert(ctx.is_valid());
    assert(!ctx.is_terminal());
    assert(!ctx.is_cleaned_up());
    assert(ctx.has_deadline());

    std::cout << "[PASS] test_construction_valid" << std::endl;
}

void test_construction_invalid() {
    // Empty request ID
    try {
        RequestContext::Config config;
        config.request_id = "";
        RequestContext ctx(config);
        assert(false && "Should have thrown RequestContextError for empty request_id");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("request_id cannot be empty") != std::string::npos);
    }

    // Negative max_new_tokens
    try {
        RequestContext::Config config;
        config.request_id = "req_err";
        config.max_new_tokens = -5;
        RequestContext ctx(config);
        assert(false && "Should have thrown RequestContextError for negative max_new_tokens");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("max_new_tokens cannot be negative") != std::string::npos);
    }

    // Negative temperature
    try {
        RequestContext::Config config;
        config.request_id = "req_err";
        config.temperature = -0.5f;
        RequestContext ctx(config);
        assert(false && "Should have thrown RequestContextError for negative temperature");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("temperature cannot be negative") != std::string::npos);
    }

    // Out of range top_p (> 1.0)
    try {
        RequestContext::Config config;
        config.request_id = "req_err";
        config.top_p = 1.5f;
        RequestContext ctx(config);
        assert(false && "Should have thrown RequestContextError for out-of-range top_p");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("top_p must be in range [0.0, 1.0]") != std::string::npos);
    }

    // Negative top_k
    try {
        RequestContext::Config config;
        config.request_id = "req_err";
        config.top_k = -1;
        RequestContext ctx(config);
        assert(false && "Should have thrown RequestContextError for negative top_k");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("top_k cannot be negative") != std::string::npos);
    }

    std::cout << "[PASS] test_construction_invalid" << std::endl;
}

void test_state_transitions_happy_path() {
    RequestContext::Config config;
    config.request_id = "req_002";
    RequestContext ctx(config);

    assert(ctx.state() == RequestState::RECEIVED);
    assert(ctx.can_transition_to(RequestState::VALIDATED));

    ctx.transition_to(RequestState::VALIDATED);
    assert(ctx.state() == RequestState::VALIDATED);

    ctx.transition_to(RequestState::TOKENIZED);
    assert(ctx.state() == RequestState::TOKENIZED);

    ctx.transition_to(RequestState::ENQUEUED);
    assert(ctx.state() == RequestState::ENQUEUED);

    ctx.transition_to(RequestState::PREFILL);
    assert(ctx.state() == RequestState::PREFILL);

    ctx.transition_to(RequestState::DECODE);
    assert(ctx.state() == RequestState::DECODE);

    ctx.transition_to(RequestState::STREAMING);
    assert(ctx.state() == RequestState::STREAMING);

    // Loop decode <-> streaming
    ctx.transition_to(RequestState::DECODE);
    assert(ctx.state() == RequestState::DECODE);

    ctx.transition_to(RequestState::STREAMING);
    assert(ctx.state() == RequestState::STREAMING);

    ctx.transition_to(RequestState::COMPLETE);
    assert(ctx.state() == RequestState::COMPLETE);
    assert(ctx.is_terminal());

    std::cout << "[PASS] test_state_transitions_happy_path" << std::endl;
}

void test_illegal_state_transitions() {
    RequestContext::Config config;
    config.request_id = "req_003";
    RequestContext ctx(config);

    // Stage skip: RECEIVED -> PREFILL
    try {
        ctx.transition_to(RequestState::PREFILL);
        assert(false && "Should have thrown InvalidStateTransitionError for stage skip");
    } catch (const InvalidStateTransitionError& e) {
        std::string msg = e.what();
        assert(msg.find("Invalid lifecycle transition") != std::string::npos);
    }

    // Transition out of terminal state
    ctx.transition_to(RequestState::VALIDATED);
    ctx.transition_to(RequestState::TOKENIZED);
    ctx.transition_to(RequestState::ENQUEUED);
    ctx.transition_to(RequestState::PREFILL);
    ctx.transition_to(RequestState::DECODE);
    ctx.transition_to(RequestState::STREAMING);
    ctx.transition_to(RequestState::COMPLETE);

    try {
        ctx.transition_to(RequestState::DECODE);
        assert(false && "Should have thrown InvalidStateTransitionError when transitioning out of terminal state");
    } catch (const InvalidStateTransitionError& e) {
        std::string msg = e.what();
        assert(msg.find("Invalid lifecycle transition") != std::string::npos);
    }

    std::cout << "[PASS] test_illegal_state_transitions" << std::endl;
}

void test_ownership_transfer() {
    RequestContext::Config config;
    config.request_id = "req_004";
    config.initial_owner = Owner::Ingress;
    RequestContext ctx(config);

    assert(ctx.current_owner() == Owner::Ingress);

    // Ingress -> Scheduler
    ctx.transfer_ownership(Owner::Scheduler, Owner::Ingress);
    assert(ctx.current_owner() == Owner::Scheduler);

    // Scheduler -> Executor
    ctx.transfer_ownership(Owner::Executor, Owner::Scheduler);
    assert(ctx.current_owner() == Owner::Executor);

    // Executor -> Backend
    ctx.transfer_ownership(Owner::Backend, Owner::Executor);
    assert(ctx.current_owner() == Owner::Backend);

    // Wrong expected owner mismatch
    try {
        ctx.transfer_ownership(Owner::Scheduler, Owner::Ingress);
        assert(false && "Should have thrown RequestOwnershipError for ownership mismatch");
    } catch (const RequestOwnershipError& e) {
        std::string msg = e.what();
        assert(msg.find("Ownership transfer failed") != std::string::npos);
    }
    // Verify owner remains unchanged after failed transfer
    assert(ctx.current_owner() == Owner::Backend);

    // Ownership transfer on terminal request
    ctx.transition_to(RequestState::CANCELLED);
    try {
        ctx.transfer_ownership(Owner::Scheduler, Owner::Backend);
        assert(false && "Should have thrown RequestOwnershipError on terminal request");
    } catch (const RequestOwnershipError& e) {
        std::string msg = e.what();
        assert(msg.find("Cannot transfer ownership of terminal request") != std::string::npos);
    }

    std::cout << "[PASS] test_ownership_transfer" << std::endl;
}

void test_cancellation_two_phase() {
    RequestContext::Config config;
    config.request_id = "req_005";
    RequestContext ctx(config);

    assert(!ctx.is_cancellation_requested());

    // Acknowledge cancellation without flag set should throw
    try {
        ctx.acknowledge_cancellation();
        assert(false && "Should have thrown RequestContextError");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("cancel flag is not set") != std::string::npos);
    }

    // Phase 1: Request cancellation (flag set, state unchanged)
    ctx.request_cancellation();
    assert(ctx.is_cancellation_requested());
    assert(ctx.state() == RequestState::RECEIVED); // State has NOT changed yet

    // Phase 2: Acknowledge cancellation (state -> CANCELLED)
    ctx.acknowledge_cancellation();
    assert(ctx.state() == RequestState::CANCELLED);
    assert(ctx.finish_reason() == CompletionReason::CANCELLED);
    assert(ctx.is_terminal());

    std::cout << "[PASS] test_cancellation_two_phase" << std::endl;
}

void test_cancellation_multithreaded() {
    RequestContext::Config config;
    config.request_id = "req_006";
    RequestContext ctx(config);

    std::thread producer([&ctx]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ctx.request_cancellation();
    });

    // Consumer polling cancellation
    while (!ctx.is_cancellation_requested()) {
        std::this_thread::yield();
    }

    producer.join();
    ctx.acknowledge_cancellation();
    assert(ctx.state() == RequestState::CANCELLED);

    std::cout << "[PASS] test_cancellation_multithreaded" << std::endl;
}

void test_timeout_expiration() {
    RequestContext::Config config;
    config.request_id = "req_007";
    config.timeout = std::chrono::milliseconds(20);
    RequestContext ctx(config);

    assert(ctx.has_deadline());
    assert(ctx.deadline_ns() > 0);

    // Before deadline
    assert(!ctx.is_expired(RequestContext::Clock::now()));
    try {
        ctx.acknowledge_timeout(RequestContext::Clock::now());
        assert(false && "Should have thrown RequestContextError when not expired");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("deadline has not passed") != std::string::npos);
    }

    // Wait until deadline passes
    std::this_thread::sleep_for(std::chrono::milliseconds(35));
    assert(ctx.is_expired(RequestContext::Clock::now()));

    ctx.acknowledge_timeout(RequestContext::Clock::now());
    assert(ctx.state() == RequestState::TIMED_OUT);
    assert(ctx.finish_reason() == CompletionReason::TIMEOUT);
    assert(ctx.is_terminal());

    std::cout << "[PASS] test_timeout_expiration" << std::endl;
}

void test_cleanup_framework() {
    RequestContext::Config config;
    config.request_id = "req_008";
    RequestContext ctx(config);

    std::vector<int> execution_order;

    ctx.register_cleanup_callback([&execution_order]() {
        execution_order.push_back(1);
    });
    ctx.register_cleanup_callback([&execution_order]() {
        execution_order.push_back(2);
    });

    // Cleanup before terminal state should throw
    try {
        ctx.cleanup();
        assert(false && "Should have thrown RequestContextError before terminal state");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("Cannot cleanup RequestContext before reaching terminal state") != std::string::npos);
    }

    ctx.transition_to(RequestState::FAILED);
    assert(ctx.is_terminal());

    // First cleanup returns true and executes callbacks in LIFO order
    bool cleaned = ctx.cleanup();
    assert(cleaned);
    assert(ctx.is_cleaned_up());
    assert(execution_order.size() == 2);
    assert(execution_order[0] == 2); // LIFO: 2 first, then 1
    assert(execution_order[1] == 1);

    // Second cleanup returns false (double cleanup prevention)
    bool second_cleanup = ctx.cleanup();
    assert(!second_cleanup);

    // Register callback on already cleaned up context should throw
    try {
        ctx.register_cleanup_callback([]() {});
        assert(false && "Should have thrown RequestContextError on cleaned up context");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("Cannot register cleanup callback") != std::string::npos);
    }

    std::cout << "[PASS] test_cleanup_framework" << std::endl;
}

void test_cleanup_callback_reentrancy_no_deadlock() {
    RequestContext::Config config;
    config.request_id = "req_reentrant_cleanup";
    RequestContext ctx(config);

    bool reentrant_call_succeeded = false;
    ctx.register_cleanup_callback([&ctx, &reentrant_call_succeeded]() {
        // A cleanup callback invoking context methods (which acquire mutex_)
        // Must NOT deadlock now that callbacks execute outside the critical section!
        RequestState s = ctx.state();
        bool cleaned = ctx.is_cleaned_up();
        if (s == RequestState::COMPLETE && cleaned) {
            reentrant_call_succeeded = true;
        }
    });

    ctx.transition_to(RequestState::VALIDATED);
    ctx.transition_to(RequestState::TOKENIZED);
    ctx.transition_to(RequestState::ENQUEUED);
    ctx.transition_to(RequestState::PREFILL);
    ctx.transition_to(RequestState::DECODE);
    ctx.transition_to(RequestState::STREAMING);
    ctx.transition_to(RequestState::COMPLETE);

    bool ok = ctx.cleanup();
    assert(ok);
    assert(reentrant_call_succeeded);

    std::cout << "[PASS] test_cleanup_callback_reentrancy_no_deadlock" << std::endl;
}

void test_destructor_non_mutating_lifecycle() {
    bool callback_executed = false;
    RequestState state_before_destruction = RequestState::DECODE;
    RequestState state_inside_callback = RequestState::RECEIVED;

    {
        RequestContext::Config config;
        config.request_id = "req_009";
        RequestContext ctx(config);
        ctx.transition_to(RequestState::VALIDATED);
        ctx.transition_to(RequestState::TOKENIZED);
        ctx.transition_to(RequestState::ENQUEUED);
        ctx.transition_to(RequestState::PREFILL);
        ctx.transition_to(RequestState::DECODE);
        state_before_destruction = ctx.state();

        ctx.register_cleanup_callback([&ctx, &callback_executed, &state_inside_callback]() {
            callback_executed = true;
            state_inside_callback = ctx.state();
        });
        // Destructor runs here without explicit cleanup or terminal transition
    }

    assert(callback_executed);
    assert(state_before_destruction == RequestState::DECODE);
    // Crucial check: destructor performed defensive cleanup WITHOUT mutating state_ to FAILED
    assert(state_inside_callback == RequestState::DECODE);

    std::cout << "[PASS] test_destructor_non_mutating_lifecycle" << std::endl;
}

void test_kv_and_device_pointers() {
    RequestContext::Config config;
    config.request_id = "req_010";
    RequestContext ctx(config);

    // Device pointers
    try {
        ctx.set_input_device_ptr(0);
        assert(false && "Should have rejected null input device pointer");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("Null device pointer rejected") != std::string::npos);
    }

    ctx.set_input_device_ptr(0x1000);
    ctx.set_output_device_ptr(0x2000);
    assert(ctx.input_device_ptr() == 0x1000);
    assert(ctx.output_device_ptr() == 0x2000);

    // KV blocks
    ctx.add_kv_block_id(10);
    ctx.add_kv_block_id(11);
    assert(ctx.kv_block_ids().size() == 2);
    assert(ctx.kv_block_ids()[0] == 10);
    assert(ctx.kv_block_ids()[1] == 11);

    ctx.set_kv_block_ids({20, 21, 22});
    assert(ctx.kv_block_ids().size() == 3);
    assert(ctx.kv_block_ids()[0] == 20);

    // KVBlockTable handle
    KVBlockTable* dummy_table = reinterpret_cast<KVBlockTable*>(0xABCD);
    ctx.set_kv_block_table_handle(dummy_table);
    assert(ctx.kv_block_table_handle() == dummy_table);

    std::cout << "[PASS] test_kv_and_device_pointers" << std::endl;
}

void test_typed_pointers_and_tokens() {
    RequestContext::Config config;
    config.request_id = "req_011";
    RequestContext ctx(config);

    // Tokenizer ref (bind once)
    Tokenizer* tok1 = reinterpret_cast<Tokenizer*>(0x1111);
    Tokenizer* tok2 = reinterpret_cast<Tokenizer*>(0x2222);

    ctx.set_tokenizer_ref(tok1);
    assert(ctx.tokenizer_ref() == tok1);

    // Same ref is OK
    ctx.set_tokenizer_ref(tok1);
    assert(ctx.tokenizer_ref() == tok1);

    // Different ref throws
    try {
        ctx.set_tokenizer_ref(tok2);
        assert(false && "Should have thrown for rebinding tokenizer_ref");
    } catch (const RequestContextError& e) {
        std::string msg = e.what();
        assert(msg.find("already bound to a different instance") != std::string::npos);
    }

    // LogicalDecodeState
    LogicalDecodeState* state_ptr = reinterpret_cast<LogicalDecodeState*>(0x3333);
    ctx.set_logical_decode_state(state_ptr);
    assert(ctx.logical_decode_state() == state_ptr);

    // Session sequence
    ctx.set_session_sequence(42);
    assert(ctx.session_sequence() == 42);

    // Generated tokens
    ctx.append_generated_token(1001);
    ctx.append_generated_token(1002);
    assert(ctx.generated_tokens().size() == 2);
    assert(ctx.generated_tokens()[0] == 1001);
    assert(ctx.generated_tokens()[1] == 1002);

    // Finish reason and error code
    ctx.set_finish_reason(CompletionReason::EOS);
    assert(ctx.finish_reason() == CompletionReason::EOS);

    ctx.set_error_code(404);
    assert(ctx.error_code() == 404);

    std::cout << "[PASS] test_typed_pointers_and_tokens" << std::endl;
}

int main() {
    std::cout << "Starting RequestContext unit tests..." << std::endl;

    test_construction_valid();
    test_construction_invalid();
    test_state_transitions_happy_path();
    test_illegal_state_transitions();
    test_ownership_transfer();
    test_cancellation_two_phase();
    test_cancellation_multithreaded();
    test_timeout_expiration();
    test_cleanup_framework();
    test_cleanup_callback_reentrancy_no_deadlock();
    test_destructor_non_mutating_lifecycle();
    test_kv_and_device_pointers();
    test_typed_pointers_and_tokens();

    std::cout << "\n==========================================" << std::endl;
    std::cout << "All RequestContext unit tests PASSED!" << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
}
