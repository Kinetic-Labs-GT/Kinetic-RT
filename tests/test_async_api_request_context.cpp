#include "Router.h"
#include "AsyncAPI.h"

#include <cassert>
#include <iostream>

void test_submit_attaches_context_with_ingress_metadata() {
    InferenceQueue queue;

    queue.submit(0x1000, 3, 17, "request-one");

    InferenceRequest request;
    assert(queue.fetch_request(request));
    assert(request.context);
    assert(request.context->request_id() == "request-one");
    assert(request.context->max_new_tokens() == 17);
    assert(request.context->current_owner() == Owner::Ingress);

    std::cout << "[PASS] test_submit_attaches_context_with_ingress_metadata" << std::endl;
}

void test_fetch_retains_submitted_context_identity() {
    InferenceQueue queue;

    queue.submit(0x2000, 5, 9, "request-two");

    InferenceRequest request;
    assert(queue.fetch_request(request));
    const RequestContext* context_before_copy = request.context.get();

    InferenceRequest copied_request = request;
    assert(copied_request.context.get() == context_before_copy);

    std::cout << "[PASS] test_fetch_retains_submitted_context_identity" << std::endl;
}

void test_separate_submissions_receive_independent_contexts() {
    InferenceQueue queue;

    queue.submit(0x3000, 2, 4, "request-three");
    queue.submit(0x4000, 7, 12, "request-four");

    InferenceRequest first_request;
    InferenceRequest second_request;
    assert(queue.fetch_request(first_request));
    assert(queue.fetch_request(second_request));
    assert(first_request.context);
    assert(second_request.context);
    assert(first_request.context.get() != second_request.context.get());

    std::cout << "[PASS] test_separate_submissions_receive_independent_contexts" << std::endl;
}

int main() {
    test_submit_attaches_context_with_ingress_metadata();
    test_fetch_retains_submitted_context_identity();
    test_separate_submissions_receive_independent_contexts();
    return 0;
}
