import sys
import os

# Add root directory to python path for kinetic_rt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import python.kinetic_rt as kinetic_rt
import ctypes

def run_inference():
    print("Initializing AOTEngine...")
    engine = kinetic_rt.AOTEngine()

    model_path = "smollm_135m_tp2.kin"
    print(f"Loading {model_path}...")
    engine.load_model(model_path)

    # Prepare dummy integer tensor representing tokenized input
    # "The capital of India is"
    seq_len = 5
    batch_size = 1

    # Let's mock a simple sequence.
    # For testing physics, the values don't matter, just the launch mechanism.
    # But since it's dummy integer, we just create a ctypes array to hold it
    # and a ctypes array for output.

    # Our launch signature is:
    # `void launch(std::vector<pybind11::object> stream_objs, std::vector<pybind11::object> buffers);`
    # We will pass a dummy python object for buffers to keep it alive.

    IntArrayType = ctypes.c_int * seq_len
    input_tokens = IntArrayType(101, 102, 103, 104, 105)

    # Dummy output logits array (vocab size typically around 32k or 50k, let's say 49152 for SmolLM)
    # We will just read the first few floats to mock it.
    vocab_size = 49152
    FloatArrayType = ctypes.c_float * (seq_len * vocab_size)
    output_logits = FloatArrayType(*([0.0] * (seq_len * vocab_size)))

    wrapper = kinetic_rt.GraphWrapper()

    stream0_ptr = 0x1000
    stream1_ptr = 0x2000

    print("Launching GraphWrapper with multithreaded TP=2 streams...")
    wrapper.begin_capture(stream0_ptr, batch_size, seq_len)
    wrapper.end_capture(stream0_ptr)

    wrapper.launch([stream0_ptr, stream1_ptr], [input_tokens, output_logits])

    # In a real engine, we'd wait for GPU to finish.
    wrapper.invalidate()

    print("Fetching output logits...")

    # Apply simple argmax to the last token
    # Since it's all 0.0 in our dummy execution, argmax is just 0
    # But let's mock reading it properly
    last_token_logits = output_logits[(seq_len-1)*vocab_size:seq_len*vocab_size]

    argmax_idx = 0
    max_val = last_token_logits[0]
    for i in range(1, vocab_size):
        if last_token_logits[i] > max_val:
            max_val = last_token_logits[i]
            argmax_idx = i

    print(f"Argmax token ID: {argmax_idx}")

if __name__ == "__main__":
    run_inference()
