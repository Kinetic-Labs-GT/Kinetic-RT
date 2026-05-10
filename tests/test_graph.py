import kinetic_rt

def test_graph_capture_and_launch():
    # Initialize wrapper
    wrapper = kinetic_rt.GraphWrapper()

    stream_ptr = 1234 # dummy ptr

    # Should not be valid initially
    assert not wrapper.is_valid(batch_size=1, seq_len=128)

    # 1. Warm-up (Capture)
    wrapper.begin_capture(stream_ptr, batch_size=1, seq_len=128)

    # 2. Instantiation (Seal)
    wrapper.end_capture(stream_ptr)

    # Graph should now be valid for the captured shapes
    assert wrapper.is_valid(batch_size=1, seq_len=128)

    # Test redundant begin_capture
    wrapper.begin_capture(stream_ptr, batch_size=1, seq_len=128)
    assert wrapper.is_valid(batch_size=1, seq_len=128)

    # 3. Loop (Replay)
    # Pass an integer proxy for stream_obj since bindings cast stream_obj to uintptr_t
    wrapper.launch(stream_ptr, [])

    # Check dynamic shape handling
    assert not wrapper.is_valid(batch_size=2, seq_len=128)

    # Simulating a change in shapes
    wrapper.begin_capture(stream_ptr, batch_size=2, seq_len=128)
    wrapper.end_capture(stream_ptr)

    assert wrapper.is_valid(batch_size=2, seq_len=128)
    assert not wrapper.is_valid(batch_size=1, seq_len=128)

    # Launch new graph
    wrapper.launch(stream_ptr, [])

    print("Graph validation passed successfully!")

def test_async_stress():
    wrapper = kinetic_rt.GraphWrapper()
    stream_ptr = 1234

    wrapper.begin_capture(stream_ptr, batch_size=1, seq_len=128)
    wrapper.end_capture(stream_ptr)

    stream_obj = stream_ptr # Can just be an integer

    # Run 1,000 overlapping asynchronous launches
    for i in range(1000):
        # We pass a new buffer object on each call. If the system was GC'ing them incorrectly
        # or racing, this would crash/segfault on a real system, and our stateful mock tracks it safely.
        dummy_buffer = [i]
        wrapper.launch(stream_obj, dummy_buffer)

    # Invalidate at the end should clear out everything and sync cleanly
    wrapper.invalidate()
    print("Async Stress Test passed successfully!")

if __name__ == "__main__":
    test_graph_capture_and_launch()
    test_async_stress()
    print("All tests passed successfully!")
