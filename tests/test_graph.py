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

    # 3. Loop (Replay)
    wrapper.launch(stream_ptr)

    # Check dynamic shape handling
    assert not wrapper.is_valid(batch_size=2, seq_len=128)

    # Simulating a change in shapes
    wrapper.begin_capture(stream_ptr, batch_size=2, seq_len=128)
    wrapper.end_capture(stream_ptr)

    assert wrapper.is_valid(batch_size=2, seq_len=128)
    assert not wrapper.is_valid(batch_size=1, seq_len=128)

    # Launch new graph
    wrapper.launch(stream_ptr)

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_graph_capture_and_launch()
