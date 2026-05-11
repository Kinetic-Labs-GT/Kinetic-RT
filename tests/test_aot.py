import kinetic_rt
import os

def test_serializer():
    serializer = kinetic_rt.Serializer()
    filepath = "test_model.kin"
    device_id = "gfx1100"
    weights_hash = 123456789
    op_graph_data = [1, 2, 3]

    # Mock hsaco that passes deep validation
    kernel_binaries = [0] * 64
    kernel_binaries[0:4] = [0x7f, ord('E'), ord('L'), ord('F')]
    kernel_binaries[4] = 2
    kernel_binaries[18:20] = [0xE0, 0x00]

    # Save file
    serializer.save_kin_file(filepath, device_id, weights_hash, op_graph_data, kernel_binaries)
    assert os.path.exists(filepath)

    # Load file
    loaded_binaries = serializer.load_kin_file(filepath)
    assert loaded_binaries == kernel_binaries

    # Try loading with a different hardware id mock
    # The mock currently returns "gfx1100"
    # So to trigger hardware mismatch, we'll save a file that expects "gfx942"
    filepath_mismatch = "test_mismatch.kin"
    serializer.save_kin_file(filepath_mismatch, "gfx942", weights_hash, op_graph_data, kernel_binaries)

    try:
        serializer.load_kin_file(filepath_mismatch)
        assert False, "Should have raised HardwareMismatch"
    except kinetic_rt.HardwareMismatch as e:
        print(f"Caught expected HardwareMismatch: {e}")
        assert "expected gfx942 but got gfx1100" in str(e)

    os.remove(filepath)
    os.remove(filepath_mismatch)

def test_serializer_error_handling():
    serializer = kinetic_rt.Serializer()

    # Test file not found
    try:
        serializer.load_kin_file("non_existent_file.kin")
        assert False, "Should have raised RuntimeError for file not found"
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")
        assert "Failed to open file for reading" in str(e)

    import tempfile
    import uuid
    # Test write failure
    nonexistent_path = os.path.join(tempfile.gettempdir(), f"invalid_dir_{uuid.uuid4()}", "test.kin")
    try:
        serializer.save_kin_file(nonexistent_path, "gfx1100", 12345, [], [])
        assert False, "Should have raised RuntimeError for write failure"
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")
        assert "Failed to open file for writing" in str(e)

def test_aot_engine():
    engine = kinetic_rt.AOTEngine()
    filepath = "aot_model.kin"
    stream_ptr = 1234

    # Compile
    engine.compile_ahead_of_time(filepath, stream_ptr)
    assert os.path.exists(filepath)

    # Load
    engine.load_model(filepath)

    os.remove(filepath)

    print("AOT Engine tests passed.")

if __name__ == "__main__":
    test_serializer()
    test_serializer_error_handling()
    test_aot_engine()
    print("All tests passed successfully!")
