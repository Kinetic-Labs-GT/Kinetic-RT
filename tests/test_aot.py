import kinetic_rt
import os

def test_serializer():
    serializer = kinetic_rt.Serializer()
    filepath = "test_model.kin"
    device_id = "gfx1100"
    weights_hash = 123456789
    op_graph_data = [1, 2, 3]
    kernel_binaries = [255, 69, 76, 70] # Mock hsaco: \x7fELF

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
    test_aot_engine()
    print("All tests passed successfully!")
