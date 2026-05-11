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

def test_bad_magic_number():
    serializer = kinetic_rt.Serializer()
    filepath = "bad_magic.kin"

    # Create a malformed .kin file with a wrong magic number
    import struct
    # KinHeader struct layout (pack):
    # uint32_t magic_number;
    # uint32_t version;
    # char device_id[256];
    # uint64_t weights_hash;
    # uint64_t op_graph_data_offset;
    # uint64_t op_graph_data_size;
    # uint64_t kernel_binaries_offset;
    # uint64_t kernel_binaries_size;

    # We will just write enough bytes to pass the size check, with a bad magic number.
    with open(filepath, "wb") as f:
        # Magic number is little endian 0x12345678 instead of 0x4B494E00
        f.write(struct.pack("<I", 0x12345678))
        f.write(b'\x00' * (256 + 4 + 8 * 5))

    try:
        serializer.load_kin_file(filepath)
        assert False, "Should have raised RuntimeError for bad magic number"
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")
        assert "bad magic number" in str(e)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

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
    test_bad_magic_number()
    test_aot_engine()
    print("All tests passed successfully!")
