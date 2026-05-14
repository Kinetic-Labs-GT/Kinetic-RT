import python.kinetic_rt as kinetic_rt
import os
from python.kinetic_rt.hardware_probe import probe_hardware

def test_serializer():
    topology, backend, arch = probe_hardware()

    # We must match the device_id that Serializer will check against.
    # Serializer fetches its device_id from C++ `hipGetDeviceProperties` internally,
    # which we've mocked in `mock_hip.h` to return "cpu" when running in headless CPU CI.
    # So we should use the actual probed `arch` for saving the mock so that loading works,
    # and only fallback for `backend` to satisfy `validate_elf_structure`'s prefix requirement.
    device_id = arch

    if backend == "CPU":
        backend = "ROCm"

    target_arch = f"{backend}_{arch}"

    serializer = kinetic_rt.Serializer()
    filepath = "test_model.kin"
    weights_hash = 123456789
    op_graph_data = [1, 2, 3]

    # Mock binary that passes deep validation based on the actual detected backend
    kernel_binaries = [0] * 64
    kernel_binaries[0:4] = [0x7f, ord('E'), ord('L'), ord('F')]
    kernel_binaries[4] = 2

    if backend == "CUDA":
        kernel_binaries[18:20] = [0xBE, 0x00] # EM_CUDA
    else:
        kernel_binaries[18:20] = [0xE0, 0x00] # EM_AMDGPU

    # Save file
    serializer.save_kin_file(filepath, device_id, target_arch, weights_hash, op_graph_data, kernel_binaries)
    assert os.path.exists(filepath)

    # Load file
    loaded_binaries = serializer.load_kin_file(filepath)
    assert loaded_binaries == kernel_binaries

    # Try loading with a different hardware id mock ("Poison Pill" Test)
    filepath_mismatch = "test_mismatch.kin"
    # Construct an intentionally wrong architecture for rejection testing
    if backend == "CUDA":
        wrong_device_id = "gfx1100"
        wrong_target_arch = "ROCm_gfx1100"
        kernel_binaries_mismatch = list(kernel_binaries)
        kernel_binaries_mismatch[18:20] = [0xE0, 0x00] # EM_AMDGPU
    else:
        wrong_device_id = "sm75"
        wrong_target_arch = "CUDA_sm75"
        kernel_binaries_mismatch = list(kernel_binaries)
        kernel_binaries_mismatch[18:20] = [0xBE, 0x00] # EM_CUDA

    serializer.save_kin_file(filepath_mismatch, wrong_device_id, wrong_target_arch, weights_hash, op_graph_data, kernel_binaries_mismatch)

    try:
        serializer.load_kin_file(filepath_mismatch)
        assert False, "Should have raised HardwareMismatchError"
    except kinetic_rt.HardwareMismatchError as e:
        print(f"Caught expected HardwareMismatchError: {e}")
        assert f"expected {wrong_device_id} but got {device_id}" in str(e)

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
        serializer.save_kin_file(nonexistent_path, "gfx1100", "ROCm_gfx1100", 12345, [], [])
        assert False, "Should have raised RuntimeError for write failure"
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")
        assert "Failed to open file for writing" in str(e)

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
        f.write(b'\x00' * (256 + 256 + 4 + 8 * 5 + 4))

    try:
        serializer.load_kin_file(filepath)
        assert False, "Should have raised RuntimeError for bad magic number"
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")
        assert "bad magic number" in str(e).lower()
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def test_aot_engine():
    # In test_aot_engine, we must ensure we use a valid architecture prefix (CUDA or ROCm)
    # because AOTEngine::validate_elf_structure explicitly expects these prefixes.
    # We will pretend we are on the detected backend, or fallback to ROCm for headless.
    topology, backend, arch = probe_hardware()
    if backend == "CPU":
        backend = "ROCm"
        arch = "gfx1100"

    target_arch = f"{backend}_{arch}"

    engine = kinetic_rt.AOTEngine()
    filepath = "aot_model.kin"
    stream_ptr = 1234

    # Compile
    engine.compile_ahead_of_time(filepath, stream_ptr, target_arch)
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
