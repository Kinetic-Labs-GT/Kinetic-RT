# Testing Kinetic-RT

This repository maintains a fully dynamic, hardware-aware test suite designed to validate the C++ Native Engine, Pybind11 boundaries, and the Python runtime orchestrator.

## Test Validation Map

All tests are located in the `tests/` directory and validate specific boundaries of the Omni-Target architecture:

- **`test_aot.py`**: Validates the native C++ `AOTEngine` and `Serializer`. Tests the full loop of compiling a `.kin` file and loading it back, verifying offset validations and handling "Poison Pill" hardware mismatch cases (e.g. attempting to load a `CUDA_sm75` artifact on an `ROCm_gfx90a` device).
- **`test_validation.py`**: Rigorously tests the deep binary validation protocols. Ensures the runtime actively rejects malformed ELF files, incorrect class identifiers (32-bit vs 64-bit), and corrupt machine types (`EM_CUDA` vs `EM_AMDGPU`).
- **`test_distributed.py`**: Verifies the orchestrator's Tensor Parallelism implementation. Confirms concurrent execution mechanics and multi-stream initialization logic across logically simulated or physical device targets.
- **`test_fusion.py`**: Tests the `fusion_forge.py` Triton wrapper, ensuring high-level operations correctly map to kernel compilation and serialization targets without throwing runtime errors.
- **`test_graph.py`**: Exercises the Pybind11 `GraphWrapper` boundary. Ensures HIP Graph instantiation, memory safety during execution via multi-threading, and correct stream context allocations are successfully propagated down to the C++ logic.

## Running the Test Suite

The test suite dynamically probes hardware and adapts expectations based on whether it is running on NVIDIA (CUDA), AMD (ROCm), or CPU (Headless). You do **not** need to manually mock architectures or skip tests when running headlessly.

To run the full suite across all validated boundaries, simply execute:

```bash
PYTHONPATH=. pytest tests/
```

### Note on "Poison Pill" Tests
Certain tests will explicitly print "Caught expected HardwareMismatchError" or similar warnings to stdout. This is intentional. The engine is deliberately verifying that illegal state accesses and corrupted `.kin` files are safely caught by the C++ engine before memory corruption can occur.

## The "First Light" Protocol

The "First Light" sequence tests the full end-to-end extraction, compilation, and serialization workflow. Follow these exact steps for standard validation using SmolLM2-135M on a fresh instance.

1.  **Export and Serialize Model:** Shard weights (e.g. `TP=1`), compile Triton ops based on hardware discovery, and generate the binary artifact.
    ```bash
    python scripts/export_hf.py --tp 1 --output_dir ./models
    ```

2.  **Execute Inference:** Launch the orchestrator to mount the serialized `.kin` file to the C++ runtime.
    ```bash
    PYTHONPATH=. python scripts/run_first_light.py --model_dir ./models
    ```
