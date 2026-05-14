# Kinetic-RT Validation Protocol

This document outlines the testing strategies, definitions, and workflows that validate the Kinetic-RT engine across multiple targets.

## Test Suite Mapping

All tests are located in the `tests/` directory:

*   **`test_aot.py`**: Verifies the core `AOTEngine` and `Serializer` components. This includes the validation of the exact ELF structural mappings and ensuring the correct metadata serialization bounds. It also houses the **"Poison Pill" Test**, which intentionally serializes an incorrect architecture string into a mock `.kin` file to assert that `HardwareMismatchError` correctly triggers.
*   **`test_distributed.py`**: Tests the `Communicator` bindings, particularly verifying that asynchronous All-Reduce operations across dummy streams correctly release the Python GIL and do not deadlock.
*   **`test_fusion.py`**: Validates the higher-level mathematical fidelity and buffer assignments generated when interpreting or compiling specific model components within Triton logic.
*   **`test_graph.py`**: Verifies the `GraphWrapper` captures bounds. Ensures validation checks (e.g., verifying validity of inputs vs batch lengths) perform correctly before executing mocked HIP graphs.
*   **`test_validation.py`**: A specialized test ensuring strict parsing and rejection states inside the Triton compilation bridge, validating missing magic headers or bit-class mismatches.
*   **`benchmark_aot.cpp`**: A natively compiled standalone test for `AOTEngine`, checking latency metrics, ensuring driver redundancies are minimized, and providing an end-to-end integration benchmark of the C++ codebase independent of the Python interpreter.

## Execution Commands

To correctly execute the Python test suite natively, you must include the local root path:

```bash
PYTHONPATH=. pytest tests/
```

To build and run the C++ standalone benchmark (in environments using the mock backend):

```bash
g++ -std=c++17 -DMOCK_HIP tests/benchmark_aot.cpp src/AOTEngine.cpp tests/mock_hip.cpp -Iinclude -I. -o benchmark_aot
./benchmark_aot
```

## The "First Light" Protocol

The "First Light" sequence tests the full end-to-end extraction, compilation, and serialization workflow. Follow these exact steps for standard validation using SmolLM2-135M.

1.  **Probe the Hardware**: Verify the topology is resolvable.
    ```bash
    python -c "import python.kinetic_rt.hardware_probe as hp; print(hp.get_topology_string())"
    ```
2.  **Export and Serialize**: Extract weights from HuggingFace, partition them (simulated TP), autodiscover target architecture, compile, and serialize everything into a single `.kin` file.
    ```bash
    PYTHONPATH=. python scripts/export_hf.py
    ```
3.  **Execute Inference**: Load the generated `.kin` file natively into the runtime wrapper and simulate a graph launch.
    ```bash
    PYTHONPATH=. python scripts/run_first_light.py
    ```
