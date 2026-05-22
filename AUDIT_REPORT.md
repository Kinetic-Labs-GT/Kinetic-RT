# Brutal Audit Report: Kinetic-RT

## Overview
This document represents a ruthless static analysis of the Kinetic-RT codebase, targeting hardcoded debris, architectural fragility, C++ memory/thread safety issues, and documentation drift. The goal is to expose technical debt and "vibe coding" that compromises this HPC library's production readiness.

---

## 🚨 FATAL Severity

### 1. Hardcoded "ROCm" and "gfx1100" Fallbacks in Core Validation
**Files:** `src/AOTEngine.cpp`, `include/AOTEngine.h`, `python/kinetic_rt/hardware_probe.py`, `tests/test_validation.py`, `tests/test_aot.py`
**Lines:**
- `include/AOTEngine.h:27-28`: Examples `char device_id[256]; // e.g., "gfx1100"` and `char target_architecture[256]; // e.g., "CUDA_sm75" or "ROCm_gfx942"`
- `python/kinetic_rt/hardware_probe.py:25`: `arch = "gfx90a" # Defaulting if missing is bad, but we try to avoid hardcoded fallbacks`
- `tests/test_validation.py:93-94`: `serializer.save_kin_file(filepath_mismatch, "gfx942", "ROCm_gfx942", 12345, [], [])` and `# AOTEngine expects "gfx1100" by default` (Actually AOTEngine device_id expects the host device id). But `src/AOTEngine.cpp:239`: `std::string expected_prefix = "ROCm";` and `python/kinetic_rt/hardware_probe.py:8`: `backend = "CUDA" if "sm" in forced_arch else "ROCm"`
**Why it's unacceptable:** An Omni-Target engine cannot default to AMD/ROCm identifiers if it intends to support NVIDIA or CPU architectures resiliently. Defaulting to `gfx90a` or unconditionally branching to `ROCm` if `sm` isn't found means on CPUs or unknown architectures, the system falsely identifies as an AMD GPU. This breaks Headless CI and CPU fallbacks.
**Architectural Fix:** Implement strict, dynamic querying. Remove all fallback assumptions. If hardware is truly missing or unsupported, fallback explicitly to `CPU` or throw an unsupported hardware exception.

### 2. Missing Thread Safety on Graph Execution (Data Races)
**File:** `src/GraphWrapper.cpp`
**Lines:** `127-147` (inside `GraphWrapper::launch`)
**Why it's unacceptable:** `workers.emplace_back([this, stream]() { CHECK_HIP(hipGraphLaunch(graph_exec_, stream)); });` spawns threads capturing `this` without any explicit memory fencing or internal synchronization around `graph_exec_` if multiple launches happen simultaneously across different python calls or if the HIP graph API itself isn't thread-safe for the same `graph_exec_`. While there is an `engine_mutex_` lock at the top of the function (`std::lock_guard<std::recursive_mutex> lock(engine_mutex_);`), locking the *entire* function defeats the purpose of spawning threads to launch asynchronously, as no other launches can occur anyway. Conversely, `in_flight_states_` and `event_pool_` are modified outside the thread but inside the lock. The architectural design here is muddled—either thread the launch while releasing the lock (and protecting only state structures), or don't use threads. Furthermore, if `workers` are joined immediately on lines 143-147 (`worker.join()`), the multithreading provides zero asynchronous benefit to the caller because it blocks waiting for all threads to submit their launches.
**Architectural Fix:** Re-architect the `launch` function. If asynchronous multi-stream submission is desired, submit to HIP asynchronously directly (HIP handles stream queues). Remove the blocking `std::thread` creation and `join()` antipattern inside a locked mutex.

### 3. Documentation Lying to User (Missing Scripts)
**File:** `README.md`, `TESTS.md`
**Lines:**
- `README.md:31`, `TESTS.md:43`: `PYTHONPATH=. python scripts/run_first_light.py`
**Why it's unacceptable:** The scripts document `run_first_light.py` outputting inference. However, in `scripts/run_first_light.py`, it literally mocks everything using integer inputs and zero'd float outputs (`FloatArrayType(*([0.0] * (seq_len * vocab_size)))`) and then executes `argmax` on zeroes. It doesn't actually execute the `.kin` file properly with real PyTorch tensors, it uses dummy `ctypes`. The README promises "highly optimized graph execution" but runs dummy data.
**Architectural Fix:** Rewrite `run_first_light.py` to actually use `KineticRuntime` from `orchestrator.py` or correctly bind PyTorch tensors. Update the README to clarify if it is just a synthetic benchmark or implement real inference.

---

## ⚡ SEVERE Severity

### 1. Build System Fragility (Race Condition on PyTorch)
**File:** `setup.py`
**Lines:** Build script does not directly import `torch`, but `python/kinetic_rt/hardware_probe.py` does. Wait, `setup.py` builds the C++ code, but `pip install -e .` runs it. If a user runs `pip install -e .` without PyTorch installed, it works because `setup.py` doesn't import torch. However, `import kinetic_rt` imports `orchestrator.py` which does `import torch` unconditionally.
**File:** `python/kinetic_rt/orchestrator.py`
**Lines:** `1`: `import torch`
**Why it's unacceptable:** `TESTS.md` states it should be a pure C++ engine. If PyTorch is required, it should be gracefully handled or explicit. `export_hf.py` imports `torch` too. The setup dependencies don't list `torch`.
**Architectural Fix:** Either add `torch` to `install_requires` in `setup.py` or make `torch` an optional dependency gracefully caught in `hardware_probe.py` and `orchestrator.py`.

### 2. Hardcoded /tmp Paths for CI Tests
**File:** `tests/test_serializer_errors.cpp`
**Lines:** `208`: `std::string bad_path = "/tmp/invalid_dir_" + timestamp + "/test.kin";`
**Why it's unacceptable:** Hardcoding `/tmp` fails on Windows systems and non-standard UNIX environments. It assumes directory structures that might not exist or might lack permissions.
**Architectural Fix:** Use a proper cross-platform temporary directory mechanism (e.g., `std::filesystem::temp_directory_path` in C++17) or pass the temp directory from the Python test runner.

### 3. Arbitrary 2GB Size Limits on Serialization
**File:** `src/AOTEngine.cpp`
**Lines:** `135`: `if (kernel_binaries_size > 1024ULL * 1024ULL * 1024ULL * 2ULL || op_graph_data_size > 1024ULL * 1024ULL * 1024ULL * 2ULL) { // 2 GB max arbitrary limit`
**Why it's unacceptable:** LLMs often have weights or kernel structures exceeding 2GB. An arbitrary 2GB limit will crash the engine on modern models.
**Architectural Fix:** Remove the arbitrary limit and replace it with dynamic memory checking, or increase the limit significantly and rely on standard `std::bad_alloc` exceptions during vector allocation.

---

## 🛠️ AMATEUR Severity

### 1. Lazy "pass" Blocks / Exception Swallowing
**File:** `src/GraphWrapper.cpp`
**Lines:** `20-22`: `catch (...) { // Suppress exceptions in destructor }`
**Why it's unacceptable:** Swallowing all exceptions in a destructor hides critical memory leaks or HIP synchronization failures during cleanup.
**Architectural Fix:** Catch specific exceptions, log them to `std::cerr` or a logger, and proceed with cleanup safely.

### 2. Vibe Coding / Unused kwargs and Magic Values
**File:** `python/kinetic_rt/fusion_forge.py`
**Lines:** `194-203`: Hardcoded byte patches `compiled_binary[18] = 0xBE`, dummy `[10, 20, 30]` op graph data, and unused `kwargs`.
**Why it's unacceptable:** Mocking compiler output with hardcoded ELF binaries inside the main compilation bridge (`compile_and_serialize`) completely defeats the purpose of the library. It isn't actually compiling anything; it's writing a fake 64-byte file.
**Architectural Fix:** Integrate actual Triton compilation (`triton.compile`) to generate the real PTX/HSACO, or explicitly move this mock logic into a testing-only mock file and separate the real compiler bridge.

### 3. Print Statements in Production Code
**File:** `python/kinetic_rt/fusion_forge.py`, `scripts/export_hf.py`, `scripts/run_first_light.py`
**Lines:** `fusion_forge.py:217`: `print(f"Fused kernel compiled and serialized to {output_filepath}")`
**Why it's unacceptable:** Global `print()` statements pollute the stdout of HPC applications.
**Architectural Fix:** Use the standard Python `logging` module.
