# Kinetic-RT: Omni-Target Multi-GPU Inference Runtime

Kinetic-RT is an Omni-Target runtime and HPC engine designed to maximize GPU performance by mapping Tensor Parallelism directly to hardware, bypassing the Python GIL.

The core architecture fuses Triton-compiled kernels within a pure, high-performance C++ engine (`src/`), seamlessly exposing APIs to Python via Pybind11 (`kinetic_rt/`).

## Architecture Highlights
- **C++ Native Engine:** High-performance core managing AOT compilation, concurrent multi-stream execution, and zero-copy tensor mapping via pure C++ memory management (safeguarded by rigorous `std::lock_guard` mutexes for thread safety).
- **Triton AOT Compilation:** Fused operations are statically compiled via Triton and serialized into `.kin` hardware-specific files, avoiding runtime JIT overhead.
- **Hardware-Agnostic & Resilient:** Dynamically probes topologies. Adapts transparently across NVIDIA (CUDA `smXX`), AMD (ROCm `gfxXX`), or cleanly falls back to `CPU` for headless execution.
- **Explicit Target Override:** The system environment variable `KINETIC_TARGET` (e.g., `export KINETIC_TARGET=sm75`) overrides hardware discovery if targeted cross-compilation is required.

## Compilation & Installation

The Python interface is an exposed wrapper to the C++ engine. Native compilation is required to bind the C++ logic to the Python `_core` namespace.

From the repository root, install dependencies and compile the extension in-place:
```bash
pip install torch triton numpy pytest transformers setuptools pybind11
pip install -e .
```
This builds the C++ backend and links it against `kinetic_rt._core`.

## First Light Quickstart

To run the end-to-end extraction, compilation, and execution pipeline:

1. **Export and Serialize the Model:**
   ```bash
   python scripts/export_hf.py --tp 1 --output_dir ./models
   ```
   *This shards weights logically, compiles the Triton fusion ops for your physical architecture, and serializes the state to a `.kin` artifact.*

2. **Run Inference:**
   ```bash
   PYTHONPATH=. python scripts/run_first_light.py --model_dir ./models
   ```
   *This bootstraps the C++ Kinetic runtime, mounts the serialized model, allocates managed memory buffers, and executes the highly optimized graph.*
