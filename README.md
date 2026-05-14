# Kinetic-RT: Omni-Target Multi-GPU Inference Runtime

Kinetic-RT is a highly optimized, cross-platform runtime for asynchronous model inference and parallelized operations across multiple GPUs. Built around a unified AOT (Ahead-Of-Time) strategy, it seamlessly bridges PyTorch/Triton compilation to low-level C++ execution.

## Architecture Overview

1.  **C++ AOT Engine (`src/AOTEngine.cpp`)**: A robust, thread-safe execution environment that handles graph capture, dynamic ELF validation, and driver-level kernel scheduling without the heavy overhead of standard Python runners.
2.  **Pybind11 Integration**: The bindings in `bindings/python_bindings.cpp` expose core engine functionalities (like `GraphWrapper`, `Serializer`, `AOTEngine`) to Python safely handling multi-threading context release (`py::gil_scoped_release`).
3.  **Hardware Auto-Discovery + Universal Compiler (`fusion_forge.py`)**: Before compilation, `hardware_probe.py` deeply inspects the local topology to find specific SM/GFX versions. The `fusion_forge.py` script then bridges Triton kernel representations, compiling them down to pure binary payloads (HSACO/PTX) tagged with the verified architecture.

## Installation

Building the engine requires compiling the underlying C++ sources natively.

### Prerequisites

Ensure you have installed all the necessary Python build tools and dependencies.

```bash
pip install -r requirements.txt # (or manually install setuptools, pytest, pybind11, torch, triton, numpy, transformers)
```

### Build Command (Crucial)

Before running any Python scripts or test suites, you **must** compile the `_core` extension natively using `setuptools`. Run the following command from the root directory:

```bash
python setup.py build_ext --inplace
# Or equivalently:
# pip install -e .
```

This ensures the `python.kinetic_rt._core` module is present and resolvable.
