from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import shutil
from pathlib import Path

# ----------------------------------------------------------------------
# Diagnostic environment checker (fixes Defect #24)
# ----------------------------------------------------------------------
def check_environment():
    """Verify that the required native toolchain is present before building.

    If MOCK_HIP=1 is set, we skip real hardware checks and allow mock build.
    Otherwise, we require either HIP (AMD) or CUDA (NVIDIA) to be properly
    installed and discoverable.
    """
    mock_hip = os.environ.get("MOCK_HIP", "0") == "1"
    if mock_hip:
        print("🔧 MOCK_HIP=1: skipping native toolchain validation (mock mode).")
        return

    # ---------- 1. Check for pybind11 headers (must be importable) ----------
    try:
        import pybind11
        pybind11.get_include()
    except ImportError:
        raise RuntimeError(
            "pybind11 is not installed or not importable. "
            "Please ensure 'pybind11>=2.5.0' is available (setup_requires should handle this)."
        )

    # ---------- 2. Locate HIP or CUDA environment ----------
    rocm_path = os.environ.get("ROCM_PATH")
    hip_path = os.environ.get("HIP_PATH")
    cuda_home = os.environ.get("CUDA_HOME")

    # Common default installation paths
    default_paths = [
        "/opt/rocm",
        "/usr/local/cuda",
        "/usr/local/hip",
    ]

    # Gather all candidate root directories
    candidates = []
    if rocm_path:
        candidates.append(rocm_path)
    if hip_path:
        candidates.append(hip_path)
    if cuda_home:
        candidates.append(cuda_home)
    for p in default_paths:
        if p not in candidates and os.path.exists(p):
            candidates.append(p)

    # ---------- 3. Look for compiler executables ----------
    def find_executable(name, search_dirs):
        """Return first existing path to executable 'name' in search_dirs or PATH."""
        for d in search_dirs:
            exe = Path(d) / "bin" / name
            if exe.is_file():
                return str(exe)
        return shutil.which(name)

    hipcc_path = find_executable("hipcc", candidates)
    nvcc_path = find_executable("nvcc", candidates)

    found_hip = hipcc_path is not None
    found_cuda = nvcc_path is not None

    if not (found_hip or found_cuda):
        # Build a diagnostic message
        diag = [
            "❌ No HIP or CUDA compiler found.",
            "   Please install the appropriate development toolkit for your GPU.",
            "",
            "   Environment variables checked:",
            f"     ROCM_PATH  = {rocm_path or '(not set)'}",
            f"     HIP_PATH   = {hip_path or '(not set)'}",
            f"     CUDA_HOME  = {cuda_home or '(not set)'}",
            "",
            "   Searched these roots:",
        ]
        for c in candidates:
            diag.append(f"     {c}")
        diag.append("")
        diag.append("   Also looked for 'hipcc' and 'nvcc' in PATH.")
        diag.append("")
        diag.append("   To build without real hardware (mock mode), set MOCK_HIP=1.")
        raise RuntimeError("\n".join(diag))

    # ---------- 4. (Optional) Warn if headers are missing ----------
    if found_hip:
        hip_root = Path(hipcc_path).parent.parent
        include_hip = hip_root / "include" / "hip" / "hip_runtime.h"
        if not include_hip.exists():
            print(f"⚠️  Warning: HIP header not found at expected location: {include_hip}")
            print("   The build may still succeed if the compiler finds headers elsewhere.")
    if found_cuda:
        cuda_root = Path(nvcc_path).parent.parent
        include_cuda = cuda_root / "include" / "cuda_runtime.h"
        if not include_cuda.exists():
            print(f"⚠️  Warning: CUDA header not found at expected location: {include_cuda}")
            print("   The build may still succeed if the compiler finds headers elsewhere.")

    print("✅ Native toolchain validation passed.")
    if found_hip:
        print(f"   Using HIP compiler: {hipcc_path}")
    elif found_cuda:
        print(f"   Using CUDA compiler: {nvcc_path}")

# ----------------------------------------------------------------------
# Remaining setup.py code
# ----------------------------------------------------------------------

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """
    def __str__(self):
        import pybind11
        return pybind11.get_include()

# Run environment validation early (before building extensions)
check_environment()

# Determine mock mode consistently (default to production)
mock_hip = os.environ.get("MOCK_HIP", "0") == "1"

# For a real ROCm environment, we'd include actual HIP paths.
# Since we are setting up a mock build for CI/testing without ROCm, we'll configure it differently.
try:
    import torch
    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_capability()
    else:
        compute_cap = None
        default_target = "sm_60"
except Exception:
    import warnings
    warnings.warn("PyTorch is not installed or CUDA is unavailable. Falling back to default mock/PTX compiler flags. Please install PyTorch for native hardware compilation.")
    compute_cap = None
    default_target = "sm_60"

macros = []
if mock_hip:
    macros.append(("MOCK_HIP", "1"))

ext_modules = [
    Extension(
        'python.kinetic_rt._core',
        ['bindings/python_bindings.cpp', 'src/AOTEngine.cpp', 'src/GraphWrapper.cpp', 'tests/mock_hip.cpp', 'src/Communicator.cpp', 'tests/mock_rccl.cpp', 'src/tensorrt/TRTEngine.cpp'],
        include_dirs=[
            get_pybind_include(),
            'include',
            'tests'
        ],
        define_macros=macros,
        language='c++'
    ),
]

# As of Python 3.6, C++11 is the default language standard.
# We explicitly set it to C++14 to ensure compatibility.
def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True

def cpp_flag(compiler):
    flags = ['-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag):
            return flag
    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='kinetic_rt',
    version='0.1.0',
    author='Jules',
    description='GraphWrapper using HIP Graphs',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=[
        'torch',
        'triton',
        'transformers',
        'fastapi',
        'pydantic',
        'uvicorn',
        'sse_starlette',
    ],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)