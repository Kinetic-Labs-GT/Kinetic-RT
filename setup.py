from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

# For a real ROCm environment, we'd include actual HIP paths.
# Since we are setting up a mock build for CI/testing without ROCm, we'll configure it differently.

try:
    import torch
    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_capability()
    else:
        compute_cap = None
except Exception:
    import warnings
    warnings.warn("PyTorch is not installed or CUDA is unavailable. Falling back to default mock/PTX compiler flags. Please install PyTorch for native hardware compilation.")
    compute_cap = None

mock_hip = os.environ.get("MOCK_HIP", "1") == "1"

macros = []
if mock_hip:
    macros.append(("MOCK_HIP", "1"))

ext_modules = [
    Extension(
        'python.kinetic_rt._core',
        ['bindings/python_bindings.cpp', 'src/AOTEngine.cpp', 'src/GraphWrapper.cpp', 'tests/mock_hip.cpp', 'src/Communicator.cpp', 'tests/mock_rccl.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            'include',
            'tests' # for mock_hip.h
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
        if has_flag(compiler, flag): return flag
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
    ],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
