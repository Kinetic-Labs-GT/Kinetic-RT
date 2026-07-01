from .hardware_probe import get_topology_string

try:
    from ._core import (
        AOTEngine,
        GraphWrapper,
        Serializer,
        Communicator,
        HardwareMismatchError,
        HardwareRouter,
        InferenceQueue,
        InferenceWorker,
        TensorDescriptor,
        DataType,
    )
except ImportError as e:
    raise ImportError("Unable to load kinetic_rt extension. Ensure the CUDA/ROCm toolkit is correctly installed and the C++ extension compiled successfully.") from e

class TopologyMismatchError(Exception):
    pass

from .orchestrator import KineticRuntime, StreamContext, validate_tensor_for_zero_copy, TensorValidationError
from .serve import serve
