import time
import logging

logger = logging.getLogger(__name__)


class KernelTuner:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self._cache = {}

    def get_warp_size(self, device_id=None):
        """Return the warp size for *device_id* (default: ``self.device_id``)."""
        try:
            import torch
            return torch.cuda.get_device_properties(device_id or self.device_id).warp_size
        except (RuntimeError, AssertionError, ImportError):
            return 32  # Default warp size for most GPUs

    def get_device_properties(self, device_id=None):
        """Return a dict of key device properties, with safe fallbacks."""
        try:
            import torch
            props = torch.cuda.get_device_properties(device_id or self.device_id)
            return {
                'warp_size': props.warp_size,
                'max_threads_per_block': props.max_threads_per_multi_processor,
                'shared_memory_per_block': props.total_memory,
                'name': props.name,
            }
        except (RuntimeError, AssertionError, ImportError):
            return {
                'warp_size': 32,
                'max_threads_per_block': 1024,
                'shared_memory_per_block': 49152,
                'name': 'unknown',
            }

    def profile_kernel(self, kernel_fn, *args, num_warmup=3, num_runs=10, **kwargs):
        """
        Profile *kernel_fn* on the GPU.  Returns average execution time in
        milliseconds.  If no GPU is available, returns ``0.0`` with a warning.
        """
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available for profiling, returning estimated time")
            return 0.0

        if not torch.cuda.is_available():
            logger.warning("No GPU available for profiling, returning estimated time")
            return 0.0

        for _ in range(num_warmup):
            kernel_fn(*args, **kwargs)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            kernel_fn(*args, **kwargs)
        torch.cuda.synchronize()

        return (time.perf_counter() - start) / num_runs * 1000  # ms

    def autotune(self, kernel_fn, configs, *args, **kwargs):
        """
        Run *kernel_fn* with each config in *configs* (list of dicts) and
        return ``(best_config, best_time_ms)``.
        """
        best_config = None
        best_time = float('inf')
        for config in configs:
            merged_kwargs = {**kwargs, **config}
            t = self.profile_kernel(kernel_fn, *args, **merged_kwargs)
            if t < best_time:
                best_time = t
                best_config = config
        return best_config, best_time
