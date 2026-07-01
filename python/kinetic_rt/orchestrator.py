from contextlib import contextmanager
from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class StreamContext:
    def __init__(self):
        # Fallback to torch.cuda if available, otherwise just mock it for headless tests
        if HAS_TORCH and torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            class MockStream:
                def cuda_stream(self):
                    return 0  # Null stream (default stream in CUDA/HIP)
            self.stream = MockStream()

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Stream context exited with error: {exc_value}")
        return False


class TensorValidationError(ValueError):
    """Raised when a tensor fails the zero-copy pre-flight validation.

    Subclassing ValueError preserves backwards compatibility with callers
    that already catch ValueError for input sanitization, while the dedicated
    type lets users distinguish memory-layout failures from ordinary value
    errors.
    """
    pass


# Human-readable names for diagnostics. We avoid str(tensor.dtype) because
# PyTorch's repr can vary across versions (e.g. "torch.float32" vs "float32").
_DTYPE_NAMES = {
    "torch.float32":  "float32",
    "torch.float16":  "float16",
    "torch.bfloat16": "bfloat16",
    "torch.int32":    "int32",
    "torch.int64":    "int64",
    "torch.int8":     "int8",
    "torch.bool":     "bool",
}


def _dtype_name(dtype) -> str:
    key = str(dtype)
    return _DTYPE_NAMES.get(key, key)


def validate_tensor_for_zero_copy(tensor, *, expected_dtype, name: str = "tensor") -> int:
    """Pre-flight validation for any tensor about to cross the Python->C++ boundary.

    The C++ execution routers dereference `tensor.data_ptr()` as raw virtual
    memory with no structural metadata. To prevent sliced, transposed, or
    mismatched-dtype tensors from silently corrupting the engine, this helper
    enforces the invariants required for a safe zero-copy handoff:

      1. Contiguity  — `tensor.is_contiguous()` must return True. Non-unit
                       stride configurations (slices, transposes, expanded
                       views) are explicitly blocked.
      2. Dtype       — `tensor.dtype` must equal `expected_dtype`. The caller
                       pins the contract (e.g. int32 for token arrays, float16
                       /bfloat16 for weights, float32 for logits).
      3. Storage     — `tensor.storage_offset()` must be strictly 0, so the
                       base pointer addresses dense linear storage from the
                       very first element.

    Args:
        tensor:         The torch.Tensor to validate.
        expected_dtype: The torch.dtype the caller has contracted to deliver
                        (e.g. torch.int32 for token arrays).
        name:           Human-readable identifier used in error messages so
                        callers can pinpoint which tensor failed.

    Returns:
        The validated `data_ptr()` as an int. Suitable for the legacy
        uintptr_t pybind11 overloads. Callers that prefer the structured
        descriptor path should additionally wrap the tensor via
        `_core.TensorDescriptor.from_tensor(tensor, name)`.

    Raises:
        TensorValidationError: if any invariant is violated.
        RuntimeError:          if PyTorch is not available.
    """
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required to validate tensors for zero-copy handoff, "
            "but `import torch` failed. Install PyTorch or run inside a "
            "torch-enabled environment."
        )

    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(
            f"{name}: expected a torch.Tensor instance, got "
            f"{type(tensor).__name__}. The zero-copy boundary only accepts "
            "torch tensors with dense linear storage."
        )

    # --- Invariant 1: Contiguity -----------------------------------------
    if not tensor.is_contiguous():
        raise TensorValidationError(
            f"{name}: tensor is not C-contiguous. "
            f"shape={tuple(tensor.shape)}, strides={tuple(tensor.stride())}. "
            "Sliced, transposed, or expanded views with non-unit strides are "
            "explicitly blocked at the Python->C++ boundary because the "
            "native execution routers assume dense row-major linear storage. "
            "Remediation: call tensor.contiguous() (or tensor.clone()) "
            "before invoking the runtime."
        )

    # --- Invariant 2: Dtype contract -------------------------------------
    if tensor.dtype != expected_dtype:
        actual = _dtype_name(tensor.dtype)
        wanted = _dtype_name(expected_dtype)
        raise TensorValidationError(
            f"{name}: dtype mismatch. tensor.dtype={actual} but the runtime "
            f"contract requires dtype={wanted}. Passing a mismatched dtype "
            "across the zero-copy boundary causes the C++ side to "
            "misinterpret element boundaries and silently corrupt the "
            "execution engine."
        )

    # --- Invariant 3: Storage offset -------------------------------------
    storage_offset = tensor.storage_offset()
    if storage_offset != 0:
        raise TensorValidationError(
            f"{name}: storage_offset={storage_offset} (elements) is non-zero. "
            "data_ptr() addresses the start of the underlying storage, but "
            "the logical view begins at a non-zero offset, so the C++ base "
            "pointer would dereference memory belonging to a sibling view. "
            "Remediation: call tensor.contiguous() to materialize a fresh "
            "dense tensor whose storage_offset is 0."
        )

    # --- Sanity: zero-length tensors have no valid base pointer -----------
    if tensor.numel() == 0:
        raise TensorValidationError(
            f"{name}: tensor has zero numel (shape={tuple(tensor.shape)}). "
            "Empty tensors cannot be forwarded across the zero-copy boundary."
        )

    ptr = tensor.data_ptr()
    if ptr == 0:
        raise TensorValidationError(
            f"{name}: data_ptr() returned 0 (null). The tensor's storage "
            "has not been allocated. Call .to(device) or ensure the tensor "
            "was materialized before invoking the runtime."
        )

    logger.debug(
        "validate_tensor_for_zero_copy(%s): dtype=%s shape=%s strides=%s "
        "storage_offset=0 data_ptr=0x%x byte_size=%d",
        name,
        _dtype_name(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        ptr,
        tensor.element_size() * tensor.numel(),
    )
    return ptr


import os


class KineticRuntime:
    def __init__(self, engine, wrapper=None, tokenizer_name=None):
        self.engine = engine  # Acts as router if wrapper is None
        self.wrapper = wrapper
        self.tp_degree = 1
        self.model_paths = []

        self.tokenizer = None
        _tok_name = tokenizer_name or "HuggingFaceTB/SmolLM2-135M"
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(_tok_name)
            logger.info(f"Loaded tokenizer: {_tok_name}")
        except (ImportError, Exception) as exc:
            logger.warning(f"Could not load tokenizer ({exc}); will use byte-level fallback")

    def load_model(self, model_dir: str):
        import glob
        from . import Serializer, TopologyMismatchError

        # In mock CI, pretend it loaded
        if os.environ.get("MOCK_HIP") == "1" and not os.path.exists(model_dir):
            self.tp_degree = 1
            self.model_paths = ["dummy.kin"]
            return

        serializer = Serializer()

        kin_files = glob.glob(os.path.join(model_dir, "*.kin"))
        if not kin_files:
            raise FileNotFoundError(f"No .kin files found in {model_dir}")

        # Read metadata from the first file to find TP degree
        self.tp_degree = serializer.get_tensor_parallel_degree(kin_files[0])

        actual_gpus = torch.cuda.device_count() if HAS_TORCH and torch.cuda.is_available() else 0
        if self.tp_degree > 1 and actual_gpus < self.tp_degree:
            raise TopologyMismatchError(f"Topology mismatch: Model exported with TP={self.tp_degree}, but found {actual_gpus} GPUs.")

        # Map shards
        self.model_paths = sorted(kin_files)
        if len(self.model_paths) < self.tp_degree:
            # We assume in a real environment we have one shard per TP degree if TP > 1,
            # or maybe one big file and we load it differently.
            # For simplicity, if we found files, we map what we have.
            logger.warning(f"Found {len(self.model_paths)} shard files but TP degree is {self.tp_degree}. Some shards may not be loaded.")

        # In a real environment we would load the shards into respective GPUs
        for i in range(min(self.tp_degree, len(self.model_paths))):
            self.engine.load_model(self.model_paths[i])

    def _convert_tensor(self, tensor: "torch.Tensor", *, expected_dtype, name: str = "tensor") -> int:
        """Validate a tensor and return its data_ptr() as an int.

        This is the single chokepoint through which every tensor pointer
        extraction must flow. The validation is enforced *before* .data_ptr()
        is read so that a layout violation can never produce a naked virtual
        memory address that escapes into the C++ binding layer.

        Args:
            tensor:         The torch.Tensor to validate.
            expected_dtype: The torch.dtype the caller has contracted to deliver.
            name:           Human-readable identifier used in error messages.

        Returns:
            The validated data_ptr() as an int.
        """
        return validate_tensor_for_zero_copy(
            tensor, expected_dtype=expected_dtype, name=name
        )

    def _make_descriptor(self, tensor: "torch.Tensor", *, expected_dtype, name: str = "tensor"):
        """Validate a tensor and return a `_core.TensorDescriptor` for the
        structured-descriptor pybind11 overloads.

        The Python-side `validate_tensor_for_zero_copy` runs first to fail
        fast with a Python-native TensorValidationError. We then construct
        the C++ TensorDescriptor, which itself re-validates on the C++ side
        (defensive: callers that bypass the Python helper and construct a
        descriptor directly are still caught).
        """
        # Fail fast on the Python side before paying for the FFI crossing.
        validate_tensor_for_zero_copy(
            tensor, expected_dtype=expected_dtype, name=name
        )
        from ._core import TensorDescriptor
        return TensorDescriptor.from_tensor(tensor, name)

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for generating text.")

        # Tokenize -> Load into Engine -> Launch Graph -> Fetch Logits -> Argmax -> Append Token

        if self.tokenizer:
            tokens = self.tokenizer.encode(prompt)
        else:
            logger.warning("No tokenizer available, using byte-level fallback")
            tokens = list(prompt.encode('utf-8'))

        vocab_size = 50257  # Standard GPT-2/SmolLM vocabulary size

        for _ in range(max_new_tokens):
            input_tensor = torch.tensor(tokens, dtype=torch.int32)

            # Use actual device tensor allocation
            device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
            input_tensor = input_tensor.to(device)
            # .to() may short-circuit when source and destination devices
            # match, but it can also produce a non-contiguous view in some
            # edge cases (e.g. after a narrow()). Defensively materialize a
            # dense tensor so the zero-copy validator never has to reject it.
            if not input_tensor.is_contiguous():
                input_tensor = input_tensor.contiguous()

            output_tensor = torch.empty(
                (len(tokens), vocab_size), dtype=torch.float32, device=device
            )
            # torch.empty() is always contiguous, but the validation helper
            # double-checks so we never rely on an internal PyTorch invariant.

            # Convert tensors to descriptors/pointers for the C++ Pybind11
            # wrapper. Every pointer extraction MUST pass through
            # _convert_tensor / _make_descriptor so the zero-copy boundary
            # stays ironclad.
            input_ptr = self._convert_tensor(
                input_tensor,
                expected_dtype=torch.int32,
                name="generate.input_tensor",
            )
            output_ptr = self._convert_tensor(
                output_tensor,
                expected_dtype=torch.float32,
                name="generate.output_tensor",
            )

            with StreamContext() as stream:
                stream_ptr = stream.cuda_stream() if hasattr(stream, "cuda_stream") else 0

                batch_size = 1
                seq_len = len(tokens)

                if self.wrapper:
                    self.wrapper.begin_capture(stream_ptr, batch_size, seq_len)
                    self.wrapper.end_capture(stream_ptr)

                    # Execute graph using pointers
                    # We also pass the original python objects to ensure they are pinned/kept alive
                    self.wrapper.launch(
                        [stream_ptr],
                        [input_ptr, output_ptr, input_tensor, output_tensor],
                    )

                    # We strictly synchronize to ensure logits are fully written by the GPU
                    if HAS_TORCH and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    self.wrapper.invalidate()
                else:
                    # Routed execution path handling HardwareRouter.
                    # Forward validated TensorDescriptors so the C++ side can
                    # re-verify shape/strides/dtype before dereferencing the
                    # raw pointer.
                    input_desc = self._make_descriptor(
                        input_tensor,
                        expected_dtype=torch.int32,
                        name="generate.input_tensor",
                    )
                    output_desc = self._make_descriptor(
                        output_tensor,
                        expected_dtype=torch.float32,
                        name="generate.output_tensor",
                    )
                    self.engine.launch(input_desc, output_desc, seq_len)
                    if HAS_TORCH and torch.cuda.is_available():
                        torch.cuda.synchronize()

            # Argmax logic against actual tensor outputs
            # Retrieve the logits for the last token in the sequence
            last_token_logits = output_tensor[-1, :]

            argmax_idx = torch.argmax(last_token_logits).item()
            tokens.append(argmax_idx)

        # Decode the generated token sequence
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        else:
            return bytes(t % 256 for t in tokens).decode('utf-8', errors='replace')
