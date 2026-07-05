from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised in torch-free environments
    HAS_TORCH = False


class StreamContext:
    """Provide a CUDA stream when one is available, otherwise a mock stream.

    The mock keeps headless CI and import-time validation paths working without
    requiring a GPU runtime. The returned object only needs to expose
    ``cuda_stream()`` for the native bindings.
    """

    def __init__(self):
        if HAS_TORCH and torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:

            class MockStream:
                def cuda_stream(self):
                    return 0

            self.stream = MockStream()

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error("Stream context exited with error: %s", exc_value)
        return False


class DeviceMismatchError(ValueError):
    """Raised when host tensors are about to cross into GPU runtime paths."""


class TensorValidationError(ValueError):
    """Raised when a tensor fails the zero-copy pre-flight validation."""


class RuntimeConfigurationError(ValueError):
    """Raised when runtime configuration is internally inconsistent."""


_DTYPE_NAMES = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.int8": "int8",
    "torch.bool": "bool",
}

_FATAL_DEVICE_MISMATCH = (
    "FATAL: Boundary Mismatch. Passed host (CPU) memory pointer to a native "
    "hardware accelerator (GPU/HIP) backend path."
)

_DEFAULT_BYTE_VOCAB = 256
_DEFAULT_TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def allows_mock_device_residency() -> bool:
    """Return True only for explicit mock/headless verification modes."""

    return _env_flag_enabled("MOCK_HIP") or _env_flag_enabled("KINETIC_ALLOW_MOCKS")


def validate_tensor_device_residency(
    tensor, *, backend_expects_accelerator: bool = True, name: str = "tensor"
) -> None:
    """Fail closed before a tensor pointer crosses into native GPU/HIP code."""

    if not backend_expects_accelerator or allows_mock_device_residency():
        return

    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to validate tensor device residency.")

    if not isinstance(tensor, torch.Tensor):
        raise DeviceMismatchError(f"{name}: expected torch.Tensor for device residency validation.")

    if tensor.device.type != "cuda" or not tensor.is_cuda:
        raise DeviceMismatchError(
            f"{_FATAL_DEVICE_MISMATCH} {name}: device={tensor.device}, "
            "expected a native CUDA/HIP device tensor. Set MOCK_HIP=1 or "
            "KINETIC_ALLOW_MOCKS=1 only for explicit mock/headless verification."
        )


def _dtype_name(dtype) -> str:
    return _DTYPE_NAMES.get(str(dtype), str(dtype))


def validate_tensor_for_zero_copy(tensor, *, expected_dtype, name: str = "tensor") -> int:
    """Validate a tensor before extracting its raw data pointer."""

    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required to validate tensors for zero-copy handoff, "
            "but `import torch` failed. Install PyTorch or run inside a torch-enabled environment."
        )

    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(
            f"{name}: expected a torch.Tensor instance, got {type(tensor).__name__}."
        )

    if not tensor.is_contiguous():
        raise TensorValidationError(
            f"{name}: tensor is not C-contiguous. shape={tuple(tensor.shape)}, strides={tuple(tensor.stride())}."
        )

    if tensor.dtype != expected_dtype:
        raise TensorValidationError(
            f"{name}: dtype mismatch. tensor.dtype={_dtype_name(tensor.dtype)} "
            f"but the runtime contract requires dtype={_dtype_name(expected_dtype)}."
        )

    storage_offset = tensor.storage_offset()
    if storage_offset != 0:
        raise TensorValidationError(
            f"{name}: storage_offset={storage_offset} (elements) is non-zero."
        )

    if tensor.numel() == 0:
        raise TensorValidationError(
            f"{name}: tensor has zero numel (shape={tuple(tensor.shape)})."
        )

    if tensor.untyped_storage().nbytes() < (tensor.numel() * tensor.element_size()):
        raise TensorValidationError(
            f"{name}: tensor logical view exceeds actual physical storage allocation bounds."
        )

    ptr = tensor.data_ptr()
    if ptr == 0:
        raise TensorValidationError(f"{name}: data_ptr() returned 0 (null).")

    logger.debug(
        "validate_tensor_for_zero_copy(%s): dtype=%s shape=%s strides=%s storage_offset=0 data_ptr=0x%x byte_size=%d",
        name,
        _dtype_name(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        ptr,
        tensor.element_size() * tensor.numel(),
    )
    return ptr


def _normalize_int(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer-compatible value.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return normalized


def _try_normalize_int(value: Any, *, name: str, minimum: int) -> Optional[int]:
    try:
        return _normalize_int(value, name=name, minimum=minimum)
    except (TypeError, ValueError):
        return None


class KineticRuntime:
    def __init__(self, engine, wrapper=None, tokenizer_name=None, vocab_size=None):
        self.engine = engine
        self.wrapper = wrapper
        self.tp_degree = 1
        self.model_paths = []
        self.vocab_size = vocab_size

        self.tokenizer = None
        self._tokenizer_name = tokenizer_name or _DEFAULT_TOKENIZER_NAME
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            logger.info("Loaded tokenizer: %s", self._tokenizer_name)
        except (ImportError, Exception) as exc:
            logger.warning("Could not load tokenizer (%s); will use byte-level fallback", exc)

    def _iter_tokenizer_vocab_candidates(self) -> Iterable[tuple[str, Any]]:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return []

        candidates = []

        try:
            candidates.append(("tokenizer.vocab_size", getattr(tokenizer, "vocab_size")))
        except Exception as exc:
            logger.debug("Tokenizer vocab_size accessor failed: %s", exc)

        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            try:
                vocab = get_vocab()
            except Exception as exc:
                logger.debug("Tokenizer get_vocab() failed: %s", exc)
            else:
                try:
                    candidates.append(("len(tokenizer.get_vocab())", len(vocab)))
                except TypeError:
                    candidates.append(("tokenizer.get_vocab()", vocab))

        try:
            candidates.append(("len(tokenizer)", len(tokenizer)))
        except (TypeError, AttributeError) as exc:
            logger.debug("Tokenizer __len__ unavailable: %s", exc)

        return candidates

    def _resolve_vocab_size(self) -> int:
        """Resolve the output vocabulary width for logit allocation."""

        if self.vocab_size is not None:
            return _normalize_int(self.vocab_size, name="explicit vocab_size", minimum=1)

        for source_name, candidate in self._iter_tokenizer_vocab_candidates():
            resolved = _try_normalize_int(candidate, name=source_name, minimum=1)
            if resolved is not None:
                return resolved

        return _DEFAULT_BYTE_VOCAB

    def _fallback_token_id(self) -> int:
        tokenizer = self.tokenizer
        if tokenizer is not None:
            for attr_name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
                try:
                    candidate = getattr(tokenizer, attr_name)
                except Exception:
                    continue
                resolved = _try_normalize_int(candidate, name=attr_name, minimum=0)
                if resolved is not None:
                    return resolved
        return 0

    def encode_prompt(self, prompt: str) -> list[int]:
        """Encode text using the active tokenizer, with a byte-level fallback."""

        if self.tokenizer:
            encode = getattr(self.tokenizer, "encode", None)
            if not callable(encode):
                raise RuntimeConfigurationError("Tokenizer does not expose an encode() method.")
            raw_tokens = encode(prompt)
        else:
            logger.warning("No tokenizer available, using byte-level fallback")
            raw_tokens = list(prompt.encode("utf-8"))

        if isinstance(raw_tokens, (str, bytes, bytearray)):
            raise RuntimeConfigurationError("Tokenizer.encode() returned an invalid token sequence type.")

        if HAS_TORCH and isinstance(raw_tokens, torch.Tensor):
            raw_tokens = raw_tokens.detach().cpu().tolist()

        tokens = []
        for index, token in enumerate(raw_tokens):
            if isinstance(token, bool):
                raise RuntimeConfigurationError(f"Token at index {index} is a bool, which is invalid.")
            try:
                tokens.append(int(token))
            except (TypeError, ValueError) as exc:
                raise RuntimeConfigurationError(
                    f"Token at index {index} is not integer-convertible: {token!r}"
                ) from exc

        if not tokens:
            tokens = [self._fallback_token_id()]

        return tokens

    def load_model(self, model_dir: str):
        import glob
        from . import Serializer, TopologyMismatchError

        if os.environ.get("MOCK_HIP") == "1" and not os.path.exists(model_dir):
            self.tp_degree = 1
            self.model_paths = ["dummy.kin"]
            return

        serializer = Serializer()

        kin_files = glob.glob(os.path.join(model_dir, "*.kin"))
        if not kin_files:
            raise FileNotFoundError(f"No .kin files found in {model_dir}")

        self.tp_degree = serializer.get_tensor_parallel_degree(kin_files[0])

        actual_gpus = torch.cuda.device_count() if HAS_TORCH and torch.cuda.is_available() else 0
        if self.tp_degree > 1 and actual_gpus < self.tp_degree:
            raise TopologyMismatchError(
                f"Topology mismatch: Model exported with TP={self.tp_degree}, but found {actual_gpus} GPUs."
            )

        self.model_paths = sorted(kin_files)
        if len(self.model_paths) < self.tp_degree:
            logger.warning(
                "Found %s shard files but TP degree is %s. Some shards may not be loaded.",
                len(self.model_paths),
                self.tp_degree,
            )

        for i in range(min(self.tp_degree, len(self.model_paths))):
            self.engine.load_model(self.model_paths[i])

    def _validate_device_residency(self, tensor: "torch.Tensor", *, name: str = "tensor") -> None:
        validate_tensor_device_residency(tensor, backend_expects_accelerator=True, name=name)

    def _convert_tensor(self, tensor: "torch.Tensor", *, expected_dtype, name: str = "tensor") -> int:
        self._validate_device_residency(tensor, name=name)
        return validate_tensor_for_zero_copy(tensor, expected_dtype=expected_dtype, name=name)

    def _launch_backend(self, input_tensor, output_tensor, seq_len: int):
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

        if self.wrapper is not None:
            if not all(hasattr(self.wrapper, attr) for attr in ("begin_capture", "end_capture", "launch", "invalidate")):
                raise RuntimeConfigurationError("wrapper backend does not expose the expected GraphWrapper methods.")

            with StreamContext() as stream:
                stream_ptr = stream.cuda_stream() if hasattr(stream, "cuda_stream") else 0
                batch_size = 1
                self.wrapper.begin_capture(stream_ptr, batch_size, seq_len)
                self.wrapper.end_capture(stream_ptr)
                self.wrapper.launch(
                    [stream_ptr],
                    [input_ptr, output_ptr, input_tensor, output_tensor],
                )
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.wrapper.invalidate()
            return

        if self.engine is None:
            raise RuntimeConfigurationError("No execution backend configured; provide either engine or wrapper.")

        if not hasattr(self.engine, "launch"):
            raise RuntimeConfigurationError("engine backend does not expose launch().")

        byte_size = int(input_tensor.numel() * input_tensor.element_size())
        self.engine.launch(input_ptr, output_ptr, seq_len, byte_size)
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for generating text.")

        max_new_tokens = _normalize_int(max_new_tokens, name="max_new_tokens", minimum=1)
        tokens = self.encode_prompt(prompt)
        vocab_size = self._resolve_vocab_size()

        for _ in range(max_new_tokens):
            device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
            input_tensor = torch.tensor(tokens, dtype=torch.int32, device=device)
            if not input_tensor.is_contiguous():
                input_tensor = input_tensor.contiguous()

            output_tensor = torch.empty((len(tokens), vocab_size), dtype=torch.float32, device=device)
            self._launch_backend(input_tensor, output_tensor, len(tokens))

            last_token_logits = output_tensor[-1, :]
            argmax_idx = int(torch.argmax(last_token_logits).item())
            tokens.append(argmax_idx)

        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        return bytes(t % 256 for t in tokens).decode("utf-8", errors="replace")
