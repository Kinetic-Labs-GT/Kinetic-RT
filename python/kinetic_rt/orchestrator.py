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

import os

class KineticRuntime:
    def __init__(self, engine, wrapper=None, tokenizer_name=None):
        self.engine = engine # Acts as router if wrapper is None
        self.wrapper = wrapper
        self.tp_degree = 1
        self.model_paths = []

        # FIX 4: Load a real tokenizer when available
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

    def _convert_tensor(self, tensor: torch.Tensor) -> int:
        return tensor.data_ptr()

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for generating text.")

        # Tokenize -> Load into Engine -> Launch Graph -> Fetch Logits -> Argmax -> Append Token

        # FIX 4: Use real tokenizer when available, byte-level fallback otherwise
        if self.tokenizer:
            tokens = self.tokenizer.encode(prompt)
        else:
            logger.warning("No tokenizer available, using byte-level fallback")
            tokens = list(prompt.encode('utf-8'))

        vocab_size = 50257 # Standard GPT-2/SmolLM vocabulary size

        for _ in range(max_new_tokens):
            input_tensor = torch.tensor(tokens, dtype=torch.int32)

            # Use actual device tensor allocation
            device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
            input_tensor = input_tensor.to(device)
            output_tensor = torch.empty((len(tokens), vocab_size), dtype=torch.float32, device=device)

            # Convert tensors to pointers for the C++ Pybind11 wrapper
            input_ptr = self._convert_tensor(input_tensor)
            output_ptr = self._convert_tensor(output_tensor)

            with StreamContext() as stream:
                stream_ptr = stream.cuda_stream() if hasattr(stream, "cuda_stream") else 0

                batch_size = 1
                seq_len = len(tokens)

                if self.wrapper:
                    self.wrapper.begin_capture(stream_ptr, batch_size, seq_len)
                    self.wrapper.end_capture(stream_ptr)

                    # Execute graph using pointers
                    # We also pass the original python objects to ensure they are pinned/kept alive
                    self.wrapper.launch([stream_ptr], [input_ptr, output_ptr, input_tensor, output_tensor])

                    # We strictly synchronize to ensure logits are fully written by the GPU
                    if HAS_TORCH and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    self.wrapper.invalidate()
                else:
                    # Routed execution path handling HardwareRouter
                    self.engine.launch(input_ptr, output_ptr, seq_len)
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
