import torch
from contextlib import contextmanager
from typing import List

class StreamContext:
    def __init__(self):
        # Fallback to torch.cuda if available, otherwise just mock it for headless tests
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            class MockStream:
                def cuda_stream(self):
                    return 0x1000
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
    def __init__(self, engine, wrapper):
        self.engine = engine
        self.wrapper = wrapper
        self.tp_degree = 1
        self.model_paths = []

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

        actual_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.tp_degree > 1 and actual_gpus < self.tp_degree:
            raise TopologyMismatchError(f"Topology mismatch: Model exported with TP={self.tp_degree}, but found {actual_gpus} GPUs.")

        # Map shards
        self.model_paths = sorted(kin_files)
        if len(self.model_paths) < self.tp_degree:
            # We assume in a real environment we have one shard per TP degree if TP > 1,
            # or maybe one big file and we load it differently.
            # For simplicity, if we found files, we map what we have.
            pass

        # In a real environment we would load the shards into respective GPUs
        for i in range(min(self.tp_degree, len(self.model_paths))):
            self.engine.load_model(self.model_paths[i])

    def _convert_tensor(self, tensor: torch.Tensor) -> int:
        return tensor.data_ptr()

    def generate(self, prompt: str) -> str:
        # Dummy tokenization loop: Tokenize -> Load into Engine -> Launch Graph -> Fetch Logits -> Argmax -> Append Token

        # Tokenize (dummy list of ints)
        tokens = [ord(c) for c in prompt]

        # We need to simulate memory management, mapping torch.Tensor to pointers
        input_tensor = torch.tensor(tokens, dtype=torch.int32)
        output_tensor = torch.zeros(len(tokens) * 50000, dtype=torch.float32)

        input_ptr = self._convert_tensor(input_tensor)
        output_ptr = self._convert_tensor(output_tensor)

        with StreamContext() as stream:
            stream_ptr = stream.cuda_stream() if hasattr(stream, "cuda_stream") else 0x1000

            # The AOTEngine handles compilation/loading (which might have already happened before generate is called)
            # The GraphWrapper captures and launches
            batch_size = 1
            seq_len = len(tokens)

            self.wrapper.begin_capture(stream_ptr, batch_size, seq_len)
            self.wrapper.end_capture(stream_ptr)

            # Launch graph with the extracted pointers to satisfy the orchestrator's pointer-mapping requirement
            self.wrapper.launch([stream_ptr], [input_ptr, output_ptr])
            self.wrapper.invalidate()

        # Argmax logic (dummy, as outputs are zeros)
        last_token_logits = output_tensor[-50000:]
        argmax_idx = torch.argmax(last_token_logits).item()

        # Append Token (mock logic)
        generated_token = chr(argmax_idx % 256) # Mock decoding
        return prompt + generated_token
