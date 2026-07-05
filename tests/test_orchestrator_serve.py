import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import pytest
import torch
from fastapi.testclient import TestClient

# Mock the C++ core extension before importing serve.py to avoid crashes in headless environments
mock_core = MagicMock()
sys.modules["python.kinetic_rt._core"] = mock_core

# Mock out the C++ classes in the module path
import python.kinetic_rt.serve as serve_module

from python.kinetic_rt.orchestrator import (
    KineticRuntime,
    ContextWindowExceededError,
    DeviceMismatchError,
    TensorValidationError,
    validate_tensor_device_residency,
    validate_tensor_for_zero_copy,
    allows_mock_device_residency,
)
from python.kinetic_rt.serve import KineticServer


class TestOrchestratorAndServe(unittest.TestCase):
    def setUp(self):
        # Ensure mock flag is enabled for tests
        self._orig_mock_hip = os.environ.get("MOCK_HIP")
        self._orig_allow_mocks = os.environ.get("KINETIC_ALLOW_MOCKS")
        os.environ["MOCK_HIP"] = "1"
        os.environ["KINETIC_ALLOW_MOCKS"] = "1"

        # Setup patches for core objects inside KineticServer
        self.router_patcher = patch("python.kinetic_rt.serve.HardwareRouter")
        self.queue_patcher = patch("python.kinetic_rt.serve.InferenceQueue")
        self.worker_patcher = patch("python.kinetic_rt.serve.InferenceWorker")

        self.mock_router_cls = self.router_patcher.start()
        self.mock_queue_cls = self.queue_patcher.start()
        self.mock_worker_cls = self.worker_patcher.start()

        # Mock instances
        self.mock_router = self.mock_router_cls.return_value
        self.mock_queue = self.mock_queue_cls.return_value
        self.mock_worker = self.mock_worker_cls.return_value

    def tearDown(self):
        self.router_patcher.stop()
        self.queue_patcher.stop()
        self.worker_patcher.stop()

        if self._orig_mock_hip is not None:
            os.environ["MOCK_HIP"] = self._orig_mock_hip
        else:
            os.environ.pop("MOCK_HIP", None)

        if self._orig_allow_mocks is not None:
            os.environ["KINETIC_ALLOW_MOCKS"] = self._orig_allow_mocks
        else:
            os.environ.pop("KINETIC_ALLOW_MOCKS", None)

    def test_mock_device_residency_enabled(self):
        self.assertTrue(allows_mock_device_residency())

    def test_tensor_device_residency_validation(self):
        # Under mock mode, validate_tensor_device_residency should return without error
        cpu_tensor = torch.zeros(10, dtype=torch.int32)
        validate_tensor_device_residency(cpu_tensor, name="test_tensor")

        # Temporarily disable mocks to test boundary mismatch exception
        os.environ["MOCK_HIP"] = "0"
        os.environ["KINETIC_ALLOW_MOCKS"] = "0"

        with self.assertRaises(DeviceMismatchError):
            validate_tensor_device_residency(cpu_tensor, name="test_tensor")

        # Set back to mock mode
        os.environ["MOCK_HIP"] = "1"
        os.environ["KINETIC_ALLOW_MOCKS"] = "1"

    def test_tensor_validation_for_zero_copy(self):
        # Contiguous, matching dtype, zero offset, non-zero element tensor should pass
        tensor = torch.zeros(10, dtype=torch.int32)
        ptr = validate_tensor_for_zero_copy(tensor, expected_dtype=torch.int32, name="test_tensor")
        self.assertEqual(ptr, tensor.data_ptr())

        # Dtype mismatch
        with self.assertRaises(TensorValidationError):
            validate_tensor_for_zero_copy(tensor, expected_dtype=torch.float32, name="test_tensor")

        # Zero size tensor
        empty_tensor = torch.zeros(0, dtype=torch.int32)
        with self.assertRaises(TensorValidationError):
            validate_tensor_for_zero_copy(empty_tensor, expected_dtype=torch.int32, name="test_tensor")

        # Non-contiguous tensor
        non_contiguous = torch.zeros((5, 2), dtype=torch.int32)[:, 0]
        self.assertFalse(non_contiguous.is_contiguous())
        with self.assertRaises(TensorValidationError):
            validate_tensor_for_zero_copy(non_contiguous, expected_dtype=torch.int32, name="test_tensor")

    def test_runtime_max_seq_len_resolution(self):
        runtime = KineticRuntime(engine=None, max_seq_len=512)
        self.assertEqual(runtime.max_seq_len, 512)

        # Invalid values fallback to default
        runtime_invalid = KineticRuntime(engine=None, max_seq_len=-5)
        self.assertEqual(runtime_invalid.max_seq_len, 2048)

    def test_runtime_context_window_exceeded(self):
        runtime = KineticRuntime(engine=None, max_seq_len=1)
        # prompt that yields tokens under tokenizer
        prompt = "abcdefghij"
        with self.assertRaises(ContextWindowExceededError):
            runtime.generate(prompt, max_new_tokens=5)

    def test_serve_endpoints_success(self):
        # Setup mock queue behavior
        responses = [
            {"request_id": "test_req_id", "token": "hello", "is_finished": False},
            {"request_id": "test_req_id", "token": " world", "is_finished": True},
        ]
        self.mock_queue.poll.side_effect = lambda: responses.pop(0) if responses else None

        # Custom UUID generation to match the mock response's request_id
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = MagicMock(hex="test_req_id")
            mock_uuid.return_value.__str__.return_value = "test_req_id"

            server = KineticServer(max_seq_len=100)
            client = TestClient(server.app)

            # Non-streaming OpenAI Chat Completion
            req_body = {
                "model": "smollm",
                "messages": [{"role": "user", "content": "test prompt"}],
                "stream": False,
                "max_tokens": 5
            }
            resp = client.post("/v1/chat/completions", json=req_body)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["choices"][0]["message"]["content"], "hello world")

    def test_serve_context_window_exceeded_openai(self):
        # Set max_seq_len to a small value so a normal prompt exceeds it
        server = KineticServer(max_seq_len=2)
        client = TestClient(server.app)

        req_body = {
            "model": "smollm",
            "messages": [{"role": "user", "content": "hello world test"}],
            "stream": False,
            "max_tokens": 5
        }
        resp = client.post("/v1/chat/completions", json=req_body)
        self.assertEqual(resp.status_code, 413)
        self.assertIn("Context window limit exceeded", resp.json()["detail"])

        # Stream version
        req_body["stream"] = True
        resp_stream = client.post("/v1/chat/completions", json=req_body)
        self.assertEqual(resp_stream.status_code, 200)
        lines = resp_stream.text.strip().split("\n\n")
        # The first chunk is roles, then error, then [DONE]
        error_chunk_found = False
        for line in lines:
            if line.startswith("data: ") and "context_length_exceeded" in line:
                error_chunk_found = True
        self.assertTrue(error_chunk_found)

    def test_serve_context_window_exceeded_anthropic(self):
        server = KineticServer(max_seq_len=2)
        client = TestClient(server.app)

        req_body = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "hello world test"}],
            "stream": False,
            "max_tokens": 5
        }
        resp = client.post("/v1/messages", json=req_body)
        self.assertEqual(resp.status_code, 413)
        self.assertIn("Context window limit exceeded", resp.json()["detail"])

        # Stream version
        req_body["stream"] = True
        resp_stream = client.post("/v1/messages", json=req_body)
        self.assertEqual(resp_stream.status_code, 200)
        lines = resp_stream.text.strip().split("\n\n")
        error_chunk_found = False
        for line in lines:
            if "context_length_exceeded" in line:
                error_chunk_found = True
        self.assertTrue(error_chunk_found)
