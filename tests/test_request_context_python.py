"""Smoke test for RequestContext Python bindings (Chapter 2 Phase 0).

Verifies reachability: import, construction, and basic cancellation.
Does not test runtime integration, scheduler, or inference.

This test mocks heavy runtime dependencies (torch, fastapi, etc.) that are
required by serve.py / orchestrator.py but irrelevant to RequestContext
binding verification.  The pattern follows tests/test_validation.py.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Stub out heavy transitive dependencies so that importing the package does
# not require a GPU runtime or web framework.  Only the native _core
# extension (already compiled) is loaded for real.
_STUBS = [
    "torch", "torch.cuda",
    "triton", "triton.language",
    "fastapi", "fastapi.responses",
    "pydantic",
    "uvicorn",
    "sse_starlette", "sse_starlette.sse",
    "transformers",
]
_saved = {}
for _mod in _STUBS:
    if _mod not in sys.modules:
        _saved[_mod] = None
        sys.modules[_mod] = MagicMock()

import python.kinetic_rt as kinetic_rt  # noqa: E402


class TestRequestContextSmoke(unittest.TestCase):

    def test_enum_values_exist(self):
        """RequestState, CompletionReason, Owner enums are importable and distinct."""
        self.assertNotEqual(kinetic_rt.RequestState.RECEIVED, kinetic_rt.RequestState.COMPLETE)
        self.assertNotEqual(kinetic_rt.CompletionReason.STREAMING, kinetic_rt.CompletionReason.EOS)
        self.assertNotEqual(kinetic_rt.Owner.Ingress, kinetic_rt.Owner.Backend)

    def test_construct_and_read_back(self):
        """Construct a RequestContext via Config and read back request_id."""
        config = kinetic_rt.RequestContextConfig()
        config.request_id = "smoke_001"
        config.prompt_text = "hello"
        ctx = kinetic_rt.RequestContext(config)
        self.assertEqual(ctx.request_id(), "smoke_001")
        self.assertEqual(ctx.state(), kinetic_rt.RequestState.RECEIVED)

    def test_cancellation_flag(self):
        """request_cancellation / is_cancellation_requested round-trip."""
        config = kinetic_rt.RequestContextConfig()
        config.request_id = "smoke_cancel"
        ctx = kinetic_rt.RequestContext(config)
        self.assertFalse(ctx.is_cancellation_requested())
        ctx.request_cancellation()
        self.assertTrue(ctx.is_cancellation_requested())

    def test_invalid_config_raises(self):
        """Empty request_id should raise."""
        config = kinetic_rt.RequestContextConfig()
        config.request_id = ""
        with self.assertRaises(Exception):
            kinetic_rt.RequestContext(config)


if __name__ == "__main__":
    unittest.main()
