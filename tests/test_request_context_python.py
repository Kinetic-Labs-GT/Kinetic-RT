import datetime
import unittest
import python.kinetic_rt as kinetic_rt


class TestRequestContextPython(unittest.TestCase):

    def test_import_and_enums(self):
        # Verify enums access at module level
        self.assertTrue(hasattr(kinetic_rt, "RequestState"))
        self.assertTrue(hasattr(kinetic_rt, "CompletionReason"))
        self.assertTrue(hasattr(kinetic_rt, "Owner"))

        self.assertEqual(kinetic_rt.RequestState.RECEIVED, kinetic_rt.RequestState.RECEIVED)
        self.assertNotEqual(kinetic_rt.RequestState.RECEIVED, kinetic_rt.RequestState.COMPLETE)
        self.assertEqual(kinetic_rt.CompletionReason.STREAMING, kinetic_rt.CompletionReason.STREAMING)
        self.assertEqual(kinetic_rt.Owner.Ingress, kinetic_rt.Owner.Ingress)

    def test_request_context_config_default(self):
        config = kinetic_rt.RequestContextConfig()
        config.request_id = "req_test_001"
        config.prompt_text = "What is Kinetic-RT?"
        config.prompt_tokens = [1, 100, 200]
        config.max_new_tokens = 32
        config.temperature = 0.8
        config.top_p = 0.95
        config.top_k = 50
        config.stop_token_ids = [2]
        config.backend_hint = "AOTEngine"
        config.timeout = datetime.timedelta(seconds=1)
        config.initial_owner = kinetic_rt.Owner.Ingress

        ctx = kinetic_rt.RequestContext(config)

        self.assertEqual(ctx.request_id(), "req_test_001")
        self.assertEqual(ctx.prompt_text(), "What is Kinetic-RT?")
        self.assertEqual(ctx.prompt_tokens(), [1, 100, 200])
        self.assertEqual(ctx.max_new_tokens(), 32)
        self.assertAlmostEqual(ctx.temperature(), 0.8, places=5)
        self.assertAlmostEqual(ctx.top_p(), 0.95, places=5)
        self.assertEqual(ctx.top_k(), 50)
        self.assertEqual(ctx.stop_token_ids(), [2])
        self.assertEqual(ctx.backend_hint(), "AOTEngine")

        # State accessors
        self.assertEqual(ctx.state(), kinetic_rt.RequestState.RECEIVED)
        self.assertEqual(ctx.finish_reason(), kinetic_rt.CompletionReason.STREAMING)
        self.assertEqual(ctx.current_owner(), kinetic_rt.Owner.Ingress)

    def test_request_context_nested_config(self):
        config = kinetic_rt.RequestContext.Config()
        config.request_id = "req_test_002"
        ctx = kinetic_rt.RequestContext(config)
        self.assertEqual(ctx.request_id(), "req_test_002")

    def test_cancellation(self):
        config = kinetic_rt.RequestContextConfig()
        config.request_id = "req_cancel_001"
        ctx = kinetic_rt.RequestContext(config)

        self.assertFalse(ctx.is_cancellation_requested())
        ctx.request_cancellation()
        self.assertTrue(ctx.is_cancellation_requested())

    def test_invalid_config(self):
        config = kinetic_rt.RequestContextConfig()
        config.request_id = ""  # empty ID should raise error
        with self.assertRaises(Exception):
            kinetic_rt.RequestContext(config)


if __name__ == "__main__":
    unittest.main()
