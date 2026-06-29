import os
import pytest
from fastapi.testclient import TestClient
from python.kinetic_rt.serve import KineticServer

@pytest.mark.skip(reason="Legacy integration test requires uninitialized theater paths; to be refactored in Phase 6 Serving rewrite.")
def test_api_server_endpoints():
    # Set custom pool configurations to verify dynamic environment parsing
    os.environ["KINETIC_KV_POOL_SIZE"] = "2048"
    os.environ["KINETIC_BLOCK_SIZE"] = "32"

    server = KineticServer()
    client = TestClient(server.app)

    # 1. Test OpenAI completion (non-streaming)
    req_body = {
        "model": "smollm",
        "messages": [{"role": "user", "content": "The capital of India is "}],
        "stream": False,
        "max_tokens": 5
    }
    resp = client.post("/v1/chat/completions", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["content"] != ""

    # 2. Test OpenAI completion (streaming)
    req_body["stream"] = True
    resp = client.post("/v1/chat/completions", json=req_body)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # 3. Test Anthropic messages (non-streaming)
    anthropic_body = {
        "model": "claude",
        "messages": [{"role": "user", "content": "The capital of India is "}],
        "stream": False,
        "max_tokens": 5
    }
    resp = client.post("/v1/messages", json=anthropic_body)
    assert resp.status_code == 200
    data = resp.json()
    assert "content" in data
    assert len(data["content"]) > 0
    assert data["content"][0]["text"] != ""

    # 4. Test Anthropic messages (streaming)
    anthropic_body["stream"] = True
    resp = client.post("/v1/messages", json=anthropic_body)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
