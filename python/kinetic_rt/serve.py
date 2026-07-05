from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict, deque
from threading import Lock
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ._core import HardwareRouter, InferenceQueue, InferenceWorker
from .orchestrator import KineticRuntime, RuntimeConfigurationError, validate_tensor_device_residency, validate_tensor_for_zero_copy

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = 100


class ChatCompletionChoiceDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: int = 100


def _normalize_positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer-compatible value.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return normalized


class KineticServer:
    def __init__(
        self,
        *,
        tokenizer_name: Optional[str] = None,
        vocab_size: Optional[int] = None,
        request_timeout_s: float = 30.0,
        poll_interval_s: float = 0.001,
    ):
        self.app = FastAPI(title="Kinetic-RT Streaming API")
        self.router = HardwareRouter()
        self.queue = InferenceQueue()
        self.worker = InferenceWorker(self.queue, self.router)
        self.runtime = KineticRuntime(self.router, None, tokenizer_name=tokenizer_name, vocab_size=vocab_size)

        self.request_timeout_s = float(request_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be > 0.")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0.")

        self.pinned_tensors: Dict[str, Any] = {}
        self._pending_responses: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
        self._pending_lock = Lock()

        self.app.post("/v1/chat/completions")(self.openai_chat_completions)
        self.app.post("/v1/messages")(self.anthropic_messages)

    def _normalize_request_tokens(self, max_tokens: Optional[int], default: int = 10) -> int:
        value = default if max_tokens is None else max_tokens
        return _normalize_positive_int(value, name="max_tokens", minimum=1)

    def _store_pending_response(self, response: Dict[str, Any]) -> None:
        request_id = response.get("request_id")
        if not request_id:
            return
        with self._pending_lock:
            self._pending_responses[request_id].append(response)

    def _pop_pending_response(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._pending_lock:
            queue = self._pending_responses.get(request_id)
            if not queue:
                return None
            response = queue.popleft()
            if not queue:
                self._pending_responses.pop(request_id, None)
            return response

    def _encode_prompt(self, prompt: str) -> List[int]:
        return self.runtime.encode_prompt(prompt)

    async def _async_generate(
        self,
        prompt: str,
        max_tokens: int,
        req_id: str,
        http_request: Optional[Request] = None,
    ) -> AsyncGenerator[str, None]:
        tokens = self._encode_prompt(prompt)

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor = torch.tensor(tokens, dtype=torch.int32, device=device)
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        self.pinned_tensors[req_id] = input_tensor

        try:
            desc_name = f"serve._async_generate[{req_id}].input_tensor"
            validate_tensor_device_residency(input_tensor, backend_expects_accelerator=True, name=desc_name)
            validate_tensor_for_zero_copy(input_tensor, expected_dtype=torch.int32, name=desc_name)

            self.queue.submit(input_tensor.data_ptr(), len(tokens), max_tokens, req_id)

            deadline = asyncio.get_running_loop().time() + self.request_timeout_s

            while True:
                if http_request is not None and await http_request.is_disconnected():
                    logger.info("Request %s disconnected; aborting generation.", req_id)
                    return

                pending = self._pop_pending_response(req_id)
                if pending is not None:
                    resp = pending
                else:
                    resp = None
                    try:
                        resp = self.queue.poll()
                    except Exception as exc:
                        raise RuntimeError(f"Queue poll failed for request {req_id}: {exc}") from exc

                    if resp:
                        if resp.get("request_id") != req_id:
                            self._store_pending_response(resp)
                            resp = None

                if resp:
                    if resp.get("error"):
                        raise RuntimeError(f"Native generation error for request {req_id}: {resp['error']}")
                    token = resp.get("token", "")
                    if resp.get("is_finished"):
                        if token:
                            yield token
                        break
                    if token:
                        yield token
                    continue

                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError(f"Generation timed out after {self.request_timeout_s:.2f}s for request {req_id}.")

                await asyncio.sleep(self.poll_interval_s)
        finally:
            self.pinned_tensors.pop(req_id, None)
            with self._pending_lock:
                self._pending_responses.pop(req_id, None)

    async def openai_chat_completions(self, request: ChatCompletionRequest, http_request: Request):
        prompt = " ".join(m.content for m in request.messages)
        max_tokens = self._normalize_request_tokens(request.max_tokens, default=10)

        if request.stream:

            async def generate():
                import time
                import uuid

                req_id = str(uuid.uuid4())
                created = int(time.time())
                yield (
                    f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                )

                try:
                    async for token in self._async_generate(prompt, max_tokens, req_id, http_request=http_request):
                        if token == "<|endoftext|>":
                            break
                        chunk = ChatCompletionChunk(
                            id="chatcmpl-123",
                            created=int(time.time()),
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChoiceDelta(content=token),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                except TimeoutError as exc:
                    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': ''}, 'finish_reason': 'timeout'}], 'error': str(exc)})}\n\n"
                    return

                yield (
                    f"data: {json.dumps({'id': 'chatcmpl-123', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        req_id = __import__("uuid").uuid4().hex
        tokens_list = []
        try:
            async for token in self._async_generate(prompt, max_tokens, req_id, http_request=http_request):
                if token == "<|endoftext|>":
                    break
                tokens_list.append(token)
        except TimeoutError as exc:
            raise HTTPException(status_code=408, detail=str(exc)) from exc
        except RuntimeConfigurationError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        full_text = "".join(tokens_list)
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": __import__("time").time_ns() // 1_000_000_000,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
        }

    async def anthropic_messages(self, request: AnthropicMessageRequest, http_request: Request):
        prompt = " ".join(m.content for m in request.messages)
        max_tokens = self._normalize_request_tokens(request.max_tokens, default=10)

        if request.stream:

            async def generate():
                import time
                import uuid

                req_id = str(uuid.uuid4())
                yield "event: message_start\ndata: {\"type\": \"message_start\", \"message\": {\"id\": \"msg_1\", \"type\": \"message\", \"role\": \"assistant\", \"content\": [], \"model\": \"claude-3\"}}\n\n"
                yield "event: content_block_start\ndata: {\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"text\", \"text\": \"\"}}\n\n"

                try:
                    async for token in self._async_generate(prompt, max_tokens, req_id, http_request=http_request):
                        if token == "<|endoftext|>":
                            break
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"
                except TimeoutError as exc:
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                    return

                yield "event: content_block_stop\ndata: {\"type\": \"content_block_stop\", \"index\": 0}\n\n"
                yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        req_id = __import__("uuid").uuid4().hex
        tokens_list = []
        try:
            async for token in self._async_generate(prompt, max_tokens, req_id, http_request=http_request):
                if token == "<|endoftext|>":
                    break
                tokens_list.append(token)
        except TimeoutError as exc:
            raise HTTPException(status_code=408, detail=str(exc)) from exc
        except RuntimeConfigurationError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        full_text = "".join(tokens_list)
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": request.model,
            "content": [{"type": "text", "text": full_text}],
        }


def serve(port: int = 8000):
    import uvicorn

    server = KineticServer()
    logger.info("Starting Kinetic-RT Streaming API on port %s", port)
    uvicorn.run(server.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    serve()
