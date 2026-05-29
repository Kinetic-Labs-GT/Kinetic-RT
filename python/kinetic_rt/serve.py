import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from .orchestrator import KineticRuntime
from ._core import HardwareRouter, InferenceQueue, InferenceWorker

logger = logging.getLogger(__name__)

# Official OpenAI schemas
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

# Anthropic schemas
class AnthropicMessageRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: int = 100

class KineticServer:
    def __init__(self):
        self.app = FastAPI(title="Kinetic-RT Streaming API")
        self.router = HardwareRouter()

        # Core C++ Inference Queue linking Async Web Layer directly to Native Hardware Router ThreadPools
        self.queue = InferenceQueue()

        # Persistent C++ background worker consuming the lock-free queue
        self.worker = InferenceWorker(self.queue, self.router)

        # For non-streaming fallback only, we preserve direct generation access
        self.runtime = KineticRuntime(self.router, None)

        # Setup routing
        self.app.post("/v1/chat/completions")(self.openai_chat_completions)
        self.app.post("/v1/messages")(self.anthropic_messages)

    async def _async_generate(self, prompt: str, max_tokens: int, req_id: str) -> AsyncGenerator[str, None]:
        # Genuine C++ Native lock-free async handoff
        self.queue.submit(prompt, max_tokens, req_id)

        # We loop and poll the lock-free C++ ring buffer natively from asyncio loop
        # ensuring the Python GIL is yielded naturally without blocking via `asyncio.sleep`
        # and entirely avoiding dynamic thread pools in Python context.
        while True:
            resp = self.queue.poll()
            if resp:
                if resp["request_id"] == req_id:
                    if resp["is_finished"]:
                        break
                    yield resp["token"]
            else:
                await asyncio.sleep(0.001)

    async def openai_chat_completions(self, request: ChatCompletionRequest):
        prompt = " ".join([m.content for m in request.messages])

        if request.stream:
            async def generate():
                import time
                import uuid
                req_id = str(uuid.uuid4())
                yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

                async for token in self._async_generate(prompt, request.max_tokens or 10, req_id):
                    chunk = ChatCompletionChunk(
                        id="chatcmpl-123",
                        created=int(time.time()),
                        model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChoiceDelta(content=token),
                            finish_reason=None
                        )]
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                yield f"data: {json.dumps({'id': 'chatcmpl-123', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Non-streaming fallback
            full_text = await asyncio.to_thread(self.runtime.generate, prompt, request.max_tokens or 10)
            return {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text[len(prompt):]},
                    "finish_reason": "stop"
                }]
            }

    async def anthropic_messages(self, request: AnthropicMessageRequest):
        prompt = " ".join([m.content for m in request.messages])

        if request.stream:
            async def generate():
                import uuid
                req_id = str(uuid.uuid4())
                yield "event: message_start\ndata: {\"type\": \"message_start\", \"message\": {\"id\": \"msg_1\", \"type\": \"message\", \"role\": \"assistant\", \"content\": [], \"model\": \"claude-3\"}}\n\n"
                yield "event: content_block_start\ndata: {\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"text\", \"text\": \"\"}}\n\n"

                async for token in self._async_generate(prompt, request.max_tokens, req_id):
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"

                yield "event: content_block_stop\ndata: {\"type\": \"content_block_stop\", \"index\": 0}\n\n"
                yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            full_text = await asyncio.to_thread(self.runtime.generate, prompt, request.max_tokens)
            return {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": request.model,
                "content": [{"type": "text", "text": full_text[len(prompt):]}]
            }

def serve(port: int = 8000):
    import uvicorn
    server = KineticServer()
    logger.info(f"Starting Kinetic-RT Streaming API on port {port}")
    uvicorn.run(server.app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    serve()
