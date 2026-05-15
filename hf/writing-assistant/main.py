# writing-assistant/main.py
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, field_validator

import config
from model import load_model, stream_response
from prompts import build_messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.tokenizer = load_model(config.MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    text: str
    mode: Literal["rewrite", "summarize", "make_formal", "make_casual"]

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text()


@app.post("/generate")
async def generate(request: GenerateRequest):
    messages = build_messages(request.text, request.mode)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_model():
        try:
            for text, chunk in stream_response(
                app.state.model, app.state.tokenizer, messages
            ):
                loop.call_soon_threadsafe(queue.put_nowait, text)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"error": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, run_model)

    async def event_stream():
        while True:
            token = await queue.get()
            if token is None:
                yield "event: done\ndata: \n\n"
                break
            if isinstance(token, dict) and "error" in token:
                yield f"event: error\ndata: {token['error']}\n\n"
                break
            yield f"data: {token}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
