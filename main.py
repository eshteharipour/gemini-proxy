"""
Gemini OpenAI-compatible proxy service.

Key features:
- OpenAI /v1/chat/completions compatible endpoint
- Key ↔ Proxy pairing: each Gemini key has a dedicated proxy
- Exhausted keys are cooled down for a configurable window
- When ALL keys are exhausted the service sleeps briefly then retries
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gemini-proxy")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _parse_key_proxy_pairs() -> list[dict]:
    """
    Supports two config styles (can be mixed):

    Style 1 – indexed env vars:
        GEMINI_KEY_0=AIza...   GEMINI_PROXY_0=http://proxy0:8080
        GEMINI_KEY_1=AIza...   GEMINI_PROXY_1=http://proxy1:8080
        (GEMINI_PROXY_N is optional; if absent the pair uses no proxy)

    Style 2 – CSV pair list:
        GEMINI_PAIRS_CSV=key1|proxy1,key2|proxy2,key3   (proxy is optional)

    Style 3 – legacy separate CSVs (keys round-robin over proxies):
        GEMINI_API_CSV=key1,key2   PROXY_CSV=proxy1,proxy2
    """
    pairs: list[dict] = []

    # Style 1
    i = 0
    while True:
        key = os.environ.get(f"GEMINI_KEY_{i}")
        if key is None:
            break
        proxy = os.environ.get(f"GEMINI_PROXY_{i}")
        pairs.append({"key": key.strip(), "proxy": proxy.strip() if proxy else None})
        i += 1

    # Style 2
    csv_pairs = os.environ.get("GEMINI_PAIRS_CSV", "")
    for token in csv_pairs.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("|", 1)
        key = parts[0].strip()
        proxy = parts[1].strip() if len(parts) > 1 else None
        pairs.append({"key": key, "proxy": proxy or None})

    # Style 3 – legacy fallback
    if not pairs:
        keys = [
            k.strip()
            for k in os.environ.get("GEMINI_API_CSV", "").split(",")
            if k.strip()
        ]
        proxies = [
            p.strip() for p in os.environ.get("PROXY_CSV", "").split(",") if p.strip()
        ]
        for idx, key in enumerate(keys):
            proxy = proxies[idx % len(proxies)] if proxies else None
            pairs.append({"key": key, "proxy": proxy})

    if not pairs:
        raise RuntimeError(
            "No Gemini API keys configured. "
            "Set GEMINI_KEY_0 / GEMINI_PROXY_0 pairs, GEMINI_PAIRS_CSV, or GEMINI_API_CSV."
        )

    return pairs


# ---------------------------------------------------------------------------
# Key manager
# ---------------------------------------------------------------------------

COOLDOWN_SECONDS = int(os.environ.get("KEY_COOLDOWN_SECONDS", "60"))
ALL_EXHAUSTED_SLEEP = float(os.environ.get("ALL_EXHAUSTED_SLEEP_SECONDS", "5"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))


class KeyEntry:
    def __init__(self, key: str, proxy: str | None):
        self.key = key
        self.proxy = proxy  # e.g. "http://user:pass@host:port" or None
        self.exhausted_until: float = 0.0  # epoch seconds
        self.total_requests: int = 0
        self.total_errors: int = 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.exhausted_until

    def mark_exhausted(self, cooldown: float = COOLDOWN_SECONDS):
        self.exhausted_until = time.monotonic() + cooldown
        logger.warning("Key ...%s cooled down for %ss", self.key[-6:], cooldown)

    def make_http_client(self) -> httpx.AsyncClient:
        proxies = None
        if self.proxy:
            proxies = {"http://": self.proxy, "https://": self.proxy}
        return httpx.AsyncClient(proxies=proxies, timeout=120)


class KeyManager:
    def __init__(self, pairs: list[dict]):
        self.entries: list[KeyEntry] = [KeyEntry(p["key"], p["proxy"]) for p in pairs]
        self._idx = -1
        self._lock = asyncio.Lock()

    async def next_available(self) -> KeyEntry:
        """Round-robin over available keys; sleeps if all are exhausted."""
        attempts = 0
        while True:
            async with self._lock:
                for _ in range(len(self.entries)):
                    self._idx = (self._idx + 1) % len(self.entries)
                    entry = self.entries[self._idx]
                    if entry.available:
                        return entry

            # All keys exhausted
            attempts += 1
            wait_until = min(e.exhausted_until for e in self.entries)
            sleep_for = max(0.1, wait_until - time.monotonic())
            logger.warning(
                "All keys exhausted (attempt %d). Sleeping %.1fs …", attempts, sleep_for
            )
            await asyncio.sleep(sleep_for)

    def stats(self) -> list[dict]:
        now = time.monotonic()
        return [
            {
                "key_suffix": e.key[-6:],
                "proxy": e.proxy,
                "available": e.available,
                "cooldown_remaining": max(0.0, round(e.exhausted_until - now, 1)),
                "total_requests": e.total_requests,
                "total_errors": e.total_errors,
            }
            for e in self.entries
        ]


# ---------------------------------------------------------------------------
# Model name mapping  (OpenAI name → Gemini name)
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    # allow pass-through of native Gemini names
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-flash-latest": "gemini-flash-latest",
    "gemini-flash-lite-latest": "gemini-flash-lite-latest",
}

DEFAULT_GEMINI_MODEL = os.environ.get(
    "DEFAULT_GEMINI_MODEL", "gemini-flash-lite-latest"
)


def resolve_model(openai_model: str) -> str:
    return MODEL_MAP.get(openai_model, openai_model)


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response schemas
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str | list[Any]


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_GEMINI_MODEL
    messages: list[Message]
    temperature: float | None = 1.0
    max_tokens: int | None = 8192
    stream: bool = False
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | str | None = None
    n: int | None = 1
    response_format: dict | None = None


# ---------------------------------------------------------------------------
# Gemini REST helpers
# ---------------------------------------------------------------------------

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _openai_messages_to_gemini(
    messages: list[Message],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI messages → (system_instruction, gemini_contents)."""
    system_instruction = None
    contents = []

    for msg in messages:
        role = msg.role
        content = (
            msg.content
            if isinstance(msg.content, str)
            else _flatten_content(msg.content)
        )

        if role == "system":
            system_instruction = content
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:  # user / tool / function
            contents.append({"role": "user", "parts": [{"text": content}]})

    return system_instruction, contents


def _flatten_content(parts: list[Any]) -> str:
    texts = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            texts.append(p["text"])
        elif isinstance(p, str):
            texts.append(p)
    return " ".join(texts)


def _build_gemini_payload(req: ChatCompletionRequest) -> dict:
    system_instruction, contents = _openai_messages_to_gemini(req.messages)

    generation_config: dict[str, Any] = {
        "temperature": req.temperature,
        "maxOutputTokens": req.max_tokens or 8192,
    }
    if req.top_p is not None:
        generation_config["topP"] = req.top_p
    if req.top_k is not None:
        generation_config["topK"] = req.top_k
    if req.stop:
        generation_config["stopSequences"] = (
            [req.stop] if isinstance(req.stop, str) else req.stop
        )

    # JSON mode
    if req.response_format and req.response_format.get("type") == "json_object":
        generation_config["responseMimeType"] = "application/json"

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    return payload


def _gemini_to_openai_response(gemini_resp: dict, model: str) -> dict:
    candidates = gemini_resp.get("candidates", [])
    choices = []
    for i, cand in enumerate(candidates):
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        finish = cand.get("finishReason", "stop").lower()
        if finish == "max_tokens":
            finish = "length"
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        )

    usage = gemini_resp.get("usageMetadata", {})
    return {
        "id": f"chatcmpl-gemini-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }


# ---------------------------------------------------------------------------
# Core request handler (with retry + key rotation)
# ---------------------------------------------------------------------------


async def call_gemini(
    key_manager: KeyManager,
    req: ChatCompletionRequest,
) -> dict:
    gemini_model = resolve_model(req.model)
    payload = _build_gemini_payload(req)
    url_tmpl = f"{GEMINI_BASE}/models/{{model}}:generateContent"

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        entry = await key_manager.next_available()
        entry.total_requests += 1
        url = url_tmpl.format(model=gemini_model)

        logger.info(
            "Attempt %d/%d | key=...%s | proxy=%s | model=%s",
            attempt,
            MAX_RETRIES,
            entry.key[-6:],
            entry.proxy,
            gemini_model,
        )

        async with entry.make_http_client() as client:
            try:
                resp = await client.post(
                    url,
                    params={"key": entry.key},
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.RequestError as exc:
                entry.total_errors += 1
                entry.mark_exhausted(cooldown=10)
                last_error = exc
                logger.warning("Network error on attempt %d: %s", attempt, exc)
                continue

        if resp.status_code == 200:
            return _gemini_to_openai_response(resp.json(), gemini_model)

        body = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        err_msg = body.get("error", {}).get("message", resp.text[:200])
        status = resp.status_code

        entry.total_errors += 1
        logger.warning("Gemini error %d on attempt %d: %s", status, attempt, err_msg)

        if status == 429 or status == 503:
            # Rate-limited or overloaded → cooldown this key
            entry.mark_exhausted()
            last_error = HTTPException(status_code=429, detail=err_msg)
        elif status == 400:
            # Bad request – no point retrying
            raise HTTPException(status_code=400, detail=err_msg)
        elif status == 401 or status == 403:
            # Invalid key – long cooldown
            entry.mark_exhausted(cooldown=3600)
            last_error = HTTPException(
                status_code=502, detail=f"Gemini auth error: {err_msg}"
            )
        else:
            last_error = HTTPException(
                status_code=502, detail=f"Gemini error {status}: {err_msg}"
            )
            await asyncio.sleep(ALL_EXHAUSTED_SLEEP)

    raise last_error or HTTPException(
        status_code=502, detail="All retry attempts failed"
    )


# ---------------------------------------------------------------------------
# Streaming support
# ---------------------------------------------------------------------------


async def call_gemini_stream(
    key_manager: KeyManager,
    req: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    gemini_model = resolve_model(req.model)
    payload = _build_gemini_payload(req)
    url = f"{GEMINI_BASE}/models/{gemini_model}:streamGenerateContent"

    entry = await key_manager.next_available()
    entry.total_requests += 1

    async with entry.make_http_client() as client:
        async with client.stream(
            "POST",
            url,
            params={"key": entry.key, "alt": "sse"},
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                entry.total_errors += 1
                entry.mark_exhausted()
                body = await resp.aread()
                raise HTTPException(status_code=502, detail=body.decode())

            chunk_index = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return

                import json

                try:
                    gemini_chunk = json.loads(data_str)
                except Exception:
                    continue

                candidates = gemini_chunk.get("candidates", [])
                for cand in candidates:
                    parts = cand.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    finish = cand.get("finishReason")
                    openai_chunk = {
                        "id": f"chatcmpl-gemini-stream-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": gemini_model,
                        "choices": [
                            {
                                "index": chunk_index,
                                "delta": {"role": "assistant", "content": text},
                                "finish_reason": finish,
                            }
                        ],
                    }
                    chunk_index += 1
                    import json as _json

                    yield f"data: {_json.dumps(openai_chunk)}\n\n"


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

key_manager: KeyManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global key_manager
    pairs = _parse_key_proxy_pairs()
    key_manager = KeyManager(pairs)
    logger.info("Loaded %d key-proxy pair(s).", len(pairs))
    for p in pairs:
        logger.info("  key=...%s  proxy=%s", p["key"][-6:], p["proxy"])
    yield


app = FastAPI(title="Gemini → OpenAI Proxy", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models():
    models = list(set(MODEL_MAP.values()))
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "google"}
            for m in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    if key_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if req.stream:
        return StreamingResponse(
            call_gemini_stream(key_manager, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await call_gemini(key_manager, req)
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    if key_manager is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    available = sum(1 for e in key_manager.entries if e.available)
    return {
        "status": "ok",
        "keys_available": available,
        "keys_total": len(key_manager.entries),
    }


@app.get("/stats")
async def stats():
    if key_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"key_stats": key_manager.stats()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
