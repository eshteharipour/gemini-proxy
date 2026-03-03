"""
Gemini → OpenAI-compatible proxy service.

Features:
- OpenAI /v1/chat/completions  (streaming supported)
- Structured output via response_schema  (pass as `schema` field or `response_format.json_schema`)
- Image inference  (OpenAI image_url content blocks → Gemini inline image parts)
- Token counting  POST /v1/token/count
- Key ↔ Proxy pairing: each Gemini key has its own dedicated proxy
- Per-key cooldown on 429/503; sleeps until a key recovers when all are exhausted
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
#
# DEBUG_MODE=1  → INFO+ goes to stderr AND every request/response is written
#                 as a structured JSON record to the file set by DEBUG_LOG_FILE
#                 (default: gemini_proxy_debug.log), one record per line (JSONL).
#
# DEBUG_MODE=0  → normal stderr-only logging, no file, no payload capture.
# ---------------------------------------------------------------------------

DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "0").strip() in ("1", "true", "yes")
DEBUG_LOG_FILE: str = os.environ.get("DEBUG_LOG_FILE", "gemini_proxy_debug.log")

_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

if DEBUG_MODE:
    # Console handler — same as before
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter(_fmt))

    # File handler — plain text log lines (not the JSONL payload records)
    _file_handler = logging.FileHandler(DEBUG_LOG_FILE, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(_fmt))

    logging.basicConfig(level=logging.DEBUG, handlers=[_console, _file_handler])
else:
    logging.basicConfig(level=logging.INFO, format=_fmt)

logger = logging.getLogger("gemini-proxy")

if DEBUG_MODE:
    logger.info("Debug mode ON — request/response payloads → %s", DEBUG_LOG_FILE)


# ---------------------------------------------------------------------------
# Debug request/response file logger
# ---------------------------------------------------------------------------

import threading as _threading

_debug_lock = _threading.Lock()


def _redact_key(key: str) -> str:
    """Keep first 4 and last 4 chars, mask the middle."""
    if len(key) <= 8:
        return "****"
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def debug_log(record: dict) -> None:
    """
    Append a single JSONL record to DEBUG_LOG_FILE.
    Each record contains:
        ts          – ISO-8601 timestamp
        event       – "request" | "response" | "stream_start" | "stream_end" | "error"
        endpoint    – Gemini REST path called
        key_suffix  – last 6 chars of the API key used
        payload     – outgoing request body  (event=request)
        response    – Gemini response body   (event=response)
        status      – HTTP status code
        elapsed_ms  – wall-clock ms for the round-trip
        ...
    Image inlineData is truncated to avoid massive log files.
    """
    if not DEBUG_MODE:
        return

    # Truncate base64 image blobs so logs stay readable
    def _scrub(obj: Any, depth: int = 0) -> Any:
        if depth > 10:
            return obj
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "data" and isinstance(v, str) and len(v) > 80:
                    out[k] = v[:40] + f"…[{len(v)} chars]"
                else:
                    out[k] = _scrub(v, depth + 1)
            return out
        if isinstance(obj, list):
            return [_scrub(i, depth + 1) for i in obj]
        return obj

    record = _scrub(record)
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))

    line = json.dumps(record, ensure_ascii=False)
    with _debug_lock:
        with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COOLDOWN_SECONDS = int(os.environ.get("KEY_COOLDOWN_SECONDS", "60"))
# Optional proxy for general outbound HTTP (image downloads, etc.) — not used for Gemini.
# Set GENERAL_PROXY=http://host:port in your env or .env file.
GENERAL_PROXY: str | None = os.environ.get("GENERAL_PROXY") or None
ALL_EXHAUSTED_SLEEP = float(os.environ.get("ALL_EXHAUSTED_SLEEP_SECONDS", "5"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
DEFAULT_GEMINI_MODEL = os.environ.get(
    "DEFAULT_GEMINI_MODEL", "gemini-flash-lite-latest"
)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _parse_key_proxy_pairs() -> list[dict]:
    """
    Three config styles (all can coexist):

    Style 1 – indexed env vars (most explicit, recommended):
        GEMINI_KEY_0=AIza...   GEMINI_PROXY_0=http://user:pass@proxy0:8080
        GEMINI_KEY_1=AIza...   # no GEMINI_PROXY_1 → direct connection

    Style 2 – single CSV:
        GEMINI_PAIRS_CSV=key1|http://proxy1:8080,key2|http://proxy2:8080,key3

    Style 3 – legacy separate CSVs (keys round-robin over proxies):
        GEMINI_API_CSV=key1,key2   PROXY_CSV=http://proxy1:8080,http://proxy2:8080
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
    for token in os.environ.get("GEMINI_PAIRS_CSV", "").split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split("|", 1)
        pairs.append(
            {
                "key": parts[0].strip(),
                "proxy": parts[1].strip() if len(parts) > 1 else None,
            }
        )

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
            "Set GEMINI_KEY_0/GEMINI_PROXY_0 pairs, GEMINI_PAIRS_CSV, or GEMINI_API_CSV."
        )
    return pairs


# ---------------------------------------------------------------------------
# Key / Proxy management
# ---------------------------------------------------------------------------


class KeyEntry:
    def __init__(self, key: str, proxy: str | None):
        self.key = key
        self.proxy = proxy  # e.g. "http://user:pass@host:port"
        self.exhausted_until: float = 0.0
        self.total_requests: int = 0
        self.total_errors: int = 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.exhausted_until

    def mark_exhausted(self, cooldown: float = COOLDOWN_SECONDS) -> None:
        self.exhausted_until = time.monotonic() + cooldown
        logger.warning("Key …%s cooled down for %.0fs", self.key[-6:], cooldown)

    def make_http_client(self, timeout: float = 120.0) -> httpx.AsyncClient:
        # httpx.AsyncClient uses `proxy=` (singular) since v0.23
        return httpx.AsyncClient(
            proxy=self.proxy,  # None → no proxy (direct connection)
            timeout=timeout,
        )


class KeyManager:
    def __init__(self, pairs: list[dict]):
        self.entries: list[KeyEntry] = [KeyEntry(p["key"], p["proxy"]) for p in pairs]
        self._idx = -1
        self._lock = asyncio.Lock()

    async def next_available(self) -> KeyEntry:
        """Round-robin; blocks until a key is out of cooldown."""
        while True:
            async with self._lock:
                for _ in range(len(self.entries)):
                    self._idx = (self._idx + 1) % len(self.entries)
                    entry = self.entries[self._idx]
                    if entry.available:
                        return entry

            # All keys exhausted — sleep until the nearest recovery
            wait_until = min(e.exhausted_until for e in self.entries)
            sleep_for = max(0.1, wait_until - time.monotonic())
            logger.warning("All keys exhausted. Sleeping %.1fs …", sleep_for)
            await asyncio.sleep(sleep_for)

    def stats(self) -> list[dict]:
        now = time.monotonic()
        return [
            {
                "key_suffix": e.key[-6:],
                "proxy": e.proxy,
                "available": e.available,
                "cooldown_remaining_s": max(0.0, round(e.exhausted_until - now, 1)),
                "total_requests": e.total_requests,
                "total_errors": e.total_errors,
            }
            for e in self.entries
        ]


# ---------------------------------------------------------------------------
# Model mapping
# ---------------------------------------------------------------------------


MODEL_MAP: dict[str, str] = {
    # allow pass-through of native Gemini names
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-flash-latest": "gemini-flash-latest",
    "gemini-flash-lite-latest": "gemini-flash-lite-latest",
}


def resolve_model(name: str) -> str:
    return MODEL_MAP.get(name, name)


# ---------------------------------------------------------------------------
# OpenAI request schemas
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str | list[Any]  # str or OpenAI content-part list


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
    # ── Structured output ────────────────────────────────────────────────────
    # Option A (OpenAI-style):
    #   response_format={"type": "json_schema", "json_schema": {"schema": {...}}}
    # Option B (direct Gemini schema, non-standard but convenient):
    #   schema={...}
    # Option C (plain JSON mode, no schema enforcement):
    #   response_format={"type": "json_object"}
    response_format: dict | None = None
    schema: dict | None = None


class TokenCountRequest(BaseModel):
    model: str = DEFAULT_GEMINI_MODEL
    messages: list[Message]


# ---------------------------------------------------------------------------
# Content conversion helpers
# ---------------------------------------------------------------------------


def _url_to_inline_data(url: str) -> dict:
    """
    Fetch an image URL (or parse a data URI) and return a Gemini inlineData part.
    """
    if url.startswith("data:"):
        # data:image/png;base64,<data>
        header, b64 = url.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        return {"inlineData": {"mimeType": mime, "data": b64}}

    # Remote URL – download synchronously (called before async context starts)
    proxy_kwargs = {"proxy": GENERAL_PROXY} if GENERAL_PROXY else {}
    with httpx.Client(timeout=30, follow_redirects=True, **proxy_kwargs) as client:
        resp = client.get(url)
        resp.raise_for_status()
    mime = resp.headers.get("content-type", "image/jpeg").split(";")[0]
    b64 = base64.b64encode(resp.content).decode()
    return {"inlineData": {"mimeType": mime, "data": b64}}


def _content_to_gemini_parts(content: str | list[Any]) -> list[dict]:
    """
    Convert an OpenAI message content value to a list of Gemini Part dicts.

    Handled OpenAI part types:
      {"type": "text",      "text": "..."}
      {"type": "image_url", "image_url": {"url": "..."}}
    """
    if isinstance(content, str):
        return [{"text": content}]

    parts: list[dict] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"text": item})
            continue
        t = item.get("type", "")
        if t == "text":
            parts.append({"text": item["text"]})
        elif t == "image_url":
            parts.append(_url_to_inline_data(item["image_url"]["url"]))
        else:
            logger.debug("Unknown content part type ignored: %s", t)
    return parts


def _messages_to_gemini(messages: list[Message]) -> tuple[str | None, list[dict]]:
    """Returns (system_instruction | None, gemini_contents)."""
    system_instruction: str | None = None
    contents: list[dict] = []

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_instruction = msg.content
            else:
                system_instruction = " ".join(
                    p.get("text", "")
                    for p in msg.content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
        else:
            gemini_role = "model" if msg.role == "assistant" else "user"
            contents.append(
                {
                    "role": gemini_role,
                    "parts": _content_to_gemini_parts(msg.content),
                }
            )

    return system_instruction, contents


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------


def _resolve_schema(req: ChatCompletionRequest) -> dict | None:
    """
    Returns:
        None          → no JSON mode
        {}            → JSON mode, no schema enforcement
        {<schema>}    → JSON mode + schema enforcement
    """
    if req.schema:
        return req.schema
    if req.response_format:
        fmt = req.response_format
        if fmt.get("type") == "json_schema" and "json_schema" in fmt:
            return fmt["json_schema"].get("schema") or {}
        if fmt.get("type") == "json_object":
            return {}
    return None


def _build_payload(req: ChatCompletionRequest) -> dict:
    system_instruction, contents = _messages_to_gemini(req.messages)
    schema = _resolve_schema(req)

    gen_cfg: dict[str, Any] = {
        "temperature": req.temperature,
        "maxOutputTokens": req.max_tokens or 8192,
    }
    if req.top_p is not None:
        gen_cfg["topP"] = req.top_p
    if req.top_k is not None:
        gen_cfg["topK"] = req.top_k
    if req.stop:
        gen_cfg["stopSequences"] = [req.stop] if isinstance(req.stop, str) else req.stop

    if schema is not None:
        gen_cfg["responseMimeType"] = "application/json"
        if schema:  # non-empty → enforce the schema
            gen_cfg["responseSchema"] = schema

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": gen_cfg,
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    return payload


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


def _gemini_to_openai(gemini: dict, model: str) -> dict:
    choices = []
    for i, cand in enumerate(gemini.get("candidates", [])):
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

    usage = gemini.get("usageMetadata", {})
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
# Core call (with retry + key rotation)
# ---------------------------------------------------------------------------


async def call_gemini(key_manager: KeyManager, req: ChatCompletionRequest) -> dict:
    gemini_model = resolve_model(req.model)
    payload = _build_payload(req)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        entry = await key_manager.next_available()
        entry.total_requests += 1

        endpoint = f"models/{gemini_model}:generateContent"
        logger.info(
            "Attempt %d/%d | key=…%s | proxy=%s | model=%s",
            attempt,
            MAX_RETRIES,
            entry.key[-6:],
            entry.proxy,
            gemini_model,
        )
        debug_log(
            {
                "event": "request",
                "endpoint": endpoint,
                "key_suffix": entry.key[-6:],
                "attempt": attempt,
                "payload": payload,
            }
        )

        t0 = time.monotonic()
        try:
            async with entry.make_http_client() as client:
                resp = await client.post(
                    f"{GEMINI_BASE}/{endpoint}",
                    params={"key": entry.key},
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
        except httpx.RequestError as exc:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            entry.total_errors += 1
            entry.mark_exhausted(cooldown=10)
            last_error = exc
            debug_log(
                {
                    "event": "error",
                    "endpoint": endpoint,
                    "key_suffix": entry.key[-6:],
                    "attempt": attempt,
                    "error": str(exc),
                    "elapsed_ms": elapsed_ms,
                }
            )
            logger.warning("Network error (attempt %d): %s", attempt, exc)
            continue

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        if resp.status_code == 200:
            gemini_body = resp.json()
            debug_log(
                {
                    "event": "response",
                    "endpoint": endpoint,
                    "key_suffix": entry.key[-6:],
                    "attempt": attempt,
                    "status": 200,
                    "elapsed_ms": elapsed_ms,
                    "response": gemini_body,
                }
            )
            return _gemini_to_openai(gemini_body, gemini_model)

        is_json = "application/json" in resp.headers.get("content-type", "")
        body = resp.json() if is_json else {}
        err_msg = body.get("error", {}).get("message", resp.text[:300])
        status = resp.status_code
        entry.total_errors += 1
        debug_log(
            {
                "event": "error",
                "endpoint": endpoint,
                "key_suffix": entry.key[-6:],
                "attempt": attempt,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "error": err_msg,
            }
        )
        logger.warning("Gemini %d (attempt %d): %s", status, attempt, err_msg)

        if status in (429, 503):
            entry.mark_exhausted()
            last_error = HTTPException(status_code=429, detail=err_msg)
        elif status == 400:
            raise HTTPException(status_code=400, detail=err_msg)
        elif status in (401, 403):
            entry.mark_exhausted(cooldown=3600)
            last_error = HTTPException(
                status_code=502, detail=f"Gemini auth error: {err_msg}"
            )
        else:
            last_error = HTTPException(
                status_code=502, detail=f"Gemini {status}: {err_msg}"
            )
            await asyncio.sleep(ALL_EXHAUSTED_SLEEP)

    raise last_error or HTTPException(
        status_code=502, detail="All retry attempts failed"
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def call_gemini_stream(
    key_manager: KeyManager, req: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    gemini_model = resolve_model(req.model)
    payload = _build_payload(req)

    entry = await key_manager.next_available()
    entry.total_requests += 1

    endpoint = f"models/{gemini_model}:streamGenerateContent"
    debug_log(
        {
            "event": "request",
            "endpoint": endpoint,
            "key_suffix": entry.key[-6:],
            "payload": payload,
        }
    )

    t0 = time.monotonic()
    stream_chunks: list[dict] = []  # collected only in DEBUG_MODE

    async with entry.make_http_client() as client:
        async with client.stream(
            "POST",
            f"{GEMINI_BASE}/{endpoint}",
            params={"key": entry.key, "alt": "sse"},
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                entry.total_errors += 1
                entry.mark_exhausted()
                body = await resp.aread()
                debug_log(
                    {
                        "event": "error",
                        "endpoint": endpoint,
                        "key_suffix": entry.key[-6:],
                        "status": resp.status_code,
                        "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                        "error": body.decode()[:300],
                    }
                )
                raise HTTPException(status_code=502, detail=body.decode())

            chunk_idx = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    debug_log(
                        {
                            "event": "stream_end",
                            "endpoint": endpoint,
                            "key_suffix": entry.key[-6:],
                            "status": 200,
                            "total_chunks": chunk_idx,
                            "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                            "chunks": stream_chunks,
                        }
                    )
                    yield "data: [DONE]\n\n"
                    return
                try:
                    gemini_chunk = json.loads(data_str)
                except Exception:
                    continue

                if DEBUG_MODE:
                    stream_chunks.append(gemini_chunk)

                for cand in gemini_chunk.get("candidates", []):
                    parts = cand.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    finish = cand.get("finishReason")
                    openai_chunk = {
                        "id": f"chatcmpl-gemini-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": gemini_model,
                        "choices": [
                            {
                                "index": chunk_idx,
                                "delta": {"role": "assistant", "content": text},
                                "finish_reason": finish,
                            }
                        ],
                    }
                    chunk_idx += 1
                    yield f"data: {json.dumps(openai_chunk)}\n\n"

            # Gemini didn't send an explicit [DONE] — emit it ourselves so clients
            # relying on the OpenAI SSE contract always see the sentinel.
            debug_log(
                {
                    "event": "stream_end",
                    "endpoint": endpoint,
                    "key_suffix": entry.key[-6:],
                    "status": 200,
                    "total_chunks": chunk_idx,
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                    "chunks": stream_chunks,
                }
            )
            yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


async def count_tokens_gemini(key_manager: KeyManager, req: TokenCountRequest) -> int:
    gemini_model = resolve_model(req.model)
    system_instruction, contents = _messages_to_gemini(req.messages)
    payload: dict[str, Any] = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    entry = await key_manager.next_available()
    entry.total_requests += 1

    endpoint = f"models/{gemini_model}:countTokens"
    debug_log(
        {
            "event": "request",
            "endpoint": endpoint,
            "key_suffix": entry.key[-6:],
            "payload": payload,
        }
    )

    t0 = time.monotonic()
    async with entry.make_http_client() as client:
        resp = await client.post(
            f"{GEMINI_BASE}/{endpoint}",
            params={"key": entry.key},
            json=payload,
            headers={"Content-Type": "application/json"},
        )
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    if resp.status_code != 200:
        is_json = "application/json" in resp.headers.get("content-type", "")
        body = resp.json() if is_json else {}
        err = body.get("error", {}).get("message", resp.text[:200])
        entry.total_errors += 1
        debug_log(
            {
                "event": "error",
                "endpoint": endpoint,
                "key_suffix": entry.key[-6:],
                "status": resp.status_code,
                "elapsed_ms": elapsed_ms,
                "error": err,
            }
        )
        raise HTTPException(status_code=502, detail=f"Token count error: {err}")

    result = resp.json()
    debug_log(
        {
            "event": "response",
            "endpoint": endpoint,
            "key_suffix": entry.key[-6:],
            "status": 200,
            "elapsed_ms": elapsed_ms,
            "response": result,
        }
    )
    return result.get("totalTokens", 0)


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
        logger.info("  key=…%s  proxy=%s", p["key"][-6:], p["proxy"])
    yield


app = FastAPI(title="Gemini → OpenAI Proxy", version="2.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models():
    models = sorted(set(MODEL_MAP.values()))
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "google"}
            for m in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if key_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if req.stream:
        return StreamingResponse(
            call_gemini_stream(key_manager, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(content=await call_gemini(key_manager, req))


@app.post("/v1/token/count")
async def token_count(req: TokenCountRequest):
    """
    Count prompt tokens for a given model + messages list.

    Request body: same shape as /v1/chat/completions minus generation params.
    Response: {"model": "gemini-2.0-flash", "total_tokens": 1234}
    """
    if key_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    total = await count_tokens_gemini(key_manager, req)
    return {"model": resolve_model(req.model), "total_tokens": total}


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
        "main_v1:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
