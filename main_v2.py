"""
Gemini → OpenAI-compatible proxy  (v2 — google-genai library backend)

v1 used raw httpx calls to the Gemini REST API directly.
v2 uses the official `google-genai` Python library instead, so it tracks
the Gemini SDK automatically as the API evolves.

Proxy handling
──────────────
The google-genai SDK has no per-client proxy API.  It reads proxy settings
from the standard environment variables (http_proxy / HTTPS_PROXY / etc.) via
urllib.request.getproxies() at Client *construction* time.

Solution: a global asyncio.Lock (_ENV_PROXY_LOCK) serialises all SDK calls.
Before each call we:
  1. Acquire the lock.
  2. Set the four proxy env vars to the value paired with the chosen key.
  3. Construct a fresh genai.Client (so it picks up the env vars).
  4. Run the synchronous SDK call in a thread-pool executor.
  5. Restore the original env vars and release the lock.

This works reliably because:
  - the lock ensures only one goroutine mutates env vars at a time
  - a new Client is created inside the lock, not cached across calls
  - different key+proxy pairs take turns; throughput ≈ Gemini latency / n_keys

Features (identical surface to v1):
- OpenAI /v1/chat/completions  (streaming supported)
- Structured output via response_schema
- Image inference  (OpenAI image_url → genai.types.Part.from_bytes)
- Token counting   POST /v1/token/count
- Key ↔ Proxy pairing with cooldown/rotation
- GENERAL_PROXY for non-Gemini HTTP (image downloads etc.)
- Debug mode  (DEBUG_MODE=1)
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Logging / debug
# ---------------------------------------------------------------------------
#
# DEBUG_MODE=1  → stderr + file handlers (DEBUG level) AND JSONL payload log
# DEBUG_MODE=0  → stderr INFO only (unchanged)

DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "0").strip() in ("1", "true", "yes")
DEBUG_LOG_FILE: str = os.environ.get("DEBUG_LOG_FILE", "gemini_proxy_debug.log")

_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

if DEBUG_MODE:
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter(_fmt))
    _file_handler = logging.FileHandler(DEBUG_LOG_FILE, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(_fmt))
    logging.basicConfig(level=logging.DEBUG, handlers=[_console, _file_handler])
else:
    logging.basicConfig(level=logging.INFO, format=_fmt)

logger = logging.getLogger("gemini-proxy")

if DEBUG_MODE:
    logger.info("Debug mode ON — request/response payloads → %s", DEBUG_LOG_FILE)

_debug_lock = threading.Lock()


def _scrub(obj: Any, depth: int = 0) -> Any:
    """Truncate long base64 blobs so log files stay readable."""
    if depth > 10:
        return obj
    if isinstance(obj, dict):
        return {
            k: (
                v[:40] + f"…[{len(v)} chars]"
                if k == "data" and isinstance(v, str) and len(v) > 80
                else _scrub(v, depth + 1)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_scrub(i, depth + 1) for i in obj]
    return obj


def debug_log(record: dict) -> None:
    """Append one JSONL record to DEBUG_LOG_FILE (no-op when debug is off)."""
    if not DEBUG_MODE:
        return
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
GENERAL_PROXY: str | None = os.environ.get("GENERAL_PROXY") or None
ALL_EXHAUSTED_SLEEP = float(os.environ.get("ALL_EXHAUSTED_SLEEP_SECONDS", "5"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
DEFAULT_GEMINI_MODEL = os.environ.get(
    "DEFAULT_GEMINI_MODEL", "gemini-flash-lite-latest"
)


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

    i = 0
    while True:
        key = os.environ.get(f"GEMINI_KEY_{i}")
        if key is None:
            break
        proxy = os.environ.get(f"GEMINI_PROXY_{i}")
        pairs.append({"key": key.strip(), "proxy": proxy.strip() if proxy else None})
        i += 1

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
# Proxy env-var context manager
# ---------------------------------------------------------------------------

# One global lock: only one SDK call at a time may mutate the process-wide
# proxy env vars.  Calls sharing the same key+proxy are naturally serialised;
# calls with different proxies take turns.  This is the only reliable way to
# handle per-key proxy routing with the google-genai SDK, which reads proxy
# settings from environment variables at Client construction time via
# urllib.request.getproxies() — there is no per-client proxy API.
_ENV_PROXY_LOCK = asyncio.Lock()
_ENV_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


@contextlib.contextmanager
def _proxy_env(proxy: str | None):
    """
    Temporarily set/unset the four standard proxy env vars.
    Must be called while the caller holds _ENV_PROXY_LOCK.
    """
    saved = {k: os.environ.get(k) for k in _ENV_VARS}
    try:
        for k in _ENV_VARS:
            if proxy:
                os.environ[k] = proxy
            else:
                os.environ.pop(k, None)
        yield
    finally:
        for k in _ENV_VARS:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


# ---------------------------------------------------------------------------
# Key / Proxy management
# ---------------------------------------------------------------------------


class KeyEntry:
    def __init__(self, key: str, proxy: str | None):
        self.key = key
        self.proxy = proxy
        self.exhausted_until: float = 0.0
        self.total_requests: int = 0
        self.total_errors: int = 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.exhausted_until

    def mark_exhausted(self, cooldown: float = COOLDOWN_SECONDS) -> None:
        self.exhausted_until = time.monotonic() + cooldown
        logger.warning("Key …%s cooled down for %.0fs", self.key[-6:], cooldown)

    def make_genai_client(self) -> genai.Client:
        """
        Create a fresh genai.Client for this key.
        Always call this *inside* _proxy_env() so the env vars are already set
        at construction time — the SDK reads them via urllib.request.getproxies()
        during __init__ and does not re-read them later.
        """
        return genai.Client(api_key=self.key)


class KeyManager:
    def __init__(self, pairs: list[dict]):
        self.entries: list[KeyEntry] = [KeyEntry(p["key"], p["proxy"]) for p in pairs]
        self._idx = -1
        self._lock = asyncio.Lock()

    async def next_available(self) -> KeyEntry:
        while True:
            async with self._lock:
                for _ in range(len(self.entries)):
                    self._idx = (self._idx + 1) % len(self.entries)
                    entry = self.entries[self._idx]
                    if entry.available:
                        return entry
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
    schema: dict | None = None


class TokenCountRequest(BaseModel):
    model: str = DEFAULT_GEMINI_MODEL
    messages: list[Message]


# ---------------------------------------------------------------------------
# Content conversion helpers  (OpenAI → genai types)
# ---------------------------------------------------------------------------


def _url_to_part(url: str) -> types.Part:
    """
    Convert an image URL (or data URI) to a genai Part.
    Remote URLs are downloaded via httpx (no proxy — images are public CDN).
    """
    if url.startswith("data:"):
        header, b64 = url.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        return types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime)

    proxy_kwargs = {"proxy": GENERAL_PROXY} if GENERAL_PROXY else {}
    with httpx.Client(timeout=30, follow_redirects=True, **proxy_kwargs) as client:
        resp = client.get(url)
        resp.raise_for_status()
    mime = resp.headers.get("content-type", "image/jpeg").split(";")[0]
    return types.Part.from_bytes(data=resp.content, mime_type=mime)


def _content_to_parts(content: str | list[Any]) -> list[types.Part]:
    if isinstance(content, str):
        return [types.Part.from_text(text=content)]

    parts: list[types.Part] = []
    for item in content:
        if isinstance(item, str):
            parts.append(types.Part.from_text(text=item))
            continue
        t = item.get("type", "")
        if t == "text":
            parts.append(types.Part.from_text(text=item["text"]))
        elif t == "image_url":
            parts.append(_url_to_part(item["image_url"]["url"]))
        else:
            logger.debug("Unknown content part type ignored: %s", t)
    return parts


def _messages_to_genai(
    messages: list[Message],
) -> tuple[str | None, list[types.Content]]:
    """Returns (system_instruction | None, list[types.Content])."""
    system_instruction: str | None = None
    contents: list[types.Content] = []

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
            role = "model" if msg.role == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=_content_to_parts(msg.content))
            )

    return system_instruction, contents


# ---------------------------------------------------------------------------
# GenerateContentConfig builder
# ---------------------------------------------------------------------------


def _resolve_schema(req: ChatCompletionRequest) -> dict | None:
    if req.schema:
        return req.schema
    if req.response_format:
        fmt = req.response_format
        if fmt.get("type") == "json_schema" and "json_schema" in fmt:
            return fmt["json_schema"].get("schema") or {}
        if fmt.get("type") == "json_object":
            return {}
    return None


def _build_config(req: ChatCompletionRequest) -> types.GenerateContentConfig:
    schema = _resolve_schema(req)

    kwargs: dict[str, Any] = {
        "temperature": req.temperature,
        "max_output_tokens": req.max_tokens or 8192,
    }
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    if req.stop:
        kwargs["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if schema is not None:
        kwargs["response_mime_type"] = "application/json"
        if schema:
            kwargs["response_schema"] = schema

    return types.GenerateContentConfig(**kwargs)


# ---------------------------------------------------------------------------
# Response conversion  (genai → OpenAI)
# ---------------------------------------------------------------------------


def _genai_to_openai(response: types.GenerateContentResponse, model: str) -> dict:
    choices = []
    for i, cand in enumerate(response.candidates or []):
        text = ""
        for part in cand.content.parts or []:
            if hasattr(part, "text") and part.text:
                text += part.text
        finish = (cand.finish_reason.name if cand.finish_reason else "stop").lower()
        if finish == "max_tokens":
            finish = "length"
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        )

    usage = response.usage_metadata
    return {
        "id": f"chatcmpl-gemini-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(usage, "total_token_count", 0) or 0,
        },
    }


# ---------------------------------------------------------------------------
# SDK call wrapper  (handles proxy env swap + error classification)
# ---------------------------------------------------------------------------


class _GeminiError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


def _classify_genai_exception(exc: Exception) -> _GeminiError:
    """
    Map genai library exceptions to a (status_code, message) pair.
    The genai SDK raises google.api_core.exceptions.* or grpc exceptions.
    We do a best-effort string match so we don't hard-depend on those imports.
    """
    name = type(exc).__name__
    msg = str(exc)
    lower = msg.lower()

    if "429" in msg or "resource_exhausted" in name.lower() or "quota" in lower:
        return _GeminiError(429, msg)
    if "503" in msg or "unavailable" in name.lower():
        return _GeminiError(503, msg)
    if (
        "401" in msg
        or "403" in msg
        or "unauthenticated" in name.lower()
        or "permission" in lower
    ):
        return _GeminiError(401, msg)
    if "400" in msg or "invalid_argument" in name.lower():
        return _GeminiError(400, msg)
    return _GeminiError(502, msg)


async def _run_with_proxy(entry: "KeyEntry", fn_builder, *args, **kwargs):
    """
    1. Acquire the global proxy lock.
    2. Set env vars to entry.proxy.
    3. Build a fresh genai.Client (so it picks up the proxy at construction).
    4. Run fn_builder(client) synchronously in an executor.
    5. Restore env vars and release the lock.

    fn_builder receives the freshly-built client and should return a callable
    that performs the actual SDK call, e.g.:
        lambda client: client.models.generate_content(...)
    """
    loop = asyncio.get_running_loop()

    async with _ENV_PROXY_LOCK:
        with _proxy_env(entry.proxy):
            client = entry.make_genai_client()
            result = await loop.run_in_executor(None, lambda: fn_builder(client))

    return result


# ---------------------------------------------------------------------------
# Core call (with retry + key rotation)
# ---------------------------------------------------------------------------


async def call_gemini(key_manager: KeyManager, req: ChatCompletionRequest) -> dict:
    gemini_model = resolve_model(req.model)
    system_instruction, contents = _messages_to_genai(req.messages)
    config = _build_config(req)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        entry = await key_manager.next_available()
        entry.total_requests += 1

        logger.info(
            "Attempt %d/%d | key=…%s | proxy=%s | model=%s",
            attempt,
            MAX_RETRIES,
            entry.key[-6:],
            entry.proxy,
            gemini_model,
        )

        # Build the payload dict for debug logging (contents are not trivially serialisable)
        debug_payload = {
            "model": gemini_model,
            "system_instruction": system_instruction,
            "contents": [
                {
                    "role": c.role,
                    "parts": [
                        (
                            {"text": p.text}
                            if hasattr(p, "text") and p.text
                            else {"inline_data": "…"}
                        )
                        for p in c.parts
                    ],
                }
                for c in contents
            ],
            "config": {
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
            },
        }
        debug_log(
            {
                "event": "request",
                "endpoint": f"models/{gemini_model}:generateContent",
                "key_suffix": entry.key[-6:],
                "attempt": attempt,
                "payload": debug_payload,
            }
        )

        # Attach system instruction to config if present
        final_config = config
        if system_instruction:
            final_config = types.GenerateContentConfig(
                **{
                    k: v
                    for k, v in {
                        "temperature": config.temperature,
                        "max_output_tokens": config.max_output_tokens,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "stop_sequences": config.stop_sequences,
                        "response_mime_type": config.response_mime_type,
                        "response_schema": config.response_schema,
                        "system_instruction": system_instruction,
                    }.items()
                    if v is not None
                }
            )

        t0 = time.monotonic()

        try:
            response = await _run_with_proxy(
                entry,
                lambda client: client.models.generate_content(
                    model=gemini_model,
                    contents=contents,
                    config=final_config,
                ),
            )
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            err = _classify_genai_exception(exc)
            entry.total_errors += 1
            debug_log(
                {
                    "event": "error",
                    "endpoint": f"models/{gemini_model}:generateContent",
                    "key_suffix": entry.key[-6:],
                    "attempt": attempt,
                    "status": err.status,
                    "elapsed_ms": elapsed_ms,
                    "error": err.message,
                }
            )
            logger.warning(
                "Gemini error (attempt %d) [%d]: %s",
                attempt,
                err.status,
                err.message[:200],
            )

            if err.status == 400:
                raise HTTPException(status_code=400, detail=err.message)
            elif err.status in (401, 403):
                entry.mark_exhausted(cooldown=3600)
                last_error = HTTPException(
                    status_code=502, detail=f"Gemini auth error: {err.message}"
                )
            elif err.status in (429, 503):
                entry.mark_exhausted()
                last_error = HTTPException(status_code=429, detail=err.message)
            else:
                last_error = HTTPException(status_code=502, detail=err.message)
                await asyncio.sleep(ALL_EXHAUSTED_SLEEP)
            continue

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        result = _genai_to_openai(response, gemini_model)
        debug_log(
            {
                "event": "response",
                "endpoint": f"models/{gemini_model}:generateContent",
                "key_suffix": entry.key[-6:],
                "attempt": attempt,
                "status": 200,
                "elapsed_ms": elapsed_ms,
                "response": result,
            }
        )
        return result

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
    system_instruction, contents = _messages_to_genai(req.messages)
    config = _build_config(req)

    entry = await key_manager.next_available()
    entry.total_requests += 1

    endpoint = f"models/{gemini_model}:streamGenerateContent"
    debug_log(
        {
            "event": "request",
            "endpoint": endpoint,
            "key_suffix": entry.key[-6:],
            "payload": {"model": gemini_model, "stream": True},
        }
    )

    if system_instruction:
        config = types.GenerateContentConfig(
            **{
                k: v
                for k, v in {
                    "temperature": config.temperature,
                    "max_output_tokens": config.max_output_tokens,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "stop_sequences": config.stop_sequences,
                    "response_mime_type": config.response_mime_type,
                    "response_schema": config.response_schema,
                    "system_instruction": system_instruction,
                }.items()
                if v is not None
            }
        )

    t0 = time.monotonic()

    # generate_content_stream is synchronous; collect all chunks in an executor
    try:
        stream_iter = await _run_with_proxy(
            entry,
            lambda client: list(
                client.models.generate_content_stream(
                    model=gemini_model,
                    contents=contents,
                    config=config,
                )
            ),
        )
    except Exception as exc:
        entry.total_errors += 1
        entry.mark_exhausted()
        err = _classify_genai_exception(exc)
        debug_log(
            {
                "event": "error",
                "endpoint": endpoint,
                "key_suffix": entry.key[-6:],
                "status": err.status,
                "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                "error": err.message,
            }
        )
        raise HTTPException(status_code=502, detail=err.message)

    chunk_idx = 0
    stream_chunks_debug: list[str] = []

    for chunk in stream_iter:
        for cand in chunk.candidates or []:
            text = ""
            for part in cand.content.parts or []:
                if hasattr(part, "text") and part.text:
                    text += part.text
            finish = cand.finish_reason.name if cand.finish_reason else None
            if finish:
                finish = finish.lower()

            if DEBUG_MODE:
                stream_chunks_debug.append(text)

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

    debug_log(
        {
            "event": "stream_end",
            "endpoint": endpoint,
            "key_suffix": entry.key[-6:],
            "status": 200,
            "total_chunks": chunk_idx,
            "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
            "chunks": stream_chunks_debug,
        }
    )
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


async def count_tokens_gemini(key_manager: KeyManager, req: TokenCountRequest) -> int:
    gemini_model = resolve_model(req.model)
    system_instruction, contents = _messages_to_genai(req.messages)

    entry = await key_manager.next_available()
    entry.total_requests += 1

    endpoint = f"models/{gemini_model}:countTokens"
    debug_log(
        {
            "event": "request",
            "endpoint": endpoint,
            "key_suffix": entry.key[-6:],
            "payload": {"model": gemini_model},
        }
    )

    # Build config with system instruction so token count matches inference
    count_config = None
    if system_instruction:
        count_config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )

    t0 = time.monotonic()

    try:
        result = await _run_with_proxy(
            entry,
            lambda client: client.models.count_tokens(
                model=gemini_model,
                contents=contents,
                config=count_config,
            ),
        )
    except Exception as exc:
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        entry.total_errors += 1
        err = _classify_genai_exception(exc)
        debug_log(
            {
                "event": "error",
                "endpoint": endpoint,
                "key_suffix": entry.key[-6:],
                "status": err.status,
                "elapsed_ms": elapsed_ms,
                "error": err.message,
            }
        )
        raise HTTPException(status_code=502, detail=f"Token count error: {err.message}")

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    total = result.total_tokens or 0
    debug_log(
        {
            "event": "response",
            "endpoint": endpoint,
            "key_suffix": entry.key[-6:],
            "status": 200,
            "elapsed_ms": elapsed_ms,
            "response": {"totalTokens": total},
        }
    )
    return total


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


app = FastAPI(
    title="Gemini → OpenAI Proxy (v2/genai)", version="2.0.0", lifespan=lifespan
)


# ---------------------------------------------------------------------------
# Routes  (identical to v1)
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
        "main_v2:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
