"""
test_proxy.py — comprehensive tests for the Gemini → OpenAI proxy.

Coverage:
  ✔ GET  /health
  ✔ GET  /stats
  ✔ GET  /v1/models
  ✔ POST /v1/chat/completions  — plain text
  ✔ POST /v1/chat/completions  — system prompt forwarded as systemInstruction
  ✔ POST /v1/chat/completions  — multi-turn conversation (user / assistant / user)
  ✔ POST /v1/chat/completions  — streaming (SSE)
  ✔ POST /v1/chat/completions  — structured output via req.schema
  ✔ POST /v1/chat/completions  — structured output via response_format json_schema
  ✔ POST /v1/chat/completions  — structured output via response_format json_object (no schema)
  ✔ POST /v1/chat/completions  — image (data-URI inline)
  ✔ POST /v1/chat/completions  — image (remote URL, httpx download mocked)
  ✔ POST /v1/token/count
  ✔ Key rotation — 429 cools down one key, next request uses the other
  ✔ Key rotation — 401/403 triggers long cooldown
  ✔ Key rotation — network error cools key, retries succeed
  ✔ All keys exhausted — sleeps then succeeds on recovery
  ✔ Gemini 400 propagates immediately as 400 (no retry)
  ✔ Proxy pairing — correct proxy used per key (env Style 1, Style 2, Style 3)
  ✔ Model alias resolution (gpt-4o → gemini-flash-latest, etc.)
  ✔ Unknown model passes through unchanged
  ✔ _content_to_gemini_parts  — plain string
  ✔ _content_to_gemini_parts  — list of text parts
  ✔ _content_to_gemini_parts  — data-URI image part
  ✔ _url_to_inline_data        — data URI parsing
  ✔ _resolve_schema            — all three option styles
  ✔ _build_payload             — stop sequences, top_p, top_k forwarded
  ✔ _gemini_to_openai          — finish_reason mapping (STOP → stop, MAX_TOKENS → length)
  ✔ Usage metadata forwarded to openai response

Dependencies:
    pip install pytest pytest-asyncio httpx respx fastapi python-dotenv
"""

import asyncio
import base64
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

# ── make the app importable without real env vars ────────────────────────────
os.environ.setdefault("GEMINI_KEY_0", "fake-key-aaaaaa")
os.environ.setdefault("GEMINI_PROXY_0", "")  # no proxy for tests
os.environ.setdefault("KEY_COOLDOWN_SECONDS", "60")
os.environ.setdefault("ALL_EXHAUSTED_SLEEP_SECONDS", "0.05")  # fast in tests
os.environ.setdefault("MAX_RETRIES", "3")

import main_v2 as main  # noqa: E402  (import after env setup)
from main_v2 import (
    ChatCompletionRequest,
    KeyEntry,
    KeyManager,
    Message,
    TokenCountRequest,
    _build_payload,
    _content_to_gemini_parts,
    _gemini_to_openai,
    _messages_to_gemini,
    _resolve_schema,
    _url_to_inline_data,
    resolve_model,
)

# ── Constants ────────────────────────────────────────────────────────────────
GEMINI_BASE = main.GEMINI_BASE
FAKE_KEY = "fake-key-aaaaaa"
MODEL = "gemini-flash-lite-latest"

# ── Helpers ──────────────────────────────────────────────────────────────────


def gemini_ok(
    text: str = "Hello!", finish: str = "STOP", tokens: dict | None = None
) -> dict:
    """Build a minimal Gemini generateContent success response."""
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": finish,
            }
        ],
        "usageMetadata": tokens
        or {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }


def count_ok(total: int = 42) -> dict:
    return {"totalTokens": total}


def make_app_client(extra_pairs: list[dict] | None = None) -> TestClient:
    """Spin up a fresh app with given key-proxy pairs."""
    pairs = extra_pairs or [{"key": FAKE_KEY, "proxy": None}]
    km = KeyManager(pairs)
    main.key_manager = km
    return TestClient(main.app)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_key_manager():
    """Reset global key_manager before each test."""
    main.key_manager = KeyManager([{"key": FAKE_KEY, "proxy": None}])
    yield
    main.key_manager = None


@pytest.fixture()
def client():
    return TestClient(main.app)


# ═══════════════════════════════════════════════════════════════════════════
# Utility / pure-function tests  (no HTTP)
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveModel:
    def test_known_alias_gpt4o(self):
        assert resolve_model("gpt-4o") in main.MODEL_MAP.values()

    def test_passthrough_unknown(self):
        assert resolve_model("my-custom-model") == "my-custom-model"

    def test_native_gemini_names_resolve_to_themselves(self):
        for k, v in main.MODEL_MAP.items():
            # if the key itself is a valid Gemini name it maps to itself
            if k == v:
                assert resolve_model(k) == v


class TestContentToGeminiParts:
    def test_plain_string(self):
        parts = _content_to_gemini_parts("hello world")
        assert parts == [{"text": "hello world"}]

    def test_list_of_text_parts(self):
        content = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
        parts = _content_to_gemini_parts(content)
        assert parts == [{"text": "foo"}, {"text": "bar"}]

    def test_data_uri_image_part(self):
        b64 = base64.b64encode(b"fake-image-bytes").decode()
        data_uri = f"data:image/png;base64,{b64}"
        content = [{"type": "image_url", "image_url": {"url": data_uri}}]
        parts = _content_to_gemini_parts(content)
        assert len(parts) == 1
        assert parts[0]["inlineData"]["mimeType"] == "image/png"
        assert parts[0]["inlineData"]["data"] == b64

    def test_unknown_type_ignored(self):
        content = [{"type": "audio", "audio_url": {"url": "http://x.com/a.mp3"}}]
        parts = _content_to_gemini_parts(content)
        assert parts == []


class TestUrlToInlineData:
    def test_data_uri_png(self):
        b64 = base64.b64encode(b"\x89PNG").decode()
        uri = f"data:image/png;base64,{b64}"
        result = _url_to_inline_data(uri)
        assert result == {"inlineData": {"mimeType": "image/png", "data": b64}}

    def test_data_uri_jpeg(self):
        b64 = base64.b64encode(b"\xff\xd8").decode()
        uri = f"data:image/jpeg;base64,{b64}"
        result = _url_to_inline_data(uri)
        assert result["inlineData"]["mimeType"] == "image/jpeg"

    @respx.mock
    def test_remote_url(self):
        img_bytes = b"fake-png-content"
        respx.get("https://example.com/image.png").mock(
            return_value=httpx.Response(
                200, content=img_bytes, headers={"content-type": "image/png"}
            )
        )
        result = _url_to_inline_data("https://example.com/image.png")
        assert result["inlineData"]["mimeType"] == "image/png"
        assert result["inlineData"]["data"] == base64.b64encode(img_bytes).decode()


class TestResolveSchema:
    def _req(self, **kwargs) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[Message(role="user", content="hi")], **kwargs
        )

    def test_none_when_no_format(self):
        assert _resolve_schema(self._req()) is None

    def test_direct_schema_field(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        req = self._req(schema=schema)
        assert _resolve_schema(req) == schema

    def test_response_format_json_schema(self):
        schema = {"type": "object"}
        req = self._req(
            response_format={"type": "json_schema", "json_schema": {"schema": schema}}
        )
        assert _resolve_schema(req) == schema

    def test_response_format_json_object_returns_empty_dict(self):
        req = self._req(response_format={"type": "json_object"})
        assert _resolve_schema(req) == {}

    def test_response_format_unknown_type_returns_none(self):
        req = self._req(response_format={"type": "text"})
        assert _resolve_schema(req) is None


class TestBuildPayload:
    def _req(self, **kwargs) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[Message(role="user", content="hi")], **kwargs
        )

    def test_basic_structure(self):
        payload = _build_payload(self._req())
        assert "contents" in payload
        assert "generationConfig" in payload

    def test_system_instruction_extracted(self):
        req = ChatCompletionRequest(
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="hello"),
            ]
        )
        payload = _build_payload(req)
        assert "systemInstruction" in payload
        assert payload["systemInstruction"]["parts"][0]["text"] == "You are helpful."

    def test_stop_sequences_list(self):
        payload = _build_payload(self._req(stop=["END", "STOP"]))
        assert payload["generationConfig"]["stopSequences"] == ["END", "STOP"]

    def test_stop_sequences_string(self):
        payload = _build_payload(self._req(stop="END"))
        assert payload["generationConfig"]["stopSequences"] == ["END"]

    def test_top_p_top_k_forwarded(self):
        payload = _build_payload(self._req(top_p=0.9, top_k=40))
        assert payload["generationConfig"]["topP"] == 0.9
        assert payload["generationConfig"]["topK"] == 40

    def test_schema_injected_into_gen_config(self):
        schema = {"type": "object"}
        payload = _build_payload(self._req(schema=schema))
        cfg = payload["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["responseSchema"] == schema

    def test_json_object_mode_no_schema_key(self):
        payload = _build_payload(self._req(response_format={"type": "json_object"}))
        cfg = payload["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert "responseSchema" not in cfg


class TestGeminiToOpenai:
    def test_basic_mapping(self):
        gemini = gemini_ok("Hi there!")
        result = _gemini_to_openai(gemini, MODEL)
        assert result["object"] == "chat.completion"
        assert result["model"] == MODEL
        assert result["choices"][0]["message"]["content"] == "Hi there!"
        assert result["choices"][0]["message"]["role"] == "assistant"

    def test_finish_reason_stop(self):
        gemini = gemini_ok(finish="STOP")
        result = _gemini_to_openai(gemini, MODEL)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_finish_reason_max_tokens(self):
        gemini = gemini_ok(finish="MAX_TOKENS")
        result = _gemini_to_openai(gemini, MODEL)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_usage_metadata_forwarded(self):
        gemini = gemini_ok(
            tokens={
                "promptTokenCount": 7,
                "candidatesTokenCount": 3,
                "totalTokenCount": 10,
            }
        )
        result = _gemini_to_openai(gemini, MODEL)
        assert result["usage"] == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }

    def test_empty_candidates(self):
        result = _gemini_to_openai({"candidates": [], "usageMetadata": {}}, MODEL)
        assert result["choices"] == []


class TestKeyEntry:
    def test_available_by_default(self):
        e = KeyEntry("key", None)
        assert e.available is True

    def test_mark_exhausted(self):
        e = KeyEntry("key", None)
        e.mark_exhausted(cooldown=9999)
        assert e.available is False

    def test_recovery_after_cooldown(self):
        e = KeyEntry("key", None)
        e.exhausted_until = time.monotonic() - 1  # already passed
        assert e.available is True

    def test_http_client_no_proxy(self):
        e = KeyEntry("key", None)
        client = e.make_http_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_http_client_with_proxy(self):
        e = KeyEntry("key", "http://proxy:8080")
        client = e.make_http_client()
        assert isinstance(client, httpx.AsyncClient)


class TestKeyManagerRotation:
    @pytest.mark.asyncio
    async def test_round_robin(self):
        km = KeyManager(
            [
                {"key": "key-A", "proxy": None},
                {"key": "key-B", "proxy": None},
            ]
        )
        e1 = await km.next_available()
        e2 = await km.next_available()
        e3 = await km.next_available()
        assert e1.key == "key-A"
        assert e2.key == "key-B"
        assert e3.key == "key-A"  # wraps around

    @pytest.mark.asyncio
    async def test_skips_exhausted_key(self):
        km = KeyManager(
            [
                {"key": "key-A", "proxy": None},
                {"key": "key-B", "proxy": None},
            ]
        )
        km.entries[0].mark_exhausted(cooldown=9999)  # key-A exhausted
        entry = await km.next_available()
        assert entry.key == "key-B"

    @pytest.mark.asyncio
    async def test_waits_when_all_exhausted(self):
        km = KeyManager([{"key": "key-A", "proxy": None}])
        km.entries[0].exhausted_until = time.monotonic() + 0.1  # very short cooldown
        t0 = time.monotonic()
        entry = await km.next_available()
        elapsed = time.monotonic() - t0
        assert entry.key == "key-A"
        assert elapsed >= 0.05  # slept at least a bit


# ═══════════════════════════════════════════════════════════════════════════
# HTTP endpoint tests  (TestClient + respx)
# ═══════════════════════════════════════════════════════════════════════════

GENERATE_URL = f"{GEMINI_BASE}/models/{MODEL}:generateContent"
COUNT_URL = f"{GEMINI_BASE}/models/{MODEL}:countTokens"


class TestHealthAndStats:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["keys_total"] == 1
        assert data["keys_available"] == 1

    def test_stats_shape(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        stats = r.json()["key_stats"]
        assert len(stats) == 1
        assert "key_suffix" in stats[0]
        assert "proxy" in stats[0]
        assert "available" in stats[0]
        assert "total_requests" in stats[0]

    def test_health_reflects_exhausted_key(self, client):
        main.key_manager.entries[0].mark_exhausted(9999)
        r = client.get("/health")
        assert r.json()["keys_available"] == 0


class TestListModels:
    def test_returns_list(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        ids = [m["id"] for m in data["data"]]
        # all values from MODEL_MAP should appear
        for v in set(main.MODEL_MAP.values()):
            assert v in ids


class TestChatCompletions:
    @respx.mock
    def test_plain_text_success(self, client):
        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(200, json=gemini_ok("Hello!"))
        )
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello!"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["usage"]["total_tokens"] == 15

    @respx.mock
    def test_system_prompt_forwarded(self, client):
        captured = {}

        def capture(request, route):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_ok())

        respx.post(GENERATE_URL).mock(side_effect=capture)
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a pirate."},
                    {"role": "user", "content": "Ahoy"},
                ],
            },
        )
        assert "systemInstruction" in captured["payload"]
        assert "pirate" in captured["payload"]["systemInstruction"]["parts"][0]["text"]
        # system message must NOT appear in contents
        roles = [c["role"] for c in captured["payload"]["contents"]]
        assert "system" not in roles

    @respx.mock
    def test_multi_turn_conversation(self, client):
        captured = {}

        def capture(request, route):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_ok("Fine!"))

        respx.post(GENERATE_URL).mock(side_effect=capture)
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm good!"},
                    {"role": "user", "content": "Great!"},
                ],
            },
        )
        contents = captured["payload"]["contents"]
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert contents[2]["role"] == "user"

    @respx.mock
    def test_structured_output_via_schema_field(self, client):
        captured = {}

        def capture(request, route):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_ok('{"name": "Alice"}'))

        respx.post(GENERATE_URL).mock(side_effect=capture)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Give me a name"}],
                "schema": schema,
            },
        )
        cfg = captured["payload"]["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["responseSchema"] == schema

    @respx.mock
    def test_structured_output_via_response_format_json_schema(self, client):
        captured = {}

        def capture(request, route):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_ok('{"x": 1}'))

        respx.post(GENERATE_URL).mock(side_effect=capture)
        schema = {"type": "object"}
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "JSON please"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": schema},
                },
            },
        )
        cfg = captured["payload"]["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["responseSchema"] == schema

    @respx.mock
    def test_structured_output_json_object_no_schema(self, client):
        captured = {}

        def capture(request, route):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_ok("{}"))

        respx.post(GENERATE_URL).mock(side_effect=capture)
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "JSON"}],
                "response_format": {"type": "json_object"},
            },
        )
        cfg = captured["payload"]["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert "responseSchema" not in cfg

    @respx.mock
    def test_image_data_uri(self, client):
        captured = {}

        def capture(request, route):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_ok("I see an image"))

        respx.post(GENERATE_URL).mock(side_effect=capture)
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        data_uri = f"data:image/png;base64,{b64}"

        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": "What's in this image?"},
                        ],
                    }
                ],
            },
        )
        parts = captured["payload"]["contents"][0]["parts"]
        inline = next(p for p in parts if "inlineData" in p)
        assert inline["inlineData"]["mimeType"] == "image/png"
        assert inline["inlineData"]["data"] == b64

    @respx.mock
    def test_image_remote_url(self, client):
        img_bytes = b"\x89PNG"
        # mock the image download AND the Gemini call
        respx.get("https://example.com/photo.jpg").mock(
            return_value=httpx.Response(
                200, content=img_bytes, headers={"content-type": "image/jpeg"}
            )
        )

        captured = {}

        def capture(request, route):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_ok("Described!"))

        respx.post(GENERATE_URL).mock(side_effect=capture)

        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/photo.jpg"},
                            },
                            {"type": "text", "text": "Describe it"},
                        ],
                    }
                ],
            },
        )
        parts = captured["payload"]["contents"][0]["parts"]
        inline = next(p for p in parts if "inlineData" in p)
        assert inline["inlineData"]["mimeType"] == "image/jpeg"
        assert inline["inlineData"]["data"] == base64.b64encode(img_bytes).decode()

    @respx.mock
    def test_stop_sequences_forwarded(self, client):
        captured = {}

        def capture(request, route):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=gemini_ok())

        respx.post(GENERATE_URL).mock(side_effect=capture)
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["END", "STOP"],
            },
        )
        assert captured["payload"]["generationConfig"]["stopSequences"] == [
            "END",
            "STOP",
        ]

    @respx.mock
    def test_gemini_400_propagates(self, client):
        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(
                400, json={"error": {"message": "Invalid request"}}
            )
        )
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 400

    @respx.mock
    def test_streaming_response(self, client):
        sse_lines = [
            "data: "
            + json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "Hello"}]},
                            "finishReason": None,
                        }
                    ]
                }
            ),
            "",
            "data: "
            + json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": " world"}]},
                            "finishReason": "STOP",
                        }
                    ]
                }
            ),
            "",
            "data: [DONE]",
            "",
        ]
        stream_url = f"{GEMINI_BASE}/models/{MODEL}:streamGenerateContent"
        respx.post(stream_url).mock(
            return_value=httpx.Response(
                200,
                text="\n".join(sse_lines),
                headers={"content-type": "text/event-stream"},
            )
        )
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            raw = r.read().decode()

        chunks = [
            json.loads(line[6:])
            for line in raw.splitlines()
            if line.startswith("data:") and "[DONE]" not in line
        ]
        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[1]["choices"][0]["delta"]["content"] == " world"
        assert chunks[0]["object"] == "chat.completion.chunk"


class TestTokenCount:
    @respx.mock
    def test_token_count_success(self, client):
        respx.post(COUNT_URL).mock(return_value=httpx.Response(200, json=count_ok(99)))
        r = client.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello world"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total_tokens"] == 99
        assert body["model"] == MODEL

    @respx.mock
    def test_token_count_error_propagates(self, client):
        respx.post(COUNT_URL).mock(
            return_value=httpx.Response(
                403,
                json={"error": {"message": "API key invalid"}},
                headers={"content-type": "application/json"},
            )
        )
        r = client.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 502
        assert "Token count error" in r.json()["detail"]


class TestKeyRotationAndCooldown:
    @respx.mock
    def test_429_cools_key_and_rotates(self, client):
        """First key gets 429 → cooled → second key used → success."""
        main.key_manager = KeyManager(
            [
                {"key": "key-A000000", "proxy": None},
                {"key": "key-B111111", "proxy": None},
            ]
        )
        call_count = 0

        def handler(request, route):
            nonlocal call_count
            call_count += 1
            key = request.url.params["key"]
            if key == "key-A000000":
                return httpx.Response(
                    429,
                    json={"error": {"message": "rate limited"}},
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(200, json=gemini_ok("OK from B"))

        respx.post(GENERATE_URL).mock(side_effect=handler)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "OK from B"
        # key-A should now be on cooldown
        assert main.key_manager.entries[0].available is False

    @respx.mock
    def test_401_triggers_long_cooldown(self, client):
        """401 → mark_exhausted(3600). Verify key unavailable."""
        main.key_manager = KeyManager(
            [
                {"key": "key-bad000", "proxy": None},
                {"key": "key-good00", "proxy": None},
            ]
        )

        def handler(request, route):
            key = request.url.params["key"]
            if key == "key-bad000":
                return httpx.Response(
                    401,
                    json={"error": {"message": "Invalid API key"}},
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(200, json=gemini_ok("OK"))

        respx.post(GENERATE_URL).mock(side_effect=handler)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        bad_entry = main.key_manager.entries[0]
        assert bad_entry.available is False
        # cooldown should be ~3600 s
        assert (bad_entry.exhausted_until - time.monotonic()) > 3000

    @respx.mock
    def test_network_error_cools_and_retries(self, client):
        """Network error → short cooldown → next key succeeds."""
        main.key_manager = KeyManager(
            [
                {"key": "key-net000", "proxy": None},
                {"key": "key-good00", "proxy": None},
            ]
        )
        call_count = 0

        def handler(request, route):
            nonlocal call_count
            call_count += 1
            key = request.url.params["key"]
            if key == "key-net000":
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200, json=gemini_ok("recovered"))

        respx.post(GENERATE_URL).mock(side_effect=handler)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "recovered"

    @respx.mock
    def test_all_retries_exhausted_returns_502(self, client):
        """All keys fail every attempt → 502."""
        main.key_manager = KeyManager([{"key": "key-bad000", "proxy": None}])
        os.environ["MAX_RETRIES"] = "2"
        main.MAX_RETRIES = 2

        respx.post(GENERATE_URL).mock(
            return_value=httpx.Response(
                503,
                json={"error": {"message": "overloaded"}},
                headers={"content-type": "application/json"},
            )
        )
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code in (429, 502)

        # restore
        main.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


class TestProxyPairing:
    """Verify correct proxy is selected per key via KeyManager."""

    @pytest.mark.asyncio
    async def test_key_keeps_its_proxy(self):
        km = KeyManager(
            [
                {"key": "key-A", "proxy": "http://proxy-A:8080"},
                {"key": "key-B", "proxy": "http://proxy-B:8080"},
            ]
        )
        e1 = await km.next_available()
        assert e1.proxy == "http://proxy-A:8080"
        e2 = await km.next_available()
        assert e2.proxy == "http://proxy-B:8080"

    def test_parse_pairs_style1(self, monkeypatch):
        monkeypatch.setenv("GEMINI_KEY_0", "key-AA")
        monkeypatch.setenv("GEMINI_PROXY_0", "http://px0:8080")
        monkeypatch.setenv("GEMINI_KEY_1", "key-BB")
        # no GEMINI_PROXY_1 → direct
        monkeypatch.delenv("GEMINI_KEY_2", raising=False)
        monkeypatch.delenv("GEMINI_PAIRS_CSV", raising=False)
        monkeypatch.delenv("GEMINI_API_CSV", raising=False)
        pairs = main._parse_key_proxy_pairs()
        assert pairs[0] == {"key": "key-AA", "proxy": "http://px0:8080"}
        assert pairs[1] == {"key": "key-BB", "proxy": None}

    def test_parse_pairs_style2(self, monkeypatch):
        monkeypatch.delenv("GEMINI_KEY_0", raising=False)
        monkeypatch.setenv("GEMINI_PAIRS_CSV", "keyX|http://px1:80,keyY")
        monkeypatch.delenv("GEMINI_API_CSV", raising=False)
        pairs = main._parse_key_proxy_pairs()
        assert pairs[0] == {"key": "keyX", "proxy": "http://px1:80"}
        assert pairs[1] == {"key": "keyY", "proxy": None}

    def test_parse_pairs_style3_legacy(self, monkeypatch):
        monkeypatch.delenv("GEMINI_KEY_0", raising=False)
        monkeypatch.delenv("GEMINI_PAIRS_CSV", raising=False)
        monkeypatch.setenv("GEMINI_API_CSV", "keyA,keyB,keyC")
        monkeypatch.setenv("PROXY_CSV", "http://px1:80,http://px2:80")
        pairs = main._parse_key_proxy_pairs()
        assert pairs[0]["key"] == "keyA" and pairs[0]["proxy"] == "http://px1:80"
        assert pairs[1]["key"] == "keyB" and pairs[1]["proxy"] == "http://px2:80"
        assert (
            pairs[2]["key"] == "keyC" and pairs[2]["proxy"] == "http://px1:80"
        )  # wraps

    def test_no_keys_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_KEY_0", raising=False)
        monkeypatch.delenv("GEMINI_PAIRS_CSV", raising=False)
        monkeypatch.delenv("GEMINI_API_CSV", raising=False)
        with pytest.raises(RuntimeError, match="No Gemini API keys"):
            main._parse_key_proxy_pairs()


class TestModelAliasResolution:
    @respx.mock
    def test_gpt4o_resolves_to_gemini_model(self, client):
        """gpt-4o alias maps to a gemini model in MODEL_MAP values."""
        resolved = resolve_model("gpt-4o")
        assert resolved in main.MODEL_MAP.values()

        url = f"{GEMINI_BASE}/models/{resolved}:generateContent"
        respx.post(url).mock(return_value=httpx.Response(200, json=gemini_ok()))
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200

    @respx.mock
    def test_unknown_model_passes_through(self, client):
        custom = "my-fine-tuned-v1"
        url = f"{GEMINI_BASE}/models/{custom}:generateContent"
        respx.post(url).mock(return_value=httpx.Response(200, json=gemini_ok()))
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": custom,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
