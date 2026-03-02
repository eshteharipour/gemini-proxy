"""
test_proxy.py — live integration tests against real Gemini API.

Requires:
    - A .env file (or env vars) with at least GEMINI_KEY_0 set
    - pip install pytest pytest-asyncio httpx fastapi uvicorn pydantic python-dotenv

Run:
    pytest test_proxy.py -v
    pytest test_proxy.py -v -k "token"          # run subset
    pytest test_proxy.py -v --tb=short          # shorter tracebacks
"""

import base64
import json
import os
import time

import httpx
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()

# ── Guard: skip entire module if no real key is configured ───────────────────
if (
    not any(os.environ.get(f"GEMINI_KEY_{i}") for i in range(10))
    and not os.environ.get("GEMINI_API_CSV")
    and not os.environ.get("GEMINI_PAIRS_CSV")
):
    pytest.skip(
        "No Gemini API key found. Set GEMINI_KEY_0 (or GEMINI_API_CSV / GEMINI_PAIRS_CSV) to run integration tests.",
        allow_module_level=True,
    )

import main  # noqa: E402 — import after env is loaded
from main import KeyManager

# ── Use a fast, cheap model for all tests ────────────────────────────────────
MODEL = os.environ.get("TEST_GEMINI_MODEL", "gemini-flash-lite-latest")

# ── Tiny 1x1 transparent PNG for image tests ────────────────────────────────
_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_PNG_DATA_URI = f"data:image/png;base64,{_PNG_1X1_B64}"


# ── Shared app client ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """Single TestClient for the whole test module — starts the app once."""
    pairs = main._parse_key_proxy_pairs()
    main.key_manager = KeyManager(pairs)
    return TestClient(main.app)


def chat(client, messages, **kwargs):
    """Helper: POST /v1/chat/completions and return the parsed JSON."""
    payload = {"model": MODEL, "messages": messages, **kwargs}
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, f"Unexpected status {r.status_code}: {r.text}"
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════
# Health / meta endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestMeta:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["keys_available"] >= 1

    def test_stats(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        stats = r.json()["key_stats"]
        assert isinstance(stats, list)
        assert len(stats) >= 1
        first = stats[0]
        assert "key_suffix" in first
        assert "available" in first
        assert "total_requests" in first

    def test_list_models(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert len(ids) >= 1
        # at least the test model (or its alias) is present
        assert any(MODEL in i or i in MODEL for i in ids)


# ═══════════════════════════════════════════════════════════════════════════
# Basic chat completions
# ═══════════════════════════════════════════════════════════════════════════


class TestChatCompletions:
    def test_simple_response(self, client):
        body = chat(
            client, [{"role": "user", "content": "Reply with exactly the word: PONG"}]
        )
        text = body["choices"][0]["message"]["content"]
        assert isinstance(text, str)
        assert len(text) > 0
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["object"] == "chat.completion"

    def test_usage_metadata_present(self, client):
        body = chat(client, [{"role": "user", "content": "Hi"}])
        usage = body["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_finish_reason_is_stop(self, client):
        body = chat(client, [{"role": "user", "content": "Say hello."}])
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_system_prompt_respected(self, client):
        body = chat(
            client,
            [
                {
                    "role": "system",
                    "content": "You are a robot. Every reply must start with 'BEEP'.",
                },
                {"role": "user", "content": "Introduce yourself."},
            ],
        )
        text = body["choices"][0]["message"]["content"]
        assert "BEEP" in text.upper()

    def test_multi_turn_conversation(self, client):
        body = chat(
            client,
            [
                {
                    "role": "user",
                    "content": "My favourite colour is electric blue. Remember it.",
                },
                {
                    "role": "assistant",
                    "content": "Got it! Your favourite colour is electric blue.",
                },
                {
                    "role": "user",
                    "content": "What is my favourite colour? One word answer.",
                },
            ],
        )
        text = body["choices"][0]["message"]["content"].lower()
        assert "blue" in text

    def test_temperature_zero_deterministic(self, client):
        """Same prompt at temp=0 should return the same (or very similar) text twice."""
        messages = [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ]
        a = chat(client, messages, temperature=0)["choices"][0]["message"][
            "content"
        ].strip()
        b = chat(client, messages, temperature=0)["choices"][0]["message"][
            "content"
        ].strip()
        assert a == b

    def test_max_tokens_limits_output(self, client):
        body = chat(
            client,
            [
                {
                    "role": "user",
                    "content": "Write a very long essay about the universe.",
                }
            ],
            max_tokens=10,
        )
        # with a tiny token budget the model must stop early
        assert body["choices"][0]["finish_reason"] in ("length", "stop")
        assert body["usage"]["completion_tokens"] <= 15  # small buffer

    def test_stop_sequence_honoured(self, client):
        body = chat(
            client,
            [{"role": "user", "content": "Count from 1 to 10, one number per line."}],
            stop=["5"],
        )
        text = body["choices"][0]["message"]["content"]
        # model should have stopped at or before "5"
        assert "6" not in text and "7" not in text


# ═══════════════════════════════════════════════════════════════════════════
# Structured output
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredOutput:
    _PERSON_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    def test_schema_field_returns_valid_json(self, client):
        body = chat(
            client,
            [
                {
                    "role": "user",
                    "content": "Return a JSON object for a fictional person named Ada who is 36 years old.",
                }
            ],
            schema=self._PERSON_SCHEMA,
        )
        text = body["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        assert "name" in parsed
        assert "age" in parsed
        assert isinstance(parsed["age"], int)

    def test_response_format_json_schema(self, client):
        body = chat(
            client,
            [
                {
                    "role": "user",
                    "content": "Return a JSON object for a person named Bob aged 25.",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": self._PERSON_SCHEMA},
            },
        )
        parsed = json.loads(body["choices"][0]["message"]["content"])
        assert "name" in parsed
        assert "age" in parsed

    def test_response_format_json_object(self, client):
        """json_object mode: no schema enforcement, but output must be valid JSON."""
        body = chat(
            client,
            [
                {
                    "role": "user",
                    "content": 'Return any JSON object with a key "ok" set to true.',
                }
            ],
            response_format={"type": "json_object"},
        )
        text = body["choices"][0]["message"]["content"]
        parsed = json.loads(text)  # must not raise
        assert isinstance(parsed, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Image inference
# ═══════════════════════════════════════════════════════════════════════════


class TestImageInference:
    def test_data_uri_image_accepted(self, client):
        body = chat(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                        {
                            "type": "text",
                            "text": "What colour is the dominant pixel in this image? One word.",
                        },
                    ],
                },
            ],
        )
        text = body["choices"][0]["message"]["content"]
        assert isinstance(text, str) and len(text) > 0

    def test_remote_image_url(self, client):
        # Public 1x1 pixel image served by httpbin
        url = "https://httpbin.org/image/png"
        body = chat(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {
                            "type": "text",
                            "text": "Describe this image in one sentence.",
                        },
                    ],
                },
            ],
        )
        text = body["choices"][0]["message"]["content"]
        assert isinstance(text, str) and len(text) > 0

    def test_mixed_text_and_image(self, client):
        body = chat(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "I'm sending you an image."},
                        {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                        {
                            "type": "text",
                            "text": "Acknowledge you received both the text and the image.",
                        },
                    ],
                },
            ],
        )
        text = body["choices"][0]["message"]["content"].lower()
        assert len(text) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Streaming
# ═══════════════════════════════════════════════════════════════════════════


class TestStreaming:
    def test_sse_chunks_received(self, client):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            raw = r.read().decode()

        data_lines = [
            line[6:]
            for line in raw.splitlines()
            if line.startswith("data:") and "[DONE]" not in line
        ]
        assert len(data_lines) >= 1

        chunks = [json.loads(d) for d in data_lines]
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk
            assert "delta" in chunk["choices"][0]

    def test_streamed_content_reassembles(self, client):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "Say 'HELLO WORLD' and nothing else."}
                ],
                "stream": True,
            },
        ) as r:
            raw = r.read().decode()

        texts = []
        for line in raw.splitlines():
            if line.startswith("data:") and "[DONE]" not in line:
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                texts.append(delta)

        full = "".join(texts).upper()
        assert "HELLO" in full or "WORLD" in full

    def test_stream_ends_with_done(self, client):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "One word: yes."}],
                "stream": True,
            },
        ) as r:
            raw = r.read().decode()

        assert "data: [DONE]" in raw


# ═══════════════════════════════════════════════════════════════════════════
# Token counting
# ═══════════════════════════════════════════════════════════════════════════


class TestTokenCount:
    def test_returns_positive_count(self, client):
        r = client.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total_tokens"] > 0
        assert body["model"] == main.resolve_model(MODEL)

    def test_longer_prompt_has_more_tokens(self, client):
        short = client.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi."}],
            },
        ).json()["total_tokens"]

        long = client.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi. " * 200}],
            },
        ).json()["total_tokens"]

        assert long > short

    def test_system_message_counted(self, client):
        without = client.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello."}],
            },
        ).json()["total_tokens"]

        with_system = client.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a very verbose assistant who always explains at length.",
                    },
                    {"role": "user", "content": "Hello."},
                ],
            },
        ).json()["total_tokens"]

        assert with_system > without


# ═══════════════════════════════════════════════════════════════════════════
# Key rotation (observable via /stats)
# ═══════════════════════════════════════════════════════════════════════════


class TestKeyRotationObservable:
    def test_request_count_increments(self, client):
        before = client.get("/stats").json()["key_stats"]
        total_before = sum(s["total_requests"] for s in before)

        chat(client, [{"role": "user", "content": "Ping"}])

        after = client.get("/stats").json()["key_stats"]
        total_after = sum(s["total_requests"] for s in after)

        assert total_after == total_before + 1

    def test_multiple_keys_both_used(self, client):
        """If there are 2+ keys, after enough requests both should have requests."""
        stats = client.get("/stats").json()["key_stats"]
        if len(stats) < 2:
            pytest.skip("Need at least 2 keys to test rotation")

        # fire enough requests to guarantee both keys are hit
        for _ in range(len(stats) * 3):
            chat(client, [{"role": "user", "content": "Hi"}])

        after = client.get("/stats").json()["key_stats"]
        used = [s for s in after if s["total_requests"] > 0]
        assert len(used) >= 2
