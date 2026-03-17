"""
test_proxy.py — live integration tests for the Gemini → OpenAI proxy.

Fires real HTTP requests at a running server. No imports from server code.

Start the server first:
    python main_v1.py   # or main_v2.py

Then run:
    pytest test_proxy.py -v
    pytest test_proxy.py -v -k "streaming"

Configuration (env vars or .env):
    PROXY_BASE_URL       Base URL of the running server  (default: http://localhost:8000)
    TEST_GEMINI_MODEL    Model to use in tests            (default: gemini-flash-lite-latest)
"""

import json
import os

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("PROXY_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL = os.getenv("TEST_GEMINI_MODEL", "gemini-flash-lite-latest")

# Tiny 1×1 transparent PNG (public domain)
_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_PNG_DATA_URI = f"data:image/png;base64,{_PNG_1X1_B64}"


# ── Shared HTTP client ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        # Verify the server is actually up before running any tests
        try:
            r = client.get("/health")
            r.raise_for_status()
        except Exception as exc:
            pytest.skip(f"Server not reachable at {BASE_URL}: {exc}")
        yield client


def chat(http, messages, **kwargs):
    """POST /v1/chat/completions and return parsed JSON. Asserts HTTP 200."""
    r = http.post(
        "/v1/chat/completions", json={"model": MODEL, "messages": messages, **kwargs}
    )
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:400]}"
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════
# Meta / health
# ═══════════════════════════════════════════════════════════════════════════


class TestMeta:
    def test_health(self, http):
        r = http.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["keys_available"] >= 1

    def test_stats(self, http):
        r = http.get("/stats")
        assert r.status_code == 200
        stats = r.json()["key_stats"]
        assert isinstance(stats, list) and len(stats) >= 1
        for s in stats:
            assert "key_suffix" in s
            assert "available" in s
            assert "total_requests" in s

    def test_list_models(self, http):
        r = http.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert len(body["data"]) >= 1
        for m in body["data"]:
            assert "id" in m


# ═══════════════════════════════════════════════════════════════════════════
# Basic chat completions
# ═══════════════════════════════════════════════════════════════════════════


class TestChatCompletions:
    def test_simple_response(self, http):
        body = chat(
            http, [{"role": "user", "content": "Reply with exactly the word: PONG"}]
        )
        assert body["object"] == "chat.completion"
        text = body["choices"][0]["message"]["content"]
        assert isinstance(text, str) and len(text) > 0
        assert body["choices"][0]["message"]["role"] == "assistant"

    def test_usage_metadata_present(self, http):
        body = chat(http, [{"role": "user", "content": "Hi"}])
        u = body["usage"]
        assert u["prompt_tokens"] > 0
        assert u["completion_tokens"] > 0
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]

    def test_finish_reason_is_stop(self, http):
        body = chat(http, [{"role": "user", "content": "Say hello."}])
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_system_prompt_respected(self, http):
        body = chat(
            http,
            [
                {
                    "role": "system",
                    "content": "You are a robot. Every reply must start with 'BEEP'.",
                },
                {"role": "user", "content": "Introduce yourself."},
            ],
        )
        assert "BEEP" in body["choices"][0]["message"]["content"].upper()

    def test_multi_turn_conversation(self, http):
        body = chat(
            http,
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
        # The model may answer "blue", "electric", or "Electric Blue" — all are correct.
        text = body["choices"][0]["message"]["content"].lower()
        assert "blue" in text or "electric" in text

    def test_temperature_zero_deterministic(self, http):
        msgs = [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ]
        a = chat(http, msgs, temperature=0)["choices"][0]["message"]["content"].strip()
        b = chat(http, msgs, temperature=0)["choices"][0]["message"]["content"].strip()
        assert a == b

    def test_max_tokens_limits_output(self, http):
        body = chat(
            http,
            [
                {
                    "role": "user",
                    "content": "Write a very long essay about the universe.",
                }
            ],
            max_tokens=10,
        )
        assert body["choices"][0]["finish_reason"] in ("length", "stop")
        assert body["usage"]["completion_tokens"] <= 15

    def test_stop_sequence_honoured(self, http):
        body = chat(
            http,
            [{"role": "user", "content": "Count from 1 to 10, one number per line."}],
            stop=["5"],
        )
        text = body["choices"][0]["message"]["content"]
        assert "6" not in text and "7" not in text


# ═══════════════════════════════════════════════════════════════════════════
# Structured output
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredOutput:
    _SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    def test_schema_field_returns_valid_json(self, http):
        body = chat(
            http,
            [
                {
                    "role": "user",
                    "content": "Return a JSON object for a fictional person named Ada who is 36.",
                }
            ],
            schema=self._SCHEMA,
        )
        parsed = json.loads(body["choices"][0]["message"]["content"])
        assert "name" in parsed and "age" in parsed
        assert isinstance(parsed["age"], int)

    def test_response_format_json_schema(self, http):
        body = chat(
            http,
            [
                {
                    "role": "user",
                    "content": "Return a JSON object for a person named Bob aged 25.",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": self._SCHEMA},
            },
        )
        parsed = json.loads(body["choices"][0]["message"]["content"])
        assert "name" in parsed and "age" in parsed

    def test_response_format_json_object(self, http):
        body = chat(
            http,
            [
                {
                    "role": "user",
                    "content": 'Return any JSON object with a key "ok" set to true.',
                }
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(body["choices"][0]["message"]["content"])
        assert isinstance(parsed, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Image inference
# ═══════════════════════════════════════════════════════════════════════════


class TestImageInference:
    def test_data_uri_image_accepted(self, http):
        body = chat(
            http,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                        {
                            "type": "text",
                            "text": "Acknowledge you received an image. One sentence.",
                        },
                    ],
                }
            ],
        )
        assert len(body["choices"][0]["message"]["content"]) > 0

    def test_remote_image_url(self, http):
        body = chat(
            http,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://httpbin.org/image/png"},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in one sentence.",
                        },
                    ],
                }
            ],
        )
        assert len(body["choices"][0]["message"]["content"]) > 0

    def test_mixed_text_and_image(self, http):
        body = chat(
            http,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "I'm sending you a small image."},
                        {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                        {
                            "type": "text",
                            "text": "Confirm you received both the text and the image.",
                        },
                    ],
                }
            ],
        )
        assert len(body["choices"][0]["message"]["content"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Streaming
# ═══════════════════════════════════════════════════════════════════════════


class TestStreaming:
    def test_sse_chunks_received(self, http):
        with httpx.Client(base_url=BASE_URL, timeout=60) as stream_client:
            with stream_client.stream(
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
        for d in data_lines:
            chunk = json.loads(d)
            assert chunk["object"] == "chat.completion.chunk"
            assert "delta" in chunk["choices"][0]

    def test_streamed_content_reassembles(self, http):
        with httpx.Client(base_url=BASE_URL, timeout=60) as stream_client:
            with stream_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Say 'HELLO WORLD' and nothing else.",
                        }
                    ],
                    "stream": True,
                },
            ) as r:
                raw = r.read().decode()

        full = "".join(
            json.loads(line[6:])["choices"][0]["delta"].get("content", "")
            for line in raw.splitlines()
            if line.startswith("data:") and "[DONE]" not in line
        ).upper()
        assert "HELLO" in full or "WORLD" in full

    def test_stream_ends_with_done(self, http):
        with httpx.Client(base_url=BASE_URL, timeout=60) as stream_client:
            with stream_client.stream(
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
    def test_returns_positive_count(self, http):
        r = http.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total_tokens"] > 0
        assert "model" in body

    def test_longer_prompt_has_more_tokens(self, http):
        short = http.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi."}],
            },
        ).json()["total_tokens"]

        long = http.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi. " * 200}],
            },
        ).json()["total_tokens"]

        assert long > short

    def test_system_message_counted(self, http):
        without = http.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello."}],
            },
        ).json()["total_tokens"]

        with_system = http.post(
            "/v1/token/count",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a very verbose assistant who explains everything at length.",
                    },
                    {"role": "user", "content": "Hello."},
                ],
            },
        ).json()["total_tokens"]

        assert with_system > without


# ═══════════════════════════════════════════════════════════════════════════
# Key rotation observable via /stats
# ═══════════════════════════════════════════════════════════════════════════


class TestKeyRotationObservable:
    def test_request_count_increments(self, http):
        before = sum(
            s["total_requests"] for s in http.get("/stats").json()["key_stats"]
        )
        chat(http, [{"role": "user", "content": "Ping"}])
        after = sum(s["total_requests"] for s in http.get("/stats").json()["key_stats"])
        assert after == before + 1

    def test_multiple_keys_both_used(self, http):
        stats = http.get("/stats").json()["key_stats"]
        if len(stats) < 2:
            pytest.skip("Need at least 2 keys configured to test rotation")

        n = len(stats) * 3
        for _ in range(n):
            chat(http, [{"role": "user", "content": "Hi"}])

        after = http.get("/stats").json()["key_stats"]
        assert sum(1 for s in after if s["total_requests"] > 0) >= 2
