# 🚀 Gemini → OpenAI Proxy

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
  
A lightweight, self-hosted proxy that exposes Google's **Gemini** API behind an
**OpenAI-compatible** `/v1/chat/completions` interface.

Drop it in front of any tool or library that speaks the OpenAI protocol and
transparently route every request to Gemini — with **multi-key rotation**,
**per-key proxy support**, **streaming (SSE)**, **vision (images)**, **structured
output**, and **token counting**.

---

## ✨ Features

| Feature | Details |
|---|---|
| **OpenAI-compatible API** | `POST /v1/chat/completions` — works with any OpenAI SDK or client |
| **Streaming** | Server-Sent Events (SSE) with `stream: true`, including `[DONE]` sentinel |
| **Vision / Images** | `data:` URIs and remote `https://` image URLs auto-converted to Gemini `inlineData` |
| **Structured Output** | `schema`, `response_format.json_schema`, and `response_format.type = "json_object"` |
| **Token Counting** | `POST /v1/token/count` returns `usage.total_tokens` |
| **Multi-Key Rotation** | Round-robin with automatic **cooldown** on 429 / 503 errors |
| **Per-Key Proxy** | Bind each Gemini API key to its own HTTP proxy (or use direct) |
| **Two Backends** | `main_v1.py` — raw REST via `httpx` · `main_v2.py` — official `google-genai` SDK |
| **Debug Logging** | Optional JSONL request/response log with base64 image redaction |
| **Docker-ready** | Minimal `Dockerfile` included |

---

## 🏛️ Architecture

```
OpenAI Client ──▶ Proxy (FastAPI) ──▶ Key Manager (round-robin) ──▶ Gemini API
                  │                       │
                  │                       ├─ Per-key proxy support
                  │                       └─ Auto-cooldown on 429/503
                  │
                  ├─ /v1/chat/completions
                  ├─ /v1/token/count
                  ├─ /health
                  └─ /stats
```

---

## 📋 Prerequisites

- **Python 3.9+**
- **Docker** *(optional — for containerized deployment)*

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/eshteharipour/gemini-proxy.git
cd gemini-proxy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set your Gemini API key(s)
```

### 3. Run

```bash
# REST backend (httpx)
python main_v1.py

# — or — SDK backend (google-genai)
python main_v2.py
```

The server starts on `http://0.0.0.0:8000` by default.

### 4. Use

Point any OpenAI-compatible client at your proxy:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="anything",          # not checked by the proxy
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",    # any valid Gemini model name
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## 🐳 Docker

```bash
docker build -t gemini-proxy .
docker run --env-file .env -p 8000:8000 gemini-proxy
```

> **Note:** The `Dockerfile` copies `main.py`. Rename your preferred backend
> (`main_v1.py` or `main_v2.py`) to `main.py` before building, or adjust the
> `Dockerfile` accordingly.

---

## 📡 API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completion. Supports both streaming and non-streaming.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | **Yes** | Gemini model name (e.g. `gemini-2.0-flash`) |
| `messages` | array | **Yes** | Array of message objects (`role` + `content`) |
| `temperature` | float | No | Sampling temperature (default `1.0`) |
| `top_p` | float | No | Nucleus sampling |
| `max_tokens` | int | No | Maximum tokens to generate |
| `stream` | bool | No | Enable SSE streaming (default `false`) |
| `schema` | object | No | JSON Schema for structured output |
| `response_format` | object | No | `{"type": "json_schema", "json_schema": {"schema": {...}}}` or `{"type": "json_object"}` |

**Non-streaming response:**

```json
{
  "id": "chatcmpl-gemini-abc123",
  "object": "chat.completion",
  "created": 1711234567,
  "model": "gemini-2.0-flash",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Hello! How can I help?" },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 6,
    "total_tokens": 14
  }
}
```

**Streaming response (SSE):**

```json
data: {"id":"chatcmpl-gemini-abc123","object":"chat.completion.chunk","created":1711234567,"model":"gemini-2.0-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-gemini-abc123","object":"chat.completion.chunk","created":1711234567,"model":"gemini-2.0-flash","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]
```

---

### `POST /v1/token/count`

| Field | Type | Required |
|---|---|---|
| `model` | string | **Yes** |
| `messages` | array | **Yes** |

**Response:**

```json
{
  "model": "gemini-2.0-flash",
  "usage": { "total_tokens": 42 }
}
```

---

### `GET /health`

```json
{ "status": "ok", "keys_available": 3, "keys_total": 3 }
```

Returns `503` with `{"status": "starting"}` while the key manager is initializing.

---

### `GET /stats`

```json
{
  "key_stats": [
    { "key": "AIza...XXX", "total_requests": 120, "total_errors": 2 }
  ]
}
```

---

## 🖼️ Vision (Image Support)

Send images exactly as you would with the OpenAI API:

```json
{
  "model": "gemini-2.0-flash",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "image_url", "image_url": { "url": "https://example.com/photo.jpg" } },
        { "type": "text", "text": "Describe this image." }
      ]
    }
  ]
}
```

Both **base64 data URIs** (`data:image/png;base64,...`) and **remote HTTPS URLs** are supported.
Remote images are fetched server-side (optionally through `GENERAL_PROXY`).

> **💡 Tip:** If you need a proxy to fetch remote images, set the `GENERAL_PROXY`
> environment variable in your `.env` file. This proxy is used *only* for image
> fetching, not for Gemini API calls.

---

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

### API Keys & Proxies

Three configuration styles are supported (use **one**):

| Style | Variables | Description |
|---|---|---|
| **1 — Indexed pairs** (recommended) | `GEMINI_KEY_0`, `GEMINI_PROXY_0`, … | Each key bound to its own proxy. Omit proxy for direct. |
| **2 — CSV pairs** | `GEMINI_PAIRS_CSV` | `key\|proxy,key\|proxy,key` (proxy optional per entry) |
| **3 — Legacy CSV** | `GEMINI_API_CSV` + `PROXY_CSV` | Keys round-robin over proxy list |

### Retry & Cooldown

| Variable | Default | Description |
|---|---|---|
| `KEY_COOLDOWN_SECONDS` | `60` | Seconds a key is benched after 429 / 503 |
| `ALL_EXHAUSTED_SLEEP_SECONDS` | `5` | Sleep before rechecking when all keys are exhausted |
| `MAX_RETRIES` | `5` | Max attempts per request before returning 502 |

### Debug

| Variable | Default | Description |
|---|---|---|
| `DEBUG_MODE` | `0` | Set to `1` for verbose logging + JSONL trace file |
| `DEBUG_LOG_FILE` | `gemini_proxy_debug.log` | Path for the JSONL debug log |

### Other

| Variable | Default | Description |
|---|---|---|
| `GENERAL_PROXY` | _(none)_ | HTTP proxy for fetching remote images (not for Gemini API calls) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

---

## 🏗️ v1 vs v2

| | `main_v1.py` | `main_v2.py` |
|---|---|---|
| **HTTP layer** | Raw `httpx` calls to Gemini REST API | Official `google-genai` SDK |
| **Proxy handling** | `httpx` client `proxy` param per request | Env-var patching with a global lock |
| **Dependencies** | Lighter (no SDK) | Requires `google-genai` |
| **API surface** | Identical | Identical |

Choose **v1** if you prefer fewer dependencies and direct control.
Choose **v2** if you want the official SDK's built-in features and future compatibility.

---

## 🧪 Testing

```bash
pip install pytest respx pytest-asyncio
pytest
```

**Test coverage includes:**

- ✅ Structured output (`schema` & `response_format`)
- ✅ Vision — data URI & remote URL
- ✅ Streaming (SSE) with `[DONE]` sentinel
- ✅ Token counting
- ✅ Health & stats endpoints
- ✅ Key rotation with cooldown logic

---

## 📄 License

[MIT](LICENSE)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repo
2. Create your feature branch: `git checkout -b feat/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feat/amazing-feature`
5. Open a Pull Request
