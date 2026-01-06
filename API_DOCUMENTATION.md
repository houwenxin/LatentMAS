# LatentMAS API Documentation

OpenAI-compatible API with latent multi-agent reasoning support.

## Quick Start

### Start the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start server with default model on single device
python server.py --model_name Qwen/Qwen2.5-7B-Instruct --devices cuda:0

# With multiple devices (load balancing across GPUs)
python server.py --model_name Qwen/Qwen2.5-7B-Instruct --devices cuda:0,cuda:1,cuda:2
```

### Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model name or path |
| `--devices` | `cuda:0` | Comma-separated list of devices for load balancing (e.g., `cuda:0,cuda:1`) |
| `--latent_steps` | `10` | Default latent reasoning steps |
| `--cache_ttl` | `1800` | Session cache TTL in seconds (30 min) |
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8000` | Port to bind to |

---

## API Endpoints

### POST `/v1/chat/completions`

OpenAI-compatible chat completions with LatentMAS extensions.

#### Request Body

```json
{
  "model": "latent-mas",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Your prompt here"}
  ],
  "mode": "normal | latent | text",
  "session_id": "optional-session-id",
  "latent_steps": 10,
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "add_think_token": false,
  "latent_space_realign": false,
  "debug_max_tokens": 50,
  "debug_continuation_prompt": null,
  "latent_only": false
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `messages` | `List[Message]` | Yes | - | Chat messages |
| `mode` | `string` | No | `"normal"` | Operation mode: `normal`, `latent`, or `text` |
| `session_id` | `string` | Conditional | `null` | Session ID for KV cache. Required for `text` mode |
| `latent_steps` | `int` | No | `null` | Fixed latent steps. If `null`, generates until EOS |
| `max_tokens` | `int` | No | `256` | Max tokens to generate (also max latent steps when `latent_steps` is null) |
| `temperature` | `float` | No | `0.7` | Sampling temperature |
| `top_p` | `float` | No | `0.95` | Top-p sampling |
| `add_think_token` | `bool` | No | `false` | Append `<think>` token for reasoning models |
| `latent_space_realign` | `bool` | No | `false` | Enable latent space realignment for better quality (only for `latent` mode) |
| `debug_max_tokens` | `int` | No | `50` | Max tokens for debug preview in latent mode |
| `debug_continuation_prompt` | `string` | No | `null` | Continuation prompt for debug text generation |
| `latent_only` | `bool` | No | `false` | Whether to keep input context in KV cache |

#### Response Body

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "latent-mas",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Generated text..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "latent_steps": 10
  },
  "session_id": "uuid-if-latent-mode"
}
```

---

## Operation Modes

### Mode: `normal`

Standard text generation without latent reasoning. Equivalent to a regular LLM API call.

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "mode": "normal",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        "max_tokens": 100
    }
)
print(response.json()["choices"][0]["message"]["content"])
# Output: "2 + 2 equals 4."
```

### Mode: `latent`

Generate latent representations and cache KV values. Used for "thinking" agents that don't produce visible output.

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "mode": "latent",
        "messages": [
            {"role": "user", "content": "Plan how to solve: What is 2 + 2?"}
        ],
        "latent_steps": 10,
        "latent_space_realign": true,  # Enable for better quality
        "debug_max_tokens": 100  # Preview what model would generate
    }
)
session_id = response.json()["session_id"]
# Content shows debug preview: "[Latent: 10 steps] The problem involves..."
```

**Key behaviors:**
- Creates/updates a session with accumulated KV cache
- Returns `session_id` for subsequent calls
- Debug text shows what model would generate (for debugging)
- If `latent_steps` is `null`, generates until EOS token
- Set `latent_space_realign: true` for better quality latent representations

### Mode: `text`

Generate text using cached KV values from previous latent calls.

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "mode": "text",
        "messages": [
            {"role": "user", "content": "Now give the final answer:"}
        ],
        "session_id": session_id,  # From previous latent call
        "max_tokens": 100
    }
)
print(response.json()["choices"][0]["message"]["content"])
# Output uses context from latent reasoning
```

---

## Session Management Endpoints

### GET `/v1/sessions/{session_id}`

Check if a session exists.

```bash
curl http://localhost:8000/v1/sessions/your-session-id
# {"session_id": "your-session-id", "exists": true}
```

### DELETE `/v1/sessions/{session_id}`

Delete a session and free its cached KV values.

```bash
curl -X DELETE http://localhost:8000/v1/sessions/your-session-id
# {"session_id": "your-session-id", "deleted": true}
```

### GET `/v1/sessions`

Get count of active sessions.

```bash
curl http://localhost:8000/v1/sessions
# {"active_sessions": 5}
```

### GET `/health`

Health check endpoint.

```bash
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true, "devices": ["cuda:0", "cuda:1"], "active_sessions": 5}
```

---

## Multi-Device Support

The API supports loading the model on multiple GPUs for automatic load balancing using round-robin scheduling.

### Configuration

```bash
# Single device (default)
python server.py --devices cuda:0

# Multiple GPUs - requests are distributed round-robin
python server.py --devices cuda:0,cuda:1,cuda:2,cuda:3

# Mix CPU and GPU
python server.py --devices cuda:0,cpu
```

### Behavior

- **Load Balancing**: Each incoming request is assigned to the next available device in round-robin order
- **Session Affinity**: Sessions (KV cache) are device-independent - any device can use cached KV values from previous calls
- **Memory Efficiency**: Each device loads a separate model instance, so memory usage scales linearly with the number of devices
- **Fault Tolerance**: If one device fails, other devices continue serving requests (requires restart to recover failed device)

### Example

```python
# Server started with: --devices cuda:0,cuda:1

# Request 1 → cuda:0
# Request 2 → cuda:1  
# Request 3 → cuda:0
# Request 4 → cuda:1
# ...and so on
```

### Monitoring

Check active devices via health endpoint:

```bash
curl http://localhost:8000/health
# Returns: {"devices": ["cuda:0", "cuda:1"], ...}
```

---

## Usage Patterns

### Pattern 1: Multi-Agent Sequential Reasoning

```python
import requests

BASE_URL = "http://localhost:8000"
question = "What is the average speed if I travel 60km at 30km/h and 60km at 60km/h?"

# Agent 1: Planner (latent)
r1 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
    "mode": "latent",
    "messages": [{"role": "user", "content": f"Plan how to solve: {question}"}],
    "latent_steps": 10
})
session_id = r1.json()["session_id"]

# Agent 2: Critic (latent, reuses session)
r2 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
    "mode": "latent",
    "messages": [{"role": "user", "content": "Review and critique the plan."}],
    "session_id": session_id,
    "latent_steps": 10
})

# Agent 3: Refiner (latent, reuses session)
r3 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
    "mode": "latent",
    "messages": [{"role": "user", "content": "Refine the plan based on feedback."}],
    "session_id": session_id,
    "latent_steps": 10
})

# Agent 4: Judger (text, produces final output)
r4 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
    "mode": "text",
    "messages": [{"role": "user", "content": f"Solve: {question}"}],
    "session_id": session_id,
    "max_tokens": 500
})
print(r4.json()["choices"][0]["message"]["content"])
```

### Pattern 2: Dynamic Latent Steps (Until EOS)

```python
# Let the model decide when to stop thinking
response = requests.post(f"{BASE_URL}/v1/chat/completions", json={
    "mode": "latent",
    "messages": [{"role": "user", "content": "Think through this problem..."}],
    # latent_steps not specified - will stop at EOS
    "max_tokens": 100,  # Maximum latent steps allowed
    "debug_max_tokens": 200
})
actual_steps = response.json()["usage"]["latent_steps"]
print(f"Model took {actual_steps} latent steps")
```

### Pattern 3: Debug Preview with Custom Continuation

```python
response = requests.post(f"{BASE_URL}/v1/chat/completions", json={
    "mode": "latent",
    "messages": [{"role": "user", "content": "Analyze this problem..."}],
    "latent_steps": 10,
    "debug_max_tokens": 100,
    "debug_continuation_prompt": "Based on my analysis, the key insight is"
})
# Debug output starts with the continuation prompt
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Server (FastAPI)                      │
│  server.py                                                       │
├─────────────────────────────────────────────────────────────────┤
│  POST /v1/chat/completions                                       │
│    ├── mode=normal  → generate_text_for_api()                   │
│    ├── mode=latent  → generate_latent_for_api() + cache         │
│    └── mode=text    → load cache + generate_text_for_api()      │
├─────────────────────────────────────────────────────────────────┤
│                      Model Wrapper (HuggingFace)                 │
│  models.py                                                       │
│    ├── generate_latent_batch_with_tokens() - latent reasoning   │
│    ├── generate_text_batch() - text generation                  │
│    └── _apply_latent_realignment() - hidden state transform     │
├─────────────────────────────────────────────────────────────────┤
│                      Cache Manager                               │
│  cache_manager.py                                                │
│    ├── {session_id: past_key_values} storage                    │
│    ├── TTL-based expiration                                     │
│    └── Thread-safe operations                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Known Limitations

1. **HuggingFace Backend Only**: Current implementation uses HF transformers. vLLM backend requires custom fork with `enable_prompt_embeds` support.

2. **Sequential Mode Only**: Hierarchical mode (parallel agents) requires embedding concatenation which isn't supported in HF backend due to position encoding issues.

3. **Prompt Structure**: Each API call sends a complete prompt. The model sees multiple system messages if you include them in every call. For best results:
   - Use system message only in the first call
   - Use minimal continuation prompts for subsequent calls

4. **KV Cache Size**: For large models (14B+), KV cache can be several hundred MB per session. Monitor memory usage with many concurrent sessions.

---

## Files Overview

| File | Description |
|------|-------------|
| `server.py` | FastAPI server with `/v1/chat/completions` endpoint |
| `models.py` | ModelWrapper with latent generation methods |
| `cache_manager.py` | Thread-safe KV cache storage with TTL |
| `test_api.py` | Test script demonstrating API usage |
