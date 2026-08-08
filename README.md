# Ollama Colab v2.6

Run Ollama (a tool for running large language models locally) on Google Colab with a public HTTPS endpoint via Cloudflare Tunnel or ngrok.

> Built as a self-teaching project exploring LLM inference, tunneling, and remote API access.

---

## What it does

- Installs Ollama on a Colab GPU instance
- Lets you browse and select any model from the Ollama library
- Exposes the Ollama API over a public HTTPS tunnel
- Supports the standard Ollama API *and* the Anthropic-compatible `/v1/messages` endpoint (Ollama v0.14.0+)

---

## Quick Start

1. Open the notebook in Google Colab
2. Set GPU runtime: `Runtime → Change runtime type → T4 GPU → Save`
3. Run `Runtime → Run all`, or execute each cell in order (Steps 1–10)
4. Copy the public URL printed in Step 8
5. Test with curl, Python, or a VS Code extension

---

## Notebook Structure

| Step | What it does |
|------|-------------|
| **1 — Install Dependencies** | Checks Python/GPU, installs missing packages, initialises shared globals (including GPU detection used for sizing) |
| **2 — Model Browser** | Interactive widget to browse/filter/select Ollama models and tags. By default only shows models/tags that fit the detected GPU's VRAM, smallest first — uncheck "Only show models that fit..." to browse the full library |
| **3 — Configure Environment** | Memory management policy, tunnel type, loads Colab Secrets |
| **4 — GPU Monitoring** | One-shot GPU snapshot; `monitor_gpu()` available for live stats |
| **5 — Install Ollama** | Downloads and installs the Ollama binary via the official install script |
| **6 — Start Ollama Server** | Launches `ollama serve` in the background, waits for the API to respond |
| **7 — Pull Model** | Downloads the selected model with streamed progress output |
| **8 — Start Tunnel** | Creates the public HTTPS endpoint (Cloudflare quick/named or ngrok) |
| **9 — Test Endpoint** | Sends a configurable test prompt and prints the response |
| **10 — Keep Server Alive** | Blocking loop; stops cell (■) to shut down cleanly |

---

## Tunnel Options

| Option | Credentials needed | URL stability |
|--------|--------------------|---------------|
| Cloudflare quick tunnel | None | Changes every run |
| Cloudflare named tunnel | `cloudflare_token` secret | Fixed hostname |
| ngrok | `ngrok_authtoken` secret | Rotates every ~2 hours (free tier) |

Add secrets via the 🔑 sidebar in Colab before running Step 3.

| Secret name | Where to get it |
|---|---|
| `cloudflare_token` | [Cloudflare dashboard](https://one.dash.cloudflare.com) → Networks → Tunnels → Create tunnel |
| `ngrok_authtoken` | [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) |

---

## API Usage

Replace `https://your-url` with the URL printed in Step 8.

### curl

```bash
# Chat
curl -X POST https://your-url/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen2.5:14b", "messages": [{"role": "user", "content": "Explain recursion"}]}'

# Anthropic-compatible endpoint (Ollama v0.14.0+)
curl -X POST https://your-url/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen2.5:14b", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Python

```python
import requests

response = requests.post(
    'https://your-url/api/chat',
    json={
        'model':    'qwen2.5:14b',
        'messages': [{'role': 'user', 'content': 'Write a Flask API'}],
        'stream':   False,
    }
)
print(response.json()['message']['content'])
```

### Structured Output

```python
schema = {
    'type': 'object',
    'properties': {
        'name':   {'type': 'string'},
        'age':    {'type': 'number'},
        'skills': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['name', 'age'],
}

response = requests.post(
    'https://your-url/api/generate',
    json={'model': 'qwen2.5:14b', 'prompt': 'Generate a developer profile',
          'format': schema, 'stream': False},
)
```

---

## Security

> ⚠️ Your Ollama instance is **publicly accessible without authentication**.

- Use for short testing sessions only
- Monitor GPU usage regularly in the Colab UI
- Do not send sensitive or personal data through the public endpoint
- For production use, add authentication (API key, OAuth, or a reverse proxy)

### Emergency shutdown

```python
for name, proc in _bg_processes.items():
    try:
        proc.terminate()
        print(f'Terminated: {name}')
    except Exception:
        pass
```

---

## Colab Limits

| Tier | Session length | GPU |
|------|---------------|-----|
| Free | A few hours | T4 (16 GB VRAM) |
| Pro | Up to 24 hours | L4 (24 GB VRAM) |
| Pro+ | Up to 24 hours | A100 (40 GB VRAM) |

---

## Recommended Models by GPU

| Model | Size on disk | Fits on T4? |
|-------|-------------|-------------|
| `qwen2.5:7b` | ~4 GB | ✅ Comfortable |
| `qwen2.5-coder:14b` | ~9 GB | ✅ Comfortable |
| `deepseek-r1:14b` | ~9 GB | ✅ Comfortable |
| `llama3.1:8b` | ~5 GB | ✅ Comfortable |
| `qwen2.5:32b` | ~20 GB | ⚠️ May exceed T4 VRAM |
| `deepseek-r1:32b` | ~20 GB | ⚠️ May exceed T4 VRAM |

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Resources

- [Ollama documentation](https://ollama.com/docs)
- [Ollama security best practices](https://github.com/ollama/ollama/blob/main/docs/security.md)
- [Colab security guide](https://research.google.com/colab/security)
- [OWASP API Security Checklist](https://owasp.org/www-project-api-security/)
