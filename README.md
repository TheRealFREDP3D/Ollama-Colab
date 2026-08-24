# Ollama Colab v2.6

Run Ollama on a free Google Colab GPU and expose a public HTTPS endpoint (Cloudflare Tunnel or ngrok) so you can use the model from anywhere — VS Code, curl, Python scripts, or Claude Code-style clients.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheRealFREDP3D/Ollama-Colab/blob/main/Ollama_Colab_v2.6.ipynb)

> Self-teaching project exploring LLM inference, reliable tunneling, GPU-aware model selection, and remote API access.

---

## What it does

- Installs Ollama on a Colab GPU instance
- Interactive model browser with live library scraping + static fallback
- GPU-fit filtering (only shows models/tags that fit the detected VRAM by default)
- Exposes the Ollama API over a public HTTPS tunnel
- Supports the standard Ollama API **and** the Anthropic-compatible `/v1/messages` endpoint (Ollama v0.14.0+)
- Tunnel readiness polling, process registry, and clean shutdown

---

## Quick Start

1. Click the **Open in Colab** badge above (or open the notebook manually)
2. Set GPU runtime: `Runtime → Change runtime type → T4 GPU → Save`
3. Run `Runtime → Run all`, or execute each cell in order (Steps 1–10)
4. Copy the public URL printed in Step 8
5. Test with curl, Python, or a VS Code extension

> ⚠️ **Security note**: The endpoint is public and unauthenticated. Use only for short experiments. See the [Security](#security) section below.

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
| **8 — Start Tunnel** | Creates the public HTTPS endpoint (Cloudflare quick/named or ngrok) and polls until it is reachable |
| **9 — Test Endpoint** | Sends a configurable test prompt and prints the response |
| **10 — Keep Server Alive** | Blocking loop; stops cell (■) to shut down cleanly |

---

## Tunnel Options

| Option | Credentials needed | URL stability |
|--------|--------------------|---------------|
| Cloudflare quick tunnel | None | Changes every run |
| Cloudflare named tunnel | `cloudflare_token` (+ optional `cloudflare_hostname`) | Fixed hostname |
| ngrok | `ngrok_authtoken` | Rotates every ~2 hours (free tier) |

Add secrets via the 🔑 sidebar in Colab **before** running Step 3.

| Secret name | Where to get it |
|-------------|-----------------|
| `cloudflare_token` | [Cloudflare dashboard](https://one.dash.cloudflare.com) → Networks → Tunnels → Create tunnel |
| `cloudflare_hostname` | Your named tunnel’s public URL (optional but recommended) |
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

### Structured Output (JSON Schema)

```python
import requests
import json

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
    json={
        'model': 'qwen2.5:14b',
        'prompt': 'Generate a developer profile',
        'format': schema,
        'stream': False,
    },
)
print(json.loads(response.json()['response']))
```

---

## Security

> ⚠️ **Your Ollama instance is publicly accessible without authentication.**

- Use for short testing sessions only — not permanent deployment
- Monitor GPU usage regularly in the Colab UI
- Do not send sensitive or personal data through the public endpoint
- For anything beyond quick experiments, add authentication (API key, Cloudflare Access / Zero Trust, or a reverse proxy)

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

The model browser (Step 2) automatically filters to models that fit the detected GPU by default.

---

## Documentation

- [User Guide](docs/USER_GUIDE.md) — step-by-step walkthrough and troubleshooting
- [Features](docs/FEATURES.md) — reliability features and design details
- [Changelog](CHANGELOG.md) — version history

---

## Resources

- [Ollama documentation](https://ollama.com/docs)
- [Ollama security best practices](https://github.com/ollama/ollama/blob/main/docs/security.md)
- [Colab security guide](https://research.google.com/colab/security)
- [OWASP API Security Checklist](https://owasp.org/www-project-api-security/)
