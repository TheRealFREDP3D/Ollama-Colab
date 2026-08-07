# Ollama Colab — User Guide

This guide walks through using the `Ollama_Colab_v2.6.ipynb` notebook end-to-end.

> Requirements: a Google account with Colab access; a GPU runtime (T4/L4/A100). The free tier gives a **T4 (16 GB VRAM)**.

---

## 1. Open and set up
1. Open `Ollama_Colab_v2.6.ipynb` in [Google Colab](https://colab.research.google.com/).
2. `Runtime → Change runtime type → T4 GPU → Save`.
3. Run cells in order (Steps 1–10), or `Runtime → Run all`.

## 2. Step-by-step

| Step | What to do | Expected result |
|---|---|---|
| 1 — Install Dependencies | Let the cell finish. | Python/GPU printed; all packages ✅. `GPU_INFO`, `_bg_processes` initialised. |
| 2 — Model Browser | Filter by capability, toggle the GPU-fit checkbox, pick a model + tag, click **✅ Use this model**. | `MODEL_NAME` is set (e.g. `deepseek-r1:1.5b`). |
| 3 — Configure Environment | Pick a tunnel + memory policy. Add secrets (see below) **before running** if using named/ ngrok. | `TUNNEL_METHOD`, `tunnel_type`, memory policy persisted to `os.environ`. |
| 4 — GPU Monitoring | One-shot snapshot. | VRAM/utilisation printed. Optionally run `monitor_gpu()` in a new cell. |
| 5 — Install Ollama | Wait for `ollama --version`. | ✅ installed. |
| 6 — Start Server | Waits for `localhost:11434` to respond. | ✅ Ollama is up. |
| 7 — Pull Model | Streams download progress. | ✅ Successfully pulled `<MODEL_NAME>`. |
| 8 — Start Tunnel | Waits until the *public* URL answers `/api/tags`. | Public `https://…` URL printed. |
| 9 — Test Endpoint | Sends `TEST_PROMPT` (editable) to `/api/chat`. | Model reply printed. |
| 10 — Keep Server Alive | Leave running while you work. | Server stays up; **■** to shut down cleanly. |

## 3. Tunnel secrets (Colab 🔑 sidebar)
| Secret | Source |
|---|---|
| `cloudflare_token` | Cloudflare dashboard → Networks → Tunnels → Create tunnel → "Install Token" |
| `ngrok_authtoken` | https://dashboard.ngrok.com/get-started/your-authtoken |
| `cloudflare_hostname` *(optional)* | Your named-tunnel hostname, e.g. `https://ollama.example.com` |

Add secrets before Step 3. The quick tunnel needs none.

## 4. Tunnel backends
- **Cloudflare quick tunnel** (default): no setup; URL changes every run.
- **Cloudflare named tunnel**: fixed hostname — best for longer sessions.
- **ngrok**: URL rotates ~2 h on the free tier.

## 5. API usage (replace `https://your-url`)

**Chat:**
```bash
curl -X POST https://your-url/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:14b","messages":[{"role":"user","content":"Explain recursion"}]}'
```

**Generate (streaming):**
```bash
curl -X POST https://your-url/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:14b","prompt":"Hello, world!","stream":true}'
```

**Anthropic-compatible** (Ollama v0.14.0+, enables Claude Code / CodeGPT-style clients):
```bash
curl -X POST https://your-url/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:14b","max_tokens":1024,"messages":[{"role":"user","content":"Hello!"}]}'
```

**Structured output (JSON schema):**
```python
import requests, json
schema = {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"number"},"skills":{"type":"array","items":{"type":"string"}}},"required":["name","age"]}
r = requests.post('https://your-url/api/generate', json={"model":"qwen2.5:14b","prompt":"Generate a developer profile","format":schema,"stream":False})
print(json.loads(r.json()["response"]))
```

## 6. VS Code integration (Continue / CodeGPT)
1. Install the extension.
2. Set provider = **Ollama**, base URL = your tunnel URL.
3. Select model e.g. `qwen2.5:14b`.

## 7. Recommended models by GPU
| GPU | Good picks |
|---|---|
| T4 (16 GB) | `qwen2.5:7b`, `qwen2.5-coder:14b`, `deepseek-r1:14b`, `llama3.1:8b` |
| L4 (24 GB) | `qwen2.5:14b`, `deepseek-r1:14b`, `llama3.1:70b` |
| A100 (40/80 GB) | `qwen2.5:32b`, `deepseek-r1:32b`, `llama3.1:70b` |

The model browser (Step 2) auto-filters to the detected GPU by default.

## 8. Troubleshooting
- **"Ollama is up" but `/api/tags` returns 404** — retry in a moment; the quick tunnel edge takes a few seconds to route after the URL appears (Step 8 already polls, but re-run Step 9 if timing was tight).
- **Model won't fit VRAM** — switch the memory policy to **Auto-unload** (Step 3) or pick a smaller tag in Step 2.
- **ngrok URL rotated** — re-run Step 8 to refresh, then update your client.
- **Cloudflare named tunnel fails** — verify the token/hostname secret and that the tunnel is running in your Cloudflare dashboard.
- **Cell errored out of order** — run Step 1 first to define `_bg_processes`, `MODEL_NAME`, `GPU_INFO`, `estimate_model_size_gb`.

## 9. Security & cost
- The endpoint is **public, unauthenticated**, and billed to your Colab GPU quota. Stop via Step 10 (■) when done.
- For anything beyond quick testing, put the tunnel behind a reverse proxy (nginx basic auth or Cloudflare Zero Trust).
- Never forward sensitive/personal data through the public URL.

See [README.md](README.md) and [FEATURES.md](FEATURES.md) for the feature matrix, and the `Ollama_Colab_v2.6_Sucessfull-Run-Example.ipynb` for a fully-executed run with sample output.
