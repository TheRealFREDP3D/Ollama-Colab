# Ollama Colab — Features

Source of truth: `Ollama_Colab_v2.6_Successful-Run-Example.ipynb`.

## Core flow
1. **Dependency & environment check (Step 1)** — detects Python, GPU, installs missing Python packages (`requests`, `beautifulsoup4`, `ipywidgets`, `tqdm`, `psutil`, `pyngrok`), and initialises shared globals: the background-process registry (`_bg_processes`), `GPU_INFO`, and `estimate_model_size_gb()`.
2. **Interactive model browser (Step 2)** — browses the Ollama library, with a curated static fallback when `ollama.com` is unreachable.
3. **GPU-fit filtering** — by default only shows model *tags* whose estimated VRAM footprint fits the detected GPU (T4/L4/A100/A100-80GB), sorted smallest-first. Toggle off to see the full library.
4. **Environment & memory policy (Step 3)** — choose a tunnel backend, set `OLLAMA_KEEP_ALIVE` memory policy (keep-loaded / auto-unload / 5 min / 30 min) which persists into `os.environ` on every change, and loads Colab secrets.
5. **GPU monitoring (Step 4)** — one-shot `nvidia-smi` snapshot; optional `monitor_gpu()` live monitor.
6. **Ollama install (Step 5)** — `curl -fsSL https://ollama.com/install.sh | sh` (plus `zstd` prereq), safe to re-run.
7. **Server start (Step 6)** — launches `ollama serve` in the background and waits for `localhost:11434` to respond; registered in `_bg_processes` for clean shutdown.
8. **Model pull (Step 7)** — streamed progress; auto-restarts `ollama serve` and retries if the server dies, re-registering the restarted process so it is cleaned up later.
9. **Public tunnel (Step 8)** — **Cloudflare quick tunnel** (no credentials), **named tunnel** (via `cloudflare_token` secret), or **ngrok** (`ngrok_authtoken` secret). Polls the *public* URL on `/api/tags` until it actually responds — covers Cloudflare's "URL printed but not yet routable" delay.
10. **Endpoint test (Step 9)** — sends a configurable test prompt through the live tunnel with backoff retries.
11. **Keep-alive (Step 10)** — blocking loop; stopping the cell terminates Ollama + tunnel cleanly via the shared process registry.

## Tunnel backends
| Backend | Credentials | URL stability |
|---|---|---|
| Cloudflare quick tunnel | none | new each run |
| Cloudflare named tunnel | `cloudflare_token` (+ optional `cloudflare_hostname`) | fixed hostname |
| ngrok | `ngrok_authtoken` | rotates ~2 h (free tier) |

## Reliability features
- **Tunnel readiness polling** — Step 8 waits for the public endpoint to actually answer, not just until the hostname is printed.
- **Endpoint reachability retry** — Step 9 retries with backoff instead of failing on the first settling-second timeout.
- **Pull retry with server restart** — Step 7 detects a dead `ollama serve`, restarts it, and re-pulls; the restarted process is tracked in `_bg_processes`.
- **Graceful scraping fallback** — the model browser falls back to a curated static list + cached default tags when `ollama.com` is unreachable.
- **Process registry** — all long-running procs (`ollama`, tunnel) live in `_bg_processes`, so both the keep-alive cell and the emergency-shutdown snippet can tear them down uniformly.
- **GPU-aware sizing** — single source of truth (`estimate_model_size_gb` + `GPU_INFO`) reused by the browser filter and the memory-safety check, so the two never disagree.

## API compatibility
- Standard Ollama REST API (`/api/chat`, `/api/generate`, `/api/generate` with JSON `format` schema, `/api/tags`).
- Anthropic-compatible `/v1/messages` endpoint (Ollama v0.14.0+).
- VS Code / CodeGPT integration via the public URL.

## Security posture
- The endpoint is **public and unauthenticated**; the notebook ships an emergency-shutdown snippet and explicit warnings.
- Recommended hardening (documented in README): reverse-proxy auth (nginx basic auth or Cloudflare Zero Trust), short sessions only.
