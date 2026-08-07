# Ollama Colab v2.6 — Notebook Source (converted from .ipynb)

> Converted from `Ollama_Colab_v2.6.ipynb` for use as a NotebookLM source.

<!-- cell 0 [markdown] -->
# Ollama Colab v2.6

Run Ollama (a tool for running large language models locally) on Google Colab with a public endpoint via ngrok or Cloudflare Tunnel. This allows users to access their own LLM instances from anywhere, integrate with VS Code extensions, or use as a backend for applications.

> Built as a self-teaching project exploring LLM inference, tunneling, and remote API access.

---

## 📋 Table of Contents

1. **Step 1: Install Dependencies** — Set up Python packages and check the environment
2. **Step 2: Model Browser** — Browse and select from the Ollama library
3. **Step 3: Configure Environment** — Memory management and tunnel settings
4. **Step 4: GPU Monitoring** — One-shot GPU snapshot + optional real-time monitor
5. **Step 5: Install Ollama** — Download and install the Ollama binary
6. **Step 6: Start Ollama Server** — Launch `ollama serve` in the background
7. **Step 7: Pull Model** — Download your chosen model with progress output
8. **Step 8: Start Tunnel** — Create the public endpoint (Cloudflare or ngrok)
9. **Step 9: Test Endpoint** — Verify the API is working end-to-end
10. **Step 10: Keep Server Alive** — Maintain the session until you are done

---

## 🚀 Quick Start

1. Set GPU runtime: `Runtime → Change runtime type → T4 GPU → Save`
2. Run **Runtime → Run all**, or execute each cell in order (Steps 1–10)
3. Copy your public URL printed in Step 8
4. Test with curl or your favourite API client

---

## 🔒 Security Advisory

> ⚠️ **Your Ollama instance is publicly accessible without authentication.**

### Immediate Risks
- **Unauthorized GPU Usage** — anyone can run inference against your Colab GPU quota
- **Cost Abuse** — malicious actors can run expensive operations at your expense
- **Prompt Injection** — adversarial prompts can manipulate model behaviour
- **Data Exfiltration** — internet-connected models could be manipulated to leak data

### Recommendations
- ✅ Monitor GPU usage regularly in the Colab UI
- ✅ Use for short testing sessions only — not permanent deployment
- ✅ Add a Cloudflare named tunnel or ngrok token for stable, audited URLs
- ✅ Set `OLLAMA_API_KEY` if you need a lightweight auth layer
- ✅ Disable tool/function-calling access for public endpoints

### Emergency Shutdown
```python
for name, proc in _bg_processes.items():
    try:
        proc.terminate()
        print(f'Terminated: {name}')
    except Exception:
        pass
```

**This notebook is for educational purposes. For production use, implement proper authentication and monitoring.**

---

### Additional Resources
- [Ollama Security Best Practices](https://github.com/ollama/ollama/blob/main/docs/security.md)
- [Colab Security Guide](https://research.google.com/colab/security)
- [OWASP API Security Checklist](https://owasp.org/www-project-api-security/)

<!-- cell 1 [markdown] -->
## Step 1: Install Dependencies

<!-- cell 2 [code] -->
```python
# ── Step 1: Install Dependencies ─────────────────────────────────────────────
# Checks Python version and GPU, then installs any missing packages.
# Also initialises shared globals used by later cells.

import subprocess
import sys
import os
import re
import threading
import time
import requests
from IPython.display import HTML, display

# ── Shared process registry (referenced by all later cells) ──────────────────
_bg_processes: dict = {}
MODEL_NAME: str | None = None
public_url: str | None = None

# ── Environment check ────────────────────────────────────────────────────────
print('🔍 Checking environment...')
pv = sys.version_info
print(f'   Python : {pv.major}.{pv.minor}.{pv.micro}')

os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia'

# ── GPU detection & model-size estimation (shared globals) ───────────────────
# Used by Step 2 (Model Browser, to filter/sort by "fits this GPU") and
# Step 3 (memory-safety check) so the sizing logic only lives in one place.
GPU_VRAM_GB: dict[str, int] = {'T4': 16, 'L4': 24, 'A100': 40, 'A100-80GB': 80}

MODEL_SIZE_GB: dict[str, float] = {  # approx VRAM footprint by tag keyword
    '405b': 240, '70b': 40, '34b': 20, '32b': 20, '27b': 17,
    '14b': 9, '13b': 8, '9b': 6, '8b': 5, '7b': 4, '6.7b': 4,
    '3b': 2, '3.8b': 2, '1.5b': 1, '1b': 1, '0.5b': 0.5,
}

def detect_gpu() -> dict:
    # Detect the Colab GPU. Returns {'name', 'vram_gb'}; defaults to T4 (free tier) if unknown.
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            for key, vram in GPU_VRAM_GB.items():
                if key in line:
                    print(f'   GPU    : {line} ✅')
                    return {'name': key, 'vram_gb': vram}
            print(f'   GPU    : {line} (unrecognised, assuming T4 VRAM) ⚠️')
        else:
            print('   GPU    : Not detected ⚠️')
    except Exception as e:
        print(f'   GPU    : Check failed ({e}) ⚠️')
    return {'name': 'T4', 'vram_gb': GPU_VRAM_GB['T4']}

def estimate_model_size_gb(text: str) -> float:
    # Best-effort VRAM size estimate (GB) from a model name/tag string.
    combined = text.lower()
    return next((v for k, v in MODEL_SIZE_GB.items() if k in combined), 7)

GPU_INFO = detect_gpu()

# ── Package installation ──────────────────────────────────────────────────────
REQUIRED = {
    'requests':       'HTTP library for API calls',
    'beautifulsoup4': 'Web scraping for model browser',
    'ipywidgets':     'Interactive UI components',
    'tqdm':           'Progress bars',
    'psutil':         'System monitoring',
    'pyngrok':        'ngrok tunnel client (pinned >=7,<8 for kwarg API stability)',
}

missing = []
for pkg, desc in REQUIRED.items():
    # bs4 is imported as beautifulsoup4 but the import name is bs4
    import_name = 'bs4' if pkg == 'beautifulsoup4' else pkg
    try:
        __import__(import_name)
        print(f'   {pkg}: already installed ✅')
    except ImportError:
        missing.append(pkg)
        print(f'   {pkg}: missing, will install...')

if missing:
    print('\n📦 Installing missing packages...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
    print('✅ All packages installed.')
else:
    print('\n✅ All dependencies already satisfied.')

print('\n🚀 Ready — proceed to Step 2.')
```

<!-- cell 3 [markdown] -->
## Step 2: Model Browser

Browse the Ollama library, filter by capability, pick a model and tag, then click **✅ Use this model**.

> If `ollama.com` is unreachable the browser falls back to a curated static list of popular models.

<!-- cell 4 [code] -->
```python
# ── Step 2: Model Browser ─────────────────────────────────────────────────────

import ipywidgets as widgets
from IPython.display import display, clear_output
from bs4 import BeautifulSoup

# ── Static fallback list (used when ollama.com is unreachable) ────────────────
STATIC_FALLBACK_MODELS = [
    {'name': 'qwen2.5',        'desc': 'General purpose language model',          'pulls': 'Very High', 'tags': ['32b', '14b', '7b', '3b']},
    {'name': 'qwen2.5-coder',  'desc': 'Code-specialised model',                  'pulls': 'Very High', 'tags': ['32b', '14b', '7b']},
    {'name': 'deepseek-r1',    'desc': 'Reasoning model with chain-of-thought',   'pulls': 'Very High', 'tags': ['70b', '32b', '14b', '8b']},
    {'name': 'llama3.1',       'desc': "Meta's open language model",              'pulls': 'Very High', 'tags': ['70b', '8b']},
    {'name': 'llama3.2',       'desc': 'Compact Meta model with strong perf',     'pulls': 'High',      'tags': ['3b', '1b']},
    {'name': 'mistral',        'desc': 'Efficient general-purpose model',         'pulls': 'High',      'tags': ['7b']},
    {'name': 'codellama',      'desc': 'Code generation specialist',              'pulls': 'Medium',    'tags': ['34b', '13b', '7b']},
    {'name': 'deepseek-coder', 'desc': 'Code-specialised model by DeepSeek',      'pulls': 'Medium',    'tags': ['33b', '6.7b']},
    {'name': 'gemma2',         'desc': "Google's open model family",              'pulls': 'Medium',    'tags': ['27b', '9b']},
    {'name': 'phi3',           'desc': "Microsoft's small but capable model",     'pulls': 'Medium',    'tags': ['14b', '3.8b']},
    {'name': 'phi4',           'desc': "Microsoft's latest compact model",        'pulls': 'Medium',    'tags': ['14b']},
    {'name': 'llava',          'desc': 'Multimodal vision-language model',        'pulls': 'Medium',    'tags': ['34b', '13b', '7b']},
    {'name': 'nomic-embed-text','desc': 'High-quality text embeddings',           'pulls': 'Medium',    'tags': ['latest']},
]

# ── Library fetcher with graceful fallback ────────────────────────────────────
def fetch_ollama_library() -> list[dict]:
    """Scrape ollama.com/library. Falls back to the static list on any error."""
    print('🔄 Fetching model list from ollama.com/library...')
    try:
        resp = requests.get('https://ollama.com/library', timeout=15,
                            headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        models = []
        seen_names: set[str] = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if not (href.startswith('/library/') and href.count('/') == 2):
                continue
            name = href.split('/')[-1]
            if name in seen_names:
                continue
            seen_names.add(name)
            desc_el  = a.select_one('p, [class*="description"]')
            pulls_el = a.find(string=lambda t: t and 'Pulls' in t)
            raw_tags = [s.get_text(strip=True) for s in a.select('span, li')
                        if s.get_text(strip=True) and len(s.get_text(strip=True)) < 20]
            unique_tags: list[str] = []
            seen_tags: set[str] = set()
            for t in raw_tags:
                if t not in seen_tags and t.lower() != name.lower():
                    seen_tags.add(t)
                    unique_tags.append(t)
            models.append({
                'name':  name,
                'desc':  desc_el.get_text(strip=True) if desc_el else '',
                'pulls': pulls_el.strip() if pulls_el else '',
                'tags':  unique_tags,
            })
        if models:
            print(f'✅ Scraped {len(models)} models from ollama.com')
            return models
        print('⚠️  Scrape returned 0 models — using static fallback list.')
    except Exception as exc:
        print(f'⚠️  Scrape failed ({exc}) — using static fallback list.')
    return STATIC_FALLBACK_MODELS


# ── Tag fetcher with cache ────────────────────────────────────────────────────
_tag_cache: dict[str, list[str]] = {}

DEFAULT_TAGS: dict[str, list[str]] = {
    'coder':     ['32b', '14b', '7b'],
    'r1':        ['70b', '32b', '14b'],
    'llama':     ['70b', '8b'],
    'embed':     ['latest'],
}

def fetch_tags(model_name: str) -> list[str]:
    """Fetch available tags for a model. Returns a cached or fallback list."""
    if model_name in _tag_cache:
        return _tag_cache[model_name]
    try:
        resp = requests.get(f'https://ollama.com/library/{model_name}/tags',
                            timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        tags: list[str] = []
        for el in soup.select("a[href*='/library/'][href*=':']"):
            tag = el['href'].split(':')[-1].split('/')[0].strip()
            if tag and ':' not in tag and tag not in tags:
                tags.append(tag)
        if tags:
            _tag_cache[model_name] = tags
            return tags
    except Exception:
        pass
    # Sensible defaults based on model name
    for keyword, defaults in DEFAULT_TAGS.items():
        if keyword in model_name.lower():
            _tag_cache[model_name] = defaults
            return defaults
    fallback = ['14b', '7b', 'latest']
    _tag_cache[model_name] = fallback
    return fallback


# ── Capability filtering ──────────────────────────────────────────────────────
CAPABILITIES = ['all', 'code', 'reasoning', 'vision', 'embedding', 'tools']
CAP_KEYWORDS = {
    'code':      ['code', 'coder', 'coding', 'starcoder', 'deepseek-coder', 'codellama'],
    'reasoning': ['reasoning', 'r1', 'thinking', 'o1'],
    'vision':    ['vision', 'multimodal', 'llava', 'moondream'],
    'embedding': ['embedding', 'embed', 'nomic'],
    'tools':     ['tools', 'function'],
}

SIZE_SUFFIXES = ('b', 'm', 'k')

def size_tags(model: dict) -> list[str]:
    return [t for t in model['tags']
            if any(t.lower().endswith(s) for s in SIZE_SUFFIXES)]

def model_matches_cap(model: dict, cap: str) -> bool:
    if cap == 'all':
        return True
    combined = f"{model['name']} {model['desc']} {' '.join(model['tags'])}".lower()
    return any(k in combined for k in CAP_KEYWORDS[cap])


# ── GPU-fit filtering (uses GPU_INFO / estimate_model_size_gb from Step 1) ────
GPU_SAFETY_MARGIN_GB = 2  # leave headroom for context + overhead, matches Step 3

def fitting_tags(model: dict) -> list[str]:
    """Tags for this model whose estimated size fits GPU_INFO with margin."""
    sizes = size_tags(model)
    return [t for t in sizes
            if estimate_model_size_gb(t) + GPU_SAFETY_MARGIN_GB <= GPU_INFO['vram_gb']]

def model_fits_gpu(model: dict) -> bool:
    """True if at least one known size tag fits; unsized models (e.g. embeddings) pass through."""
    sizes = size_tags(model)
    return True if not sizes else bool(fitting_tags(model))

def smallest_size_gb(model: dict) -> float:
    sizes = size_tags(model)
    return min((estimate_model_size_gb(t) for t in sizes), default=0.0)


# ── Widgets ───────────────────────────────────────────────────────────────────
_style  = {'description_width': '80px'}
_layout = widgets.Layout(width='100%')

w_search    = widgets.Text(placeholder='Search models...', description='Search:', style=_style, layout=_layout)
w_cap       = widgets.ToggleButtons(options=CAPABILITIES, value='all', description='Filter:',
                                    style={'button_width': '100px', 'description_width': '80px'}, layout=_layout)
w_fit_gpu   = widgets.Checkbox(
    value=True,
    description=f"Only show models that fit {GPU_INFO['name']} ({GPU_INFO['vram_gb']} GB VRAM)",
    indent=False, layout=_layout,
)
w_list      = widgets.Select(options=[], rows=12, description='Model:', style=_style,
                             layout=widgets.Layout(width='100%'))
w_tag_label = widgets.HTML('<b>Tag / variant:</b>')
w_tags      = widgets.RadioButtons(options=[], description='', layout=_layout)
w_info      = widgets.HTML('')
w_apply     = widgets.Button(description='✅  Use this model', button_style='success',
                             layout=widgets.Layout(width='220px'))
w_out       = widgets.Output()


def refresh_list(*_) -> None:
    query    = w_search.value.strip().lower()
    cap      = w_cap.value
    fit_only = w_fit_gpu.value
    filtered = [m for m in ALL_MODELS
                if model_matches_cap(m, cap)
                and (not query or query in m['name'].lower() or query in m['desc'].lower())
                and (not fit_only or model_fits_gpu(m))]
    if fit_only:
        filtered.sort(key=smallest_size_gb)  # smallest / easiest-to-run first
    labels = []
    for m in filtered:
        sizes     = size_tags(m)[:4]
        size_str  = f"  {' '.join(sizes)}" if sizes else ''
        pulls     = m['pulls'].replace(' Pulls', '').strip() if m['pulls'] else ''
        pulls_str = f'  [{pulls}]' if pulls else ''
        labels.append(f"{m['name']}{size_str}{pulls_str}")
    w_list.options = labels
    if labels:
        w_list.value = labels[0]


def on_model_select(change: dict) -> None:
    if not w_list.value:
        return
    model_name = w_list.value.split()[0]
    w_info.value = '<i>Loading tags...</i>'
    w_tags.options = []
    tags = fetch_tags(model_name)

    warning = ''
    if w_fit_gpu.value:
        fitting = [t for t in tags if estimate_model_size_gb(t) + GPU_SAFETY_MARGIN_GB <= GPU_INFO['vram_gb']]
        if fitting:
            tags = fitting
        else:
            warning = f"<br><span style='color:#c62828'>⚠️ No known tag comfortably fits {GPU_INFO['name']} ({GPU_INFO['vram_gb']}GB) — showing all tags.</span>"

    # Smallest-first, with 'latest' pinned to the front since its real size is unknown
    ordered = sorted(tags, key=lambda t: (t != 'latest', estimate_model_size_gb(t)))[:20]
    w_tags.options = ordered
    if ordered:
        w_tags.value = ordered[0]
    meta  = next((m for m in ALL_MODELS if m['name'] == model_name), None)
    desc  = meta['desc']  if meta else ''
    pulls = meta['pulls'] if meta else ''
    w_info.value = (
        f'<b>{model_name}</b>'
        + (f' — {desc}'             if desc  else '')
        + (f'<br><small>{pulls}</small>' if pulls else '')
        + warning
    )


def on_apply(_) -> None:
    global MODEL_NAME
    if not w_list.value or not w_tags.value:
        return
    model_base = w_list.value.split()[0]
    MODEL_NAME = f'{model_base}:{w_tags.value}'
    with w_out:
        clear_output()
        print(f'✅  MODEL_NAME set to: {MODEL_NAME}')
        print('Continue to Step 3 when ready.')


w_search.observe(refresh_list,    names='value')
w_cap.observe(refresh_list,       names='value')
w_fit_gpu.observe(refresh_list,   names='value')
w_fit_gpu.observe(on_model_select, names='value')  # re-filter tags for the current model too
w_list.observe(on_model_select,   names='value')
w_apply.on_click(on_apply)

# ── Initial render ────────────────────────────────────────────────────────────
ALL_MODELS = fetch_ollama_library()
refresh_list()

display(widgets.VBox([
    widgets.HTML('<h4>🦙 Ollama Model Browser</h4>'),
    widgets.HTML('<small><i>Tip: If web scraping fails the browser uses a built-in fallback list. '
                 'Uncheck the GPU filter below to browse the full library.</i></small><br>'),
    w_search, w_cap, w_fit_gpu, w_list,
    w_tag_label, w_tags, w_info,
    w_apply, w_out,
], layout=widgets.Layout(padding='8px', border='1px solid #ccc', border_radius='6px')))
```

<!-- cell 5 [markdown] -->
## Step 3: Configure Environment

### Memory management

| Option | `keep_alive` | When to use |
|---|---|---|
| Keep loaded | `-1` (forever) | Small models, fastest responses |
| Auto-unload | `0` (immediate) | Large models or limited VRAM |
| 5 minutes | `300 s` | Most interactive sessions |
| 30 minutes | `1800 s` | Extended single-model sessions |

### Tunnel credentials

Add secrets via the 🔑 sidebar in Colab:

| Secret name | Where to get it |
|---|---|
| `cloudflare_token` | [Cloudflare dashboard](https://one.dash.cloudflare.com) → Networks → Tunnels → Create tunnel |
| `ngrok_authtoken`  | [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) |

<!-- cell 6 [code] -->
```python
# ── Step 3: Configure Environment ────────────────────────────────────────────

# @title Tunnel & memory configuration { run: "auto" }
TUNNEL_METHOD = 'Cloudflare quick tunnel (no credentials)'  # @param ["Cloudflare quick tunnel (no credentials)", "Cloudflare named tunnel (token)", "ngrok (token)"]

import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import userdata

# ── Guard: model must have been selected in Step 2 ───────────────────────────
if not MODEL_NAME:
    raise RuntimeError(
        'MODEL_NAME is not set.\n'
        "Go back to Step 2, choose a model and tag, then click '✅ Use this model'."
    )

# ── GPU info / model sizing ───────────────────────────────────────────────────
# GPU_INFO and estimate_model_size_gb() come from Step 1 (shared globals) so
# this cell, the Step 2 Model Browser, and the memory check all agree on one
# GPU reading and one sizing table instead of three separate copies.

def check_model_memory_safety(model_name: str, tag: str) -> dict:
    size_gb = estimate_model_size_gb(f"{model_name} {tag}")
    return {
        'model_size_gb': size_gb,
        'gpu_vram_gb':   GPU_INFO['vram_gb'],
        'gpu_name':      GPU_INFO['name'],
        'safe':          size_gb + 2 <= GPU_INFO['vram_gb'],
        'tight':         size_gb + 2 > GPU_INFO['vram_gb'] - 4,
    }


# ── Memory widget ─────────────────────────────────────────────────────────────
MEMORY_OPTIONS = {
    'keep_loaded': {'keep_alive': -1,   'desc': 'Fastest — keep model in VRAM permanently'},
    'auto_unload': {'keep_alive': 0,    'desc': 'Safest — unload after every request'},
    '5min':        {'keep_alive': 300,  'desc': 'Good balance for interactive sessions'},
    '30min':       {'keep_alive': 1800, 'desc': 'Extended single-model sessions'},
}

w_memory = widgets.RadioButtons(
    options=[
        ('Keep loaded (fast, risky)',  'keep_loaded'),
        ('Auto-unload (safe, slow)',   'auto_unload'),
        ('Keep 5 minutes',            '5min'),
        ('Keep 30 minutes',           '30min'),
    ],
    value='keep_loaded',
    description='Memory:',
    style={'description_width': '80px'},
    layout=widgets.Layout(width='100%'),
)
w_mem_info = widgets.HTML('')

def persist_memory_config() -> None:
    """Write the currently selected memory policy to os.environ."""
    global MEMORY_CONFIG
    MEMORY_CONFIG = MEMORY_OPTIONS[w_memory.value]
    os.environ['OLLAMA_KEEP_ALIVE'] = str(MEMORY_CONFIG['keep_alive'])


def update_mem_info(*_) -> None:
    model_part, tag_part = (MODEL_NAME.split(':', 1) if ':' in MODEL_NAME else (MODEL_NAME, 'latest'))
    s   = check_model_memory_safety(model_part, tag_part)
    cfg = MEMORY_OPTIONS[w_memory.value]
    if not s['safe']:
        colour, icon, title = '#ffebee', '⚠️', 'GPU Memory Warning'
        body = f"Model (~{s['model_size_gb']}GB) may exceed {s['gpu_name']} VRAM ({s['gpu_vram_gb']}GB). Consider Auto-unload."
    elif s['tight']:
        colour, icon, title = '#fff3e0', '⚡', 'High Memory Usage'
        body = f"Model (~{s['model_size_gb']}GB) uses most of {s['gpu_name']} VRAM ({s['gpu_vram_gb']}GB)."
    else:
        colour, icon, title = '#e8f5e8', '✅', 'Safe Memory Usage'
        body = f"Model (~{s['model_size_gb']}GB) fits comfortably in {s['gpu_name']} VRAM ({s['gpu_vram_gb']}GB)."
    w_mem_info.value = (
        f'<div style="background:{colour};padding:8px;border-radius:4px;margin:4px 0">'
        f'<strong>{icon} {title}</strong><br>{body}<br><em>Policy: {cfg["desc"]}</em></div>'
    )
    # Persist immediately so a mid-session change actually takes effect,
    # instead of only being applied the one time this cell finishes running.
    persist_memory_config()


w_memory.observe(update_mem_info, names='value')
update_mem_info()  # initial render (also persists the default policy)

print(f"GPU: {GPU_INFO['name']} ({GPU_INFO['vram_gb']} GB VRAM)")

display(widgets.VBox(
    [widgets.HTML('<h4>🧠 Memory Management</h4>'), w_memory, w_mem_info],
    layout=widgets.Layout(padding='10px', border='1px solid #ddd', border_radius='5px'),
))

# ── Resolve tunnel credentials ────────────────────────────────────────────────
_method_map = {
    'Cloudflare quick tunnel (no credentials)': 'quick',
    'Cloudflare named tunnel (token)':          'cloudflare',
    'ngrok (token)':                            'ngrok',
}
tunnel_type  = _method_map[TUNNEL_METHOD]
tunnel_token = None

if tunnel_type == 'cloudflare':
    try:
        tunnel_token = userdata.get('cloudflare_token')
        if not tunnel_token:
            raise ValueError("Secret 'cloudflare_token' is empty.")
        print('Cloudflare named tunnel — token loaded ✅')
    except Exception as exc:
        print(f'ERROR: {exc}')
        print("Add 'cloudflare_token' to Colab Secrets, or switch to the quick tunnel option.")
        raise

elif tunnel_type == 'ngrok':
    try:
        tunnel_token = userdata.get('ngrok_authtoken')
        if not tunnel_token:
            raise ValueError("Secret 'ngrok_authtoken' is empty.")
        print('ngrok — token loaded ✅')
        print('NOTE: ngrok free-tier URLs rotate every ~2 hours.')
    except Exception as exc:
        print(f'ERROR: {exc}')
        print("Add 'ngrok_authtoken' to Colab Secrets, or switch to the quick tunnel option.")
        raise

else:
    print('Cloudflare quick tunnel — no credentials needed ✅')
    print('NOTE: The public URL changes every time you run Step 8.')

# ── Persist remaining environment settings ─────────────────────────────────────
# (OLLAMA_KEEP_ALIVE is already kept in sync by persist_memory_config(),
#  called from update_mem_info() above -- both at init and on every change.)
os.environ['OLLAMA_CONTEXT_LENGTH'] = '16384'
os.environ['OLLAMA_HOST']           = '0.0.0.0'

print(f'\nModel  : {MODEL_NAME}')
print(f'Tunnel : {TUNNEL_METHOD}')
print(f'Memory : {w_memory.value} (keep_alive={MEMORY_CONFIG["keep_alive"]}s)')
```

<!-- cell 7 [markdown] -->
## Step 4: GPU Monitoring

Prints a one-shot GPU snapshot. For live monitoring run `monitor_gpu()` in a separate cell.

<!-- cell 8 [code] -->
```python
# ── Step 4: GPU Monitoring ────────────────────────────────────────────────────

def get_gpu_snapshot() -> None:
    """Print a single GPU information snapshot."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print('⚠️  Could not query GPU.')
            return
        print('🖥️  GPU Information')
        print('─' * 50)
        for line in result.stdout.strip().splitlines():
            idx, name, driver, vram = [p.strip() for p in line.split(',')]
            print(f'  GPU {idx}: {name}')
            print(f'    Driver : {driver}')
            print(f'    VRAM   : {vram} MB')
        print('─' * 50)
    except Exception as exc:
        print(f'⚠️  GPU snapshot failed: {exc}')


def monitor_gpu(interval: int = 2, duration: int = 60) -> None:
    """Print real-time GPU stats. Runs in the calling cell until stopped (■) or timeout."""
    from IPython.display import clear_output
    print(f'📊 GPU Monitor — {interval}s interval, {duration}s max. Stop cell (■) to exit.')
    end = time.time() + duration
    try:
        while time.time() < end:
            result = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                print('⚠️  nvidia-smi error')
                break
            clear_output(wait=True)
            rows = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 8:
                    continue
                idx, name, g_util, m_util, m_used, m_total, temp, power = parts[:8]
                mem_pct = int(m_used) / int(m_total) * 100 if m_total else 0
                rows.append(
                    f'  GPU {idx} | {name:<20} | GPU {g_util:>3}% | '
                    f'Mem {m_used}/{m_total} MB ({mem_pct:.0f}%) | {temp}°C | {power}W'
                )
            elapsed = int(end - time.time())
            print(f'🖥️  Live GPU — {elapsed}s remaining')
            print('─' * 80)
            print('\n'.join(rows) or '  (no data)')
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    print('\n✅ GPU monitor stopped.')


get_gpu_snapshot()
print("\n💡 For live monitoring, run `monitor_gpu(interval=2, duration=120)` in a new cell.")
```

<!-- cell 9 [markdown] -->
## Step 5: Install Ollama

Downloads and installs the Ollama binary. Safe to re-run — it upgrades if already installed.

<!-- cell 10 [code] -->
```python
import subprocess
# ── Step 5: Install Ollama ────────────────────────────────────────────────────

print('📦 Installing Ollama...')

# Install zstd first, as it's required by Ollama installer
print('Installing zstd...')
zstd_install = subprocess.run(
    ['sudo', 'apt-get', 'install', '-y', 'zstd'],
    capture_output=True, text=True
)
if zstd_install.returncode == 0:
    print('✅ zstd installed.')
else:
    print('❌ zstd installation failed.')
    print('--- STDOUT ---')
    print(zstd_install.stdout)
    print('--- STDERR ---')
    print(zstd_install.stderr)
    raise RuntimeError('zstd installation failed.')

install = subprocess.run(
    'curl -fsSL https://ollama.com/install.sh | sh',
    shell=True, capture_output=True, text=True
)
if install.returncode == 0:
    # Confirm binary is on PATH
    version = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
    print(f'✅ Ollama installed: {version.stdout.strip()}')
else:
    print('❌ Installation failed.')
    print('--- STDOUT ---')
    print(install.stdout)
    print('--- STDERR ---')
    print(install.stderr)
    raise RuntimeError('Ollama installation failed. Check the output above.')
```

<!-- cell 11 [markdown] -->
## Step 6: Start Ollama Server

Launches `ollama serve` in the background and waits until the API responds.

<!-- cell 12 [code] -->
```python
import subprocess
import threading
import time
import requests

# ── Step 6: Start Ollama Server ───────────────────────────────────────────────

def stream_output(process: subprocess.Popen, label: str = '') -> None:
    """Read process stdout in a background thread and print each line."""
    prefix = f'[{label}] ' if label else ''
    for line in process.stdout:
        line = line.strip()
        if line:
            print(f'{prefix}{line}')

def wait_for_ollama(timeout: int = 60) -> None:
    """Block until the Ollama HTTP API responds or the process dies."""
    for i in range(timeout):
        proc = _bg_processes.get('ollama')
        if proc and proc.poll() is not None:
            raise RuntimeError(
                f'ollama serve exited early (code {proc.returncode}). '
                'Check above — there may be a GPU driver issue.'
            )
        try:
            r = requests.get('http://localhost:11434', timeout=30)
            if r.status_code in (200, 404):
                print(f'✅ Ollama is up (after {i + 1}s).')
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError('Ollama did not start within 60 s.')

print('Killing any existing ollama processes...')
subprocess.run(['killall', '-q', 'ollama'], capture_output=True)
time.sleep(1) # Give a moment for processes to terminate

print('🚀 Starting ollama serve...')
ollama_proc = subprocess.Popen(
    ['ollama', 'serve'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
)
_bg_processes['ollama'] = ollama_proc
threading.Thread(target=stream_output, args=(ollama_proc, 'ollama'), daemon=True).start()

wait_for_ollama()
```

<!-- cell 13 [markdown] -->
## Step 7: Pull Model

Downloads the model selected in Step 2. Large models (14B+) may take 5–15 minutes.

<!-- cell 14 [code] -->
```python
import subprocess
import threading
import time

# ── Step 7: Pull Model ─────────────────────────────────────────────────────────

# Ensure MODEL_NAME is set. If not, default to a common model for demonstration.
if 'MODEL_NAME' not in globals() or not MODEL_NAME:
    print("MODEL_NAME was not set in Step 2. Defaulting to 'deepseek-r1:1.5b'.")
    MODEL_NAME = 'deepseek-r1:1.5b'

print(f'📥 Pulling {MODEL_NAME} ...')
print('⏱️  This may take 5–15 minutes depending on model size and connection speed.\n')

pull = subprocess.Popen(
    ['ollama', 'pull', MODEL_NAME],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
)
try:
    for line in pull.stdout:
        line = line.strip()
        if line:
            print(line)
except KeyboardInterrupt:
    pull.terminate()
    raise

rc = pull.wait()
if rc == 0:
    print(f'\n✅ Successfully pulled {MODEL_NAME}')
else:
    # If the server stopped, try to restart and pull again (basic retry)
    print(f'\n⚠️ Model pull failed (exit code {rc}). The Ollama server might have stopped.')
    print('Attempting to restart Ollama server and retry pull...')
    # Kill any existing ollama processes
    subprocess.run(['killall', '-q', 'ollama'], capture_output=True)
    time.sleep(1)

    # Restart server the same way Step 6 does, so it's tracked in
    # _bg_processes and gets cleaned up by Step 10 / the emergency shutdown snippet.
    print('Restarting ollama serve in the background...')
    ollama_proc = subprocess.Popen(
        ['ollama', 'serve'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    _bg_processes['ollama'] = ollama_proc
    threading.Thread(target=stream_output, args=(ollama_proc, 'ollama'), daemon=True).start()
    wait_for_ollama()

    # Retry pull
    print(f'Retrying pull for {MODEL_NAME}...')
    pull_retry = subprocess.Popen(
        ['ollama', 'pull', MODEL_NAME],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    try:
        for line in pull_retry.stdout:
            line = line.strip()
            if line:
                print(line)
    except KeyboardInterrupt:
        pull_retry.terminate()
        raise

    rc_retry = pull_retry.wait()
    if rc_retry == 0:
        print(f'\n✅ Successfully pulled {MODEL_NAME} after retry.')
    else:
        raise RuntimeError(f'Model pull failed even after retry (exit code {rc_retry}). Please check the Ollama server logs.')
```

<!-- cell 15 [markdown] -->
## Step 8: Start Tunnel

Creates the public endpoint using the tunnel type chosen in Step 3.

<!-- cell 16 [code] -->
```python
import subprocess
import threading
import time
import requests
import re

# ── Step 8: Start Tunnel ──────────────────────────────────────────────────────

public_url = None

def wait_for_tunnel_ready(url: str, timeout: int = 30, interval: int = 2) -> bool:
    # Poll the PUBLIC url until it actually responds -- not just until the
    # hostname exists. Cloudflare's own quick-tunnel banner warns "it may
    # take some time to be reachable" after the URL is printed: there's a
    # real gap between the edge allocating the hostname and it finishing
    # the proxy handshake back to this cloudflared process. Hitting the URL
    # during that gap causes a hang until the caller's read timeout fires,
    # not an immediate connection error.
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f'{url}/api/tags', timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False

# Install cloudflared if not already present and a Cloudflare tunnel is selected
if TUNNEL_METHOD in ['Cloudflare quick tunnel (no credentials)', 'Cloudflare named tunnel (token)']:
    try:
        subprocess.run(['which', 'cloudflared'], check=True, capture_output=True)
        print('✅ cloudflared is already installed.')
    except subprocess.CalledProcessError:
        print('📦 Installing cloudflared...')
        try:
            subprocess.run([ # Download the latest cloudflared binary for Linux AMD64
                'wget', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'
            ], check=True)
            subprocess.run(['chmod', '+x', 'cloudflared-linux-amd64'], check=True) # Make it executable
            subprocess.run(['sudo', 'mv', 'cloudflared-linux-amd64', '/usr/local/bin/cloudflared'], check=True) # Move to PATH
            print('✅ cloudflared installed.')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Failed to install cloudflared: {e.stderr.decode()}') from e

# ── Cloudflare named tunnel ───────────────────────────────────────────────────
if tunnel_type == 'cloudflare':
    print('Starting Cloudflare named tunnel...')
    cf_proc = subprocess.Popen(
        ['cloudflared', 'tunnel', '--edge-ip-version', '4', '--no-autoupdate',
         'run', '--token', tunnel_token],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
    )
    _bg_processes['tunnel'] = cf_proc
    threading.Thread(target=stream_output, args=(cf_proc, 'cf'), daemon=True).start()

    print('Waiting up to 30 s for tunnel to connect...')
    for _ in range(15):
        time.sleep(2)
        if cf_proc.poll() is not None:
            raise RuntimeError(
                f'cloudflared exited early (code {cf_proc.returncode}). '
                'Check that your tunnel token is valid in the Cloudflare dashboard.'
            )
        try:
            requests.get('http://localhost:11434/api/tags', timeout=3)
            print('Cloudflare named tunnel is running.')
            print('Use the hostname configured in your Cloudflare dashboard as the public URL.')
            public_url = 'https://<your-cloudflare-hostname>'
            break
        except Exception:
            pass

# ── Cloudflare quick tunnel ───────────────────────────────────────────────────
elif tunnel_type == 'quick':
    print('Starting Cloudflare quick tunnel (no credentials)...')
    cf_proc = subprocess.Popen(
        ['cloudflared', 'tunnel', '--edge-ip-version', '4',
         '--url', 'http://localhost:11434', '--no-autoupdate'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
    )
    _bg_processes['tunnel'] = cf_proc

    print('Waiting for public URL...')
    for line in cf_proc.stdout:
        line = line.strip()
        if line:
            print(f'[cf] {line}')
        match = re.search(r'(https://[\w-]+\.trycloudflare\.com)', line)
        if match:
            public_url = match.group(1)
            break

    # Hand off remaining stdout to background thread so cell completes
    threading.Thread(target=stream_output, args=(cf_proc, 'cf'), daemon=True).start()

    if not public_url:
        raise RuntimeError('Could not find a Cloudflare quick-tunnel URL in output.')

    print(f'URL found: {public_url}')
    print('Waiting for the tunnel to become reachable '
          '(Cloudflare warns this can take a few seconds after the URL appears)...')
    if wait_for_tunnel_ready(public_url, timeout=30):
        print('✅ Tunnel is live and responding.')
    else:
        print('⚠️  Tunnel URL exists but has not responded within 30s.')
        print('    It may still come up shortly -- re-run Step 9 in a few seconds, '
              'or re-run this cell for a fresh URL.')

# ── ngrok ─────────────────────────────────────────────────────────────────────
elif tunnel_type == 'ngrok':
    from pyngrok import ngrok as _ngrok
    _ngrok.set_auth_token(tunnel_token)
    _ngrok.kill()
    time.sleep(1)
    print('Starting ngrok tunnel...')
    tunnel   = _ngrok.connect(11434, 'http', options={'host_header': 'localhost:11434'})
    public_url = tunnel.public_url
    print(f'ngrok tunnel open: {public_url}')
    print('Waiting for the tunnel to become reachable...')
    if wait_for_tunnel_ready(public_url, timeout=15):
        print('✅ Tunnel is live and responding.')
    else:
        print('⚠️  Tunnel URL exists but has not responded within 15s. Re-run Step 9 shortly.')

# ── Summary ───────────────────────────────────────────────────────────────────
if public_url:
    print('\n' + '=' * 60)
    print('  ✅ Ollama is live!')
    print(f'  Model  : {MODEL_NAME}')
    print(f'  URL    : {public_url}')
    print(f'  Tunnel : {tunnel_type}')
    print('=' * 60)
    if tunnel_type == 'quick':
        print('\nNOTE: This URL is temporary — it changes every time you run this cell.')
    elif tunnel_type == 'ngrok':
        print('\nNOTE: ngrok free-tier URLs rotate every ~2 hours.')
else:
    print('\nPublic URL could not be determined. Check the output above.')
```

<!-- cell 17 [markdown] -->
## Step 9: Test Endpoint

Sends a quick prompt to verify the API is working end-to-end.

<!-- cell 18 [code] -->
```python
TEST_PROMPT = 'Write a hello world program in Python with a comment on each line.'  # @param {type:"string"}

# Retries with backoff instead of failing on one attempt: the first hit
# to a fresh tunnel URL (especially a Cloudflare quick tunnel) can time out
# even after Step 8's own readiness check, since the two checks happen
# seconds apart and the tunnel/model can still be settling in.
REACHABILITY_RETRIES = 4
REACHABILITY_BACKOFF_S = 3

if not public_url or public_url.startswith('https://<your'):
    print('ℹ️  Set public_url to your actual tunnel hostname before running this cell.')
else:
    # Check endpoint is reachable, retrying a few times before giving up
    r = None
    last_exc = None
    for attempt in range(1, REACHABILITY_RETRIES + 1):
        try:
            r = requests.get(f'{public_url}/api/tags', timeout=10)
            r.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            break
        except Exception as exc:
            last_exc = exc
            if attempt < REACHABILITY_RETRIES:
                print(f'⏳ Attempt {attempt}/{REACHABILITY_RETRIES} not reachable yet '
                      f'({exc.__class__.__name__}) -- retrying in {REACHABILITY_BACKOFF_S}s...')
                time.sleep(REACHABILITY_BACKOFF_S)

    if r is None:
        print(f'❌ Endpoint still not reachable after {REACHABILITY_RETRIES} attempts: {last_exc}')
        print('Re-run this cell, or re-run Step 8 for a fresh tunnel URL if this keeps happening.')
    else:
        models = [m['name'] for m in r.json().get('models', [])]
        print(f'✅ Endpoint reachable. Available models: {models}')
        print(f'\n💬 Sending test prompt to {MODEL_NAME}...\n')
        try:
            resp = requests.post(
                f'{public_url}/api/chat',
                json={
                    'model':    MODEL_NAME,
                    'messages': [{'role': 'user', 'content': TEST_PROMPT}],
                    'stream':   False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            print(resp.json()['message']['content'])
        except Exception as exc:
            print(f'❌ Chat request failed: {exc}')
```

<!-- cell 19 [markdown] -->
## Step 10: Keep Server Alive

Blocks until you stop the cell (■) or the Colab session disconnects.
Stopping this cell terminates Ollama and the tunnel cleanly.

<!-- cell 20 [code] -->
```python
# ── Step 10: Keep Server Alive ────────────────────────────────────────────────

if not public_url:
    print('Server not started — run previous cells first.')
else:
    print(f'Server running at : {public_url}')
    print(f'Model             : {MODEL_NAME}')
    print(f'Tunnel            : {tunnel_type}')
    print(f'Context window    : 16 384 tokens')
    print('\nKeeping server alive. Stop this cell (■) to shut down.')

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print('\n🛑 Shutting down...')
        for name, proc in _bg_processes.items():
            try:
                proc.terminate()
                print(f'  Terminated {name}')
            except Exception:
                pass
        print('Done.')
```

<!-- cell 21 [markdown] -->
---

## Usage Examples

Replace `https://your-url` with the URL printed in Step 8.

### curl

```bash
# Chat
curl -X POST https://your-url/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen2.5:14b", "messages": [{"role": "user", "content": "Explain recursion"}]}'

# Single-turn generate
curl -X POST https://your-url/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen2.5:14b", "prompt": "Hello, world!", "stream": false}'

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
import json
print(json.loads(response.json()['response']))
```

### VS Code (Continue / CodeGPT)

1. Install the extension
2. Set provider to **Ollama**
3. Set base URL to your tunnel URL
4. Select model (e.g. `qwen2.5:14b`)

---

## Notes

- **Quick tunnel**: URL changes every run — not suitable for persistent integrations.
- **ngrok free tier**: URL rotates every ~2 hours.
- **Cloudflare named tunnel**: Fixed hostname — best for longer-running use.
- **Colab free tier**: Sessions last a few hours; daily GPU quota applies.
- **Colab Pro**: Sessions up to 24 hours with L4 GPU access.
- **Security**: The endpoint is public with no authentication. Avoid sending sensitive data.
- **Anthropic API**: `/v1/messages` is available from Ollama v0.14.0+, enabling Claude Code-style clients.

