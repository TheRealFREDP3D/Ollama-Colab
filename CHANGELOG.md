# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.6] - 2026-08-05

### Added
- Step 8 now polls the actual public tunnel URL until it responds before declaring the tunnel ready, instead of trusting the printed URL alone — Cloudflare's own quick-tunnel banner warns it "may take some time to be reachable" after the hostname appears, and hitting it too early causes a hang until the caller's timeout fires
- Step 9's reachability check now retries a few times with a short backoff instead of failing on a single attempt
- Model Browser (Step 2) can filter the library down to models that fit the detected GPU's VRAM, sorted smallest-first — on by default, toggle off to see everything
- GPU detection and model-size estimation moved into Step 1 as shared globals (`GPU_INFO`, `estimate_model_size_gb()`), reused by both Step 2 and Step 3 instead of three separate copies of the same logic

### Fixed
- Notebook Structure table now lists Steps 6/7 in the order they actually appear (Start Server → Pull Model)
- `pyngrok` added to the Step 1 dependency installer (previously only installed implicitly, causing `ModuleNotFoundError` for ngrok users)
- Memory policy widget now writes `OLLAMA_KEEP_ALIVE` to `os.environ` on every change, not just once at cell-execution time
- Step 7's retry path registers the restarted `ollama serve` process in `_bg_processes`, so it's picked up by Step 10 / emergency shutdown instead of leaking
- Removed leftover deprecated cell from Step 7

## [2.5] - 2026-08-04

### Fixed
- Duplicate model browser definitions removed
- Missing `ollama serve` step added (Step 7)
- Missing Ollama install step added (Step 5)
- Security advisory was in a code cell (now markdown only)
- `_bg_processes` and globals now initialised in Step 1 to prevent `NameError` on out-of-order execution
- GPU monitor HTML table now built as a single string before `display()` call
- Step numbers now match the Table of Contents (1–10, no gaps or duplicates)

### Improved
- Memory config widget auto-applies on change and persists to `os.environ`
- Type hints and docstrings throughout
- `check_model_memory_safety` uses a single dict lookup instead of nested if-chains

## [2.4] - 2026-08-03

### Added
- Initial public release
- Interactive model browser with Ollama library scraping and static fallback
- Cloudflare quick tunnel, named tunnel, and ngrok support
- GPU monitoring widget
- Memory management policy widget
- DeepSeek-r1 model support

## [1.0] - 2026-07-XX

### Added
- Initial working version with qwen2.5-coder-32b support
- Basic Colab notebook structure
- ngrok tunnel integration
- GPU monitoring capabilities

[Unreleased]: https://github.com/TheRealFREDP3D/Ollama-Colab/compare/v2.6...HEAD
[2.6]: https://github.com/TheRealFREDP3D/Ollama-Colab/compare/v2.5...v2.6
[2.5]: https://github.com/TheRealFREDP3D/Ollama-Colab/compare/v2.4...v2.5
[2.4]: https://github.com/TheRealFREDP3D/Ollama-Colab/compare/v1.0...v2.4
[1.0]: https://github.com/TheRealFREDP3D/Ollama-Colab/releases/tag/v1.0-Working
