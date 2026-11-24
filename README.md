# cheap-llm-stack

A self-contained, CPU-only LLM inference node for the **Orange Pi Zero 3W**
(Allwinner A733: 2× Cortex-A76 @2.0 GHz + 6× Cortex-A55 @1.79 GHz, 12 GB LPDDR5,
aarch64). One `sudo ./install.sh` builds everything, downloads the models and
starts an OpenAI-compatible endpoint.

```
client ──► :10000  llm-router          semantic orchestrator (Python, stdlib)
              │      rules → embeddings → default
              ▼
           :10001  llama-swap          model supervisor (Go, one hot model at a time)
              ├── embedding-model       Qwen3-Embedding-0.6B  Q8_0   8k   always on
              ├── router-0.6b           Qwen3-0.6B            Q8_0   8k   always on
              ├── assistant             Qwen3.5-2B UD-Q4_K_XL 16k  text+vision  warm
              ├── assistant-large       Qwen3.5-4B UD-Q4_K_XL 16k  text+vision  on demand
              └── vision-docs           Qwen3-VL-2B  UD-Q4_K_XL 16k OCR/tables   on demand
```

## Why this shape

* **Two A76 cores are the whole budget.** Generation is bound by memory
  bandwidth, so bigger models are strictly slower. Measured on this exact board
  with `llama-bench` (q8_0 KV, flash-attn on, `--threads 2`):

  | model | file | tg (tok/s) | pp (tok/s) |
  | --- | --- | --- | --- |
  | Qwen3-0.6B Q8_0 | 0.60 GB | **16.9** | 107 (`-t 2`) |
  | Qwen3.5-2B UD-Q4_K_XL | 1.24 GB | **6.3** | 34 (`-t 8`) |
  | Qwen3.5-4B UD-Q4_K_XL | 2.70 GB | **3.0** | 12.7 (`-t 8`) |
  | Qwen3-VL-2B UD-Q4_K_XL | 1.05 GB | **7.5** | — |

  A single 8–9B model would land at ~1.5 tok/s: one 300-token answer would take
  over three minutes, so it is deliberately not part of the stack.
* **Model swaps are the real cost.** Weights live on the microSD (~20–45 MB/s),
  a cold load costs 40–70 s. The stack is therefore short (5 entries) and the
  router keeps every conversation **sticky** on one model. Frequently used
  weights stay in page cache after the first load, which is why a "swap" usually
  costs 1–3 s.
* **Vision is free.** Qwen3.5-2B is natively multimodal: one resident 2B model
  answers text *and* images, no separate VLM in RAM. A second, dedicated VLM
  (Qwen3-VL-2B) handles document-grade images: measured on a synthetic invoice,
  `assistant` described the image at 6.2 tok/s while `vision-docs` transcribed it
  verbatim at 7.5 tok/s.
* **The NPU is not usable.** The A733's NPU has no mainline driver, no
  `llama.cpp` backend and no `/dev/galcore`; `ggml` CPU (NEON dotprod + fp16) is
  the only path. `i8mm` is absent on A76, so the build uses
  `-march=armv8.2-a+dotprod+fp16`.

## Measured settings

| knob | value | why |
| --- | --- | --- |
| `--threads` | **2** | one per A76; more threads contend for bandwidth and slow generation |
| `--threads-batch` | **8** | prefill parallelises across all eight cores (+35–60 % pp) |
| KV cache | **q8_0** (`--flash-attn on`) | half the KV footprint, negligible quality loss |
| chat ctx | **16384** | 32k costs ~2× KV reads for no real gain at these speeds |
| embedding ctx | **8192** | mandatory model, always resident: keep its KV small |
| eviction | `q8_0` + `--mlock` on resident models | weights never get dropped under pressure |

## Endpoints

| URL | what |
| --- | --- |
| `POST /v1/chat/completions` | `model: auto` → router picks; any explicit id bypasses routing |
| `POST /v1/embeddings` | always forced to `embedding-model` |
| `GET /v1/models` | proxy to llama-swap |
| `GET /health` | router + upstream status, per-route map |
| `GET /stats` | per-route counters, sticky sessions, prototype state |

Response headers `X-Route`, `X-Route-Model`, `X-Route-By`, `X-Route-Ms` show what
was chosen and why (`client`, `sticky`, `image`, `image+ocr-hint`, `code`,
`reasoning`, `trivial`, `semantic(0.71/0.09)`, `default`).

## Operations

```bash
sudo ./install.sh                 # build + models + units  (--skip-build, --skip-models, --dry-run)
sudo ./install.sh --tune          # same, plus OS tuning
sudo ./scripts/tune.sh --check    # show unapplied OS tuning
sudo ./scripts/tune.sh --apply    # swappiness, noatime, governor, zram checks
./healthcheck.sh                  # quick liveness of both tiers
./benchmark.sh --all              # pp/tg per model, embedding latency
sudo ./update-llama.cpp.sh        # rebuild llama.cpp at the latest tag (timer runs it weekly)
```

Model downloads are resumable (`curl -C -`); re-running `install.sh` only fetches
what is missing. llama.cpp is rebuilt every Sunday at 03:00 by
`llama-swap-update.timer` and the service restarted only when the tag changes.

## Layout

```
/opt/cheap-llm-stack
├── bin/            llama-server, llama-bench, llama-swap
├── models/         GGUF weights and mmproj projectors
├── llama.cpp/      pinned source tree used for the build
├── router/         llm_router.py   (the orchestrator)
├── scripts/        tune.sh
├── config.yaml     llama-swap config
└── systemd units   installed to /etc/systemd/system
```
