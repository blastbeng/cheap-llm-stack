# cheap-llm-stack

A self-contained, CPU-only LLM inference node for the **Orange Pi Zero 3W**
(Allwinner A733: 2× Cortex-A76 @2.0 GHz + 6× Cortex-A55 @1.79 GHz, 12 GB LPDDR5,
aarch64). One `sudo ./install.sh` builds everything, downloads the models and
starts an OpenAI-compatible endpoint.

```
client ──► :10000  llm-router          semantic orchestrator (Python, stdlib)
              │      rules → embeddings → default
              ▼
           :10001  llama-swap          model supervisor (Go, groups: core / heavy)
              ├── embedding-model       Qwen3-Embedding-0.6B  Q8_0   8k   core, always on
              ├── router-0.6b           Qwen3-0.6B            Q8_0   8k   core, fast tier
              ├── assistant             Qwen3.5-2B UD-Q4_K_XL 16k  text+vision  core
              ├── reasoner              LFM2-8B-A1B  MoE Q4_K_M 16k text workhorse  heavy
              ├── deep                  Qwen3.5-4B UD-Q4_K_XL 16k  thinking, opt-in  heavy
              └── vision-docs           Qwen3-VL-2B  UD-Q4_K_XL 16k  OCR/tables      heavy
```

## Why this shape

* **Two A76 cores are the whole budget.** Generation is bound by memory
  bandwidth, so bigger dense models are strictly slower. Measured on this exact
  board with `llama-bench` (q8_0 KV, flash-attn on, `--threads 2`):

  | model | file | tg (tok/s) | pp (tok/s) |
  | --- | --- | --- | --- |
  | Qwen3-0.6B Q8_0 | 0.60 GB | **16.9** | 107 (`-t 2`) |
  | Qwen3.5-2B UD-Q4_K_XL | 1.24 GB | **6.3** | 34 (`-t 8`) |
  | **LFM2-8B-A1B (MoE) Q4_K_M** | 5.04 GB | **10.7** | 24.8 (`-t 8`) |
  | OLMoE-1B-7B q4_k_m | 4.21 GB | 13.2 | 30.6 (`-t 8`) |
  | Qwen3.5-4B UD-Q4_K_XL | 2.70 GB | **3.0** | 12.7 (`-t 8`) |
  | Qwen3-VL-2B UD-Q4_K_XL | 1.05 GB | **7.5** | — (HTTP, ~2 s/image) |

* **MoE is the key finding.** `LFM2-8B-A1B` holds 8.4B of weights but activates
  only 1.1B per token, so it decodes **10.7 tok/s — 1.7× faster than the dense
  2B and 3.6× faster than the dense 4B**, while being a much stronger model. On
  a bandwidth-bound board this is the only way to get 8B-class quality without
  paying 8B-class latency. A single *dense* 8–9B model would land at ~1.5 tok/s
  (one 300-token answer = 3+ minutes), so dense is deliberately absent.
* **Quality was measured, not assumed.** Six prompts in Italian (summary, code,
  arithmetic, strict JSON, email, explanation), three models, anonymised and
  graded blind: **LFM2-8B-A1B won 5/6, Qwen3.5-4B 1/6, OLMoE-1B-7B 0/6**. OLMoE
  is the fastest of the three but lost on quality, so it is not shipped.
* **Thinking costs 2 minutes at 2.9 tok/s.** Qwen3.5-4B spent 349 tokens on
  `17*23` unconstrained (~2 min). It is therefore kept as `deep`, with
  `--reasoning-budget 256` and a budget message that stops the model mid-thought
  instead of truncating it, and it is **never auto-selected** — ask for it with
  `model: deep`.
* **Speculative decoding does not pay here.** Measured on the 2B with a 0.8B
  draft: 6.16 tok/s → 4.79 (n=3) → 3.09 (n=8); `ngram-*` gave no change. Draft
  verification is bandwidth-bound on the same two cores, so it is disabled.
* **Model swaps are the real cost.** Weights live on the microSD (~20–45 MB/s),
  a cold load costs 40–70 s. The resident set (embedding + 0.6B + 2B VLM + the
  5 GB MoE ≈ 9 GB) keeps everything a normal turn needs hot; the two cold models
  live in the `heavy` swap group so they can never push RAM into swap. After a
  first load, weights stay in page cache, so re-loading costs 1–3 s.
* **Vision is free.** Qwen3.5-2B is natively multimodal and resident, so images
  are answered with no swap: on a synthetic invoice `assistant` described it at
  6.2 tok/s, while the dedicated `vision-docs` (Qwen3-VL-2B) transcribed it
  verbatim at 7.5 tok/s.
* **The NPU is not usable.** The A733's NPU has no mainline driver, no
  `llama.cpp` backend and no `/dev/galcore`; `ggml` CPU (NEON dotprod + fp16) is
  the only path. `i8mm` is absent on A76, so the build uses
  `-march=armv8.2-a+dotprod+fp16`.

## Measured settings

| knob | value | why |
| --- | --- | --- |
| `--threads` | **2** | one per A76; more threads contend for bandwidth and slow generation (pinned to A55: 8.8 tok/s vs 16.8) |
| `--threads-batch` | **8** | prefill parallelises across all eight cores (+35–60 % pp) |
| KV cache | **q8_0** (`--flash-attn on`) | half the KV footprint, negligible quality loss |
| chat ctx | **8192** (deep/vision-docs 16384) | 8k prefill already costs ~5 min at 25 tok/s; bigger windows only burn RAM |
| embedding ctx | **4096** | mandatory model, always resident: keep its KV small |
| `--parallel` | **1** | llama.cpp sizes compute buffers per slot; the default 4 cost ~1 GB/model |
| load mode | `mmap+mlock` core, `mmap` heavy | core never pages out; the 5 GB MoE stays reclaimable |

### RAM, measured (this is what forced the sizes above)

`VmRSS` + `VmSwap` per llama-server with `--parallel 1`:

| model | weights (file) | anon (KV + buffers) | ctx |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | 0.62 GB | 1.0 GB at 8k → 0.5 GB at 4k | 4096 |
| Qwen3-0.6B | 0.62 GB | 1.16 GB at 8k → 0.6 GB at 4k | 4096 |
| Qwen3.5-2B + mmproj | 1.30 GB | 2.1 GB at 16k → 1.0 GB at 8k | 8192 |
| LFM2-8B-A1B | 5.04 GB (mmap) + 0.63 resident | **5.0 GB at 16k → 1.2 GB at 8k** | 8192 |

At 16k the four resident models needed ~12.5 GB and the board went into zram swap
(5.7 GB swapped, 400 MB available) — the MoE then took 5 minutes to answer its
health check. At the sizes above the same set fits in ~7 GB and leaves 4–5 GB of
page cache for the cold models.

`--load-mode mmap+mlock` (core) pins weights so they never page out; the heavy
group uses plain `mmap` so its 5 GB stay reclaimable instead of forcing swap.

## Groups

| group | swap | members | effect |
| --- | --- | --- | --- |
| `core` | no, `persistent` | embedding-model, router-0.6b, assistant | always loaded, never evicted |
| `heavy` | yes | reasoner, deep, vision-docs | one at a time ⇒ worst case ≈ 9 GB, never OOM |

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
`reasoning`, `trivial`, `semantic(0.71/0.09)`, `default`). Routes: `fast` →
router-0.6b, `assistant`/`reason` → reasoner, `vision` → assistant, `vision-docs`
→ vision-docs, `embedding` → embedding-model.

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
