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
              ├── embedding-model       Qwen3-Embedding-0.6B  Q8_0   4k   CORE, always on
              ├── reasoner              LFM2-8B-A1B  MoE Q4_0   8k   CORE, text workhorse
              ├── router-0.6b           Qwen3-0.6B            Q8_0   4k   heavy, fast tier, preloaded
              ├── assistant             Qwen3.5-2B UD-Q4_K_XL 4k   heavy, text+vision
              └── vision-docs           Qwen3-VL-2B  UD-Q4_K_XL 4k   heavy, OCR/tables
```

## Why this shape

* **Two A76 cores are the whole budget.** Generation is bound by memory
  bandwidth, so bigger dense models are strictly slower. Measured on this exact
  board with `llama-bench` (q8_0 KV, flash-attn on, `--threads 2`):

  | model | file | tg (tok/s) | pp (tok/s) |
  | --- | --- | --- | --- |
  | Qwen3-0.6B Q8_0 | 0.60 GB | **16.9** | 107 (`-t 2`) |
  | Qwen3.5-2B UD-Q4_K_XL | 1.24 GB | **6.3** | 34 (`-t 8`) |
  | **LFM2-8B-A1B (MoE) Q4_0** | 4.41 GB | **11.7** (`-t 2`) | **35.0** (`-t 8`) |
  | LFM2-8B-A1B (MoE) Q4_K_M | 5.04 GB | 10.7 (`-t 2`) | 25.0 (`-t 8`) | not shipped |
  | OLMoE-1B-7B q4_k_m | 4.21 GB | 13.2 | 30.6 (`-t 8`) | not shipped |
  | Qwen3.5-4B UD-Q4_K_XL | 2.70 GB | 3.0 | 12.7 (`-t 8`) | not shipped |
  | Qwen3-VL-2B UD-Q4_K_XL | 1.05 GB | **7.5** | — (HTTP, ~2 s/image) | shipped |

  The same MoE at Q4_K_M bought ~0.5 GB more quality for **−29 % prefill**
  (25.0 vs 35.0 tok/s) and −9 % decode, so Q4_0 ships. Both quantizations were
  re-measured after the last rebuild; the whole board runs ~7 % under the very
  first numbers, which is board state (page cache / thermals), not the build.

* **MoE is the key finding.** `LFM2-8B-A1B` holds 8.4B of weights but activates
  only 1.1B per token, so it decodes **11.7 tok/s — 1.7× faster than the dense
  2B and 3.6× faster than the dense 4B**, while being a much stronger model. On
  a bandwidth-bound board this is the only way to get 8B-class quality without
  paying 8B-class latency. A single *dense* 8–9B model would land at ~1.5 tok/s
  (one 300-token answer = 3+ minutes), so dense is deliberately absent.
* **Quality was measured, not assumed.** Six prompts in Italian (summary, code,
  arithmetic, strict JSON, email, explanation), three models, anonymised and
  graded blind: **LFM2-8B-A1B won 5/6, Qwen3.5-4B 1/6, OLMoE-1B-7B 0/6**. OLMoE
  is the fastest of the three but lost on quality, so it is not shipped.
* **There is no thinking tier, and that is a measurement.** A 4B dense model
  decodes at 3.0 tok/s here, a 2B at ~6: with `--reasoning on --reasoning-budget 256`
  the 4B spent **170 s** on one definition question, and the 2B with a 96-token
  budget still ran to 500 tokens = **83 s**. Nothing interactive fits in that, and
  every use of the 4B evicted the MoE, which then took **121 s** to come back from
  microSD. The 4B is therefore not shipped at all. The MoE answers the same
  questions correctly in ~11 s without a visible thinking pass.
* **The MoE is also the accuracy tier.** Six Italian factual prompts (MoE, Dante,
  `17*23`, HTTP/2, first iPhone, RAM vs swap) run against the MoE and the 2B:
  both got 6/6 right, the MoE was ~2x faster and had better grammar (the 2B wrote
  *"è un architetture"*, invented *"Dante Alciat"* and a wrong Purgatorio date).
  No split tier was worth adding. Its one real weakness is a bare acronym it does
  not know — asked `MoE` without the expansion it invented *"Model of Explanation"*,
  so the router sends definition-shaped questions to it only with the term intact.
* **Speculative decoding does not pay here.** Measured on the 2B with a 0.8B
  draft: 6.16 tok/s → 4.79 (n=3) → 3.09 (n=8); `ngram-*` gave no change. Draft
  verification is bandwidth-bound on the same two cores, so it is disabled.
* **Model swaps are the real cost.** Weights live on the microSD (~20–45 MB/s),
  a cold load costs 40–110 s (a VLM is 2.2 GB with its projector). The resident
  set is embedding + MoE + the 0.6B fast tier ≈ 7.5 GB, which covers every text
  request and leaves ~3.5 GB for one VLM; the VLMs live in the `heavy` swap group
  so they can never push RAM into swap. After a first load, weights stay in page
  cache, so re-loading costs 13 s.
* **Vision works, but the load and the encoder are the cost.** Two VLMs share the
  heavy group: `assistant` (Qwen3.5-2B, conversational) and `vision-docs`
  (Qwen3-VL-2B, OCR). Both read a synthetic invoice correctly — `assistant`
  transcribed `FATTURA N. 2026-114 / Data: 15/12/2026 / Imponibile: 1.036,69 EUR /
  IVA 22%: 228,07 EUR / Totale: 1.284,50 EUR` line by line, `vision-docs` answered
  the invoice number just as well — so the split is about the *kind* of question,
  not about one model being unable to see.
  Measured end-to-end on the invoice: **8.0 s** warm (same image, prompt cache),
  **68.8 s** when the VLM had to be loaded, ~100 s when the router also had to
  swap one VLM for the other. The image encoder (mmproj ViT) is the other half:
  185 image tokens took **48 s at `-t 2` and 30 s at `-t 8`**, so the VLMs alone
  use eight threads. The router therefore keeps the two VLMs from flapping:
  "trascrivi / estrai il testo / fattura / screenshot / tabella" go to
  `vision-docs`, everything else to `assistant`, and an explicit `model:` id
  always wins.
* **The NPU is not usable.** The A733's NPU has no mainline driver, no
  `llama.cpp` backend and no `/dev/galcore`; `ggml` CPU (NEON dotprod + fp16) is
  the only path. `i8mm` and `SVE` are absent, so the build pins
  `-march=armv8.2-a+dotprod+fp16`.
* **Two build experiments, both rejected by measurement** (llama-bench on
  LFM2-8B-A1B Q4_0, q8_0 KV, `-fa on`):

  | build | t2 pp512 | t2 tg96 | t8 pp512 | t8 tg96 |
  | --- | --- | --- | --- | --- |
  | `armv8.2-a+dotprod+fp16` (shipped) | **22.2** | **11.7** | **35.0** | **9.8** |
  | `GGML_NATIVE=ON` (cortex-a76.cortex-a55) | 22.1 | 11.7 | 34.8 | 9.8 |
  | above + `GGML_CPU_KLEIDIAI=ON` | 21.8 | 11.7 | 33.0 | 9.7 |

  KleidiAI's int8 micro-kernels need `i8mm` to beat ggml's own repack, and this
  A76 has none, so it is off. `--ubatch-size` was swept too: 256 and 128 are
  slower than the default 512.

## Measured settings

| knob | value | why |
| --- | --- | --- |
| `--threads` | **2** | one per A76; more threads contend for bandwidth and slow generation (pinned to A55: 8.8 tok/s vs 16.8) |
| `--threads-batch` | **8** | prefill parallelises across all eight cores (+35–60 % pp) |
| KV cache | **q8_0** (`--flash-attn on`) | half the KV footprint, negligible quality loss |
| chat ctx | **8192** | 8k prefill already costs ~5 min at 25 tok/s; 16k on the MoE costs +1.2 GB of KV and would leave no headroom for a VLM |
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
| LFM2-8B-A1B | 4.41 GB (mmap) + 0.63 resident | **5.0 GB at 16k → 1.2 GB at 8k** | 8192 |
| Qwen3.5-4B + mmproj | 2.91 GB | 3.1 GB at 16k | not shipped |

At 16k the four resident models needed ~12.5 GB and the board went into zram swap
(5.7 GB swapped, 400 MB available) — the MoE then took 5 minutes to answer its
health check. At the sizes above the same set fits in ~7 GB and leaves 4–5 GB of
page cache for the cold models.

`--load-mode mmap+mlock` (core) pins weights so they never page out; the heavy
group uses plain `mmap` so its 5 GB stay reclaimable instead of forcing swap.

Measured worse case, for the record: with the 4B resident next to the MoE the
kernel pushed **`VmSwap` 5.00 GB** of the MoE into zram (its `RssAnon` fell to
0.7 MB). That is the state the removed tier used to create, and it is why the
group model alone could not be trusted to keep the stack fast.

## Groups

| group | swap | members | effect |
| --- | --- | --- | --- |
| `core` | no, `persistent` | embedding-model, reasoner | always loaded, never evicted (~5.7 GB anon) |
| `heavy` | yes | router-0.6b, assistant, vision-docs | one at a time; all three preloaded except the VLMs |

Why the VLMs are *not* resident: 12 GB sits exactly at the limit. With
embedding + 0.6B + VLM + MoE loaded the board had 226 MB available and the
kernel swapped the whole router-0.6b (896 MB) and 2 GB of the MoE's KV into
zram. Resident = embedding + MoE leaves 3.7 GB available and zero swap.

The fast tier *is* preloaded even though it is evictable, because it answers the
most frequent requests: the first `ciao` after a reboot spent **141 s** waiting
for a cold 0.62 GB read from microSD, versus **1.1 s** once preloaded. A vision
load still evicts it; the warm reload then costs 13 s.

## Endpoints

| URL | what |
| --- | --- |
| `POST /v1/chat/completions` | `model: auto` → router picks; any explicit id bypasses routing |
| `POST /v1/embeddings` | always forced to `embedding-model` |
| `GET /v1/models` | proxy to llama-swap |
| `GET /health` | router + upstream status, per-route map |
| `GET /stats` | per-route counters, sticky sessions, prototype state |

Response headers `X-Route`, `X-Route-Model`, `X-Route-By`, `X-Route-Ms` show what
was chosen and why (`client`, `sticky`, `image`, `image+ocr-hint`, `chitchat`,
`code`, `reasoning`, `trivial`, `translate-short`, `semantic(0.71/0.09)`,
`semantic-fast-rejected`, `default`). Routes: `fast` → router-0.6b,
`assistant`/`reason` → reasoner, `vision` → assistant, `vision-docs` →
vision-docs, `embedding` → embedding-model.

Three routing guards exist because the 0.6B failed on them in testing: a request
that *writes* code is never sent to it however short it is (`scrivi una funzione
python …`, 9 words, produced broken docstring prose), one that asks what a
technical term *is* is not either (it answered `MoE = "Model of Explanation"`),
and neither is a short *factual* question — `Chi ha scritto la Divina Commedia e
in che anno è ambientata?` is 11 words and starts with `chi`, so it used to be
"trivial" and the 0.6B answered *"periodo 1301-1310"*. A question that contains a
proper noun or a `chi ha scritto` / `in che anno` / `who wrote` shape now goes to
the MoE.
Small talk goes the other way — `ciao, come stai?` now lands on the tiny tier in
1.0 s instead of waking the 5 GB model, with a one-line system prompt that keeps
it in the user's language (it used to answer Italian with Spanish).

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
