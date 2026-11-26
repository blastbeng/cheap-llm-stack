# cheap-llm-stack

A self-contained, CPU-only LLM inference node for the **Orange Pi Zero 3W**
(Allwinner A733: 2× Cortex-A76 @2.0 GHz + 6× Cortex-A55 @1.79 GHz, 12 GB LPDDR5,
aarch64). One `sudo ./install.sh` builds everything, downloads the models and
starts an OpenAI-compatible endpoint that speaks **chat, vision, embeddings and
tool calling**.

```
client ──► :10000  llm-router          content-aware orchestrator (Python, stdlib)
              │      tools → rules → embeddings → default
              ▼
           :10001  llama-swap          model supervisor (Go, groups: core / heavy)
              ├── embedding-model       Qwen3-Embedding-0.6B  Q8_0   4k   CORE, always on
              ├── reasoner              LFM2.5-8B-A1B  MoE Q4_0  16k   CORE, text + TOOLS
              ├── router                LFM3-1.7B     QAD-Q4_0   8k   CORE, chitchat / one-liners
              └── vision                LFM3-VL-2B    QAD-Q4_0   6k   heavy, images / OCR
```

Four ids, one job each (plus the optional 0.6B). Anything that needs to think, write or call a tool goes
to the 8B MoE; greetings, one-liners and short translations go to the 1.7B (with the
0.6B kept installed for when 1 s latency matters more than prose); anything with a
pixel in it goes to the VLM; embeddings are always on.

## Why this shape

* **Two A76 cores are the whole budget.** Generation is bound by memory
  bandwidth, so bigger dense models are strictly slower. Measured on this exact
  board with `llama-bench` (q8_0 KV, flash-attn on, `--threads 2`):

  | model | file | tg (tok/s) | pp (tok/s) | note |
  | --- | --- | --- | --- | --- |
  | Qwen3-0.6B Q8_0 | 0.62 GB | **16.2** | 92 (`-t 2`) / 125 (`-t 8`) | optional 1 s tier |
  | **LFM2.5-8B-A1B (MoE) Q4_0** | 4.85 GB | **11.6** | 22.6 (`-t 2`) / 34.8 (`-t 8`) | shipped, default + tools |
  | **LFM3-1.7B QAD-Q4_0** | 1.05 GB | **10.1** | 41.4 (`-t 2`) / 62.9 (`-t 8`) | shipped, fast tier |
  | **LFM3-VL-2B QAD-Q4_0** | 1.05 GB | **8.1** | 33.1 (`-t 8`) | shipped, vision |
  | Qwen3.5-2B UD-Q4_K_XL | 1.24 GB | 6.3 | 34 (`-t 8`) | not shipped |
  | LFM2-8B-A1B (MoE) Q4_0 | 4.41 GB | 11.7 | 35.0 (`-t 8`) | replaced by LFM2.5 |
  | Qwen3-VL-2B UD-Q4_K_XL | 1.05 GB | 7.5 | — | replaced by LFM3-VL |
  | Qwen3.5-4B UD-Q4_K_XL | 2.70 GB | 3.0 | 12.7 (`-t 8`) | not shipped |

  The MoE at Q4_K_M bought ~0.5 GB more quality for **−29 % prefill** and −9 %
  decode, so Q4_0 ships.

* **MoE is the key finding.** `LFM2.5-8B-A1B` holds 8.3B of weights but activates
  only ~1.5B per token, so it decodes **11.6 tok/s — 1.8× faster than the dense
  2B and 3.9× faster than the dense 4B**, while being a much stronger model. A
  single *dense* 8–9B model would land at ~1.5 tok/s (one 300-token answer = 3+
  minutes), so dense is deliberately absent.

* **Tool calling decided the workhorse.** An agent client (Open WebUI, or anything
  speaking OpenAI + MCP) needs a model that emits structured `tool_calls` and then
  reads its own tool results. Measured on the same 7-case battery (5 tools, one
  request that must fan out to two tools, one chitchat case that must call nothing,
  Italian and English):

  | model | score | per call | verdict |
  | --- | --- | --- | --- |
  | **LFM2.5-8B-A1B, `reasoning_budget_tokens=32`** | **7/7** | 17.9 s | shipped: `reasoner` |
  | LFM2.5-8B-A1B, budget 96 | 7/7 | 24.1 s | same score, 35 % slower |
  | LFM2.5-8B-A1B, budget 0 | 5/7 | 52 s | thinking leaks into `content` → unusable |
  | Qwen3-0.6B Q8 | 7/7 | 14–21 s | mangles arguments, poor prose |
  | LFM2.5-1.2B-Instruct Q8 | 6/7 | 28 s | refuses `save_memory` |
  | LFM3-1.7B Q4_K_M | 6/7 | 37–89 s | one empty answer |
  | LFM3-VL-2B QAD-Q4_0 | 6/7 | — | also handles image + tools |
  | MiniCPM5-2B | 5/7 | — | invents a tool call on chitchat |
  | **LFM2-8B-A1B (previous workhorse)** | **1/7** | — | never emits `tool_calls`, answers instead |

  That last row is why the workhorse was replaced: LFM2.5 knows when *not* to
  answer and calls a tool instead, which is the whole point of an agent stack.
  Its tool calls stream as `tool_calls` deltas over SSE (5–9 chunks, correct
  `name` + `arguments`), which is what Open WebUI's native function calling reads.
  This build has a dedicated native handler for the LFM2.5 chat template
  (`common/chat.cpp: common_chat_params_init_lfm2_5`), so calls are generated
  through a PEG grammar rather than the lossy generic parser.

### Model sweep, 26 Nov 2026 (8-case battery = the 7 above + the synthesis turn)

Everything new that fits an 8B-A1B budget was tried: custom fine-tunes, abliterated
and uncensored merges, imatrix quants, the Ling-3.0-tiny family, the new LFM3
backbone. Same harness for all (`ctx 12288`, q8_0 KV, `--threads 2/8`, budget 32,
temperature 0.6, seeded tool battery; six fixed quality cases):

| model | quant | file | t2 pp512 / tg64 | tools | verdict |
| --- | --- | --- | --- | --- | --- |
| **LFM2.5-8B-A1B** | Q4_0 | 4.50 GiB | 22.98 / 10.37 | **8/8** | baseline, still the best text+tools |
| **LFM3-1.7B** | QAD-Q4_0 | 0.99 GiB | **41.44 / 10.13** | 7/8 | **shipped as the fast tier**: 1.8× prefill, same decode, 1/4.5 the RAM, 256K ctx |
| LFM3-1.7B | Q8_0 | 1.85 GiB | 29.77 / 6.11 | 8/8 | 8/8 tools but 40 % less decode — Q4_0 wins |
| Moon-Beam-8B-A1B (Ling-3.0-tiny + engram) | IQ4_XS | 4.49 GiB | 20.31 / 8.39 | 8/8 | good prose, but 20 % less decode and reasoning leaks into `content` |
| edge0-8b (Ling-3.0-tiny) | Q4_1 | 4.65 GiB | 12.90 / 8.56 | 8/8 | slowest prefill of the 8B class, same leak |
| LFM2.5-8B-A1B-Uncensored | IQ4_XS | 4.29 GiB | 15.14 / 9.65 | 8/8 | works, but −34 % prefill and it answers Italian prompts in English |
| LFM2.5-8B-A1B-Hermes-Abliterated | IQ3_XXS | 3.33 GB | bench stalls | — | unusable on this build |
| LFM2.5-Q3.8-TBrilliance-NEO-MAX | IQ4_XS | 4.55 GiB | 15.04 / **2.53** | — | 4× slower decode, misattributes the Commedia → reject |
| EuroMoE-2.6B-A0.6B | Q4_K_M | 1.62 GB | — | — | `ggml-impl.h` fatal error: unsupported quant/arch |

Tool numbers are not the whole story: all of these models are the same generation
and share the same flaws (the shipped MoE also leaks a little thinking into
`content` on extraction, and also got the 14:35+2h50 arithmetic wrong), but on the
six quality cases the MoE is the only one that gets the Italian facts right *and*
it has the fastest decode. That, plus 20–34 % less prefill for every IQ4_XS
candidate, is why nothing replaced the workhorse.

* **The thinking budget is a real knob, not a formality.** `reasoning_budget_tokens`
  (per request, `--reasoning-budget` server-side) is what keeps LFM2.5 honest:
  at `0` it answers immediately but writes its monologue into `content`
  (*“The user's request is to send a message on Telegram…”*) and drops to 5/7;
  at `32` it ends the thinking block cleanly and still picks the right tool 7/7.
  The router sets it per route: **96 for plain chat**, 32 for tools and vision,
  256 for the rare `reason` request, nothing for the fast tier. Tools keep 32
  because a tool decision needs no plan (and 96 costs 6 s per call); chat gets 96
  because 32 cuts the thought mid-sentence on anything that needs a plan and the
  leftover is then returned inside `content` — measured on *"spiega passo per passo
  come funziona TCP"*: at 32, 150 chars of thought landed in `reasoning_content`
  and the rest of the monologue in `content`; at 96 it is clean, and on typical
  prompts it is 0–3 s *faster*, because the model stops re-thinking.

* **One VLM replaced two.** `LFM3-VL-2B` reads the same synthetic invoice as the
  old `vision-docs` (Qwen3-VL-2B) in **24.0 s vs 113.4 s**, describes an image in
  **24.8 s vs 145.7 s** (Qwen3.5-2B), and takes 1.25 GB of weights instead of
  ~2.4 GB per model. It also does tool calls (6/7), so one turn can carry both an
  image and tools.

  Image size is the knob that decides whether vision is usable. Measured, same
  prompt, same model (tokens ≈ pixels/1000, and prefill dominates):

  | image | image tokens | wall |
  | --- | --- | --- |
  | invoice PNG, 27 KB | 335 | 32 s (first call after the VLM was evicted: includes the load) |
  | photo 800×600, 92 KB | 498 | 38 s |
  | photo 1600×1200, 299 KB | **1923** | **145 s** |

  So: downscale to ~1 MP (≈1000 tokens, ~60–90 s) before sending, and treat a
  straight-from-the-phone 12 MP photo as a 10-minute request that will also eat the
  context. It is not a bug in the stack — it is 1923 tokens of prefill at ~25 tok/s.

  It is also *literal* about wording, which is worth knowing before blaming the stack.
  `Estrai: fornitore, numero, data, totale. … solo i campi` returns the four labels
  (or labels then values); the same request phrased as *"estrai i **valori** di …,
  una riga per campo, formato campo: valore"* returns all four values in 24 s,
  deterministically. Measured: the thinking budget does not matter here (0, 32 and
  128 produce the same extraction, 27–37 s), so vision keeps 32 like the rest.

* **Agent payloads are prefill-heavy, and the prefix cache is the fix.** A real
  client sends a system prompt plus tool schemas on every turn: ~1k tokens cold
  cost **43 s** on the MoE, the next turn in the same conversation cost **28–30 s**
  instead of the 24 s baseline. That is why `--parallel 1` matters twice: it keeps
  the compute buffers small *and* keeps the tool-schema block in the slot's prompt
  cache. It is also why `reasoner` has `ttl: 0` — an unload throws that cache away.

* **Allocating context is nearly free; filling it is not.** `LFM2.5-8B-A1B` has
  attention KV in only 6 of its 24 layers — the other 18 are gated shortconv with a
  constant-size state — so the window costs **~4.8 MB per 1000 tokens** and changes
  nothing else. Measured with the same 1.5k-token prompt (`--threads 2/8`,
  `--parallel 1`, q8_0 KV):

  | `--ctx-size` | 4096 | 8192 | 12288 | 16384 | 24576 | 32768 | 40960 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `Anonymous` | 4705 MB | 4725 | 4744 | 4764 | 4803 | 4842 | 4881 |
  | pp / tg | 34.3 / 9.03 | 34.3 / 9.03 | 34.3 / 9.03 | 34.3 / 9.01 | 34.3 / 9.01 | 34.2 / 9.01 | 34.3 / 9.03 |

  What costs is the prompt: prefill and decode fall with depth, and the needle at
  the start of the prompt was still retrieved at every one of these points, so the
  quality holds while the latency does not:

  | prompt tokens | 1.5k | 1.9k | 2.6k | 5.7k | 7.7k (93 % of 8k) | 9.6k | 14.3k |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | pp (tok/s) | 34.3 | 33.1 | 31.2 | 23.4 | 17.7 | 17.0 | 13.7 |
  | tg (tok/s) | 9.03 | 8.68 | 8.20 | 6.03 | 4.52 | 4.77 | 3.76 |

  Hence **16384**, not 12288: +19 MB anon for 4k more room, because an agent client
  burns context with tool schemas and tool results. 24576 and 32768 are affordable
  too (+59 / +98 MB) — they are a policy choice, not a memory one: a full 16k turn
  is a 6–10 minute prefill, which is why tool results are capped at 6000 chars
  upstream. The model itself was trained for 128k.

* **There is no thinking tier, and that is a measurement.** A 4B dense model
  decodes at 3.0 tok/s here, a 2B at ~6: with `--reasoning-budget 256` the 4B
  spent **170 s** on one definition question, and the 2B with a 96-token budget
  still ran to 500 tokens = **83 s**. Nothing interactive fits in that, and every
  use of the 4B evicted the MoE, which then took **121 s** to come back from
  microSD. The MoE answers the same questions correctly in ~11 s.

* **Speculative decoding does not pay here.** Measured on the 8B MoE with its
  328M DSpark drafter: 10.2 tok/s → 3.36–5.64 (verification is bandwidth-bound on
  the same two cores). An earlier 0.8B drafter on the 2B went 6.16 → 4.79 (n=3) →
  3.09 (n=8). Drafts are rejected.

* **Model swaps are the real cost.** Weights live on the microSD (~20–45 MB/s), a
  cold load costs 40–110 s. The resident set is embedding + 1.7B + MoE ≈ 6.4 GB,
  which covers every text and tool request and leaves several GB of page cache for
  the VLM; the VLM lives in the `heavy` swap group so it can never push RAM into
  swap. After a first load, weights stay in page cache, so reloading costs ~13 s.

* **The NPU is not usable.** The A733's NPU has no mainline driver, no
  `llama.cpp` backend and no `/dev/galcore`; `ggml` CPU (NEON dotprod + fp16) is
  the only path. `i8mm` and `SVE` are absent, so the build pins
  `-march=armv8.2-a+dotprod+fp16`.

* **Q4_0 is the fastest 4-bit format here, and repacking is why.** Same
  LFM2.5-8B-A1B weights, same q8_0 KV, `llama-bench -t 2,8 -p 512 -n 64`:

  | quant | file | t2 pp512 | t2 tg64 | t8 pp512 | t8 tg64 |
  | --- | --- | --- | --- | --- | --- |
  | **Q4_0 (shipped)** | 4.50 GiB | **22.98** | **10.37** | **37.12** | **9.07** |
  | Q4_K_M (official) | 4.79 GiB | 16.44 | 9.39 | 24.96 | 8.14 |
  | IQ4_XS (community) | 4.29 GiB | 15.14 | 9.65 | 24.50 | 8.97 |
  | Q4_1 (community) | 4.65 GiB | 12.90 | 8.56 | 21.54 | 7.66 |

  `--repack 0` on the shipped file drops it to **11.48 pp / 7.49 tg**, so weight
  repacking is worth 2× prefill and +37 % decode; Q4_0's repacked kernels
  (`q4_0_8x8` NEON+dotprod) beat both the K-quants and the imatrix quants on this
  CPU, and Q4_0 also uses ~330 MB less anonymous memory than Q4_K_M. A community
  fine-tune that only ships IQ4_XS therefore starts with a 34 % prefill handicap.
  Two consequences: `-fa on` is mandatory (`--flash-attn off` cannot create a
  context with quantized KV: *failed to create context*), and `-b`/`-ub` are not
  worth tuning (`-ub 1024` → 33.07 t/s vs 33.08 at the default).

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
| chat ctx | **16384** | 12288→16384 is +19 MB anon (4.8 MB per 1000 tokens, measured up to 40960) and zero pp/tg change while prompts are short; buys the room an agent client burns on tool schemas and results. The tradeoff is filling it: 5.7k tokens = 23.4 pp / 6.0 tg, 9.6k = 17.0 / 4.8, 14.3k = 13.7 / 3.8 |
| embedding ctx | **4096** | mandatory model, always resident: keep its KV small |
| fast tier ctx | **8192** | LFM3-1.7B: 4k strangled a client that sends a system prompt plus a conversation; 8k is ~50 MB of KV and the first turn pays the prefill once |
| VLM ctx | **6144** | 4096 strangled any client that sends a system prompt plus a conversation, and a VLM also pays ~1k tokens for one image; the 2k extra are cheap in RAM and later turns ride the prompt cache |
| `--parallel` | **1** | compute buffers are sized per slot (default 4 cost ~1 GB/model) and one slot keeps the tool-schema prompt cache alive |
| `--reasoning-budget` | **32** | measured sweet spot: clean `content`, 7/7 tool calling, 17.9 s/call |
| load mode | `mmap+mlock` core, `mmap` heavy | core never pages out; the 4.9 GB MoE stays reclaimable |

### RAM, measured (this is what forced the sizes above)

`VmRSS` + `VmSwap` per llama-server with `--parallel 1`:

| model | weights (file) | `Anonymous` (KV + buffers) | ctx |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | 0.62 GB | 0.70 GB | 4096 |
| LFM3-1.7B | 1.05 GB | ~1.0 GB | 8192 |
| LFM2.5-8B-A1B | 4.85 GB (mmap) | **4.67 GB** | 16384 |
| LFM3-VL-2B + mmproj | 1.05 + 0.20 GB | 1.28 GB | 6144 |
| Qwen3-0.6B (optional) | 0.62 GB | 1.00 GB | 6144 |

`Anonymous` from `smaps_rollup` with all four loaded: 7.6 GB, `free` reports
9.5 GB used and 2.4 GB available, no OOM, 64 °C. The ctx bump costs 59 KB/token on
the 0.6B — measured 905 MB at 4096 vs 1025 MB at 6144, which is exactly the q8_0 KV
size, so it is the only part that scales. Both layers that cap a prompt agree on the
new number: with a prompt above the window, the proxy's clamp reports
`"context_tokens": 6144` for `cheap/router`, and `cheap/vision` reaches the board and
is refused with `available context size (6144 tokens)` — nothing still assumes 4096.

The earlier "16k → zram swap" incident was **not** the KV cache: it was the default
`--parallel 4` compute buffers (~1 GB per model), which `--parallel 1` removed. With
one slot the MoE's whole window costs 4.8 MB per 1000 tokens, so 16384 adds 19 MB
over 12288 and the resident set above fits with GBs of page cache to spare for the
cold VLM.

## Groups

| group | swap | members | effect |
| --- | --- | --- | --- |
| `core` | no, `persistent` | embedding-model, router, reasoner | always loaded, never evicted (~6.5 GB anon) |
| `heavy` | yes | vision | loads on demand, `ttl: 900` |

The VLM is *not* resident: with embedding + 0.6B + VLM + MoE loaded the board once
had 226 MB available and the kernel swapped the whole `router` tier (896 MB) and
2 GB of the MoE's KV into zram.

The fast tier *is* kept warm even though it is evictable, because it answers the
most frequent requests: the first `ciao` after a reboot spent **141 s** waiting
for a cold microSD read (measured on the 0.62 GB 0.6B it replaced; LFM3-1.7B is
1.05 GB), versus **1.1 s** with the weights in page cache.

**Warming is done by the router, not by llama-swap's `hooks.on_startup.preload`,
and that hook is deliberately absent from `config.yaml`.** The hook is a synthetic
`GET /` routed to the model, and the status of that GET is what llama-swap reports
as success (upstream `internal/server/api.go`, `startPreload`). llama.cpp built
with `-DLLAMA_USE_PREBUILT_UI=OFF` registers no `/` route, so every start logged
`[ERROR] failed to preload model <id>: status 404` while the model loaded and
passed its health check — reproduced against a stub upstream, and a permanent
false alarm in the only monitor this board has is worse than the silence it
replaces. Turning the UI on to satisfy the probe would add a Hugging Face bundle
download to every weekly rebuild. `router/llm_router.py: warm_loop` does the job
properly instead: one 1-token request per core model through llama-swap, at boot
**and every time the llama-swap process is replaced** — the case the hook
cannot cover, because the weekly rebuild restarts llama-swap and the first tool
case of the update gate then measured the MoE load (**93 s**) rather than the
build (**17 s**). A 1-token completion also warms the prompt cache, which `GET /`
never touches. The restart is detected by reading the pid and start time of the
llama-swap process out of `/proc`, not by polling `/health`: a `systemctl restart`
is over in 1-2 s, so a 20 s liveness poll misses the window and warms nothing —
the first version of this loop did exactly that, silently.

`reasoner` has `ttl: 0` and no eviction: a cold load is ~100 s, and an unload also
throws away the prompt cache that makes agent traffic affordable. (An earlier
`ttl: 1800` silently unloaded it at 03:07 and the next request paid the full load.)

## Endpoints

| URL | what |
| --- | --- |
| `POST /v1/chat/completions` | `model: auto` → router picks; any explicit id bypasses routing |
| `POST /v1/embeddings` | always forced to `embedding-model` |
| `GET /v1/models` | llama-swap's list **plus** a synthetic `auto` (tools + vision) |
| `GET /health` | router + upstream status, per-route map and budgets |
| `GET /stats` | per-route counters, sticky sessions, prototype state |

Response headers `X-Route`, `X-Route-Model`, `X-Route-By`, `X-Route-Ms`,
`X-Route-Budget` show what was chosen and why (`client`, `sticky`, **`tools`**,
`image`, `image+ocr-hint`, `chitchat`, `code`, `reasoning`, `trivial`,
`translate-short`, `semantic(0.71/0.09)`, `semantic-fast-rejected`, `default`).

Routes: `fast` → router, `assistant`/`reason`/**`tools`** → reasoner,
`vision`/`vision-docs` → vision, `embedding` → embedding-model.

### Tool-calling routing

Tool turns are decided **before** everything else, including stickiness:

* `tools` (or legacy `functions`) in the request → route `tools`;
* a `role: "tool"` message, an assistant `tool_calls` entry, or a tool-result
  content part in the history → route `tools` (we are mid-loop and must keep
  hitting the model that issued the call);
* the session's sticky entry is re-pinned to `tools`, so a conversation that
  started as `ciao` cannot stay on the 0.6B once a tool appears.

An explicit model id is normally forwarded verbatim — that is how you debug one
tier. The one exception is the fast tier: a request addressed to `router`,
`router-0.6b`, `fast` or `tiny` that carries `tools` (or a tool history) is
answered by the `tools` route instead, and reports `X-Route-By: tools-on-tiny`.
Tool orchestration lives on the MoE: LFM3-1.7B does emit calls (7/8 at Q4_0, 8/8 at
Q8_0) but the MoE is 8/8 at the same 10.4 tok/s, and the 0.6B mangles the arguments.
`router` as a *routing* id (`auto`, `router`, `orchestrator`) never reaches this
check — `tools` wins before the model is picked.

The tool payload is passed through verbatim (`tools`, `tool_choice`,
`parallel_tool_calls`), no system prompt is injected (it would shift the cached
prefix), the thinking budget is set to 32, the answer ceiling is raised to at least
800 tokens so a client's `max_tokens: 200` cannot truncate the call instead of the
prose, and **temperature is raised to 0.6** (see below). A client that tries to
switch thinking *off* (`chat_template_kwargs.enable_thinking: false`,
`reasoning_effort: "none"`) is overruled, because on a reasoning-only model that is
what pushes the monologue into `content`.

Why the temperature floor. Two of the seven cases are temperature-sensitive, and
not in the direction you would guess — a *cold* sampler answers a tool request in
prose instead of emitting the call. Measured, seed 1234, same prompt, `tg` = "manda
un messaggio su Telegram" and `none` = "raccontami una barzelletta" (must call
nothing):

| budget | temp | tg | none | result |
| --- | --- | --- | --- | --- |
| 32 / 64 / 96 / 192 | 0.2 | no call | calls `web_search` | 0/2 |
| 96 | 0.6 | ok | no call | **2/2** |
| 32 (full 7-case battery) | 0.6 | ok | no call | **7/7**, 17–19 s/call |

The budget is therefore irrelevant here (that is why it stays at 32, the fastest) and
the floor is what fixes it. It only *raises*: a client asking for 0.8 or 1.0 keeps
its value. `ROUTER_TOOLS_MIN_TEMPERATURE=0` disables it, which is also what makes
"5/7 direct, 7/7 through the router" a reproducible A/B:
`scripts/toolcheck.py --only tg,none --url http://127.0.0.1:10001 --model reasoner`
bypasses the router, and therefore the floor, on purpose.

### Routing guards

Three exist because the 0.6B failed on them in testing: a request that *writes*
code is never sent to it however short it is (`scrivi una funzione python …`, 9
words, produced broken docstring prose), nor is one that asks what a technical
term *is* (it answered `MoE = "Model of Explanation"`), nor a short *factual*
question — `Chi ha scritto la Divina Commedia e in che anno è ambientata?` is 11
words and starts with `chi`, so it used to be "trivial" and the 0.6B answered
*"periodo 1301-1310"*. A question containing a proper noun or a `chi ha scritto` /
`in che anno` / `who wrote` shape now goes to the MoE.

Small talk goes the other way — `ciao, come stai?` lands on the tiny tier in 1.0 s
instead of waking the 5 GB model, with a one-line system prompt that keeps it in
the user's language (it used to answer Italian with Spanish).

### Known limitation: identifiers inside tool arguments

Every model tested mangles exotic identifiers in tool arguments — `aiderdesk_global_memory`
came back as `hcierdesk_…`, `hpperdesk_…`, `helperdesk_…` from the 8B, the 0.6B and
the 1.2B alike, while the *tool name* and the JSON shape were always correct. On a
2 GB-token vocabulary a made-up compound noun is cheap to mis-spell. Mitigate on
the client side (declare the collection as an `enum` in the tool schema, or
normalise/repair in the MCP gateway) — a bigger model does not fix it.
`toolcheck.py` reports it as a warning and does not fail the run.

## Open WebUI recipe

In production Open WebUI does not point at the board directly: it goes through
LiteLLM on the media host (`http://192.168.1.13:4000/v1`), so the ids it sees are
`cheap/auto`, `cheap/router`, `cheap/reasoner`, `cheap/vision` and
`cheap/embedding-model`. Two properties of that client are not obvious, and both
of them bit us:

* **Native function calling is expensive here and it is ON by default.** With
  the builtin tool categories untouched, a plain `hello` carried 28 schemas and
  **5,430 prompt tokens**; adding one MCP server made it 31 schemas and **7,640
  tokens** — 95 % of the prompt. On the tiny tier (4k then, 6k now) that is a
  guaranteed `413 prompt_too_large`, on the 12,288-token MoE it leaves room for
  one turn.
  The context limit is not the problem; the schemas are.
* **Raising `--ctx-size` does not fix schemas.** Measured prefill on this board is
  21–35 tok/s, so a 7.6k prompt costs 4–6 minutes *per request*. More context only
  buys room for the schemas to fill. It was still worth going 4096 → 6144 on the
  tiny/VLM tiers (see Measured settings), because once the surface is down to two
  tools the *fixed* part of the prompt is ~1k and the cache makes later turns
  nearly free — the extra room then goes to the conversation, which is the point.

So, for every model that shadows a connection id (Workspace → Models, or
`POST /api/v1/models/create` with `base_model_id: null` and the connection id as
`id`):

1. `meta.builtinTools` — every category `false`: `time`, `user_input`, `files`,
   `knowledge`, `chats`, `subagents`, `memory`, `web_search`,
   `image_generation`, `code_interpreter`, `notes`, `channels`, `skills`,
   `tools`, `tasks`, `automations`, `calendar`, `notifications`. Keep
   `meta.capabilities.builtin_tools` **true** — that flag also enables inlining
   of attached file content, and with every category off it emits no schema.
2. `meta.capabilities` — `terminal`, `web_search`, `image_generation`,
   `code_interpreter`, `memory` false; `vision` true only on `cheap/vision`.
3. Attach **`mcp-tools-lite`** (`server:mcp:mcp-tools-lite`): 2 meta-tools
   (`find_tool`, `run_tool`), ~640 tokens, and the whole 109-tool gateway behind
   one lookup. In Open WebUI
   a global `inlet` filter injects that id into every chat, so `TOOL_IDS` there
   must equal the connection's `info.id`. That filter is also where the
   orchestration is consolidated: it *strips* every tool id for `cheap/router`
   (the tiny tier never gets one, not even when picked by hand) and only
   auto-injects on `cheap/auto`, `cheap/reasoner` and the non-cheap models;
   `cheap/vision` can still opt in manually for the image+tools turn.
4. Leave `chat.tool_search` and `chat.context_compaction` off — both add a pass
   over the prompt, and compaction cannot fit a 6k model anyway.

Measured with this profile: `cheap/router` carries **50 prompt tokens** and no
tools at all, `cheap/reasoner` carries **819** with the two meta-tools, 15.7 s
first-turn prefill and **0.3 s on the following turn** — with a constant 2-tool
surface the chat prefix stays byte-identical, llama.cpp reuses its cache and only
the new message is prefilled. That cache hit is the real payoff of a small tool
surface, more than the tokens themselves. `cheap/auto` answers the same shape of
request in 990 tokens and routes through the orchestrator. `auto`, `router` and
`orchestrator` are the magic ids that mean "route this" — `router` included, it
does not pin the 0.6B; `reasoner` and `vision` are the direct ids, useful for
debugging only.

### Which id to choose in the model picker

`cheap/auto` — this is the one. It reaches the **router on :10000**, which
inspects the request and forwards to the right llama-swap model: tools, images,
code/translate/reason and the embedding-similarity tier are decided in Python,
and the choice is pinned per session so a conversation does not bounce. Every
other `cheap/*` id in the picker is a *direct* llama-swap model that bypasses
that decision.

| picker id | what it really is | when to use |
|---|---|---|
| `cheap/auto` | router, picks per request | always; it is the only id that can reach the tools tier |
| `cheap/reasoner` | LFM2.5-8B-A1B, pinned | when you know you want the MoE and want to skip routing |
| `cheap/vision` | LFM3-VL-2B | images, OCR |
| `cheap/embedding-model` | Qwen3-Embedding-0.6B | never — it is an embedding model, not a chatbot |
| `cheap/router` | LFM3-1.7B fast tier | **hidden** (`is_active 0`): the name collides with the router's own magic id `router`, which means "route this", so picking it looked like routing and instead pinned the fast tier |

`auto`, `router` and `orchestrator` are the three ids the router itself treats as
"decide for me"; `cheap/router` through LiteLLM arrived as the literal model id
`router`, so the two meanings met in the same picker. Keeping the fast tier
reachable as `fast` / `tiny` / `lfm3` (and the 0.6B as `router-0.6b`) while hiding
`cheap/router` removes the ambiguity without losing either tier.

### Why "search the web" used to hang (measured 26 Nov)

Four faults stacked into one symptom — an answer that never arrives:

1. **The meta-tools were named after the mechanism, not the effect.**
   `tools_search` / `tool_call` read as "search" / "maybe call something", and the
   MoE answered exactly that: it searched in round 1 *and* round 2 and then
   stopped with nothing. Renamed to **`find_tool`** (returns schemas, runs
   nothing) and **`run_tool`** (the only way to get real data); the descriptions
   say that in those words and the `query` description asks for the *action*
   ("search the web for X"), not the user's question. Same two tools, same ~640
   prompt tokens. Old names still dispatch, so existing prompts keep working.
2. **Ranking lost to verbose descriptions.** Cosine similarity over
   `name + description` put `query-docs` above `web_search` for "search the web
   for the latest news" (4/15 correct top-1 on a labelled set). Two server-side
   changes — an intent-alias table (`search/web/news/latest/cerca/notizie → web_search`,
   12 domains) worth 0.08, plus a light lexical top-up (0.05) — bring it to
   **13/15 top-1**. This costs **zero prompt tokens**: it is ranking, not prompt.
   Tunables: `MCP_LITE_ALIAS_WEIGHT`, `MCP_LITE_LEXICAL_WEIGHT`.
3. **The synthesis turn was starved.** A tool turn is two jobs: emit the call
   (budget 32 is measured 7/7) and write the answer from the tool result. At
   budget 32 the second one returns `finish=stop` with an **empty** `content` in
   6 s — the thought eats the budget and there is nothing left to say, which is
   exactly what a client shows as a hang. The router now sends
   `ROUTER_BUDGET_TOOLS_SYNTH=96` (the chat budget) when the last message is a
   tool result.
4. **A tool result is re-sent on every following turn.** 12 000 characters of web
   markdown both overflow the 12k window (empty prompt) and cost seconds of
   prefill per round. `ROUTER_TOOL_RESULT_MAX_CHARS=6000` trims head+tail with an
   explicit marker, so the model knows text is missing.

**The iteration cap is deliberately left at -1 (unlimited)** in Open WebUI.
Capping it at 8 does terminate the bad loop, but it also kills the long agentic
tasks this stack exists for: Open WebUI then answers "Tool-call limit reached (8
iterations)". The loop has to end because the tools work, not because the client
gives up. One chat row did measure 939 s with `content: ""` before these fixes —
that row was the model re-searching, not the cap.

`scripts/toolcheck.py` has an eighth case, **`synth`**, that replays a
question + tool call + tool result and requires a non-empty answer with no
further call: that is the regression test for faults 3 and 4.

### The other half: the board is shared, and an abandoned request owns it

`llama-swap` answers **429 `concurrency_limit`** instead of queueing, and
`globalConcurrencyLimit` is 4. Behind that, each model runs with `--parallel 1`:
one slot.

The dangerous case is not load, it is an **abandoned request**. If the client goes
away (stop button, its own timeout) or nobody reads the stream, `llama-server`
keeps decoding to `max_tokens` and holds its only slot. One 4096-token request at
4.4 tok/s blocks the reasoner for ~15 minutes, and every other caller — Open
WebUI included — gets an instant 429 that it renders as a spinner.

Measured during the investigation: 20 429s against 19 chat requests in ten
minutes. Most of those 429s were **LiteLLM retrying 3×** for a single failed
call, not a client polling. The other client was Home Assistant's Assist agent
`conversation.soggiorno_llama_server_pc_openrouter` (the `llama_cpp`
integration, `base_url` = LiteLLM, the same key Open WebUI uses) — a legitimate
conversation agent that simply could not get a slot.

Practical consequences, none of them inside the router:

* a background or voice client sharing `cheap/auto` competes with chat for the
  same slot: give it its own LiteLLM key with an `rpm/tpm` limit, or a different
  model id, if it must not starve the interactive path;
* keep `max_tokens` small on anything automated (the router floors tool turns at
  800, but it never *lowers* a client's 4096);
* `scripts/toolcheck.py` retries a 429 for up to 6×10 s, because a busy board
  should not read as a tool-calling regression.

Reasoning shows up as `reasoning_content`; Open WebUI renders it as the thinking
block. At budget 32 it is short by design.

## Operations

```bash
sudo ./install.sh                 # build + models + units  (--skip-build, --skip-models, --dry-run)
sudo ./install.sh --tune          # same, plus OS tuning
sudo ./install.sh --prune-models  # delete GGUFs that are not part of the stack
sudo ./scripts/tune.sh --check    # show unapplied OS tuning
sudo ./scripts/tune.sh --apply    # swappiness, noatime, governor, zram checks
./healthcheck.sh                  # quick liveness of both tiers
./healthcheck.sh --tools          # + the 8-case tool-calling battery
./scripts/toolcheck.py            # tool calling on its own (--quick, --stream, --model)
./benchmark.sh --all              # pp/tg per model, embedding latency
sudo ./update-llama.cpp.sh        # rebuild llama.cpp at the tip of master (the timer runs it weekly)
sudo LLAMA_CPP_SOURCE=release ./update-llama.cpp.sh   # newest release tag instead
sudo LLAMA_CPP_TAG=v1.0.0 ./update-llama.cpp.sh       # one exact ref, bypasses the comparison
```

Model downloads are resumable (`curl -C -`); re-running `install.sh` only fetches
what is missing. llama.cpp is rebuilt every Sunday at 03:00 by
`llama-swap-update.timer`, which tracks **the tip of master** and does nothing when
it has not moved.

**Why master and not releases.** The newest llama.cpp release is v1.0.0
(2026-11-13), the build this board ships, so a release tracker would compile
nothing for months and the gate below would never run. The regressions that matter
here — chat-template handlers, tool parsers, ARM kernels — land on master and are
fixed there in days. Weekly master behind a gate that rolls back has the risk
profile of a release and the content of a year, and a broken tip is *skipped*, not
installed (see step 6).

**The weekly rebuild is why `healthcheck.sh --tools` exists.** Tool calling depends
on the model's chat template being recognised by a native handler in
`common/chat.cpp`; when that regresses, LFM2.5 falls back to the generic parser and
starts emitting the call as plain text instead of `tool_calls`. Nothing else in the
stack notices — `/health` is fine, chat is fine — so the battery would go 7/7 → 3/7
and nobody would know.

So the update script gates the new build on the battery instead of trusting it:

1. before compiling, `bin/` (62 MB) and the version id are copied to
   `.build-backup/`, replacing the previous backup;
2. the build runs through `install.sh` (same flags as a first install). On master
   the tip sha is pinned, so the binary that is verified is the commit that was
   read from `ls-remote`, even if master moves during the 40 min compile;
3. llama-swap is restarted and must answer `/v1/models` within 2 min; the workhorse
   is warmed with a 1-token request so the battery measures the build, not a cold
   model load (the router's warm loop does the same for the whole resident set);
4. `toolcheck.py --only tg,web,mem,wx,multi,none` runs **through the router** (a
   direct call to `:10001` would skip the reasoning budget and the tool-turn
   temperature floor, i.e. test a configuration no client uses);
5. on failure the battery is retried once, then `.build-backup/bin` is restored,
   llama-swap is restarted, the battery is run again to prove the rollback, and the
   unit exits non-zero so `systemctl --failed` shows it;
6. the sha that failed (compile failure, no start, or a failed battery) goes to
   `.llama-cpp-broken`. The next run compares it against the tip: same sha means
   the build is skipped with a warning and exit 0, so a broken master does not
   cost 40 min of compile every Sunday. It clears by itself when master moves, and
   a verified build clears it.

A bad build therefore costs one night's tools, not a week's. `.build-backup/` is
also the manual rollback:

```bash
sudo rm -rf /opt/cheap-llm-stack/bin && sudo cp -a /opt/cheap-llm-stack/.build-backup/bin /opt/cheap-llm-stack/bin
sudo cp /opt/cheap-llm-stack/.build-backup/version /opt/cheap-llm-stack/.llama-cpp-version
sudo systemctl restart llama-swap && ./healthcheck.sh --tools
```

If the rollback also fails the battery, the script says so explicitly: the build is
not the cause (look at the model files, the config, or the router).

## Layout

```
/opt/cheap-llm-stack
├── bin/            llama-server, llama-bench, llama-swap
├── models/         GGUF weights and the mmproj projector (4 models, nothing else)
├── llama.cpp/      pinned source tree used for the build
├── router/         llm_router.py   (the orchestrator)
├── scripts/        tune.sh, toolcheck.py
├── config.yaml     llama-swap config
└── systemd units   installed to /etc/systemd/system
```
