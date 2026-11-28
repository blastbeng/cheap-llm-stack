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
              ├── reasoner              Moon-Beam-8B-A1B MoE IQ4_XS 16k  text + TOOLS
              ├── router                LFM3-1.7B     QAD-Q4_0   8k   CORE, chitchat / one-liners
              └── vision                LFM3-VL-2B    QAD-Q4_0   6k   heavy, images / OCR
```

Four ids, one job each. Anything that needs to think, write or call a tool goes to
the 8B MoE; greetings, one-liners and short translations go to the 1.7B; anything
with a pixel in it goes to the VLM; embeddings are always on.

## Why this shape

* **Two A76 cores are the whole budget.** Generation is bound by memory
  bandwidth, so bigger dense models are strictly slower. Measured on this exact
  board with `llama-bench` (q8_0 KV, flash-attn on, `--threads 2`):

  | model | file | tg (tok/s) | pp (tok/s) | note |
  | --- | --- | --- | --- | --- |
  | Qwen3-0.6B Q8_0 | 0.62 GB | **16.2** | 92 (`-t 2`) / 125 (`-t 8`) | retired: replaced by LFM3-1.7B |
  | **Moon-Beam-8B-A1B (MoE) IQ4_XS** | 5.00 GB | **8.4** | 20.3 (`-t 2`) / 30.8 (`-t 8`) | shipped, default + tools |
  | LFM2.5-8B-A1B (MoE) Q4_0 | 4.85 GB | 10.4 | 23.2 (`-t 2`) / 37.3 (`-t 8`) | previous default, 8/9 tools |
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
  | LFM2.5-8B-A1B, `reasoning_budget_tokens=32` | 7/7 | 17.9 s | short battery prompts only — starves the real Open WebUI one, see below |
  | **LFM2.5-8B-A1B, budget 96** | **7/7** | 24.1 s | shipped for tool turns: same score, +35 % here, but the only reliable value once the prompt carries a system message |
  | LFM2.5-8B-A1B, budget 0 | 5/7 | 52 s | thinking leaks into `content` → unusable |
  | Qwen3-0.6B Q8 (retired) | 7/7 | 14–21 s | mangles arguments, poor prose |
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
| LFM2.5-8B-A1B | Q4_0 | 4.50 GiB | 22.98 / 10.37 | 8/8 | the default until the 9-case battery below: same flaws, twice the latency per call |
| **LFM3-1.7B** | QAD-Q4_0 | 0.99 GiB | **41.44 / 10.13** | 7/8 | **shipped as the fast tier**: 1.8× prefill, same decode, 1/4.5 the RAM, 256K ctx |
| LFM3-1.7B | Q8_0 | 1.85 GiB | 29.77 / 6.11 | 8/8 | 8/8 tools but 40 % less decode — Q4_0 wins |
| **Moon-Beam-8B-A1B (Ling-3.0-tiny + engram)** | IQ4_XS | 4.49 GiB | 20.29 / 8.39 | **9/9** | **shipped**: fewer tokens of thought per call make it the fastest 8B in a loop (7.6-11.3 s vs 16.6-18.3 s) |
| edge0-8b (Ling-3.0-tiny) | Q4_1 | 4.65 GiB | 12.90 / 8.56 | 8/8 | slowest prefill of the 8B class, same leak |
| LFM2.5-8B-A1B-Uncensored | IQ4_XS | 4.29 GiB | 15.14 / 9.65 | 8/8 | works, but −34 % prefill and it answers Italian prompts in English |
| LFM2.5-8B-A1B-Hermes-Abliterated | IQ3_XXS | 3.33 GB | bench stalls | — | unusable on this build |
| LFM2.5-Q3.8-TBrilliance-NEO-MAX | IQ4_XS | 4.55 GiB | 15.04 / **2.53** | — | 4× slower decode, misattributes the Commedia → reject |
| EuroMoE-2.6B-A0.6B | Q4_K_M | 1.62 GB | — | — | `ggml-impl.h` fatal error: unsupported quant/arch |

Tool numbers are not the whole story: all of these models are the same generation
and share the same flaws — both MoEs leak a little thinking into `content` when
the reasoning budget is smaller than the thought (measured: at budget 96 the
Moon-Beam prompt "3 corridoi × 7 scaffali × 12 scatole − 47" answers **173**, at
256 it answers **205**; the router already gives the answering turn 256). What
separated them was not the bench but the battery: on the 9-case battery the
Moon-Beam MoE scores 9/9 and per-call latency is set by how much the model
thinks, not by tok/s — so it replaced the workhorse despite decoding 20 % slower
and costing 57 MB more KV at double the context.

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

* **A thinking tier cannot be the default, and that is a measurement.** A 4B dense
  model decodes at 3.0 tok/s here, a 2B at ~6: with `--reasoning-budget 256` the 4B
  spent **170 s** on one definition question, and the 2B with a 96-token budget
  still ran to 500 tokens = **83 s**. Nothing interactive fits in that, and every
  use of the 4B evicted the MoE, which then took **121 s** to come back from
  microSD. The MoE answers the same questions correctly in ~11 s. The one exception
  is the `deep` escalation (long *and* reasoning, measured at 265–431 s, first
  turn or mid-chat): see the quality-tier section.

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

### A/B/C/D pass, 27 Nov 2026: quant, grammar, context

Everything here was measured with the board otherwise idle — one model at a time,
`llama-swap` and the router stopped — because a bench that competes with a resident
4.8 GB model measures swap, not the model. `llama-bench -fa on -ctk q8_0 -ctv q8_0
-t 2,8 -p 512 -n 128 -r 2`, production serving flags (`--threads 2 --threads-batch 8
--parallel 1`, reasoning on, budget 32), `scripts/toolcheck.py` 9-case battery at
seed 1234 / temperature 0.6.

| # | change | measured | verdict |
| --- | --- | --- | --- |
| A | LFM2.5-8B-A1B **Q5_K_M** (6.03 GB) instead of Q4_0 | pp 10.56/16.57, tg 6.74/6.83, 8/9, load 120 s, 33-37 s/call | **reject**: same score, half the speed, +1.1 GB |
| B | `grammar` (GBNF) to constrain the tool call | `400 Cannot use custom grammar constraints with tools` | **not available** — and not needed, the template already constrains it |
| B2 | `json_schema` + `tools` | accepted, but the call arrives as JSON in `content` (no `tool_calls`) | **reject**: it bypasses the channel a loop needs |
| C | **Moon-Beam-8B-A1B IQ4_XS** | pp 20.29/30.79, tg 8.39/7.78, **9/9**, 7.6-11.3 s/call, load 91 s | **shipped** as the reasoner |
| D | LFM3-1.7B QAD-Q4_0 | pp 41.46/62.53, tg 10.19/9.92, 7/9, load **22 s** | confirms the fast tier (5.9-12.6 s/call) |
| ref | Nemotron-3.5-Bolt-E4B Q4_K_M | pp 8.50/14.18, tg 3.06/2.94, 9/9, 19-27 s/call, 98-116 s synthesis | confirms the deep tier: correct, 3x the wall clock |

The `grammar` refusal is the interesting one: with `--jinja` on, llama.cpp installs a
grammar for the tool call itself, so a second one is rejected outright. That is also
why every battery failure is a *wrong choice* and never malformed JSON — the call
channel is already constrained, so B was dropped rather than worked around.

### Re-evaluation, 28 Nov 2026: Qwen3.5, threads, OpenBLAS, depth

The question was whether a model released after the ones shipped (the Qwen3.5
family: hybrid Gated DeltaNet + sparse MoE, 262k native context) changes any seat,
and whether the serving knobs left any speed on the table. Same harness as above
(`llama-bench -fa on -ctk q8_0 -ctv q8_0 -d 2048 -p 512 -n 128 -r 2 -t 2,8`), but
**at depth 2048** — a real turn is never a 512-token prompt, and depth is exactly
where the architectures separate: a model with attention in every layer pays for
every token in the KV, a hybrid one does not.

| model | file | pp512 @ d2048 t2 / t8 | tg128 @ d2048 t2 / t8 | note |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B Q4_0 | 1.12 GiB | **24.2 / 38.5** | 6.81 / 6.71 | hybrid; 8/9 battery; the best long-context engine measured here |
| Qwen3.5-4B Q4_0 | 2.40 GiB | 7.2 / 12.9 | 2.88 / 3.01 | reject: 1k-token prefill = 78 s |
| Qwen3.5-9B Q4_0 | 5.00 GiB | 5.0 / 8.7 | 1.73 / 1.78 | reject: 1.7 tok/s |
| **Moon-Beam-8B-A1B IQ4_XS** (shipped) | 4.49 GiB | 11.0 / 17.4 | 5.98 / 6.08 | **9/9** again today — still no rival at 9/9 |
| LFM2.5-8B-A1B Q4_0 | 4.50 GiB | 14.4 / 22.9 | 8.24 / 7.89 | 6/9 at temperature 0.2 (8/9 at 0.6, the Nov-27 run) |
| **LFM3-1.7B QAD-Q4_0** (shipped) | 0.99 GiB | 24.5 / 38.3 | **8.53 / 8.26** | 8/9, 12-22 s per call |
| Nemotron-3.5-Bolt E4B Q4_K_M (shipped deep) | 5.74 GiB | 6.8 / 11.3 | 2.53 / 2.47 | 9/9 at 3x the wall clock |

Two things came out of this. First, **the MoE is still the right shape**: the 1.7B
dense decodes faster than the 8B MoE (8.5 vs 6.0) and every dense model above 2B
collapses — 3.0 tok/s at 4B, 1.8 at 9B — so "bigger but dense" is a downgrade on
this board, exactly as the Nov-27 pass said. Second, **depth hurts the incumbents
far more than the new hybrid**: at 8k depth LFM3 falls to 5.95 tok/s and Moon-Beam
to 3.51, while Qwen3.5-2B still does 5.37 and prefills 12.6 tok/s against
Moon-Beam's 4.64. The hybrid keeps KV in only a few layers (gated shortconv
elsewhere) — the trick that made 16k affordable on the MoE, taken further.

Depth ladder (r=1, `-d 0,2048,8192`), t2 pp / tg. This is the answer to "how much
context can we afford", because allocating context is free and *filling* it is not:

| model | d0 | d2048 | d8192 |
| --- | --- | --- | --- |
| Moon-Beam | 18.1 / 7.12 | 11.0 / 6.19 | 4.64 / 3.51 |
| LFM3-1.7B | 41.0 / 10.11 | 24.3 / 8.56 | 10.8 / 5.95 |
| Qwen3.5-2B | 33.8 / 7.41 | 24.0 / 6.81 | **12.6 / 5.37** |

Nothing here argues for moving the windows: **16384 for the workhorse and 8192 for
the fast tier stay**, because the wall clock of a full window is set by prefill and
the incumbent's prefill at depth is what it always was. What the ladder does say is
that a 16k turn on Moon-Beam is a ~10-minute prefill — which is why tool results are
capped at 6000 chars upstream, and why Qwen3.5-2B is the model to reach for if a
client ever genuinely needs 32k.

The battery (9 cases, seed 1234, production flags, 16k ctx) decided the seats. Two
budgets, because the router sends 96 for a call turn and 256 for the answer:

| model | budget 32 | budget 96 | the failing case |
| --- | --- | --- | --- |
| **Moon-Beam** | **9/9** | — | — |
| Qwen3.5-2B | 8/9 | 8/9 | `none`: answers "ciao, come stai?" with a `telegram_send_message` call |
| LFM3-1.7B | 8/9 | 8/9 | `synth` at 32, `synthweb` at 96 — re-calls a tool instead of answering |
| LFM2.5-8B-A1B | 6/9 | — | `tg` produced no call at all, plus `none` and `synthweb` |

So **no seat moved.** A model that calls Telegram on "ciao" cannot hold the
`assistant` route, and the fast tier is already faster per call with the same
score. Qwen3.5-2B/Q4_0 stays on the board as the documented long-context
alternative (`models/Qwen3.5-2B-Q4_0.gguf`, deliberately not wired into
`config.yaml`); the 4B and 9B are kept only as re-bench material — a plain quant,
not the `UD-*` dynamic ones, because the upcast layers cost CPU speed.

The knobs, measured on Moon-Beam at d2048 unless said otherwise:

| lever | measured | verdict |
| --- | --- | --- |
| `-t 2,4` instead of `-t 2,8` | pp 10.96 → 13.09 (+19 %), tg 6.06 → 5.16 (−15 %) | reject: a turn is decode-dominated |
| `-C 0xC0 --cpu-strict 1` (A76 only) | t2 identical (10.94 / 5.98), t8 collapses to 0.49 tok/s | reject: production `-t 2` already lands on the A76s |
| `-ub 1024` | pp 17.43, tg 6.13 (+0.4 % / +0.8 %) | reject: noise, keep the default 512 |
| `--spec-type ngram-simple` (n-gram drafting, 220-token answer, `-r` 218-iteration repeats) | a repetitive JSON prompt 5.66 → 6.11 tok/s (+8 %, 102/194 drafts accepted); an ordinary Italian prose prompt **7.32 → 6.48 (−11 %, 1/48 accepted)**; `ngram-mod` is worse than baseline on the JSON prompt too (5.01, 6/192) | reject: it pays only on boilerplate, and the flag is per *server*, so the same `reasoner` that drafts tool JSON also writes the answer. The 53 % acceptance on JSON is exactly the shape the grammar already produces for free |
| **OpenBLAS** (`-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`, otherwise the same flags, isolated build) | 2B: pp 24.39/37.97, tg 6.87/6.80 (was 24.20/38.47, 6.81/6.71); Moon-Beam: pp 10.82/16.66, tg 6.06/6.15 (was 10.96/17.36, 5.98/6.08) | **reject**: a wash, and it would put a library dependency into the weekly rebuild. `ggml-blas` only calls `cblas_sgemm` for matmuls with batch ≥ 32 and dequantizes the weights to f32 first — on a bandwidth-bound board that is the one thing it cannot afford |
| **`vm.swappiness` 100 → 10** (`scripts/tune.sh --apply`, which had never been run on this board) | 2B tg 6.20 → 6.71-6.81 at d2048; `vmstat` si/so 715/492 → 0, `Swap` 4 GB → 0 | **shipped** — the only real win of the whole pass, and it was the OS, not a model: at 100 the kernel evicts the mmap'd weights to zram mid-run. Shipping it took a second fix: `armbian-config`'s `99-zz-armbian-tuning-profile.conf` sorts after a `90-*` drop-in and puts the value back to 100, so `tune.sh` now writes **`99-zzz-cheap-llm-stack.conf`** |

That last line is now a permanent guard: `healthcheck.sh` warns when
`/proc/sys/vm/swappiness` disagrees with what `tune.sh` writes, and when more than
20 % of swap is in use, because both mean model pages are being evicted.

### Context: 16k is the ceiling, and time is what sets it

Doubling the window costs **+57 MB** on this model (3.20 GB at 16k → 3.26 GB at 32k,
q8_0 KV, GQA) and leaves the cold load at 91 s for both. Filling it is the cost:

| prompt | model | prefill | wall clock for a 24-40 token answer |
| --- | --- | --- | --- |
| 5 497 tok | LFM2.5-8B-A1B Q4_0 | 21.6 tok/s (254 s) | 259 s |
| 6 485 tok | Moon-Beam-8B-A1B | ≈14 tok/s (455 s) | 455 s |
| 24 000 tok | Moon-Beam-8B-A1B | >900 s (the client gave up) | — |

A long-context request is minutes of prefill whatever the window, so the lever is
keeping the prompt small — the flat surface caps a tool result at 5×500 chars for
exactly this reason — not paying RAM for a 32k window nothing can fill in reasonable
time. 16384 stays.

### Quality tier, measured 27 Nov 2026: NVIDIA-Nemotron-3.5-Bolt-E4B-4GB

The one candidate that is a different *kind* of model rather than another 8B-A1B
merge: `gemma4` architecture (this build knows it — `gemma4` is in `libllama`),
7.52 B params, 42 blocks, 64k trained context, Q4_K_M built from an imatrix,
5.74 GiB, plus a 0.99 GiB vision mmproj if wanted. The "E4B" in the name is about
*embedding* parameters (per-layer input embeddings, 2560 d + 256 per layer), not
sparse compute: per token it is dense.

| measurement | Nemotron-3.5-Bolt-E4B | LFM2.5-8B-A1B (shipped) |
| --- | --- | --- |
| file / quant | 5.74 GiB Q4_K_M | 4.50 GiB Q4_0 (repacked) |
| llama-bench t2 pp512 / tg32 | 8.07 / **2.91** | 22.98 / **10.37** |
| llama-bench t8 pp512 / tg32 | 13.15 / 2.70 | 35.0 / 9.8 |
| cold load | 105 s | ~100 s |
| tool call (3-tool surface, think off) | correct call, 21.7 s for 19 tok | 7/7 battery at ~18 s |
| `scripts/toolcheck.py` subset (tg/web/none/synth) | **4/4** (20.8 s web, 28.5 s none, 60 s synth) | 9/9 |
| synthesis from a `web_search` result | calls `web_url_read` instead of answering (26.4 s) | same reflex — the loop guard is what ends it for both |
| the same turn with `tools` withdrawn | correct 167-token answer, 73.3 s | correct, 30-50 s |
| thinking on, `reasoning_budget_tokens: 32` | honoured, 64.6 s for 140 tok | 96 / 256 depending on route |
| "chi è Gerry Scotti?" with no tool result | **hallucinates** (journalist and writer) | also wrong on facts it does not have |

Verdict: **opt-in weights, automatic routing.** It is not better at the thing that
was broken (it takes the same "read one more page" step), it is 3.5× slower, and at
5.74 GiB its weights cannot coexist with the MoE in 12 GB — loading it evicts the
workhorse and the next MoE request pays the cold load again. So:

* it lives in a `text` swap group **with** the MoE (`swap: true`, not persistent;
  `core` keeps only embedding + fast tier so `bolt` cannot evict those),
  `ttl: 900` so it unloads itself;
* it is reachable deliberately as well: `model: bolt` (aliases `quality`, `deep`,
  `nemotron`, so `cheap/bolt` in LiteLLM works through the wildcard);
* **`auto` escalates to it.** `ROUTER_DEEP_TIER=1` (set in the unit) turns the
  `deep` route on. A request is deep when it is long (now **>400 chars or ≥ 25
  words**, `ROUTER_DEEP_MIN_CHARS` / `ROUTER_DEEP_MIN_WORDS`) **and** uses reasoning
  vocabulary, judged on **the last user turn** — not on the whole concatenation,
  because Open WebUI ships a ~1.5 kB system prompt describing the tool surface on
  every request: it is longer than any threshold and always mentions HTTP, docker
  and tokens, so the old gate measured the client's prompt, not the question. The
  session is then pinned to `bolt` (`sticky-deep`), so the tool results coming back
  are read by the model that asked for them and a follow-up does not pay the ~100 s
  reload. The route also sends a 128-token thinking budget — a 256-token thought
  at 2.91 tok/s is 88 s before the first word.

  **600/40 was measured too high: 14 days of logs, zero `deep`, and `bolt` never
  loaded from an Open WebUI chat.** The gate only ever fired on essays; a real design
  question is ~30 words. Two changes came with the new numbers:

  * the gate reads the last user turn (above), and
  * **mid-chat escalation** (`ROUTER_DEEP_ESCALATE=1`): a chat that starts ordinary
    is pinned to `tools` on turn 1, and until now only turn 1 could reach the quality
    tier, so a deep question on turn 4 was answered by the MoE. Now it can escalate
    once per session (`deep-escalation`) — never while a tool result is in flight
    and never on an image turn, because those belong to the model that started them.

  Measured end to end through the router (27–28 Nov):

  | case | route (why) | wall clock |
  | --- | --- | --- |
  | long M/M/1 design question + the 7-tool MCP surface | `deep` (`deep-first-turn`), 800 tok | **411.8 s** |
  | the same question with no tools | `deep` (`reasoning-deep`), 400 tok | **265.1 s** |
  | tool result inside that session | `deep` (`sticky-deep`), 800 tok | **290.5 s** |
  | 30-word design question + tools, OWUI-shaped system prompt | `deep` (`deep-first-turn`), follow-up `sticky-deep` | 54.3 s warm, 148 chunks |
  | **escalation:** chitchat turn, then a 34-word design question, same session | `tools` → `deep` (`deep-escalation`) | 107.5 s + **431.2 s** |
  | short technical question + tools + the long system prompt | `tools` → `reasoner` (the gate reads the question, not the client's prompt) | 124 s (800 tok) |
  | new short question, new session | `tools` → `reasoner`, 111 tok | 109.8 s (the reload) |
  | `scripts/toolcheck.py`, full battery, `model: auto` | 9/9 on 27 Nov, 8/9 on 28 Nov, every case `tools`/`reasoner` | 12–120 s/case |

  The deep rows are the price: one deep answer is 4–7 minutes, and getting back to
  the MoE costs a swap when the session ends. Ordinary tool traffic is untouched —
  the battery still routes every case to `tools`/`reasoner` (8/9 on 28 Nov: the
  `synthweb` case now turns the synthesis turn into another `web_search` at the
  default seed, reproducible, and its routing is unchanged), and a short tool
  question is still ~24 s.

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
  | `--reasoning-budget` | **32** | server default only: the router overrides it per route (96 call turn, 256 synthesis, 128 on the `deep` tier — 256 there is 88 s of thinking before the first word) |
| load mode | `mmap+mlock` core, `mmap` heavy | core never pages out; the 5.0 GB MoE stays reclaimable |

### RAM, measured (this is what forced the sizes above)

`VmRSS` + `VmSwap` per llama-server with `--parallel 1`:

| model | weights (file) | `Anonymous` (KV + buffers) | ctx |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | 0.62 GB | 0.70 GB | 4096 |
| LFM3-1.7B | 1.05 GB | ~1.0 GB | 8192 |
| Moon-Beam-8B-A1B | 5.00 GB (mmap) | **2.44 GB** (Rss 6.97 GB with the mapped weights) | 16384 |
| LFM3-VL-2B + mmproj | 1.05 + 0.20 GB | 1.28 GB | 6144 |

`Anonymous` from `smaps_rollup` with all four loaded: 7.6 GB, `free` reports
9.5 GB used and 2.4 GB available, no OOM, 64 °C. (That set still counted the 0.64 GB
0.6B, retired since: removing it buys that disk back, not RAM — it was never
resident.) The ctx bump costs 59 KB/token on that model — measured 905 MB at 4096 vs
1025 MB at 6144, exactly the q8_0 KV size, so it is the only part that scales. Both
layers that cap a prompt agree on the new number: with a prompt above the window,
the proxy's clamp reports
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
| `core` | no, `persistent` | embedding-model, router | always loaded, never evicted (~1.7 GB anon) |
| `text` | yes | reasoner, bolt | one at a time: the MoE (4.8 GB) and the quality tier (5.7 GiB) do not fit together |
| `heavy` | yes | vision | loads on demand, `ttl: 900` |

The MoE sits in `text` rather than in `core` because `persistent` groups can never
be evicted: with it in `core`, a `bolt` request could not load at all (6.6 + 5.7
GB against 12). `ttl: 0` still keeps it loaded; only a `bolt` request unloads it —
explicit, or the `deep` route escalating a long reasoning turn — and the next MoE
request reloads it (~100 s cold, seconds while the page cache is warm). Measured
after a `bolt` turn: `reasoner ready`,
`embedding-model ready`, `router ready`, 3.7 GB available.

The VLM is *not* resident: with embedding + fast tier + VLM + MoE loaded the board
once had 226 MB available and the kernel swapped the whole `router` tier (896 MB) and
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
| `GET /health` | router + upstream status, per-route map and budgets, plus the live `tools_synth_budget` / `tool_result_max_chars` |
| `GET /stats` | per-route counters, sticky sessions, prototype state |

Response headers `X-Route`, `X-Route-Model`, `X-Route-By`, `X-Route-Ms`,
`X-Route-Budget` show what was chosen and why (`client`, `sticky`, **`tools`**,
`image`, `image+ocr-hint`, **`deep-first-turn`**, **`deep-escalation`**,
  **`reasoning-deep`**,
**`sticky-deep`**, `chitchat`, `code`, `reasoning`, `trivial`,
`translate-short`, `semantic(0.71/0.09)`, `semantic-fast-rejected`, `default`).

Routes: `fast` → router, `assistant`/`reason`/**`tools`** → reasoner,
`vision`/`vision-docs` → vision, **`deep` → bolt**, `embedding` → embedding-model.

### Tool-calling routing

Tool turns are decided **before** everything else, including stickiness:

* `tools` (or legacy `functions`) in the request → route `tools`;
* a `role: "tool"` message, an assistant `tool_calls` entry, or a tool-result
  content part in the history → route `tools` (we are mid-loop and must keep
  hitting the model that issued the call);
* the session's sticky entry is re-pinned to `tools`, so a conversation that
  started as `ciao` cannot stay on the fast tier once a tool appears.

Two rules sit in front of that branch, and both exist because Open WebUI attaches
the flat tool surface to *every* request, so "has tools" stopped meaning "is an
agent turn":

* **images win over tools.** LFM3-VL handles tools (6/7 on the battery) and neither
  text model can see the picture, so a turn carrying an image goes to
  `vision`/`vision-docs` even with tools in the body. Before this rule the router
  sent every image to the MoE, blind: measured, `image+tools` now answers the text
  in the picture in 43.1 s on `by=image`.
* **a deep question wins over tools, on the first turn and once mid-chat.** Long
  (>400 chars or ≥ 25 words, judged on the last user turn) *and* reasoning
  vocabulary *and* no tool history → `deep` on `bolt`, and the session is pinned
  there (`sticky-deep`) so the tool results that come back are read by the model that
  asked for them. Mid-chat escalation (`ROUTER_DEEP_ESCALATE=1`) fires at most once
  per session, and never while a tool result is in flight. See the quality-tier
  section.

An explicit model id is normally forwarded verbatim — that is how you debug one
tier. The one exception is the fast tier: a request addressed to `router`, `fast`,
`tiny` or `lfm3` that carries `tools` (or a tool history) is answered by the `tools`
route instead, and reports `X-Route-By: tools-on-tiny`.
Tool orchestration lives on the MoE: LFM3-1.7B does emit calls (7/9 at Q4_0, 8/8 at
Q8_0) but the MoE is 9/9 on the current 9-case battery and its per-call latency is
the same (7.6-11.3 s against 5.9-12.6 s).
`router` as a *routing* id (`auto`, `router`, `orchestrator`) never reaches this
check — `tools` wins before the model is picked.

The tool payload is passed through verbatim (`tools`, `tool_choice`,
`parallel_tool_calls`), no system prompt is injected (it would shift the cached
prefix), the answer ceiling is raised to at least 800 tokens so a client's
`max_tokens: 200` cannot truncate the call instead of the prose, and **temperature
is raised to 0.6** (see below). The thinking budget is **96** when the model has
to *emit* a call and **256** when the last message is a tool result and it has to
*write the answer* (`ROUTER_BUDGET_TOOLS_SYNTH`); the end-of-turn markers are
appended to `stop` so the GGUF's literal `</|end_of_sentence|>` cannot leak into
the answer. A client that tries to switch thinking *off*
(`chat_template_kwargs.enable_thinking: false`, `reasoning_effort: "none"`) is
overruled, because on a reasoning-only model that is what pushes the monologue
into `content`.

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

Three exist because the (then) 0.6B failed on them in testing: a request that *writes*
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
  tiny/VLM tiers (see Measured settings), because once the surface is down to
  10 tools the *fixed* part of the prompt is 1.8k (0.7k with the two meta-tools
  alone) and the cache makes later turns nearly free — the extra room then goes
  to the conversation, which is the point.

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
3. Attach **`mcp-tools-lite`** (`server:mcp:mcp-tools-lite`): 10 tools, 7,055
   characters of schema / **1,831 prompt tokens** — eight flat everyday tools
   (`web_search`, `web_url_read`, `search_and_read`, `send_message`,
   `read_image`, `chroma_query_documents`, `terminal_run`, `terminal_file`)
   plus the two meta-tools (`find_tool`, `run_tool`) that keep the whole
   109-tool gateway reachable. In Open WebUI a global `inlet` filter injects that id into
   every chat, so `TOOL_IDS` there must equal the connection's `info.id`. That
   filter is also where the orchestration is consolidated: it *strips* every
   tool id for `cheap/router` (the tiny tier never gets one, not even when
   picked by hand). The image+tools turn no longer needs opting in: the router
   sends any turn carrying an image to the vision tier, which handles tools
   itself. See the next section for why a MoE needs the flat five and a large
   model does not.
4. Leave `chat.tool_search` and `chat.context_compaction` off — both add a pass
   over the prompt, and compaction cannot fit a 6k model anyway.

### One surface, flat tools next to the meta-tools (`mcp-tools-lite`)

The meta-tool surface is a good trade for a large model: 678 prompt tokens buy
109 tools. It is a **bad** trade for the 8B-A1B MoE, because it turns one job
into two decisions and the second one keeps failing:

* `find_tool` → schema, then `run_tool` with the arguments: the model has to
  carry an opaque payload across a turn. Measured on the same 6-case web
  battery: **meta-tools 0/6, one flat tool 6/6** — the failures were not bad
  arguments, they were `find_tool` in round 1, `find_tool` again in round 2,
  `finish=stop` with nothing.
* `find_tool("Gerry Scotti")` ranks `ask` (the model council) above
  `web_search` and answers **14,027 characters** at k=3 (2,967 at k=1): the
  lookup that was supposed to save tokens is the largest prompt in the chat.
  The full `find_tool → run_tool → answer` chain is **1/3 usable** (LOOP, LOOP,
  empty answer), even with `k=1`.
* `run_tool` also hides the real schema, so the arguments arrive as prose
  (`{"name": "web_search", "arguments": "query=..."}`) instead of a parsed
  call.

So the same proxy advertises the everyday tools **flat, next to** the
meta-tools: one connection, one surface, nothing else to keep in sync.
`tools/list` answers 10 tools / 7,055 characters of schema / **1,831 prompt
tokens** (`cl100k_base`), against 678 for the two meta-tools alone and 71 for
none.

| flat tool | why it is in the set | schema |
|---|---|---|
| `web_search` | the original failure; curated `query` + `count` | 828 chars |
| `web_url_read` | the second step of a search (`url` + `maxLength`) | 528 |
| `search_and_read` | the proxy's own: search **and** read the top hit in one call, one hop instead of two | 790 |
| `send_message` | Telegram notify; `text` only, the chat is fixed | 331 |
| `read_image` | one path in, ImageContent out | 321 |
| `chroma_query_documents` | "what did we decide before?"; `query` only | 412 |
| `terminal_run` | a shell on the media host (Open Terminal); `command` + optional `process_id` to follow a long job | 876 |
| `terminal_file` | read/write one text file through the same API | 733 |

`web_search` and the other hand-written specs are in `proxy.py`; anything named
in `MCP_LITE_FLAT_TOOLS` without one gets a compact schema generated from the
catalogue (`_compact_spec`: name, description trimmed to 200 chars, parameters
with type/enum/default). Three details are not cosmetic:

* **The upstream schema invites invented arguments.** `send_message` has a
  `chatId` override and the MoE filled it with `"blast"` in 3/3 runs, which
  would override the configured chat; the flat schema does not offer the field.
  Same for `freshness`/`site` on `web_search` — invented as `freshness:"day"`,
  `site:"italiano"`, which then filters the answer away.
* **`chroma_query_documents` needs a collection name, and a model cannot guess
  one** (0/3 on "cerca nella memoria del progetto" with the generated schema).
  The flat schema takes `query` and the proxy fills in
  `MCP_LITE_MEMORY_COLLECTION` (`aiderdesk_cheap-llm-stack_memory`).
* **Results are rendered, not dumped.** `web_search` answers
  `title / url / snippet` per result plus `(N more results not shown)`;
  `chroma_query_documents` answers `- note` lines. Raw JSON (lanes, timings,
  ids, distances, `null` arrays) is what the synthesis turn copies instead of
  the content, and it is re-sent on every following turn (see fault 4).

**`terminal_run` / `terminal_file` (28 Nov).** In Open WebUI the terminal is not
a tool server: attaching it to a model costs its whole 22-tool surface (17.6 kB
of schema, **4,619 tokens**) on every turn. The same Open Terminal REST API as
two flat tools costs **1,609 chars / 422 tokens**, and no new process:
`mcp-tools-lite` joins the external network `open-terminal-net`, holds the
bearer key and forwards; the commands run inside the `open-terminal` container
itself (uid 1000, cwd `/home/user`, no docker socket), i.e. the same sandbox the
panel already uses. Measured on `cheap/auto` through the router, "quanti file e
directory ci sono in /opt/projects/personal-ai-stack?": hop 1 calls
`terminal_run` twice (`find … | wc -l`), hop 2 answers 1.875 file / 326
directory — **2 hops, 115 s**, 2,202 prompt tokens of which 2,096 cached. Output
is cut at 1,400 chars with an explicit "… not shown: rerun a narrower command
(grep, head, tail)" marker, just under the router's `TOOL_RESULT_MAX_CHARS`, so
the model is never handed half a log that looks whole.

Enabled per deployment, so the proxy stays generic:

```yaml
  mcp-tools-lite:
    ports: ["8814:8814"]         # one container, one index cache
    environment:
      MCP_LITE_FLAT_TOOLS: "web_search,web_url_read,search_and_read,send_message,read_image,chroma_query_documents,terminal_run,terminal_file"
      MCP_LITE_MEMORY_COLLECTION: "aiderdesk_cheap-llm-stack_memory"
      MCP_LITE_TERMINAL_URL: "http://open-terminal:8000"   # empty = kill switch
      MCP_LITE_FLAT_MAX_RESULTS: "5"
      MCP_LITE_FLAT_SNIPPET: "500"
```

With `MCP_LITE_FLAT_TOOLS` empty the surface is the old two meta-tools, so any
other client of :8814 is unaffected. The proxy logs the surface once per change
(`surface: 10 tools, 7055 chars of schema (...)`).

Measured on the production sentence ("Cerca sul web chi è Gerry Scotti e dimmi
chi è in due frasi"), streaming, seeds 5/15/25, three everyday asks:

```
web       -> web_search             {"query":"Gerry Scotti","count":3}   3/3
telegram  -> send_message           {"text":"Testi completati."}         3/3
memory    -> chroma_query_documents {"query":"router","n_results":3}     3/3
```

The call turn is 13–22 s (the first one pays the prefill) and the whole chain —
call → real `tools/call` → answer — returns an Italian answer in ~40–50 s at
the synthesis budget, 60 s for the slowest seed. Without the flat tools the same
three asks produce `find_tool`, prose, or `web_search` on the memory question:
**0/3, 0/3, 0/3**.

In Open WebUI this is one connection (`mcp-tools-lite`, same bearer) and one id
in the filter (v0.4.2) for every model, `cheap/*` included — a model never sees
two surfaces, and their sum costs twice the prompt and confuses the choice:

```python
TOOL_IDS      = ["server:mcp:mcp-tools-lite"]
DROP_TOOL_IDS = ("server:mcp:mcp-cheap-flat",)   # removed server: stale chats clean themselves
```

`cheap/router` still gets **no** tools at all. Nothing outside the connection
changes — a cloud model (`anthropic/claude-sonnet-4.5`, `openai/gpt-5`, …) keeps
exactly the tool ids it had and drives the same 10-tool surface through
`find_tool` / `run_tool` as before.

A second container (`mcp-tools-lite-flat` on :8815, one flat tool, no
meta-tools) existed for a day while this was measured; it is gone, with its
cache copy, because two containers meant two 109-tool indexes (192 MB each),
two connections to keep aligned and two surfaces that could drift.

Measured with this profile: `cheap/router` carries **50 prompt tokens** and no
tools at all; the MoE carries 1,831 on the first turn and **0.3 s on the
following turn** — with a constant surface the chat prefix stays byte-identical,
llama.cpp reuses its cache and only the new message is prefilled. That cache
hit is the real payoff of a small tool surface, more than the tokens
themselves. `cheap/auto` routes through the orchestrator. `auto`, `router` and
`orchestrator` are the magic ids that mean "route this" — `router` included, it
does not pin the fast tier; `reasoner` and `vision` are the direct ids, useful for
debugging only.

### Which id to choose in the model picker

`cheap/auto` — this is the one. It reaches the **router on :10000**, which
inspects the request and forwards to the right llama-swap model: tools, images,
code/translate/reason, long reasoning (the `deep` tier on `bolt`) and the
embedding-similarity tier are decided in Python, and the choice is pinned per
session so a conversation does not bounce. Every other `cheap/*` id in the picker
is a *direct* llama-swap model that bypasses that decision.

| picker id | what it really is | when to use |
|---|---|---|
  | `cheap/auto` | router, picks per request | always; it is the only id that can reach the tools *and* `deep` tiers |
| `cheap/reasoner` | LFM2.5-8B-A1B, pinned | when you know you want the MoE and want to skip routing |
| `cheap/bolt` | Nemotron-3.5-Bolt-E4B, pinned | when you want the quality tier by hand and want to skip routing; it supports tools, and it swaps the MoE out (105 s cold load, 2.9 tok/s, ~40 s before the first word) |
| `cheap/vision` | LFM3-VL-2B | images, OCR |
| `cheap/embedding-model` | Qwen3-Embedding-0.6B | never — it is an embedding model, not a chatbot |
| `cheap/router` | LFM3-1.7B fast tier | **hidden** (`is_active 0`): the name collides with the router's own magic id `router`, which means "route this", so picking it looked like routing and instead pinned the fast tier |

`auto`, `router` and `orchestrator` are the three ids the router itself treats as
"decide for me"; `cheap/router` through LiteLLM arrived as the literal model id
`router`, so the two meanings met in the same picker. Keeping the fast tier
reachable as `fast` / `tiny` / `lfm3` while hiding `cheap/router` removes the
ambiguity without losing the tier.

### Why "search the web" used to hang (measured 26 Nov)

Five faults stacked into one symptom — an answer that never arrives:

1. **The meta-tools were named after the mechanism, not the effect.**
   `tools_search` / `tool_call` read as "search" / "maybe call something", and the
   MoE answered exactly that: it searched in round 1 *and* round 2 and then
   stopped with nothing. Renamed to **`find_tool`** (returns schemas, runs
   nothing) and **`run_tool`** (the only way to get real data); the descriptions
   say that in those words and the `query` description asks for the *action*
   ("search the web for X"), not the user's question. Same two tools, same ~640
   prompt tokens. Old names still dispatch, so existing prompts keep working.
   This was necessary but **not sufficient**: the indirection itself is beyond
   the MoE, so the same surface now advertises five flat everyday tools next to
   the meta-tools (0/6 → **6/6**, see the "One surface, flat tools"
   section).
2. **Ranking lost to verbose descriptions.** Cosine similarity over
   `name + description` put `query-docs` above `web_search` for "search the web
   for the latest news" (4/15 correct top-1 on a labelled set). Two server-side
   changes — an intent-alias table (`search/web/news/latest/cerca/notizie → web_search`,
   12 domains) worth 0.08, plus a light lexical top-up (0.05) — bring it to
   **13/15 top-1**. This costs **zero prompt tokens**: it is ranking, not prompt.
   Tunables: `MCP_LITE_ALIAS_WEIGHT`, `MCP_LITE_LEXICAL_WEIGHT`.
3. **Both turns were starved, and the call turn worst of all.** A tool turn is
   two jobs: emit the call and write the answer from the tool result. On the
   call turn, 32 tokens looked right on the short battery prompts, but on the
   real Open WebUI request (MCP instructions as a system message + "cerca sul
   web chi è Gerry Scotti e dimmi chi è in due frasi") the budget runs out
   *while the model is planning the call* and the leftover thought lands in
   `content` — `"The user asks in Italian: Search the web who is…"`, no
   `tool_calls`, and then a confident answer from memory (and a wrong one: it
   invented a singing career). Measured on the flat surface, streaming, seeds
   15/25/47/55/65: `32 → 3/5` calls, `64 → 1/5`, **`96 → 5/5` (12–14 s)**, `128 → 5/5`.
   On the answer turn the same starvation returns an **empty** `content` in 6 s
   (`32 → 0/3`, `96 → 0/3`, `192 → 2/3`, **`256 → 3/3`**, `384 → 3/3` at 54 s),
   which is exactly what a client shows as a hang. The router therefore sends
   **96** to emit a call and `ROUTER_BUDGET_TOOLS_SYNTH=256` to write the
   answer; the asymmetry *is* the fix.
   Note on the obvious alternatives: this llama.cpp build **ignores
   `tool_choice`** (named object *and* `"required"` — a named object measured
   2/5, worse than leaving it alone) and **ignores `response_format`
   json_schema** (0/5 produced JSON), so the budget is the only working lever.
   The cost is paid on every chat too, because every `cheap/*` chat carries the
   flat tool and therefore goes through the `tools` route: on the flat surface a
   chitchat turn went 12.9 s (32) → 20.6 s (96) and "chi è Dante Alighieri" 10.3 s
   → 12.2 s. That is the price of not answering a search from memory; the tool
   call itself is 12–16 s either way.
4. **A tool result is re-sent on every following turn, and its size is what the
   model thinks about.** The raw 1.5 kB `web_search` JSON (lanes, timings,
   engines) produced an empty answer at budgets 96–128 (0/3, 1/3); two or three
   compact `title / url / snippet` lines answer at 256. `ROUTER_TOOL_RESULT_MAX_CHARS`
   is therefore **1500** (it was 6000), trimmed head+tail with an explicit
   marker so the model knows text is missing and can re-query. Above ~1500 the
   JSON also costs seconds of prefill per round, and 12 000 characters of web
   markdown overflow the window outright.
5. **The end-of-turn marker leaked as text.** The GGUF carries
   `</|end_of_sentence|>` as ordinary text pieces, so a degenerate turn *writes*
   it into the answer — that is the `…dimmi chi è</｜end▁of▁sentence>` row Open
   WebUI stored on 26 Nov. The router now appends the model's own stop markers
   (both spellings, plus `<|im_end|>`) to every reasoner request: free, and it
   never fires on a well-formed turn.

6. **The page was navigation, not an article.** `web_url_read` (mcp-searxng
   behind the gateway) converts HTML to markdown but keeps the chrome: on
   `it.wikipedia.org/wiki/Gerry_Scotti` the first 2 000 characters are
   `[Vai al contenuto] / Menu principale / Navigazione / Ricerca / Aspetto`.
   A model that asked for 2 000 characters therefore read *menus*, concluded it
   needed the other language edition, read more menus, and on the fourth turn
   spent its whole 256-token thinking budget planning a third fetch:
   `content: ""`, `usage: {"incomplete": true}`, three turns of 54.6 / 58.4 /
   23.8 s (Open WebUI chat `1f7d4872`, 27 Nov). The fix is in the flat tool, not
   in the model: it now asks the gateway for a 60 000-character window, keeps the
   article (`_readable`: start at the first block that is not a menu label, a link
   farm or a short non-sentence; drop those blocks everywhere else), flattens
   inline links to their label (`[Gerry Scotti](https://…)` → `Gerry Scotti`,
   which on a Wikipedia article is more than half the bytes) and only then
   truncates to the size the caller asked for. Measured: 40 000 chars of page →
   **15 449 of article**, and the first 2 000 now start at "Virginio Scotti,
   detto Gerry …". `section` / `readHeadings` / `paragraphRange` still pass
   through untouched, because those slices are already filtered upstream.
   The same change makes search results usable on their own (**4 × 300 chars**
   instead of 3 × 220, 1 200 chars total, under the router's 1 500 cap): the first
   snippet for an Italian query is now real prose from the article.

**A turn that already carries ten tool results answers with them.**
`ROUTER_TOOL_LOOP_MAX_RESULTS` (default **10**, `0` disables) now ends the loop
*with the schemas still in the body*: `tool_choice` is downgraded to `auto` when
the client forced `required`/`any`, and one extra user message — an answer-now
note (`ROUTER_LOOP_GUARD_NOTE`, it names the result count) — is appended at the
**end** of the conversation. The guard also forces the 256-token synthesis
budget, even though `awaiting_tool_answer()` would normally infer it from the
last message.

Why not remove the tools, as this router did until 28 Nov. `llama.cpp` installs
the model's tool-call parser only when `tools` is in the request, and with the
parser gone the model's native Hermes markup
(`<tool_call>web_search<arg_key>…`) leaves as ordinary `content`: Open WebUI
renders it verbatim, executes nothing, and closes the run with
`usage.incomplete` (chat `63fdf696`, 28 Nov). `tool_choice: "none"` is not the
way around it — the server drops the schemas from the rendered prompt all the
same (probe: prompt_n 132 → 4, markup leaked). Appending the note instead of
rewriting the message list keeps the cached prefix byte-identical: measured on
the board, the first guarded turn prefilled 661 tokens and the repeat of the
same request **4**. The ceiling moved 3 → 10 because three results were often
too few for the question that started the loop (the same chat kept searching);
with ten, the model has the evidence and the ~1 060-token tool surface is not
thrown away for nothing.

**When a client strips the tools anyway, the leaked call is repaired.**
`ROUTER_REPAIR_TEXT_CALLS` (default `1`) rewrites a text
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` block
into a real `tool_calls` entry, with a deterministic `call_<sha1>` id and JSON
arguments (`<arg_value>10</arg_value>` is the number 10), leaving any prose
around it as `content`. Only names the conversation already knows are accepted
— the request's `tools`/`functions` or a call already present in the history —
so a chat that never had tools cannot be given an invented one. Streaming
needs a one-prefix hold: the markup arrives as ordinary content deltas, so the
first characters of each message are held until `<tool_call>` is ruled in or
out, then the calls are attached to the finish chunk with
`finish_reason: "tool_calls"`; usage and `[DONE]` pass through untouched.
Measured live on the board, 28 Nov: a 5-message tool history with no `tools`
and an explicit "usa il tool web_search" produced a 104-chunk SSE stream with
`content: ""` and one repaired `web_search` call, and the router logged
`repaired a text tool call into tool_calls (web_search)`; the non-streaming
request returned the same call as JSON. Plain prose and byte-faithful
passthrough are covered by the local harness.

Measured end to end through the real 7-tool surface (`e2e`: router → llama-swap →
MoE, tools executed on `mcp-tools-lite`):

| scenario | before | after |
| --- | --- | --- |
| "chi è Gerry Scotti? cerca sul web" | 3 turns, 55 s each, empty answer | search 13.1 s → read article 41.6 s → **answer, 99.4 s total** |
| same, second page tried first | `content: ""`, `incomplete` | 2 hops, answer |
| "confronta le due Wikipedia, leggi entrambe" | loop until the budget dies | 3 results → `loop guard: 3 tool results already`, answer (234 s, 800-token answer; measured with the old ceiling of 3) |
| `scripts/toolcheck.py` 9-case battery | 9/9 | **9/9** (20-27 s per case, `synthweb` 83 s) |

The same turn was then replayed through **Open WebUI's own chat-completions
pipeline** (28 Nov, `POST /api/chat/completions` with an API key — `ENABLE_API_KEYS`
is now on in the Open WebUI compose for exactly this test — and `tool_ids:
["server:mcp:mcp-tools-lite"]`, the MCP server resolved from
`tool_server.connections`, so the tool surface is the production one and the model
sees the real `mcp-tools-lite_*` names). The whole loop, end to end, executed by
hand because Open WebUI stops at the call on a non-saved chat (no `chat_id`, so no
websocket events, no `pause_for_tool_approval` — the call is simply returned to the
caller):

| round | router | wall | what came back |
| --- | --- | --- | --- |
| user asks "cerca sul web chi è Gerry Scotti…" | `route=tools budget=96` | 15.3 s | one native `tool_calls`: `mcp-tools-lite_web_search{"query": "Gerry Scotti chi è programma adesso", "count": 3}`, no markup in `content`, 69 tokens, prompt 2214 of which **2179 cached** |
| result fed back | `route=tools by=sticky-loop budget=256` | 36.6 s | **another `web_search`** with a re-worded query — the `synthweb` failure, live |
| second result fed back | `route=tools by=sticky-loop budget=256` | 61.1 s | `finish_reason: stop`, no call, and the answer is right: quiz/varietà host, *Caduta libera*, with the Max Giusti handover noted |

113 s for a two-hop search that a human would call one question. The temperature
raise (0.2 → 0.6 on the tool turn) fired on every round, and the KV prefix survived
LiteLLM → router → llama-swap on all three (2217 tokens cached on rounds two and
three), which is the whole point of keeping the tool surface byte-identical.

The guard was re-measured on the board with the schemas kept: a 10-result tool
history answered in 40 s (`finish: stop`, **0 calls**, 661 prompt / 136 completion
tokens) instead of looping, and the repeat of the same turn prefilled only 4
tokens against the 143 s full re-prefill the strip-and-rewrite guard used to pay.

**The iteration cap is deliberately left at -1 (unlimited)** in Open WebUI.
Capping it at 8 does terminate the bad loop, but it also kills the long agentic
tasks this stack exists for: Open WebUI then answers "Tool-call limit reached (8
iterations)". The loop has to end because the tools work, not because the client
gives up. One chat row did measure 939 s with `content: ""` before these fixes —
that row was the model re-searching, not the cap.

`scripts/toolcheck.py` has two synthesis cases, **`synth`** (a small weather
result) and **`synthweb`** (a 1.5 kB `web_search` payload, the production
shape): each replays a question + tool call + tool result and requires a
non-empty answer with no further call. Those are the regression tests for
faults 3, 4 and 5.

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
sudo ./install.sh --with-quality  # also fetch the opt-in Nemotron tier (5.74 GiB)
sudo ./scripts/tune.sh --check    # show unapplied OS tuning
sudo ./scripts/tune.sh --apply    # swappiness, noatime, governor, zram checks
./healthcheck.sh                  # quick liveness of both tiers
./healthcheck.sh --tools          # + the 9-case tool-calling battery
./scripts/toolcheck.py            # tool calling on its own (--quick, --stream, --model)
./benchmark.sh --all              # pp/tg per model, embedding latency
sudo ./update-llama.cpp.sh        # rebuild llama.cpp at the tip of master (the timer runs it weekly)
sudo LLAMA_CPP_SOURCE=release ./update-llama.cpp.sh   # newest release tag instead
sudo LLAMA_CPP_TAG=v1.0.0 ./update-llama.cpp.sh       # one exact ref, bypasses the comparison
```

`--check` is not cosmetic: a plain `install.sh` does not imply `--tune`, and this
board served for weeks with `vm.swappiness=100` because of it. At 100 the kernel
evicts the mmap'd weights to zram mid-conversation (measured on 28 Nov: −9 % decode
and 715/492 pages/s of swap I/O), which looks exactly like "the model got slower".
`healthcheck.sh` now warns when the sysctl disagrees with `tune.sh` or when more
than 20 % of swap is in use — and it is worth reading: it caught this board back at
100 within minutes of the drop-in being written, because `armbian-config`'s
`99-zz-armbian-tuning-profile.conf` (swappiness 100, dirty ratios 20/1, written for
flash endurance) sorts *after* `90-*` and any `sysctl --system` re-applies it. Ours
therefore lives in `99-zzz-cheap-llm-stack.conf`, after theirs; a 90-* file is
visibly written and visibly ignored.

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
