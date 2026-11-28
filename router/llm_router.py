#!/usr/bin/env python3
"""
llm-router — the "orchestrator" of cheap-llm-stack.

A dependency-free (stdlib only) OpenAI-compatible facade in front of llama-swap.
It decides *which* model should answer a request, so that the expensive act of
switching models happens rarely and always on purpose.

Why a router and not "one big model":
  * the text workhorse is a MoE (LFM2-8B-A1B): 8.4B of weights but only 1.1B active
    per token, so it decodes at 10.7 tok/s — faster than the dense 2B (6.3) and far
    faster than the dense 4B (2.9), while being better than both on blind grading;
  * llama-swap can only swap by request *model name*, it cannot inspect content;
  * therefore content-aware routing belongs in this small proxy: keep the cheap,
    always-resident models for what they are good at, and pay a swap only when the
    request genuinely needs the 5 GB MoE, the OCR VLM or the thinking model.

Decision cascade (fastest first):
  0. tools in the request, or a tool result in the history -> tools (the MoE agent)
  1. explicit model in the request  -> pass through untouched (no routing at all)
  2. images present                 -> vision (single VLM) ; OCR hints only relabel it
  3. deterministic keyword rules     -> code / translate / summarize / reason / trivial
  4. embedding similarity            -> reuses the always-on embedding-model, 0 extra RAM
  5. default                         -> reasoner (MoE workhorse)

Tool calling is the reason step 0 exists. An agent client (Open WebUI, or anything
that speaks OpenAI + MCP/tools) sends `tools` on every turn and expects the same
model to keep interpreting its own tool results, so a tool conversation must never
be downgraded to the fast tier and must never bounce between models: the split
decision below is made once and re-pinned in the sticky table. Measured on the
7-case battery (5 tools, parallel calls, one negative case, IT+EN) the MoE scores
7/7 at reasoning_budget_tokens=32, while the old 0.6B tier scored 7/7 but mangled
tool arguments and wrote poor prose, and the previous workhorse scored 1/7.
`reasoning_budget_tokens` is injected per route because it is the one sampling knob
that changes behaviour here: at 0 the MoE spills its thinking into `content`
(5/7, 52 s per call), at 32 it is 7/7 in 17.9 s, at 96 still 7/7 but 24.1 s — so
tool turns keep 32, while plain chat runs at 96 because at 32 a prompt that needs a
real plan gets its thought cut mid-sentence and the leftover shows up in `content`.

There is no "big model" escalation: on this board it would be a 4B at 2.9 tok/s,
and a bounded thinking pass still costs 170 s (measured) — a tier nobody waits for.
A six-question factual battery (MoE, Dante, 17*23, HTTP/2, first iPhone, RAM vs swap)
came out equal on accuracy and ~2x faster on the MoE, with better Italian grammar,
so the MoE is both the default and the quality tier.

Every route is sticky per session (X-Session-Id header, else a hash of the first
user message, else client IP+UA) for STICKY_TTL seconds, and the decision is made
once, at the front of the loop: after a tool call has been issued the session keeps
the model that issued it (`sticky-loop`), because a swap is the most expensive
thing this router can do — 88-120 s of cold load on the 8B tiers, plus the prompt
cache holding the tool-schema block. A switch happens only when it is unavoidable:
an image turn for a model with no vision, the first tool turn of a session that
started on another route, or a deep request (on its first turn, and once mid-chat
when ROUTER_DEEP_ESCALATE=1).

Response headers: X-Route, X-Route-Model, X-Route-By, X-Route-Ms, X-Route-Budget.

The router also advertises a synthetic model id, `auto`, in GET /v1/models. Without
it a client can only pick a raw llama-swap id (router, reasoner, vision,
embedding-model), which is passed through verbatim and therefore *bypasses* every
decision above. `auto`, `router` and `orchestrator` are the three magic ids that
mean "route this request"; the fast tier also answers to `fast`, `tiny` and `lfm3`
so it can still be addressed directly when debugging.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_HOST = os.environ.get("LLAMA_SWAP_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("LLAMA_SWAP_PORT", "10001"))
LISTEN_HOST = os.environ.get("ROUTER_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("ROUTER_PORT", "10000"))

# ROUTER_DEBUG_BODY=1 logs the exact JSON handed upstream. It exists for one job:
# a tool-call case that passes straight at the MoE and fails through this proxy is
# only explainable by what the proxy changed in the payload, and guessing that is a
# waste of an afternoon at 20 s per case. Off by default — a request body carries
# user text.
DEBUG_BODY = os.environ.get("ROUTER_DEBUG_BODY", "0") == "1"

STICKY_TTL = int(os.environ.get("ROUTER_STICKY_TTL", "1800"))
SEM_THRESHOLD = float(os.environ.get("ROUTER_SEM_THRESHOLD", "0.45"))
SEM_MIN_MARGIN = float(os.environ.get("ROUTER_SEM_MARGIN", "0.02"))
MAX_STICKY = 1024

# reasoning_budget_tokens per route (llama.cpp's per-request thinking cap). -1 would
# mean "use the server default"; we always say it explicitly so a route change is one
# edit here and not a service restart. Values are measured, see the module docstring:
#   96  for plain chat. 32 was enough for short answers but not for a prompt that
#       needs a real plan: the cap then cuts the thought mid-sentence and the leftover
#       is returned inside `content` (" thinking We need to answer in Italian...") —
#       measured on "spiega passo per passo come funziona TCP" (150 ch of thought in
#       reasoning_content, the rest dumped in content). At 96 the same prompt is clean
#       and often *faster* (26.0 s vs 32.1 s on a Rust/Go comparison) because the
#       model stops re-thinking; typical chat prompts cost 0-3 s more.
#   96  for the tool-call turn as well. 32 looked right on the short battery prompts
#       (7/7 at 17-19 s) but starves the real Open WebUI request: with the MCP
#       instructions as a system message plus "cerca sul web chi è Gerry Scotti e
#       dimmi chi è in due frasi", 32 cuts the thought *while it is planning the
#       call* and dumps "The user asks in Italian: Search the web who is…" into
#       content — no tool_calls at all. Measured on the flat surface, streaming,
#       seeds 15/25/47/55/65: 32 -> 3/5 calls, 64 -> 1/5, 96 -> 5/5 (12-14 s),
#       128 -> 5/5 (12-16 s). The call needs ~90 tokens of thought, not 32.
#   256 for the rare "prove it / design it" request, which is the only place a long
#       thinking pass pays for itself at 10 tok/s
ROUTE_BUDGET = {
    "assistant": int(os.environ.get("ROUTER_BUDGET_CHAT", "96")),
    "reason": int(os.environ.get("ROUTER_BUDGET_REASON", "256")),
    "tools": int(os.environ.get("ROUTER_BUDGET_TOOLS", "96")),
    "vision": int(os.environ.get("ROUTER_BUDGET_VISION", "32")),
    "vision-docs": int(os.environ.get("ROUTER_BUDGET_VISION", "32")),
    # Only used from the `deep` route: at 2.9 tok/s a 256-token thought is 88 s
    # before the first word of the answer.
    "deep": int(os.environ.get("ROUTER_BUDGET_DEEP", "128")),
}
# A tool call is ~120 tokens of reasoning + ~60 of arguments; a client that caps the
# answer at 200 tokens truncates the call instead of the prose, which the agent then
# never sees. Raise the ceiling on tool turns only, never lower a client's value.
TOOLS_MIN_MAX_TOKENS = int(os.environ.get("ROUTER_TOOLS_MIN_MAX_TOKENS", "800"))
# A tool turn is two different jobs and they want opposite budgets.
#   * emitting a call  -> 32 tokens of thinking is measured 7/7 at 17-19 s;
#   * writing the answer from the tool result -> the thinking has to FIT, or the
#     model stops with an EMPTY answer (the whole budget is spent thinking and
#     nothing is left to say). Measured on a replayed synthesis turn with real
#     web_search output (LFM2.5-8B-A1B, Open WebUI payload, 3 runs per point):
#       budget  32 -> 0/3 answers   (6 s: pure loss)
#       budget  96 -> 0/3 answers   (the old value)
#       budget 192 -> 2/3 answers   (20-35 s)
#       budget 256 -> 3/3 answers with a compact result (33 s)
#       budget 384 -> 3/3 answers, 54 s
#     An empty answer is exactly what Open WebUI shows as a hung chat, and a
#     32-token cap on the *call* turn plus a 256-token cap on the *answer* turn
#     is the pair that makes both jobs work.
TOOLS_SYNTH_BUDGET = int(os.environ.get("ROUTER_BUDGET_TOOLS_SYNTH", "256"))
# A tool result is re-sent on every following turn, and its size is what the
# model thinks about: the raw 1.5 kB web_search JSON (lanes, timings) produced an
# empty answer at budget 96-128 (0/3, 1/3), while two or three compact
# "title / url / snippet" lines answer at budget 256. Keep head + tail but keep
# it small: the head carries the match, the tail carries the date/source footer.
TOOL_RESULT_MAX_CHARS = int(os.environ.get("ROUTER_TOOL_RESULT_MAX_CHARS", "1500"))
# Below this the MoE answers a tool request in prose instead of emitting the call.
# Measured, see apply_route_knobs. 0 disables the floor.
TOOLS_MIN_TEMPERATURE = float(os.environ.get("ROUTER_TOOLS_MIN_TEMPERATURE", "0.6"))
# Loop guard. An 8B model on a slow host can keep asking for "one more page":
# measured on Open WebUI chat 1f7d4872 (web_search -> web_url_read EN -> IT,
# 55 s per turn) it never stopped because every result was page chrome, and the
# fourth turn spent its whole 256-token thinking budget planning another fetch
# and returned no content at all (empty answer, usage.incomplete). So after this
# many tool results the turn is a synthesis turn whether the model agrees or not.
#
# What the guard must NOT do is take the schemas away. Measured here on 28 Nov,
# replayed history, same seed: llama.cpp installs the model's tool-call parser
# only when `tools` is in the request, and `tool_choice: "none"` is not a way
# around it — the server drops the schemas from the rendered prompt all the same
# (prompt_n 132 -> 4). Either way the model's native
# `<tool_call>/<arg_key>/<arg_value>` markup leaks into `content` as plain text,
# Open WebUI renders it verbatim and the run ends with usage.incomplete — that is
# exactly how the 08:09 automation died after three calls. Keeping the schemas
# and appending one user message instead: 2/2 prose answers, 15 s, no call. It is
# also cheaper, because the cached prefix survives and only the appended line is
# prefilled, where dropping the tools re-prefilled all 3 011 tokens (143 s).
#
# 10 results leaves a real run (search -> read -> read -> ...) room before the
# synthesis is forced; 0 disables the guard.
TOOL_LOOP_MAX_RESULTS = int(os.environ.get("ROUTER_TOOL_LOOP_MAX_RESULTS", "10"))
LOOP_GUARD_NOTE = os.environ.get(
    "ROUTER_LOOP_GUARD_NOTE",
    "[router] %d tool results are already in this turn and no further tool result "
    "will be available. Do not call a tool: answer now with what you have, and say "
    "explicitly what you could not verify.")
# The quality tier (llama-swap model `bolt`, Nemotron-3.5-Bolt-E4B) is reached by
# `auto` when a request is plainly a deep one: long *and* asking for reasoning,
# judged on the last user turn (a client's system prompt is long and technical on
# every turn, so the concatenation is not a signal). The session is then pinned
# there (sticky-deep) so a follow-up does not pay the ~100 s reload back to the MoE.
# With ROUTER_DEEP_ESCALATE=1 a session that started ordinary can still escalate
# mid-chat, once, as long as no tool result is in flight and there is no image.
# Measured on this board: 2.91 tok/s against the MoE's 10.4, 5.74 GiB of weights,
# 412 s for one 800-token answer -- which is why the escalation is narrow and off
# until the unit sets ROUTER_DEEP_TIER=1.
DEEP_ENABLED = os.environ.get("ROUTER_DEEP_TIER", "0") == "1"
# The "long enough to be worth 3.5x the latency" bar, judged on the last user turn
# only. 600 chars / 40 words was measured too high for Open WebUI: a real design
# question ("progetta un indice semantico per 109 tool, quali trade-off...") is
# ~30 words, so `auto` never left the MoE and `bolt` sat idle for 14 days.
# Both numbers are knobs because the right value is a latency budget, not a fact:
# raise them again and `auto` goes back to MoE-only, without a code change.
DEEP_MIN_CHARS = int(os.environ.get("ROUTER_DEEP_MIN_CHARS", "400"))
DEEP_MIN_WORDS = int(os.environ.get("ROUTER_DEEP_MIN_WORDS", "25"))
# Mid-chat escalation: a session pinned to `tools` because Open WebUI attaches the
# flat surface to every request can still be asked a deep question on turn 3.
# Off by default (it costs one cold load of 5.7 GB when it fires); the unit sets it.
DEEP_ESCALATE = os.environ.get("ROUTER_DEEP_ESCALATE", "0") == "1"
# How long an escalated session is remembered *beyond* the sticky TTL, so that an
# expiry cannot re-escalate the same chat and start a bolt <-> MoE swap loop
# (~100 s each way on the microSD).
DEEP_SEEN_TTL = int(os.environ.get("ROUTER_DEEP_SEEN_TTL", "21600"))
CONTEXT_LENGTH = int(os.environ.get("ROUTER_CONTEXT_LENGTH", "12288"))

DEFAULT_ROUTE = "assistant"

# route -> llama-swap model id
# "assistant", "reason" and "tools" all land on the MoE: it is the best text model on
# the board *and* the fastest of the big ones, so there is nothing to split between a
# small default, a quality escalation and an agent tier for text.
ROUTE_MODEL = {
    "fast": "router",
    "assistant": "reasoner",
    "reason": "reasoner",
    "tools": "reasoner",
    "vision": "vision",
    "vision-docs": "vision",
    "deep": "bolt",
    "embedding": "embedding-model",
}

# The fast tier and every alias that resolves to it. A tool turn addressed to one of
# these is redirected to the tools route (see Handler.do_POST): the MoE scores 9/9
# on the 9-case battery and LFM3-1.7B 7/9, and per-call latency is within 2 s of each
# other (7.6-11.3 s vs 5.9-12.6 s), so the MoE's accuracy is what decides.
TINY_MODELS = {"router", "fast", "tiny", "lfm3"}

# --------------------------------------------------------------- prototypes ---
# One short utterance per intent, in English and Italian (this stack is used in
# Italian). They are embedded once at startup with the always-on embedding model.
PROTOTYPES = {
    "fast": [
        "what is the capital of france",
        "convert 10 km to miles",
        "say hello in three words",
        "capitalize this sentence",
        "give me a one line answer",
        "qual è la capitale dell'italia",
        "quanto fa 12 per 8",
        "rispondi in una riga",
    ],
    "assistant": [
        "explain how photosynthesis works",
        "write a short email to my landlord",
        "summarize this article for me",
        "what are the pros and cons of electric cars",
        "spiegami come funziona la fotosintesi",
        "scrivimi una email per il mio padrone di casa",
        "riassumi questo testo",
    ],
    "reason": [
        "prove that the square root of 2 is irrational step by step",
        "debug this stack trace and tell me the root cause",
        "design a fault tolerant architecture for a queue service",
        "refactor this module and explain the trade-offs",
        "analyze the complexity of this algorithm",
        "dimostra passo passo perché questo limite esiste",
        "analizza questo codice e trova il bug",
        "progetta l'architettura di un sistema distribuito",
    ],
    "vision": [
        "what is in this photo",
        "describe this picture",
        "what breed is the dog in the image",
        "che cosa si vede in questa foto",
        "descrivi questa immagine",
    ],
    "vision-docs": [
        "extract the text from this screenshot",
        "read the numbers in this table and turn them into csv",
        "transcribe this receipt into json",
        "what does this scanned invoice say",
        "estrai il testo da questo screenshot",
        "leggi questa tabella e convertila in csv",
        "trascrivi questa ricevuta in json",
    ],
}

# ---------------------------------------------------------------- rule hints ---
# An actual snippet or crash in the message, not the *name* of a technology. The
# difference matters: "riassumi in 5 punti i vantaggi di Rust rispetto a Go" is 10
# words and used to hit this rule through the word "rust", land on the (then) 0.6B
# and come back with three shallow lines. Technology vocabulary lives in TECH_RE and
# now only guards the fast tier; a shortcut to it needs code you can point at.
CODE_RE = re.compile(
    r"```|~~|\bdef\s+\w+\s*\(|\bclass\s+\w+|\bfunction\s+\w+\(|\bimport\s+\w+|#include"
    r"|\bTraceback \(most recent call last\)|\b\w+Error:|\bsegfault\b|\bnull pointer\b"
    r"|\bSELECT\b.+?\bFROM\b|\bCREATE TABLE\b",
    re.IGNORECASE,
)
REASON_RE = re.compile(
    r"\b(prove|derive|step[ -]by[ -]step|trade-?offs?|architect(ure|ural)|root cause"
    r"|algorithm|complexity|optimi[sz]e|refactor|benchmark|dimostra|dimostrazione"
    r"|passo[ -]per[ -]passo|passo passo|architettura|complessità|ottimizza|rifattorizza"
    r"|causa (radice|del problema))\b",
    re.IGNORECASE,
)
TRANSLATE_RE = re.compile(r"\b(translate|traduci|traduzione|in inglese|in italiano)\b", re.IGNORECASE)
SUMMARY_RE = re.compile(r"\b(summar(y|ize|ise)|tl;?dr|riassum(i|ere|to)|sintesi)\b", re.IGNORECASE)
OCR_RE = re.compile(
    r"\b(ocr|screenshot|receipt|invoice|scanned|scan\b|table|spreadsheet|csv|form\b|document"
    r"|scontrino|ricevuta|fattura|tabella|foglio|modulo|documento"
    # reading text *out of* a picture is the document model's job even when the
    # sentence never names a document: "trascrivi il testo, riga per riga" used
    # to fall through to the conversational VLM, which then had to be swapped
    # for the OCR one on the very next turn (~100 s of microSD load each way).
    r"|trascriv\w*|transcri\w*|estrai il testo|extract text|leggi il testo|read the text)\b",
    re.IGNORECASE,
)
SHORT_HINT_RE = re.compile(
    r"^(what|who|when|where|which|how much|how many|define|convert|translate|capital of"
    r"|che|chi|quando|dove|quanto|quanti|qual|quale|quali|definisci|converti)\b",
    re.IGNORECASE,
)
# Greetings, thanks, "are you there": the only requests where the fast tier is
# unambiguously good enough. Kept narrow on purpose — small talk is cheap only if it
# stays small.
CHITCHAT_RE = re.compile(
    r"^\W*(ciao|salve|hey|hi|hello|buongiorno|buonasera|grazie|thanks|thank you|ok|okay"
    r"|sei (li|connesso)|ci sei|come stai|how are you|che ore|what time)\b",
    re.IGNORECASE,
)
# Asking the model to *produce* or *change* code is never a fast-tier job, however
# short the sentence: "scrivi una funzione python ..." is 9 words and used to land
# on the 0.6B, where it came back as broken docstring prose.
GENERATE_RE = re.compile(
    r"\b(write|create|implement|generate|refactor|fix|add|rewrite|convert .* (to|into)"
    r"|scrivi|crea|implementa|genera|rifattorizza|correggi|aggiungi|riscrivi|sistema)\b",
    re.IGNORECASE,
)
# Anything that asks for an explanation, a comparison or a why/how does NOT belong on
# the fast tier: the (then) 0.6B answered "Un MoE (Moyen d'Équipe)" — 2 s of
# latency saved are not worth a hallucinated acronym.
EXPLAIN_RE = re.compile(
    r"\b(explain|describe|summar|compare|difference|why|how (does|do|is|are)|tell me about"
    r"|spiega|spiegami|descrivi|riassum|confronta|differenza|perch[eé]|come (funziona|si)\b)"
    r"|\bcos'?\s?[eè] (un|una|il|la|lo|i|gli|le|l')\b|\bche cos'?[eè]\b|\bwhat(?:'s| is) a\b",
    re.IGNORECASE,
)
# Vocabulary that turns even a 5-word question into a job for the MoE: a 0.6B-tier
# model invents acronyms when asked about anything technical ("MoE = Model of
# Explanation").
TECH_RE = re.compile(
    r"\b(moe|llm|model of experts|mixture of experts|transformer|attention|embedding|token"
    r"|quantizzazion\w*|quantization|gguf|lora|qlora|kv[- ]cache|context window|thread|mmap"
    r"|npu|gpu|cpu|ram|vram|swap|zram|docker|kubernetes|systemd|api|http|tls|vpn|ssh|dns"
    r"|regex|sql|git|python|rust|golang|javascript|typescript|kernel|filesystem)\b",
    re.IGNORECASE,
)
# Questions that need stored world knowledge, not a one-liner. "Chi ha scritto la
# Divina Commedia e in che anno è ambientata?" is 11 words and starts with "chi",
# so it matched the trivial rule and the (then) 0.6B answered "periodo 1301-1310".
FACTUAL_RE = re.compile(
    r"\b(chi ha (scritto|detto|fatto|inventato|creato|dipinto|composto)"
    r"|in che (anno|secolo|periodo|epoca)"
    r"|quando (è|e) (uscito|nato|morto|ambientat|avvenut)"
    r"|dove (è|e) (nato|ambientat|situat)"
    r"|who (wrote|invented|discovered|painted|composed)"
    r"|when was .{0,40}(born|released|published))",
    re.IGNORECASE,
)
# A capitalised word that is not the first token is a name, a place or a title: the
# request is about an entity, so the tier that has room for facts answers it.
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def has_proper_noun(text: str) -> bool:
    return any(w[0].isupper() and len(w) > 2 for w in WORD_RE.findall(text)[1:])


def log(msg: str) -> None:
    sys.stderr.write("%s llm-router %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg))
    sys.stderr.flush()


# ------------------------------------------------------------------ upstream ---
def upstream_json(path: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    url = "http://%s:%d%s" % (UPSTREAM_HOST, UPSTREAM_PORT, path)
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def upstream_ok(path: str = "/health", timeout: float = 5.0) -> bool:
    """True when llama-swap answers 2xx. It replies with the plain text "OK",
    not JSON, so this must not try to parse a body."""
    url = "http://%s:%d%s" % (UPSTREAM_HOST, UPSTREAM_PORT, path)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def embed(texts: list[str]) -> list[list[float]]:
    out = upstream_json(
        "/v1/embeddings",
        {"model": "embedding-model", "input": texts, "encoding_format": "float"},
        timeout=60,
    )
    return [d["embedding"] for d in out["data"]]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


# ------------------------------------------------------------------ warming ---
# The resident set is warmed here, not by llama-swap's preload hook, and the hook
# is deliberately absent from config.yaml.
#
# llama-swap implements hooks.on_startup.preload as a synthetic `GET /` routed to
# the model, and the status of that GET is what it calls success (upstream:
# internal/server/api.go, startPreload). llama.cpp built with
# -DLLAMA_USE_PREBUILT_UI=OFF (our build; the option fetches a UI bundle from a
# Hugging Face bucket at build time) registers no `/` route, so
# every preload logged
#     [ERROR] failed to preload model <id>: status 404
# while the model loaded fine. Verified on this board against a stub upstream:
# health passes, then GET / answers 404 and the preload is declared failed. A
# permanent false alarm in the only monitor this box has is worse than the
# silence it replaces, and a preload hook cannot do the two jobs that matter:
#   * it runs only when llama-swap itself starts. The weekly rebuild restarts
#     llama-swap, so the first request after an update paid the MoE load: measured
#     93 s for the first tool case of the battery against 17 s for the rest.
#   * `GET /` warms nothing but the process. A 1-token completion also warms the
#     prompt cache and the KV slots, which is what makes a tools+system prefix
#     cost 4-6 s warm instead of 43 s cold.
WARM_ENABLED = os.environ.get("ROUTER_WARM", "1") == "1"
# The interval only bounds how long after a llama-swap restart the warm-up starts;
# between restarts the loop costs one /proc scan and one HTTP GET.
WARM_POLL_SEC = int(os.environ.get("ROUTER_WARM_POLL", "20"))
# Order matters: the embedding model is what Prototypes.build is waiting for, and
# the MoE is the slow load, so it goes last and never blocks the front door.
WARM_MODELS = ("embedding-model", "router", "reasoner")
# A cold MoE read from microSD is minutes; a warm one is seconds.
WARM_TIMEOUT = float(os.environ.get("ROUTER_WARM_TIMEOUT", "900"))


def warm_model(model: str) -> bool:
    """One-token request that makes llama-swap load `model` and keeps its caches
    hot. reasoning_budget_tokens=0 because the reply is thrown away and the
    cheapest generation is the right one; the budget-0 thinking leak into
    `content` is not read by anyone here."""
    t0 = time.time()
    try:
        if model.startswith("embedding"):
            embed(["warm"])
        else:
            upstream_json("/v1/chat/completions",
                          {"model": model,
                           "messages": [{"role": "user", "content": "ok"}],
                           "max_tokens": 1, "temperature": 0.0, "stream": False,
                           "reasoning_budget_tokens": 0},
                          timeout=WARM_TIMEOUT)
        log("warm %s in %.1fs" % (model, time.time() - t0))
        return True
    except Exception as exc:
        # Deliberately not fatal: an unwarm model is slow, not broken, and the
        # next request loads it anyway.
        log("warn: warm %s failed after %.1fs: %s" % (model, time.time() - t0, exc))
        return False


def llama_swap_identity() -> tuple[int, int] | None:
    """(pid, start time) of the running llama-swap, or None when it is not running.

    /proc/<pid>/stat field 22 is the process start time in clock ticks since boot,
    so the pair changes whenever the service is replaced. The identity is *read*
    from /proc rather than polled over HTTP because a `systemctl restart` is over
    in 1-2 s: a liveness poll every 20 s misses the down window entirely and the
    warm-up silently never runs (measured — the first version of this loop watched
    /health and logged nothing across a real restart). Scanning /proc also avoids
    a `systemctl show` subprocess and any dependency on the unit name.
    """
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/comm" % entry) as fh:
                if fh.read().strip() != "llama-swap":
                    continue
            with open("/proc/%s/stat" % entry) as fh:
                # "pid (comm) state ppid ..." — the comm field can contain spaces
                # and parentheses, so split on the last ") ".
                tail = fh.read().rsplit(") ", 1)[1].split()
            return int(entry), int(tail[19])      # field 22 == starttime
        except (FileNotFoundError, PermissionError, ProcessLookupError,
                IndexError, ValueError, OSError):
            continue
    return None


def warm_loop() -> None:
    """Warm the resident set at boot and again every time the llama-swap process
    is replaced — a restart evicts every model, and that is the case a preload
    hook cannot cover: the weekly rebuild restarts llama-swap, and the update
    gate's first tool case measured the MoE load (93 s) instead of the build
    (17 s)."""
    seen = None
    while True:
        ident = llama_swap_identity()
        if ident is None:
            seen = None                    # llama-swap is gone; warm when it is back
        elif ident != seen:
            if upstream_ok(timeout=3.0):
                log("warming the resident set (%s)"
                    % ("boot" if seen is None else "llama-swap pid %d" % ident[0]))
                seen = ident
                for model in WARM_MODELS:
                    warm_model(model)
            # not answering yet: leave `seen` alone and try again next poll
        time.sleep(WARM_POLL_SEC)


class Prototypes:
    """Prototype vectors, built once the embedding model is answering."""

    def __init__(self) -> None:
        self.vectors: dict[str, list[tuple[str, list[float]]]] = {}
        self.ready = threading.Event()

    def build(self) -> None:
        texts, index = [], []
        for route, samples in PROTOTYPES.items():
            for s in samples:
                texts.append(s)
                index.append(route)
        for attempt in range(1, 121):
            try:
                vecs = embed(texts)
                built: dict[str, list[tuple[str, list[float]]]] = {}
                for route, text, vec in zip(index, texts, vecs):
                    built.setdefault(route, []).append((text, vec))
                self.vectors = built
                self.ready.set()
                log("prototypes ready (%d utterances, %d routes)" % (len(texts), len(built)))
                return
            except Exception as exc:  # embedding model still loading
                if attempt in (1, 6, 20, 60):
                    log("waiting for embedding-model (%s): %s" % (attempt, exc))
                time.sleep(5)
        log("ERROR: could not build prototypes; falling back to rules only")

    def best_route(self, text: str) -> tuple[str | None, float, float]:
        if not self.ready.is_set() or not text.strip():
            return None, 0.0, 0.0
        try:
            (vec,) = embed([text[:2000]])
        except Exception as exc:
            log("embedding lookup failed: %s" % exc)
            return None, 0.0, 0.0
        scored = []
        for route, samples in self.vectors.items():
            scored.append((max(cosine(vec, v) for _, v in samples), route))
        scored.sort(reverse=True)
        if not scored:
            return None, 0.0, 0.0
        best = scored[0][0]
        margin = best - (scored[1][0] if len(scored) > 1 else 0.0)
        return scored[0][1], best, margin


PROTO = Prototypes()

# ------------------------------------------------------------------ sessions ---
_sticky: dict[str, tuple[str, float]] = {}
_sticky_lock = threading.Lock()
_stats: dict[str, int] = {}
_stats_lock = threading.Lock()


def sticky_get(key: str) -> str | None:
    now = time.time()
    with _sticky_lock:
        item = _sticky.get(key)
        if item and now - item[1] < STICKY_TTL:
            return item[0]
        _sticky.pop(key, None)
    return None


def sticky_put(key: str, route: str) -> None:
    with _sticky_lock:
        if len(_sticky) > MAX_STICKY:
            oldest = sorted(_sticky.items(), key=lambda kv: kv[1][1])[: len(_sticky) // 4]
            for k, _ in oldest:
                _sticky.pop(k, None)
        _sticky[key] = (route, time.time())


def bump(route: str) -> None:
    with _stats_lock:
        _stats[route] = _stats.get(route, 0) + 1


# Sessions that have already been on the quality tier at least once. Kept apart from
# _sticky because it outlives it: the sticky entry expires after STICKY_TTL (30 min),
# and without this memory a long chat could escalate, fall back, escalate again --
# two model swaps per turn pair, 100 s each on this board.
_deep_seen: dict[str, float] = {}
_deep_seen_lock = threading.Lock()


def deep_seen(key: str) -> bool:
    now = time.time()
    with _deep_seen_lock:
        stamp = _deep_seen.get(key)
        if stamp is not None and now - stamp < DEEP_SEEN_TTL:
            return True
        _deep_seen.pop(key, None)
    return False


def deep_mark(key: str) -> None:
    with _deep_seen_lock:
        if len(_deep_seen) > MAX_STICKY:
            oldest = sorted(_deep_seen.items(), key=lambda kv: kv[1])[: len(_deep_seen) // 4]
            for k, _ in oldest:
                _deep_seen.pop(k, None)
        _deep_seen[key] = time.time()


# --------------------------------------------------------------- language -----
# Small models drift between languages: the 0.6B tier asked "grazie, sei gentile"
# in Italian answered in Spanish. One short system line fixes it, ~20 prompt tokens.
LANG_HINT = "Rispondi nella stessa lingua dell'utente. Sii breve e concreto."


def keep_language(body: dict) -> dict:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return body
    if any(isinstance(m, dict) and m.get("role") == "system" for m in msgs):
        return body
    body["messages"] = [{"role": "system", "content": LANG_HINT}] + msgs
    return body


# ------------------------------------------------------------------ routing ---
def message_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for chunk in c:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    parts.append(chunk.get("text", ""))
    return "\n".join(parts).strip()


def last_user_text(messages: list[dict]) -> str:
    """Text of the newest user turn -- the request the decision is actually about.

    `message_text` concatenates system, assistant and tool messages too, and an
    Open WebUI turn carries a ~1.5 kB system prompt describing the tool surface:
    it mentions HTTP, docker and tokens before the user has typed anything, and it
    is longer than any deep threshold. Judged on that text every chat looks deep;
    judged on the last user turn only, the gate measures the question.
    """
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return message_text([m])
    return ""


def has_image(messages: list[dict]) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for chunk in c:
                if isinstance(chunk, dict) and chunk.get("type") in ("image_url", "input_image", "image"):
                    return True
    return False


def has_tools(body: dict) -> bool:
    """The client offered tools on this turn."""
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        return True
    # legacy OpenAI "functions" API, still emitted by a few clients
    functions = body.get("functions")
    return isinstance(functions, list) and bool(functions)


def has_tool_history(messages: list[dict]) -> bool:
    """A tool has already been called in this conversation (we are mid-loop)."""
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") in ("tool", "function"):
            return True
        if m.get("tool_calls") or m.get("function_call"):
            return True
        content = m.get("content")
        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and str(chunk.get("type", "")).startswith("tool"):
                    return True
    return False


def normalize_reasoning(body: dict) -> dict:
    """Stop a client from switching thinking *off* on a reasoning-only model.

    LFM2.5 always thinks; asked to answer without a thinking block it writes the
    monologue into `content` (measured: 5/7 on the tool battery, replies like
    "The user's request is to send a message on Telegram..."). We keep the budget
    in charge instead, which ends the block cleanly.
    """
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and kwargs.pop("enable_thinking", None) is False:
        body["chat_template_kwargs"] = kwargs
        if not kwargs:
            body.pop("chat_template_kwargs", None)
        log("dropped chat_template_kwargs.enable_thinking=false (budget governs thinking)")
    if str(body.get("reasoning_effort", "")).lower() in ("none", "off", "0"):
        body.pop("reasoning_effort", None)
        log("dropped reasoning_effort=none (budget governs thinking)")
    return body


# The MoE's chat template ends a turn with `</|end_of_sentence|>`, but its GGUF
# carries that marker as ordinary text pieces rather than as a control token: on
# a degenerate turn the model *writes* it into the answer. That is exactly what
# Open WebUI stored on 26 Nov ("Cerca gerry scotti sul web e dimmi chi è</｜
# end▁of▁sentence>") and showed to the user. A stop string is free and keeps the
# garbage out of the client; it never fires on a well-formed turn because a real
# tool call arrives as a parsed `tool_calls`, not as text.
STOP_MARKERS = ("</|end_of_sentence|>", "</｜end▁of▁sentence>", "<|im_end|>")


def add_stop_markers(body: dict) -> dict:
    """Append the model's own end-of-turn markers to whatever stop list exists."""
    stops = body.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    elif not isinstance(stops, list):
        stops = []
    added = [m for m in STOP_MARKERS if m not in stops]
    if added:
        body["stop"] = stops + added
    return body


def awaiting_tool_answer(messages: list[dict]) -> bool:
    """The last turn handed back tool output: this turn writes the answer."""
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("tool", "function"):
            return True
        content = m.get("content")
        if role == "user" and isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and str(chunk.get("type", "")).startswith("tool"):
                    return True
        return False
    return False


def tool_result_count(messages: list[dict]) -> int:
    """How many tool answers this turn already carries.

    Three shapes reach this router: `role: tool` (Open WebUI chat completions),
    `role: function` (legacy) and a Responses-style user message carrying
    `function_call_output` items. All of them mean "a tool has already run".
    """
    total = 0
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("tool", "function"):
            total += 1
            continue
        content = m.get("content")
        if isinstance(content, list):
            for chunk in content:
                if not isinstance(chunk, dict):
                    continue
                kind = str(chunk.get("type", ""))
                if kind.startswith("tool") or kind == "function_call_output":
                    total += 1
    return total


def cap_tool_results(messages: list[dict], limit: int = TOOL_RESULT_MAX_CHARS) -> int:
    """Trim `role: tool` payloads in place; returns how many were cut.

    Nothing is invented: a head/tail excerpt plus an explicit marker, so the model
    knows text is missing and can ask the tool again with a narrower query.
    """
    if limit <= 0:
        return 0
    cut = 0

    def _cap(value):
        nonlocal cut
        if not isinstance(value, str) or len(value) <= limit:
            return value
        head = int(limit * 0.8)
        tail = limit - head
        cut += 1
        return ("%s\n\n[... %d characters cut by the router: re-run the tool with a "
                "narrower query if the missing part matters ...]\n\n%s"
                % (value[:head], len(value) - limit, value[-tail:]))

    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("tool", "function"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            m["content"] = _cap(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    chunk["text"] = _cap(chunk.get("text"))
    return cut


# ------------------------------------------------------ text tool-call repair ---
# A client can strip `tools` on its own synthesis turn, and a chat template can
# regress; llama.cpp then passes the model's native
# `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` block
# through as ordinary `content`. Measured on Open WebUI 28 Nov: it is shown as
# text, nothing is executed, the run dies with usage.incomplete. The router knows
# which tools the conversation is allowed to call, so it can turn that dead end
# back into a real call — streaming included, where the markup arrives as ordinary
# content deltas and the first chars have to be held back until it is decided.
REPAIR_TEXT_CALLS = os.environ.get("ROUTER_REPAIR_TEXT_CALLS", "1") == "1"
TEXT_CALL_RE = re.compile(r"<tool_call>\s*([^\s<]+)\s*(.*?)</tool_call>", re.S)
TEXT_ARG_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.S)


def known_tools(body: dict) -> set:
    """Tool names this conversation may call: the schema, plus what it already did."""
    names = set()
    for entry in body.get("tools") or []:
        fn = (entry or {}).get("function") or {}
        if isinstance(fn.get("name"), str):
            names.add(fn["name"])
    for entry in body.get("functions") or []:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.add(entry["name"])
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        for call in m.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            if isinstance(fn.get("name"), str):
                names.add(fn["name"])
    return names


def _arg_value(raw: str):
    """`<arg_value>10</arg_value>` is the number 10; anything else stays a string."""
    try:
        return json.loads(raw)
    except Exception:
        return raw


def parse_text_calls(text: str, allowed: set):
    """(calls, leftover) out of the model markup; calls is None when there is none.

    Only names the conversation already knows are accepted: the point is to
    rescue a call the server failed to parse, not to invent one for a chat that
    never had tools.
    """
    if not isinstance(text, str) or "<tool_call>" not in text:
        return None, text
    calls = []
    for match in TEXT_CALL_RE.finditer(text):
        name = match.group(1).strip()
        if not name or name not in allowed:
            continue
        args = {}
        for key, value in TEXT_ARG_RE.findall(match.group(2) or ""):
            args[key.strip()] = _arg_value(value.strip())
        digest = hashlib.sha1(("%s:%s" % (name, json.dumps(args, sort_keys=True)))
                              .encode()).hexdigest()[:24]
        calls.append({"id": "call_" + digest, "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(args, ensure_ascii=False)}})
    if not calls:
        return None, text
    return calls, TEXT_CALL_RE.sub("", text).strip()


def repair_message(message: dict, allowed: set) -> bool:
    """Rewrite `content` markup into `tool_calls` in place. True if it fired."""
    if not isinstance(message, dict) or message.get("tool_calls"):
        return False
    calls, leftover = parse_text_calls(message.get("content"), allowed)
    if not calls:
        return False
    message["tool_calls"] = calls
    message["content"] = leftover
    return True


class TextCallSniffer:
    """Holds the first content chars of an SSE message until it knows what they are.

    A leaked call starts at the very beginning of `content` and arrives as many
    ordinary deltas, so those deltas cannot be forwarded while they may still
    turn out to be a call. The hold is one short prefix, once per message.
    """

    MARK = "<tool_call>"

    def __init__(self, allowed: set):
        self.allowed = allowed
        self.state = "sniff"       # sniff -> pass, or sniff -> collect -> pass
        self.held = ""
        self.calls = None

    def feed(self, text: str) -> str:
        """The part of `text` that may go out now."""
        if self.state == "pass":
            return text
        self.held += text
        while True:
            if self.state == "sniff":
                probe = self.held.lstrip()
                if not probe:
                    return ""
                if probe.startswith(self.MARK):
                    self.state = "collect"
                    continue
                if len(probe) <= len(self.MARK) and self.MARK.startswith(probe):
                    return ""
                out, self.held, self.state = self.held, "", "pass"
                return out
            if self.state == "collect":
                if "</tool_call>" not in self.held:
                    return ""
                calls, leftover = parse_text_calls(self.held, self.allowed)
                if not calls:
                    out, self.held, self.state = self.held, "", "pass"
                    return out
                self.calls, self.held, self.state = calls, "", "pass"
                return leftover
            return text

    def pending(self) -> str:
        """Whatever is still held (a truncated markup) — flush at end of message."""
        out, self.held = self.held, ""
        return out

    def take(self):
        calls, self.calls = self.calls, None
        return calls


def apply_route_knobs(body: dict, route: str) -> dict:
    """Per-route sampling knobs: thinking budget, a token floor, a temperature floor."""
    msgs = body.get("messages") or []
    guarded = False
    if route in ("tools", "deep") and body.get("tools") and TOOL_LOOP_MAX_RESULTS > 0:
        used = tool_result_count(msgs)
        if used >= TOOL_LOOP_MAX_RESULTS:
            # Enough evidence: answer with it — but keep the schemas. Dropping
            # them (or tool_choice:"none", which the server turns into the same
            # thing) is what leaks the native `<tool_call>` markup into `content`
            # and kills the client's run; see TOOL_LOOP_MAX_RESULTS. The ask goes
            # in a message at the END of the conversation, so the cached prefix
            # stays byte-identical and only the new line is prefilled.
            note = LOOP_GUARD_NOTE % used if "%d" in LOOP_GUARD_NOTE else LOOP_GUARD_NOTE
            body["messages"] = list(msgs) + [{"role": "user", "content": note}]
            if body.get("tool_choice") in ("required", "any"):
                body["tool_choice"] = "auto"
            guarded = True
            log("loop guard: %d tool results already, answering with the schemas kept" % used)
    if route == "tools" and (guarded or awaiting_tool_answer(msgs)):
        # Budget is only injected when the client has not chosen one, same rule as
        # every other route.
        if "reasoning_budget_tokens" not in body and "thinking_budget_tokens" not in body:
            body["reasoning_budget_tokens"] = TOOLS_SYNTH_BUDGET
    if "reasoning_budget_tokens" not in body and "thinking_budget_tokens" not in body:
        budget = ROUTE_BUDGET.get(route)
        if budget is not None:
            body["reasoning_budget_tokens"] = budget
    if route == "tools" or (route == "deep" and body.get("tools")):
        # A cold sampler (0.2, the usual client default) collapses this model onto the
        # prose continuation instead of the tool-call branch: measured on the 7-case
        # battery at seed 1234, temperature 0.2 scored 5/7 identically on budgets
        # 32/64/96/192, while 0.6 scored 7/7 at budget 32 — the Telegram case emitted
        # no call at all and the joke case called web_search. So raise, never lower:
        # a client that asks for 0.8 or 1.0 keeps it.
        temp = body.get("temperature")
        if isinstance(temp, (int, float)) and temp < TOOLS_MIN_TEMPERATURE:
            body["temperature"] = TOOLS_MIN_TEMPERATURE
            log("raised temperature %.2f -> %.2f on a tool turn"
                % (float(temp), TOOLS_MIN_TEMPERATURE))
        # Only raise a ceiling the client actually set; never introduce a second
        # token-limit field next to one that is already there.
        fields = [f for f in ("max_tokens", "max_completion_tokens") if f in body] or ["max_tokens"]
        for field in fields:
            value = body.get(field)
            if not isinstance(value, int) or value < TOOLS_MIN_MAX_TOKENS:
                body[field] = TOOLS_MIN_MAX_TOKENS
    return body


def session_key(headers, body: dict, client_ip: str) -> str:
    sid = headers.get("X-Session-Id")
    if sid:
        return "sid:" + sid
    msgs = body.get("messages") or []
    first_user = ""
    for m in msgs:
        if m.get("role") == "user":
            first_user = message_text([m])[:400]
            break
    seed = (first_user or "") + "\x00" + client_ip
    return "h:" + hashlib.sha1(seed.encode()).hexdigest()[:16]


def looks_deep(text: str) -> bool:
    """A request worth 3.5x the latency: long *and* asking for reasoning.

    Short prompts are never escalated, however technical they look: at 2.91 tok/s
    the quality tier costs 40 s before the first word, and a one-liner does not
    need it. The thresholds match the `deep` rule in _decide_cold.
    """
    if not DEEP_ENABLED or not text:
        return False
    if not (REASON_RE.search(text) or TECH_RE.search(text)):
        return False
    return len(text) > DEEP_MIN_CHARS or len(text.split()) >= DEEP_MIN_WORDS


def decide(body: dict, headers, client_ip: str) -> tuple[str, str]:
    """Return (route, reason)."""
    msgs = body.get("messages") or []
    text = message_text(msgs)
    key = session_key(headers, body, client_ip)

    # A session that already went deep stays there. Switching back to the MoE
    # re-reads 4.8 GB from storage and throws away the quality tier's prompt
    # cache, and a follow-up ("ok, e il caso n=3?") is the same job as the turn
    # before it.
    sticky = sticky_get(key)
    if sticky == "deep" and DEEP_ENABLED:
        return "deep", "sticky-deep"

    # 0a. an image turn goes to the vision tier even when the client attached
    #     tools: LFM3-VL handles tools (6/7 on the battery), while neither text
    #     model can see the picture at all -- before this branch Open WebUI's
    #     global tool injection sent every image to the MoE, blind. A session
    #     that already ran tools stays where it is: the model that issued the
    #     call is the one that has to read the result back.
    if has_image(msgs) and not has_tool_history(msgs):
        route, reason = _decide_cold(msgs, text, body)
        sticky_put(key, route)
        return route, reason

    # 0b. agent turns win over everything else, including stickiness: a conversation
    #    that started as chitchat and now carries tools must not stay on the fast tier,
    #    and a tool loop must keep hitting the model that issued the call (it is the
    #    model that has to interpret its own tool results, and the one whose prompt
    #    cache holds the tool schema block).
    if has_tools(body) or has_tool_history(msgs):
        # The deep gate is judged on the last user turn, not on the concatenation:
        # Open WebUI sends a long system prompt with the tool surface on every turn,
        # so `text` is always over any length threshold and always mentions HTTP,
        # docker and tokens. That made the gate depend on how wordy the client's
        # system prompt was instead of on the question.
        ask = last_user_text(msgs)
        # ...except the FIRST turn of a session that is plainly a deep reasoning
        # request: Open WebUI attaches the flat tool surface to every request, so
        # without this branch `auto` would never reach the quality tier at all.
        # Only on the first turn: once the session is pinned, the branch above
        # keeps every further turn (tool results included) on the same model.
        # An image request stays on the vision tier: `bolt` has no vision, and
        # the deep rule would otherwise swallow the picture.
        if (sticky is None and looks_deep(ask) and not has_tool_history(msgs)
                and not has_image(msgs)):
            sticky_put(key, "deep")
            deep_mark(key)
            return "deep", "deep-first-turn"
        # A loop that has already issued a call is not re-homed. The decision is
        # made once, at the front of the loop; after that a switch is the most
        # expensive thing this router can do. Measured on this board: a cold load
        # is 88-120 s per model (22 s only for the 1 GB fast tier), and switching
        # also throws away the prompt cache that holds the tool-schema block, so
        # the next turn re-reads ~1.5 k tokens at ~20 tok/s (~75 s) instead of the
        # 4-6 s a warm continuation costs. The model that wrote the call is the one
        # that has to read its result back.
        if sticky and has_tool_history(msgs):
            return sticky, "sticky-loop"
        # Mid-chat escalation. A chat whose earlier turns were ordinary ("ciao, che
        # giorno e' oggi?") is pinned to the MoE, and Open WebUI's tool injection
        # means the tools branch above owns every later turn: before this, a deep
        # question on turn 4 could never reach `bolt`, only turn 1 could.
        # Guarded on all four sides: it must still be the front of a turn (no tool
        # result in flight -- that model has to read its own result back), no image
        # (`bolt` is blind), the last user message itself must be the deep one, and
        # each session may escalate only once, so a TTL expiry cannot start a
        # bolt <-> MoE swap loop.
        if (DEEP_ESCALATE and sticky != "deep" and not has_tool_history(msgs)
                and not has_image(msgs) and looks_deep(ask) and not deep_seen(key)):
            sticky_put(key, "deep")
            deep_mark(key)
            log("escalating session %s: %s -> deep for this turn (one cold load, "
                "~100 s, 2.9 tok/s from here)" % (key[:12], sticky or "cold"))
            return "deep", "deep-escalation"
        if sticky and sticky != "tools":
            # First tool turn of a session that started elsewhere: this switch is
            # paid once per session, and it is the last one (sticky-loop holds).
            log("switching session %s: %s -> tools (one cold load, ~90 s, and the "
                "prompt cache is lost)" % (key[:12], sticky))
        sticky_put(key, "tools")
        return "tools", "tools"

    if sticky:
        return sticky, "sticky"

    route, reason = _decide_cold(msgs, text, body)
    sticky_put(key, route)
    return route, reason


def _decide_cold(msgs: list[dict], text: str, body: dict) -> tuple[str, str]:
    low = text.lower().strip()
    words = len(low.split())

    # 1. images
    if has_image(msgs):
        if OCR_RE.search(text) or OCR_RE.search(json.dumps(msgs)[:4000]):
            return "vision-docs", "image+ocr-hint"
        return "vision", "image"

    # 2. code
    if CODE_RE.search(text):
        # asking *about* a two-line snippet is a fine job for the tiny tier; asking
        # it to *write* code is not, no matter how short the sentence is.
        tiny_ok = words < 12 and len(text) < 160 and not GENERATE_RE.search(text)
        return ("fast", "code-short") if tiny_ok else ("reason", "code")

    # 3. reasoning / design
    # 18 words / 160 chars, not 25/200: "Spiega passo per passo come funziona il
    # protocollo TCP e perché il three-way handshake è necessario, con un esempio
    # concreto di connessione." is 24 words / 145 chars and fell through to the
    # default route, where the chat budget cut its thought and leaked it into content.
    if (DEEP_ENABLED and REASON_RE.search(text)
            and (words >= DEEP_MIN_WORDS or len(text) > DEEP_MIN_CHARS)):
        return "deep", "reasoning-deep"
    if REASON_RE.search(text) and (words > 18 or len(text) > 160):
        return "reason", "reasoning"

    # 4. summarization of a long input
    if SUMMARY_RE.search(text) and (len(text) > 2500 or words > 400):
        return "reason", "summarize-long"

    # 5a. small talk: greetings, thanks, "are you there"
    if words <= 10 and CHITCHAT_RE.search(low) and not EXPLAIN_RE.search(text):
        return "fast", "chitchat"

    # 5b. trivial / short factual
    if (
        words <= 12
        and len(text) <= 90
        and SHORT_HINT_RE.search(low)
        and not EXPLAIN_RE.search(text)
        and not TECH_RE.search(text)
        # facts and named entities are not trivia: a 1.7B gets them wrong
        and not FACTUAL_RE.search(text)
        and not has_proper_noun(text)
    ):
        return "fast", "trivial"

    # 6. translate of short strings -> fast tier is enough
    if TRANSLATE_RE.search(text) and words <= 40:
        return "fast", "translate-short"

    # 7. semantic similarity against prototypes
    sem_route, score, margin = PROTO.best_route(text)
    if sem_route == "fast" and (words > 12 or EXPLAIN_RE.search(text) or TECH_RE.search(text)):
        # a short-answer prototype must not capture an explanation request
        return DEFAULT_ROUTE, "semantic-fast-rejected"
    if sem_route and score >= SEM_THRESHOLD and margin >= SEM_MIN_MARGIN:
        return sem_route, "semantic(%.2f/%.2f)" % (score, margin)

    # 8. default: the MoE text workhorse (8.4B weights, 1.1B active, 10.7 tok/s)
    return DEFAULT_ROUTE, "default"


# -------------------------------------------------------------------- server ---
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llm-router/1.0"

    def log_message(self, fmt, *args):  # quieter than the default access log
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    # -- helpers ---------------------------------------------------------------
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _upstream(self, method: str, path: str):
        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=900)
        fwd = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        for h in ("Accept", "Authorization"):
            if self.headers.get(h):
                fwd[h] = self.headers[h]
        return conn, fwd

    # -- endpoints -------------------------------------------------------------
    def do_GET(self):
        if self.path.split("?")[0] in ("/v1/models", "/models"):
            return self._models()
        if self.path.startswith(("/props", "/slots", "/metrics")):
            return self._passthrough("GET", self.path)
        if self.path == "/health":
            body = {"status": "ok", "upstream": "http://%s:%d" % (UPSTREAM_HOST, UPSTREAM_PORT),
                    "routes": ROUTE_MODEL, "budgets": ROUTE_BUDGET,
                    "tools_synth_budget": TOOLS_SYNTH_BUDGET,
                    "tool_result_max_chars": TOOL_RESULT_MAX_CHARS,
                    "tool_loop_max_results": TOOL_LOOP_MAX_RESULTS,
                    "repair_text_calls": REPAIR_TEXT_CALLS,
                    "deep_tier": DEEP_ENABLED,
                    "deep_min_chars": DEEP_MIN_CHARS,
                    "deep_min_words": DEEP_MIN_WORDS,
                    "deep_escalate": DEEP_ESCALATE,
                    "context_length": CONTEXT_LENGTH,
                    "prototypes_ready": PROTO.ready.is_set()}
            if not upstream_ok("/health"):
                body["status"] = "degraded"
                body["upstream_error"] = "llama-swap did not answer /health"
            return self._json(200 if body["status"] == "ok" else 503, body)
        if self.path == "/stats":
            with _stats_lock:
                return self._json(200, {"routes": dict(_stats), "sticky_sessions": len(_sticky),
                                        "prototypes_ready": PROTO.ready.is_set()})
        return self._json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self):
        raw = self._read_body()
        path = self.path
        if path.startswith("/v1/embeddings"):
            # embeddings always go to the always-on embedding model
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                return self._json(400, {"error": {"message": "invalid json"}})
            body["model"] = "embedding-model"
            return self._forward(json.dumps(body).encode(), path, "embedding", "embedding", "forced", 0.0)

        if path.startswith(("/v1/chat/completions", "/v1/completions", "/api/chat")):
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                return self._json(400, {"error": {"message": "invalid json"}})
            cut = cap_tool_results(body.get("messages") or [])
            if cut:
                log("capped %d tool result(s) at %d chars" % (cut, TOOL_RESULT_MAX_CHARS))
            # Names the answer may legitimately turn back into a call (see
            # parse_text_calls); empty means no repair for this request.
            allowed = known_tools(body) if REPAIR_TEXT_CALLS else set()
            requested = str(body.get("model") or "auto")
            if requested and requested.lower() not in ("auto", "router", "orchestrator"):
                # A concrete id is forwarded verbatim (that is how a client debugs one
                # tier), with one exception: a tool request must never land on the fast
                # tier. LFM3-1.7B does emit calls (7/9 at Q4_0) but the 8B MoE is
                # 9/9 at the same per-call latency.
                # Tool orchestration lives on the MoE, so this is not a downgrade.
                if requested.lower() in TINY_MODELS and (
                    has_tools(body) or has_tool_history(body.get("messages") or [])
                ):
                    model = ROUTE_MODEL["tools"]
                    body["model"] = model
                    body = apply_route_knobs(body, "tools")
                    body = add_stop_markers(normalize_reasoning(body))
                    bump("tools")
                    log("route=tools model=%s by=tools-on-tiny (client asked %s)"
                        % (model, requested))
                    return self._forward(json.dumps(body).encode(), path, model, "tools",
                                         "tools-on-tiny", 0.0,
                                         body.get("reasoning_budget_tokens"), allowed)
                return self._forward(raw, path, requested, requested, "client", 0.0, None,
                                     allowed)

            t0 = time.time()
            route, why = decide(body, self.headers, self.address_string())
            model = ROUTE_MODEL.get(route, ROUTE_MODEL["assistant"])
            body["model"] = model
            body = apply_route_knobs(body, route)
            if route != "fast":
                body = add_stop_markers(normalize_reasoning(body))
            if route == "fast":
                body = keep_language(body)
            elapsed = (time.time() - t0) * 1000.0
            bump(route)
            log("route=%s model=%s by=%s budget=%s max_tokens=%s %.0fms"
                % (route, model, why, body.get("reasoning_budget_tokens"),
                   body.get("max_tokens") or body.get("max_completion_tokens"), elapsed))
            if DEBUG_BODY:
                log("forwarded: %s" % json.dumps(body, ensure_ascii=False)[:2000])
            return self._forward(json.dumps(body).encode(), path, model, route, why, elapsed,
                                 body.get("reasoning_budget_tokens"), allowed)

        # everything else: transparent
        return self._forward(raw, path, "passthrough", "passthrough", "passthrough", 0.0)

    # -- proxying --------------------------------------------------------------
    def _forward(self, raw: bytes, path: str, model: str, route: str, why: str, ms: float,
                 budget=None, allowed: set | None = None):
        conn, headers = self._upstream("POST", path)
        try:
            conn.request("POST", path, body=raw, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:
            conn.close()
            return self._json(502, {"error": {"message": "llama-swap unreachable: %s" % exc,
                                              "type": "upstream_error"}})

        self.send_response(resp.status)
        ctype = resp.getheader("Content-Type", "application/json")
        self.send_header("Content-Type", ctype)
        self.send_header("X-Route", route)
        self.send_header("X-Route-Model", model)
        self.send_header("X-Route-By", why)
        self.send_header("X-Route-Ms", "%.1f" % ms)
        if budget is not None:
            self.send_header("X-Route-Budget", str(budget))
        self.send_header("X-Accel-Buffering", "no")
        streaming = "text/event-stream" in ctype or resp.getheader("Transfer-Encoding") == "chunked"
        if streaming:
            # connection-close framing: simplest reliable way to relay SSE verbatim
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            try:
                if allowed:
                    self._relay_repair(resp, allowed)
                else:
                    while True:
                        chunk = resp.read(2048)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                conn.close()
            return

        data = resp.read()
        conn.close()
        if allowed:
            data = self._repair_json(data, allowed)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _repair_json(self, data: bytes, allowed: set) -> bytes:
        """Non-streaming: `content` holding a text tool call becomes `tool_calls`."""
        try:
            payload = json.loads(data or b"{}")
        except Exception:
            return data
        fired = 0
        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if repair_message(choice.get("message") or {}, allowed):
                choice["finish_reason"] = "tool_calls"
                fired += 1
        if not fired:
            return data
        log("repaired %d text tool call(s) into tool_calls" % fired)
        return json.dumps(payload, ensure_ascii=False).encode()

    def _relay_repair(self, resp, allowed: set) -> None:
        """SSE relay that rewrites a leaked `<tool_call>` block into a real call.

        Everything else is byte-identical to the old pass-through relay: the only
        thing this touches is `choices[].delta.content`, and only while the first
        chars are still undecided (see TextCallSniffer).
        """
        sniffer = TextCallSniffer(allowed)
        while True:
            line = resp.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped.startswith(b"data:"):
                self.wfile.write(line)
                self.wfile.flush()
                continue
            payload = stripped[5:].strip()
            if payload == b"[DONE]":
                calls, tail = sniffer.take(), sniffer.pending()
                if calls or tail:
                    self.wfile.write(self._sse_chunk({"content": tail} if tail else {}, calls))
                self.wfile.write(line)
                self.wfile.flush()
                continue
            try:
                event = json.loads(payload)
            except Exception:
                self.wfile.write(line)
                self.wfile.flush()
                continue
            if self._rewrite_event(event, sniffer):
                line = b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n"
            self.wfile.write(line)
            self.wfile.flush()

    @staticmethod
    def _sse_chunk(delta: dict, calls=None) -> bytes:
        body = dict(delta)
        if calls:
            body["tool_calls"] = calls
        event = {"choices": [{"index": 0, "delta": body,
                              "finish_reason": "tool_calls" if calls else None}]}
        return b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n"

    def _rewrite_event(self, event: dict, sniffer: TextCallSniffer) -> bool:
        """Patch one SSE event in place; True when it must be re-serialised."""
        touched = False
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                keep = sniffer.feed(delta["content"])
                if keep != delta["content"]:
                    touched = True
                    if keep:
                        delta["content"] = keep
                    else:
                        delta.pop("content", None)
            finishing = bool(choice.get("finish_reason"))
            calls = sniffer.take() if finishing else None
            tail = sniffer.pending() if finishing else ""
            if calls or tail:
                body = dict(choice.get("delta") or {})
                if tail:
                    body["content"] = (body.get("content") or "") + tail
                if calls:
                    body["tool_calls"] = calls
                    choice["finish_reason"] = "tool_calls"
                    log("repaired a text tool call into tool_calls (%s)"
                        % (calls[0]["function"]["name"]))
                else:
                    body["content"] = tail
                choice["delta"] = body
                touched = True
        return touched

    def _models(self):
        """Advertise llama-swap's models plus the orchestrator itself.

        Clients pick from this list, and a concrete id is forwarded verbatim, so
        without an `auto` entry every request would bypass routing. The entry
        declares function calling and vision because the orchestrator may pick a
        model that has them.
        """
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=60)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            data = json.loads(resp.read().decode() or "{}")
            conn.close()
        except Exception as exc:
            return self._json(502, {"error": {"message": "llama-swap unreachable: %s" % exc,
                                              "type": "upstream_error"}})
        models = data.get("data")
        if isinstance(models, list) and not any(
                isinstance(m, dict) and m.get("id") == "auto" for m in models):
            models.insert(0, {
                "id": "auto",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cheap-llm-stack",
                "name": "auto · orchestrator (tools + vision, routes per request)",
                "context_length": CONTEXT_LENGTH,
                "capabilities": {"function_calling": True, "vision": True, "embeddings": False},
                "supported_parameters": ["tools", "tool_choice", "parallel_tool_calls",
                                         "reasoning_budget_tokens", "max_tokens", "stream"],
            })
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _passthrough(self, method: str, path: str):
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=60)
            conn.request(method, path)
            resp = conn.getresponse()
            data = resp.read()
            conn.close()
        except Exception as exc:
            return self._json(502, {"error": {"message": str(exc), "type": "upstream_error"}})
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


def main() -> None:
    log("listening on %s:%d -> llama-swap %s:%d" % (LISTEN_HOST, LISTEN_PORT, UPSTREAM_HOST, UPSTREAM_PORT))
    if WARM_ENABLED:
        threading.Thread(target=warm_loop, name="warm", daemon=True).start()
    threading.Thread(target=PROTO.build, name="proto-build", daemon=True).start()
    Server((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
