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
  1. explicit model in the request  -> pass through untouched (no routing at all)
  2. images present                 -> assistant (VLM, resident) or vision-docs (OCR/table)
  3. deterministic keyword rules     -> code / translate / summarize / reason / trivial
  4. embedding similarity            -> reuses the always-on embedding-model, 0 extra RAM
  5. default                         -> reasoner (MoE workhorse)

The thinking model ("deep") is never chosen automatically: at 2.9 tok/s a bounded
think costs ~90 s, so it must be an explicit `model: deep`. Because it cannot share
the board with the MoE (measured: 5 GB of the MoE pushed into zram), the router
unloads whichever of the two is not about to run — the group model in llama-swap
cannot do it, the MoE is `persistent` and so exempt from `heavy`'s swap.

Every route is sticky per session (X-Session-Id header, else a hash of the first
user message, else client IP+UA) for STICKY_TTL seconds. Stickiness matters: a
model swap costs 1-40 s depending on page-cache state, so a conversation must not
bounce between models.

Response headers: X-Route, X-Route-Model, X-Route-By, X-Route-Ms.
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

STICKY_TTL = int(os.environ.get("ROUTER_STICKY_TTL", "1800"))
SEM_THRESHOLD = float(os.environ.get("ROUTER_SEM_THRESHOLD", "0.45"))
SEM_MIN_MARGIN = float(os.environ.get("ROUTER_SEM_MARGIN", "0.02"))
MAX_STICKY = 1024

DEFAULT_ROUTE = "assistant"

# route -> llama-swap model id
# "assistant" and "reason" both land on the MoE: it is the best text model on the
# board *and* the fastest of the big ones, so there is nothing to split between a
# small default and a quality escalation for text.
ROUTE_MODEL = {
    "fast": "router-0.6b",
    "assistant": "reasoner",
    "reason": "reasoner",
    "deep": "deep",
    "vision": "assistant",
    "vision-docs": "vision-docs",
    "embedding": "embedding-model",
}

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
CODE_RE = re.compile(
    r"```|~~|\bdef\s+\w+\s*\(|\bclass\s+\w+|\bfunction\s+\w+\(|\bimport\s+\w+|#include"
    r"|\bTraceback \(most recent call last\)|\b\w+Error:|\bsegfault\b|\bnull pointer\b"
    r"|\bSELECT\b.+\bFROM\b|\bCREATE TABLE\b|\bgit\s+(commit|rebase|merge|cherry-pick)\b"
    r"|\bdocker(-compose)?\s+(run|build|up)\b|\bkubernetes\b|\bregex\b|\bunit test\b"
    r"|\bAPI endpoint\b|\bpython\b|\brust\b|\bgolang\b|\btypescript\b|\bjavascript\b"
    r"|\bSQL\b|\bcmake\b|\bmakefile\b|\bsystemd\b",
    re.IGNORECASE,
)
REASON_RE = re.compile(
    r"\b(prove|derive|step[ -]by[ -]step|trade-?offs?|architect(ure|ural)|root cause"
    r"|algorithm|complexity|optimi[sz]e|refactor|benchmark|dimostra|dimostrazione"
    r"|passo passo|architettura|complessità|ottimizza|rifattorizza|causa (radice|del problema))\b",
    re.IGNORECASE,
)
TRANSLATE_RE = re.compile(r"\b(translate|traduci|traduzione|in inglese|in italiano)\b", re.IGNORECASE)
SUMMARY_RE = re.compile(r"\b(summar(y|ize|ise)|tl;?dr|riassum(i|ere|to)|sintesi)\b", re.IGNORECASE)
OCR_RE = re.compile(
    r"\b(ocr|screenshot|receipt|invoice|scanned|scan\b|table|spreadsheet|csv|form\b|document"
    r"|scontrino|ricevuta|fattura|tabella|foglio|modulo|documento)\b",
    re.IGNORECASE,
)
SHORT_HINT_RE = re.compile(
    r"^(what|who|when|where|which|how much|how many|define|convert|translate|capital of"
    r"|che|chi|quando|dove|quanto|quanti|qual|quale|quali|definisci|converti)\b",
    re.IGNORECASE,
)
# Greetings, thanks, "are you there": the only requests where 0.6B is unambiguously
# good enough. Kept narrow on purpose — small talk is cheap only if it stays small.
CHITCHAT_RE = re.compile(
    r"^\W*(ciao|salve|hey|hi|hello|buongiorno|buonasera|grazie|thanks|thank you|ok|okay"
    r"|sei (li|connesso)|ci sei|come stai|how are you|che ore|what time)\b",
    re.IGNORECASE,
)
# Asking the model to *produce* or *change* code is never a 0.6B job, however short
# the sentence: "scrivi una funzione python ..." is 9 words and was routed to the
# tiny tier, which then wrote broken docstring prose.
GENERATE_RE = re.compile(
    r"\b(write|create|implement|generate|refactor|fix|add|rewrite|convert .* (to|into)"
    r"|scrivi|crea|implementa|genera|rifattorizza|correggi|aggiungi|riscrivi|sistema)\b",
    re.IGNORECASE,
)
# Anything that asks for an explanation, a comparison or a why/how does NOT belong on
# the 0.6B: asked to explain a MoE it answered "Un MoE (Moyen d'Équipe)" — 2 s of
# latency saved are not worth a hallucinated acronym.
EXPLAIN_RE = re.compile(
    r"\b(explain|describe|summar|compare|difference|why|how (does|do|is|are)|tell me about"
    r"|spiega|spiegami|descrivi|riassum|confronta|differenza|perch[eé]|come (funziona|si)\b)"
    r"|\bcos'?\s?[eè] (un|una|il|la|lo|i|gli|le|l')\b|\bche cos'?[eè]\b|\bwhat(?:'s| is) a\b",
    re.IGNORECASE,
)
# Vocabulary that turns even a 5-word question into a job for the MoE: the 0.6B
# invents acronyms when asked about anything technical ("MoE = Model of Explanation").
TECH_RE = re.compile(
    r"\b(moe|llm|model of experts|mixture of experts|transformer|attention|embedding|token"
    r"|quantizzazion\w*|quantization|gguf|lora|qlora|kv[- ]cache|context window|thread|mmap"
    r"|npu|gpu|cpu|ram|vram|swap|zram|docker|kubernetes|systemd|api|http|tls|vpn|ssh|dns"
    r"|regex|sql|git|python|rust|golang|javascript|typescript|kernel|filesystem)\b",
    re.IGNORECASE,
)


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


# ------------------------------------------------------------ memory policy ---
# The MoE (5 GB of weights, copied into anonymous memory) and the 4B (3.4 GB of
# weights + ~1.5 GB of compute buffers) do not fit together in 12 GB. Measured on
# the board with both resident: VmSwap of the MoE = 5.0 GB, i.e. the kernel pushed
# the *whole workhorse* into zram and every later MoE request crawled. llama-swap
# groups cannot express this exclusion on their own, because the MoE lives in the
# persistent `core` group and is therefore exempt from `heavy`'s swap. The router
# is the only component that knows which model is about to run, so it unloads the
# other half of the pair first: `deep` is opt-in, and paying a 25 s MoE reload
# after an explicit 170 s thinking request is the cheap side of that trade.
HEAVY_ALTERNATIVES = {"reasoner": ("deep",), "deep": ("reasoner",)}
_room_lock = threading.Lock()


def running_models() -> set[str]:
    """Cheap: llama-swap answers from its own state, no model is touched."""
    try:
        return {m["model"] for m in upstream_json("/running", timeout=5).get("running", [])}
    except Exception:
        return set()


def make_room(model: str) -> None:
    others = HEAVY_ALTERNATIVES.get(model)
    if not others:
        return
    with _room_lock:  # serialise: two parallel requests must not fight over the pair
        running = running_models()
        for other in others:
            if other not in running:
                continue
            try:
                req = urllib.request.Request(
                    "http://%s:%d/api/models/unload/%s" % (UPSTREAM_HOST, UPSTREAM_PORT, other),
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=180).close()
                log("room: unloaded %s to make space for %s" % (other, model))
            except Exception as exc:
                log("room: could not unload %s: %s" % (other, exc))


# --------------------------------------------------------------- language -----
# The 0.6B drifts between languages: asked "grazie, sei gentile" in Italian it
# answered in Spanish. One short system line fixes it and costs ~20 prompt tokens.
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


def has_image(messages: list[dict]) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for chunk in c:
                if isinstance(chunk, dict) and chunk.get("type") in ("image_url", "input_image", "image"):
                    return True
    return False


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


def decide(body: dict, headers, client_ip: str) -> tuple[str, str]:
    """Return (route, reason)."""
    msgs = body.get("messages") or []
    text = message_text(msgs)
    key = session_key(headers, body, client_ip)

    sticky = sticky_get(key)
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
    if REASON_RE.search(text) and (words > 25 or len(text) > 200):
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
        if self.path.startswith(("/v1/models", "/props", "/slots", "/metrics")):
            return self._passthrough("GET", self.path)
        if self.path == "/health":
            body = {"status": "ok", "upstream": "http://%s:%d" % (UPSTREAM_HOST, UPSTREAM_PORT),
                    "routes": ROUTE_MODEL, "prototypes_ready": PROTO.ready.is_set()}
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
            requested = str(body.get("model") or "auto")
            if requested and requested.lower() not in ("auto", "router", "orchestrator"):
                make_room(requested)
                return self._forward(raw, path, requested, requested, "client", 0.0)

            t0 = time.time()
            route, why = decide(body, self.headers, self.address_string())
            model = ROUTE_MODEL.get(route, DEFAULT_ROUTE)
            body["model"] = model
            if route == "fast":
                body = keep_language(body)
            make_room(model)
            elapsed = (time.time() - t0) * 1000.0
            bump(route)
            log("route=%s model=%s by=%s %.0fms" % (route, model, why, elapsed))
            return self._forward(json.dumps(body).encode(), path, model, route, why, elapsed)

        # everything else: transparent
        return self._forward(raw, path, "passthrough", "passthrough", "passthrough", 0.0)

    # -- proxying --------------------------------------------------------------
    def _forward(self, raw: bytes, path: str, model: str, route: str, why: str, ms: float):
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
        self.send_header("X-Accel-Buffering", "no")
        streaming = "text/event-stream" in ctype or resp.getheader("Transfer-Encoding") == "chunked"
        if streaming:
            # connection-close framing: simplest reliable way to relay SSE verbatim
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            try:
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
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
    threading.Thread(target=PROTO.build, name="proto-build", daemon=True).start()
    Server((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
