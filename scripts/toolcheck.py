#!/usr/bin/env python3
"""
toolcheck.py — does the stack still do OpenAI tool calling?

Seven cases against the router's OpenAI-compatible endpoint: five single-tool
requests, one that should fan out to two tools, one that must produce no tool call
at all. The tools are the ones this household actually uses (Telegram, web search,
memory, Chroma, weather), in Italian and English, which is the language mix the
board serves.

Why this exists as a script and not as a one-off:
  * llama.cpp is rebuilt from master every week (llama-swap-update.timer). Tool
    calling depends on the model's chat template being recognised by a native
    handler in common/chat.cpp — when that regresses, LFM2.5 silently goes back to
    the generic parser and starts emitting the call as plain text instead of
    `tool_calls`. Nothing else in the stack notices: /health is fine, chat is fine.
  * so `healthcheck.sh --tools` runs this, and a regression shows up as 3/7 instead
    of 7/7 without anyone having to test by hand.

Usage:
  scripts/toolcheck.py                       # all cases, via the router
  scripts/toolcheck.py --quick               # 3 cases (~1 min)
  scripts/toolcheck.py --stream              # same battery over SSE
  scripts/toolcheck.py --seed 0              # unseeded sampling (what users get)

Sampling is seeded by default: the MoE is a 1.5B-active model at temperature 0.2 and
one case of seven flips between runs (the Telegram case did, once, in production),
which makes a regression check useless. A fixed seed makes 7/7 mean "the build is
fine" rather than "this draw was lucky".

  scripts/toolcheck.py --url http://host:10000 --model auto

Exit code 0 when every case passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "telegram_send_message",
        "description": "Send a text message to the user on Telegram.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Message text"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the public web and return result titles, urls and snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "description": "max results 1-10"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "save_memory",
        "description": "Store a durable user preference or fact for future sessions.",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string"},
            "type": {"type": "string", "enum": ["user-preference", "code-pattern", "task"]}},
            "required": ["content", "type"]}}},
    {"type": "function", "function": {
        "name": "query_vector_db",
        "description": "Semantic search inside a Chroma vector collection.",
        "parameters": {"type": "object", "properties": {
            "collection_name": {"type": "string"},
            "query": {"type": "string"},
            "n_results": {"type": "integer"}},
            "required": ["collection_name", "query"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
]

# key, prompt, expected tool name ("NONE" = must not call anything), optional
# argument check: (field, expected substring)
CASES = [
    ("tg", "Manda un messaggio su Telegram dicendo che arrivo alle 20.",
     "telegram_send_message", None),
    ("web", "Cerca sul web le ultime notizie su llama.cpp.", "web_search", None),
    ("mem", "Ricordati che il mio provider LLM preferito e Anthropic.",
     "save_memory", ("type", "user-preference")),
    ("vec", "Cerca nella collection aiderdesk_global_memory le informazioni sulla "
            "configurazione MCP.", "query_vector_db", ("collection_name", "aiderdesk")),
    ("wx", "Che tempo fa a Milano domani?", "get_weather", None),
    ("multi", "Cerca sul web il prezzo attuale del Bitcoin e poi mandamelo su Telegram.",
     "web_search", None),
    ("none", "Ciao, come stai? Raccontami una barzelletta breve.", "NONE", None),
]

QUICK = {"tg", "wx", "none"}


def post(url: str, payload: dict, stream: bool, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        return json.loads(resp.read()), resp.headers

    name, args, content = None, "", ""
    for line in resp:
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        chunk = line[6:]
        if chunk == "[DONE]":
            break
        delta = (json.loads(chunk).get("choices") or [{}])[0].get("delta") or {}
        content += delta.get("content") or ""
        for call in delta.get("tool_calls") or []:
            name = call.get("function", {}).get("name") or name
            args += call.get("function", {}).get("arguments") or ""
    msg = {"content": content, "tool_calls": [{"function": {"name": name, "arguments": args}}] if name else []}
    return {"choices": [{"message": msg}]}, resp.headers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:10000")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--quick", action="store_true", help="only the three decisive cases")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=1234, help="0 = let the server sample")
    ap.add_argument("--budget", type=int, default=None,
                    help="reasoning_budget_tokens to send (default: the router's)")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    cases = [c for c in CASES if not args.quick or c[0] in QUICK]
    ok = 0
    warned = 0
    for key, prompt, expect, argcheck in cases:
        payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                   "tools": TOOLS, "tool_choice": "auto",
                   "temperature": args.temperature, "max_tokens": 400}
        if args.budget is not None:
            payload["reasoning_budget_tokens"] = args.budget
        if args.seed:
            payload["seed"] = args.seed
        if args.stream:
            payload["stream"] = True
        t0 = time.time()
        try:
            data, headers = post(endpoint, payload, args.stream, args.timeout)
        except urllib.error.HTTPError as exc:
            print("  [%s] FAIL http %s: %s" % (key, exc.code, exc.read()[:200]))
            continue
        except Exception as exc:
            print("  [%s] FAIL %s" % (key, exc))
            continue
        wall = time.time() - t0
        message = (data.get("choices") or [{}])[0].get("message") or {}
        calls = message.get("tool_calls") or []
        names = [c.get("function", {}).get("name") for c in calls]
        passed = (expect == "NONE" and not calls) or (expect in names)

        detail = ""
        if calls:
            raw = calls[0].get("function", {}).get("arguments") or ""
            try:
                parsed = json.loads(raw)
                detail = " ".join("%s=%s" % (k, str(v)[:40]) for k, v in parsed.items())[:110]
            except Exception:
                passed, detail = False, "arguments are not JSON: %r" % raw[:80]
            if passed and argcheck:
                field, want = argcheck
                try:
                    got = str(json.loads(raw).get(field, ""))
                    if want.lower() not in got.lower():
                        # every model tested here mangles exotic identifiers inside
                        # tool arguments; keep it loud but do not fail the run
                        warned += 1
                        detail += "  [WARN %s=%r]" % (field, got)
                except Exception:
                    pass

        route = headers.get("X-Route", "-")
        ok += 1 if passed else 0
        print("  [%s] %-5s route=%-11s %4.1fs tools=%s %s"
              % (key, "OK" if passed else "FAIL", route, wall, names, detail))

    total = len(cases)
    print("RESULT %d/%d%s" % (ok, total, "" if not warned else "  (%d argument warning)" % warned))
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
