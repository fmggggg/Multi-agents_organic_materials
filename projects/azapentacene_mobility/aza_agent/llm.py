"""Minimal OpenAI Responses-API client with function tools (stdlib only), same pattern as
poreflow/scripts/agent/llm.py. Keys are read from files (config.py; never logged or committed).
Every call appends token usage to a ledger."""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import OPENAI_KEY_FILE, OPENROUTER_KEY_FILE

KEY = Path(OPENAI_KEY_FILE)


def respond(model, input_items, instructions=None, tools=None, prev=None, ledger=None, effort=None, tag="",
            retries=5, timeout=900):
    """-> (response id, text, [(call_id, name, args dict)])"""
    body = dict(model=model, input=input_items)
    if instructions:
        body["instructions"] = instructions
    if tools:
        body["tools"] = [dict(type="function", **t) for t in tools]
    if prev:
        body["previous_response_id"] = prev
    if effort:
        body["reasoning"] = dict(effort=effort)
    key = os.environ.get("OPENAI_API_KEY") or KEY.read_text().strip()
    req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:500]
            if (e.code == 429 or e.code >= 500) and k < retries - 1:  # incl. Cloudflare 520-524
                time.sleep(20 * (k + 1))
                continue
            raise RuntimeError(f"OpenAI HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError):
            if k == retries - 1:
                raise
            time.sleep(20 * (k + 1))
    if ledger:
        u = out.get("usage") or {}
        with open(ledger, "a") as h:
            h.write(json.dumps(dict(t=time.time(), model=model, tag=tag, input=u.get("input_tokens"),
                                    output=u.get("output_tokens"),
                                    cached=(u.get("input_tokens_details") or {}).get("cached_tokens"),
                                    reasoning=(u.get("output_tokens_details") or {}).get("reasoning_tokens"))) + "\n")
    text, calls = "", []
    for it in out.get("output", []):
        if it["type"] == "message":
            text += "".join(c.get("text", "") for c in it["content"] if c["type"] == "output_text")
        elif it["type"] == "function_call":
            calls.append((it["call_id"], it["name"], json.loads(it["arguments"] or "{}")))
    return out["id"], text, calls


# ---------------------------------------------------------------- OpenRouter (OpenAI-compatible Chat Completions)
OR_KEY = Path(OPENROUTER_KEY_FILE)
OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat(model, messages, tools=None, ledger=None, effort=None, tag="", retries=4, timeout=600):
    """One Chat Completions call. -> (assistant message dict to append, text, [(call_id, name, args or None, raw)])
    Free models allow ~50 requests/day on a free-tier key, so retries are few and only for transient errors."""
    body = dict(model=model, messages=messages)
    if tools:
        body["tools"] = [dict(type="function", function=t) for t in tools]
    if effort:
        body["reasoning"] = dict(effort=effort)
    key = os.environ.get("OPENROUTER_API_KEY") or OR_KEY.read_text().strip()
    req = urllib.request.Request(OR_URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "X-Title": "aza-agent"})
    out = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            if "error" in out and not out.get("choices"):  # provider error delivered with HTTP 200
                raise urllib.error.HTTPError(OR_URL, int(out["error"].get("code") or 502), str(out["error"])[:500],
                                             None, None)
            break
        except urllib.error.HTTPError as e:
            try:
                msg = e.read().decode()[:500]
            except Exception:
                msg = str(e.msg)[:500]
            daily = "free-models-per-day" in msg
            if (e.code == 429 or e.code >= 500) and not daily and k < retries - 1:
                time.sleep(30 * (k + 1))
                continue
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError):
            if k == retries - 1:
                raise
            time.sleep(30 * (k + 1))
    if ledger:
        u = out.get("usage") or {}
        with open(ledger, "a") as h:
            h.write(json.dumps(dict(t=time.time(), model=model, served_by=out.get("provider"), tag=tag,
                                    input=u.get("prompt_tokens"), output=u.get("completion_tokens"),
                                    cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                                    reasoning=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                    cost=u.get("cost"))) + "\n")
    m = out["choices"][0]["message"]
    calls = []
    for c in m.get("tool_calls") or []:
        raw = c["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = None
        calls.append((c["id"], c["function"]["name"], args, raw))
    keep = dict(role="assistant", content=m.get("content") or "")
    if m.get("tool_calls"):
        keep["tool_calls"] = m["tool_calls"]
    return keep, m.get("content") or "", calls
