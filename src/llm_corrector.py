"""Context-aware re-ranking of uncertain predictions with an LLM.

Safety rule: the LLM may only choose among the model's candidate phrases.
Its answer is checked against the candidate IDs; anything else is ignored
and the model's top-1 guess is used instead. The LLM never receives images,
video or landmarks, only phrase text and probabilities.

Providers are called over plain HTTPS with the standard library, so no
vendor SDK is needed: claude, gemini, openai, groq, or a local model via ollama.
"""
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import config

PROMPT = """You are helping a hospital patient who cannot speak and communicates by mouthing
phrases. A lip-reading model produced these candidate phrases:

{candidates}

Recent things the patient said, oldest first:
{history}

Choose the single most likely candidate given the context. You must choose
one of the candidates above and must not invent new phrases.
Reply with JSON only: {{"choice": "<phrase_id>", "reason": "<one short sentence>"}}"""

# Groq chat models to fall back to (in order) if the configured one has been retired.
GROQ_PREFERRED = ["qwen/qwen3", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3", "llama"]

_KEY_ENV = {
    "claude": ["LLM_API_KEY", "ANTHROPIC_API_KEY"],
    "gemini": ["LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["LLM_API_KEY", "OPENAI_API_KEY"],
    "groq": ["LLM_API_KEY", "GROQ_API_KEY"],
    "ollama": [],
}


def build_prompt(candidates, history, phrases):
    cand_lines = "\n".join(
        f'{i}. [{pid}] "{phrases[pid]["text"]}" (probability {p:.2f})'
        for i, (pid, p) in enumerate(candidates, 1))
    hist_lines = "\n".join(f'- "{phrases[pid]["text"]}"' for pid in history if pid in phrases)
    return PROMPT.format(candidates=cand_lines, history=hist_lines or "- (nothing yet)")


def parse_choice(text, candidates, phrases):
    """Return (phrase_id, reason) if the reply names a valid candidate, else None."""
    valid = [pid for pid, _ in candidates]
    data = None
    start = text.find("{")
    while start != -1 and data is None:
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    if isinstance(data, dict):
        choice = str(data.get("choice", "")).strip().strip("[]").upper()
        reason = str(data.get("reason", "")).strip()
        if choice in valid:
            return choice, reason
        for pid in valid:  # the model answered with the phrase text instead of the ID
            if choice.lower() == phrases[pid]["text"].lower():
                return pid, reason
    ids = [pid for pid in valid if re.search(rf"\b{pid}\b", text)]
    if len(ids) == 1:
        return ids[0], ""
    return None


class LLMCorrector:
    def __init__(self, provider=None, model=None, timeout_s=None, api_key=None):
        # .env may choose the provider and model for this laptop (LLM_PROVIDER, LLM_MODEL)
        self.provider = (provider or os.environ.get("LLM_PROVIDER") or config.LLM_PROVIDER).lower()
        if self.provider not in _KEY_ENV:
            raise ValueError(f"Unknown LLM provider {self.provider!r}")
        self.model = model or (os.environ.get("LLM_MODEL") if not provider else None) or config.LLM_MODELS[self.provider]
        self.timeout_s = config.LLM_TIMEOUT_S if timeout_s is None else timeout_s
        self.api_key = api_key or next(
            (os.environ[k] for k in _KEY_ENV[self.provider] if os.environ.get(k)), None)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")

    @property
    def available(self):
        return self.provider == "ollama" or bool(self.api_key)

    @property
    def unavailable_reason(self):
        if self.available:
            return ""
        return f"no API key for {self.provider} (set LLM_API_KEY in .env)"

    # -- public API ---------------------------------------------------------

    def submit(self, candidates, history, phrases):
        """Start a re-ranking call in the background; returns a Future.

        The future resolves to (phrase_id, reason) or None on any failure.
        """
        return self._pool.submit(self._choose, list(candidates), list(history), phrases)

    def choose(self, candidates, history, phrases):
        """Blocking version with the timeout applied. Falls back to top-1."""
        future = self.submit(candidates, history, phrases)
        try:
            result = future.result(timeout=self.timeout_s)
        except FutureTimeout:
            result = None
        if result is None:
            return candidates[0][0], "LLM unavailable or too slow - used top-1", "fallback"
        return result[0], result[1], "llm"

    # -- internals ----------------------------------------------------------

    def _choose(self, candidates, history, phrases):
        prompt = build_prompt(candidates, history[-config.LLM_CONTEXT_PHRASES:], phrases)
        try:
            reply = self._call(prompt)
        except Exception as e:  # network errors, bad keys, retired models...
            print(f"[llm] {self.provider} call failed: {e}")
            return None
        return parse_choice(reply, candidates, phrases)

    def _chat_body(self, prompt):
        body = {"model": self.model, "max_completion_tokens": 120, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}
        if self.provider == "groq":
            # Groq's current models "think" before answering; keep that short or off.
            body["max_completion_tokens"] = 400
            if "qwen" in self.model:
                body["reasoning_effort"] = "none"
            elif "gpt-oss" in self.model:
                body["reasoning_effort"] = "low"
        return body

    def _switch_groq_model(self):
        """Pick a chat model this key can use. Returns True if the model changed."""
        if getattr(self, "_switched", False):
            return False
        self._switched = True
        url = config.LLM_ENDPOINTS["groq"].replace("/chat/completions", "/models")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}",
                                                   "User-Agent": "SilentSpeech/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s + 3) as resp:
                ids = [m["id"] for m in json.loads(resp.read().decode()).get("data", [])]
        except Exception:
            return False
        for prefix in GROQ_PREFERRED:
            match = sorted(i for i in ids if i.startswith(prefix))
            if match and match[-1] != self.model:
                print(f"[llm] Groq model {self.model} is no longer available; using {match[-1]}")
                self.model = match[-1]
                return True
        return False

    def _post(self, url, body, headers):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "SilentSpeech/1.0", **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s + 1) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None

    def _call(self, prompt):
        if self.provider == "claude":
            data = self._post(
                config.LLM_ENDPOINTS["claude"],
                {"model": self.model, "max_tokens": 120,
                 "messages": [{"role": "user", "content": prompt}]},
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
            return "".join(b.get("text", "") for b in data.get("content", []))

        if self.provider in ("openai", "groq"):     # Groq speaks the OpenAI format
            try:
                data = self._post(config.LLM_ENDPOINTS[self.provider], self._chat_body(prompt),
                                  {"Authorization": f"Bearer {self.api_key}"})
            except RuntimeError as e:
                # Providers retire model names often; on Groq, switch to one that exists.
                if self.provider != "groq" or "model_not_found" not in str(e) or not self._switch_groq_model():
                    raise
                data = self._post(config.LLM_ENDPOINTS[self.provider], self._chat_body(prompt),
                                  {"Authorization": f"Bearer {self.api_key}"})
            return data["choices"][0]["message"]["content"] or ""

        if self.provider == "gemini":
            data = self._post(
                config.LLM_ENDPOINTS["gemini"].format(model=self.model),
                {"contents": [{"parts": [{"text": prompt}]}],
                 "generationConfig": {"responseMimeType": "application/json",
                                      "maxOutputTokens": 120,
                                      "thinkingConfig": {"thinkingBudget": 0}}},
                {"x-goog-api-key": self.api_key})
            return data["candidates"][0]["content"]["parts"][0]["text"]

        # ollama: a local model, fully offline and private
        data = self._post(f"{config.OLLAMA_URL}/api/chat",
                          {"model": self.model, "stream": False, "format": "json",
                           "options": {"temperature": 0},
                           "messages": [{"role": "user", "content": prompt}]}, {})
        return data["message"]["content"]

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
