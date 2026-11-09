"""The LLM layer against a local mock server that speaks each provider's format.

Checks the real HTTP code path (headers, request body, response parsing),
the timeout fallback, error fallback and the never-invent-a-phrase rule,
without needing an API key or internet.
"""
import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.llm_corrector import LLMCorrector  # noqa: E402
from src.utils import load_phrases  # noqa: E402

CANDS = [("P10", 0.48), ("P28", 0.41), ("P11", 0.06)]


class MockLLM(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):          # Groq-style model list
        if self.path.endswith("/models"):
            self._send(200, {"data": [{"id": "whisper-large-v3"}, {"id": "qwen/qwen3.8-27b"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        mode = self.server.mode
        if mode == "slow":
            time.sleep(1.5)
        if mode == "error":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error": "invalid key"}')
            return
        if body.get("model") == "retired-model":
            self._send(404, {"error": {"message": "The model does not exist", "code": "model_not_found"}})
            return
        text = json.dumps({"choice": "P99" if mode == "invent" else "P10", "reason": "pain, so the nurse"})
        if self.path.startswith("/v1/messages"):
            resp = {"content": [{"type": "text", "text": text}]}
        elif "chat/completions" in self.path:
            resp = {"choices": [{"message": {"content": text}}]}
        elif ":generateContent" in self.path:
            resp = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        else:
            resp = {"message": {"content": text}}
        data = json.dumps(resp).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:   # the client already gave up (timeout test)
            pass

    def log_message(self, *args):
        pass


class TestLLMHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockLLM)
        cls.server.requests, cls.server.mode = [], "ok"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.saved = (dict(config.LLM_ENDPOINTS), config.OLLAMA_URL)
        config.LLM_ENDPOINTS.update({
            "claude": base + "/v1/messages",
            "openai": base + "/v1/chat/completions",
            "groq": base + "/openai/v1/chat/completions",
            "gemini": base + "/v1beta/models/{model}:generateContent",
        })
        config.OLLAMA_URL = base
        cls.phrases = load_phrases(phrase_set="full")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        config.LLM_ENDPOINTS.clear()
        config.LLM_ENDPOINTS.update(cls.saved[0])
        config.OLLAMA_URL = cls.saved[1]

    def setUp(self):
        self.server.requests.clear()
        self.server.mode = "ok"

    def choose(self, provider, timeout=1.0):
        llm = LLMCorrector(provider, api_key="test-key", timeout_s=timeout)
        try:
            return llm.choose(CANDS, ["P09"], self.phrases)
        finally:
            llm.close()

    def test_each_provider_round_trip(self):
        expected_header = {"claude": ("x-api-key", "test-key"), "openai": ("authorization", "Bearer test-key"),
                           "groq": ("authorization", "Bearer test-key"),
                           "gemini": ("x-goog-api-key", "test-key"), "ollama": None}
        for provider, header in expected_header.items():
            with self.subTest(provider=provider):
                self.server.requests.clear()
                pid, reason, source = self.choose(provider)
                self.assertEqual((pid, source), ("P10", "llm"))
                self.assertIn("nurse", reason)
                path, headers, body = self.server.requests[0]
                if header:
                    self.assertEqual(headers.get(header[0]), header[1])
                prompt = json.dumps(body)
                self.assertIn("Call the nurse", prompt)
                self.assertIn("I am in pain", prompt)       # the conversation context
                if provider == "claude":
                    self.assertEqual(headers.get("anthropic-version"), "2023-06-01")
                    self.assertEqual(body["model"], config.LLM_MODELS["claude"])
                if provider == "gemini":
                    self.assertIn(config.LLM_MODELS["gemini"], path)
                if provider == "groq":
                    self.assertEqual(body["reasoning_effort"], "none")     # qwen: answer without "thinking"

    def test_groq_switches_away_from_a_retired_model(self):
        llm = LLMCorrector("groq", model="retired-model", api_key="test-key", timeout_s=2.0)
        try:
            pid, _, source = llm.choose(CANDS, ["P09"], self.phrases)
            self.assertEqual((pid, source), ("P10", "llm"))
            self.assertEqual(llm.model, "qwen/qwen3.8-27b")
        finally:
            llm.close()

    def test_invented_phrase_falls_back_to_top1(self):
        self.server.mode = "invent"
        self.assertEqual(self.choose("claude")[0::2], ("P10", "fallback"))

    def test_http_error_falls_back(self):
        self.server.mode = "error"
        self.assertEqual(self.choose("openai")[0::2], ("P10", "fallback"))

    def test_slow_reply_times_out_quickly(self):
        self.server.mode = "slow"
        t0 = time.perf_counter()
        pid, _, source = self.choose("gemini", timeout=0.4)
        self.assertEqual((pid, source), ("P10", "fallback"))
        self.assertLess(time.perf_counter() - t0, 1.2)

    def test_no_key_means_unavailable(self):
        llm = LLMCorrector("claude", api_key=None)
        llm.api_key = None
        self.assertFalse(llm.available)
        self.assertIn("API key", llm.unavailable_reason)
        llm.close()


if __name__ == "__main__":
    unittest.main()
