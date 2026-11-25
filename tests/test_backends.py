"""Tests for OllamaBackend and OpenAICompatibleBackend over real local HTTP servers."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from invocation_protocol.backends import (
    DeterministicStubBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
)
from invocation_protocol.types import InvocationError, RunConfig, ValidationError


class _CapturedState:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, object]]] = []
        self.status_code: int = 200
        self.raw_body: bytes = b"{}"
        self.block_event: threading.Event | None = None


def _make_handler(state: _CapturedState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw) if raw else {}
            headers = {k.lower(): v for k, v in self.headers.items()}
            state.requests.append((self.path, headers, payload))
            if state.block_event is not None:
                state.block_event.wait(timeout=2.0)
            try:
                self.send_response(state.status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(state.raw_body)
            except OSError:
                return

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    return Handler


class TestBackends(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _CapturedState()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        if self.state.block_event is not None:
            self.state.block_event.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def test_ollama_backend_request_response_and_errors(self) -> None:
        cfg = RunConfig(temperature=0.0, seed=17, max_tokens=256, timeout_seconds=5.0)
        backend = OllamaBackend(model="llama3.2", base_url=self.base_url)
        self.assertEqual(backend.name, "ollama:llama3.2")

        self.state.status_code = 200
        self.state.raw_body = json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"ok"}',
                    "thinking": "step 1: verify",
                    "tool_calls": [
                        {"function": {"name": "lookup_fact", "arguments": {"key": "protocol_steps"}}}
                    ],
                }
            }
        ).encode("utf-8")

        resp = backend.generate([{"role": "user", "content": "hi"}], config=cfg)
        self.assertEqual(resp.content, '{"answer":"ok"}')
        self.assertEqual(resp.thinking, "step 1: verify")
        self.assertEqual(len(resp.tool_requests), 1)
        self.assertEqual(resp.tool_requests[0].tool_name, "lookup_fact")
        self.assertEqual(resp.tool_requests[0].args, {"key": "protocol_steps"})

        path, _headers, sent = self.state.requests[-1]
        self.assertEqual(path, "/api/chat")
        self.assertFalse(sent["stream"])
        self.assertEqual(
            sent["options"],
            {"temperature": 0.0, "seed": 17, "num_predict": 256, "num_ctx": 4096},
        )
        self.assertNotIn("tools", sent)

        # Content-encoded tool fallback when message.tool_calls is absent and allowed_tools is non-empty
        self.state.raw_body = json.dumps(
            {"message": {"role": "assistant", "content": '{"name": "lookup_fact", "arguments": {"key": "k"}}'}}
        ).encode("utf-8")
        resp_raw_fallback = backend.generate(
            [{"role": "user", "content": "hi"}], config=cfg, allowed_tools=("lookup_fact",)
        )
        self.assertEqual(len(resp_raw_fallback.tool_requests), 1)
        self.assertEqual(resp_raw_fallback.tool_requests[0].tool_name, "lookup_fact")
        self.assertEqual(resp_raw_fallback.tool_requests[0].args, {"key": "k"})

        # Ignored when allowed_tools is empty
        resp_no_tools = backend.generate([{"role": "user", "content": "hi"}], config=cfg, allowed_tools=())
        self.assertEqual(resp_no_tools.tool_requests, ())

        # Fenced tool JSON fallback
        self.state.raw_body = json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": '```json\n{"name": "write_memory", "arguments": {"key": "x"}}\n```',
                }
            }
        ).encode("utf-8")
        resp_fenced = backend.generate(
            [{"role": "user", "content": "hi"}], config=cfg, allowed_tools=("lookup_fact",)
        )
        self.assertEqual(len(resp_fenced.tool_requests), 1)
        self.assertEqual(resp_fenced.tool_requests[0].tool_name, "write_memory")

        # Half-fenced, empty fence, non-tool JSON, blank name, and non-dict args do NOT parse as tool requests
        for bad_content in (
            "",
            '```json\n{"name": "lookup_fact", "arguments": {}}\nextra',
            'extra\n{"name": "lookup_fact", "arguments": {}}\n```',
            "```json\n```",
            '{"answer": "done"}',
            '{"name": "   ", "arguments": {}}',
            '{"name": "lookup_fact", "arguments": []}',
        ):
            self.state.raw_body = json.dumps({"message": {"role": "assistant", "content": bad_content}}).encode("utf-8")
            bad_resp = backend.generate(
                [{"role": "user", "content": "hi"}], config=cfg, allowed_tools=("lookup_fact",)
            )
            self.assertEqual(bad_resp.tool_requests, (), msg=bad_content)

        # Custom context_length forwards as num_ctx and tools are included when allowed_tools is non-empty
        backend.generate(
            [{"role": "user", "content": "hi"}],
            config=RunConfig(context_length=2048),
            allowed_tools=("lookup_fact", "count_words", "list_sources"),
        )
        self.assertEqual(self.state.requests[-1][2]["options"]["num_ctx"], 2048)  # type: ignore[index]
        sent_tools = self.state.requests[-1][2]["tools"]
        self.assertEqual(
            [item["function"]["name"] for item in sent_tools],  # type: ignore[index]
            ["lookup_fact", "count_words", "list_sources"],
        )

        # Slow response timeout raises InvocationError rather than raw TimeoutError
        gate = threading.Event()
        self.state.block_event = gate
        try:
            with self.assertRaises(InvocationError) as timeout_ctx:
                backend.generate(
                    [{"role": "user", "content": "hi"}],
                    config=RunConfig(timeout_seconds=0.05),
                )
            self.assertIn("timed out", str(timeout_ctx.exception))
        finally:
            gate.set()
            self.state.block_event = None

        # HTTP 500 error
        self.state.status_code = 500
        self.state.raw_body = b"internal error"
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "hi"}], config=cfg)

        # Malformed JSON response
        self.state.status_code = 200
        self.state.raw_body = b"not-json"
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "hi"}], config=cfg)

        # Missing message.content
        self.state.raw_body = b'{"message":{}}'
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "hi"}], config=cfg)

    def test_ollama_backend_streaming_ndjson_and_errors(self) -> None:
        cfg = RunConfig(temperature=0.0, seed=17, max_tokens=256, timeout_seconds=5.0)
        tokens: list[str] = []
        backend = OllamaBackend(model="llama3.2", base_url=self.base_url, on_token=tokens.append)

        self.state.status_code = 200
        self.state.raw_body = (
            b"\n"
            + json.dumps({"message": {"content": '{"ans', "thinking": "think "}}).encode("utf-8")
            + b"\n"
            + json.dumps(
                {
                    "message": {
                        "content": 'wer":"ok"}',
                        "thinking": "done",
                        "tool_calls": [{"function": {"name": "count_words", "arguments": {"text": "a b"}}}],
                    }
                }
            ).encode("utf-8")
            + b"\n"
            + json.dumps({"message": {"content": "", "thinking": ""}, "done": True}).encode("utf-8")
            + b"\n"
            + json.dumps({"message": {"content": None, "thinking": 123}}).encode("utf-8")
            + b"\n"
        )

        resp = backend.generate([{"role": "user", "content": "hi"}], config=cfg)
        self.assertEqual(tokens, ['{"ans', 'wer":"ok"}'])
        self.assertNotIn("", tokens)
        self.assertEqual(resp.content, '{"answer":"ok"}')
        self.assertEqual(resp.thinking, "think done")
        self.assertEqual(len(resp.tool_requests), 1)
        self.assertEqual(resp.tool_requests[0].tool_name, "count_words")
        self.assertTrue(self.state.requests[-1][2]["stream"])

        # Malformed NDJSON chunk fails cleanly
        self.state.raw_body = b"{bad-ndjson\n"
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "hi"}], config=cfg)

        # HTTP 500 on streaming fails cleanly
        self.state.status_code = 500
        self.state.raw_body = b"stream error"
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "hi"}], config=cfg)

        # Streaming timeout raises InvocationError
        self.state.status_code = 200
        gate = threading.Event()
        self.state.block_event = gate
        try:
            with self.assertRaises(InvocationError) as ctx:
                backend.generate([{"role": "user", "content": "hi"}], config=RunConfig(timeout_seconds=0.05))
            self.assertIn("timed out", str(ctx.exception))
        finally:
            gate.set()
            self.state.block_event = None

    def test_openai_compatible_backend_request_response_and_errors(self) -> None:
        cfg = RunConfig(temperature=0.2, seed=99, max_tokens=128, timeout_seconds=5.0)
        backend = OpenAICompatibleBackend(model="gpt-local", base_url=self.base_url, api_key="secret-key")
        self.assertEqual(backend.name, "openai-compatible:gpt-local")

        self.state.status_code = 200
        self.state.raw_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"function": {"name": "count_words", "arguments": '{"text": "one two"}'}}
                            ],
                        }
                    }
                ]
            }
        ).encode("utf-8")

        resp = backend.generate([{"role": "user", "content": "count"}], config=cfg)
        self.assertEqual(resp.content, "")
        self.assertEqual(len(resp.tool_requests), 1)
        self.assertEqual(resp.tool_requests[0].args, {"text": "one two"})

        path, headers, sent = self.state.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers.get("authorization"), "Bearer secret-key")
        self.assertEqual(sent["temperature"], 0.2)
        self.assertEqual(sent["seed"], 99)
        self.assertEqual(sent["max_tokens"], 128)
        self.assertNotIn("tools", sent)

        backend.generate(
            [{"role": "user", "content": "count"}],
            config=cfg,
            allowed_tools=("lookup_fact", "count_words", "list_sources"),
        )
        oa_tools = self.state.requests[-1][2]["tools"]
        self.assertEqual(
            [item["function"]["name"] for item in oa_tools],  # type: ignore[index]
            ["lookup_fact", "count_words", "list_sources"],
        )

        # Empty choices and missing message fail cleanly
        self.state.raw_body = b'{"choices":[]}'
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "count"}], config=cfg)

        self.state.raw_body = b'{"choices":[{}]}'
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "count"}], config=cfg)

        # Invalid tool_calls arguments and missing function.name fail cleanly
        self.state.raw_body = json.dumps(
            {"choices": [{"message": {"content": "ok", "tool_calls": [{"function": {"name": "f", "arguments": "{bad"}}]}}]}
        ).encode("utf-8")
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "count"}], config=cfg)

        self.state.raw_body = json.dumps(
            {"choices": [{"message": {"content": "ok", "tool_calls": [{"function": {}}]}}]}
        ).encode("utf-8")
        with self.assertRaises(InvocationError):
            backend.generate([{"role": "user", "content": "count"}], config=cfg)

        # Blank/None api_key omits Authorization header
        no_key_backend = OpenAICompatibleBackend(model="gpt-local", base_url=self.base_url, api_key="   ")
        self.state.raw_body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
        no_key_backend.generate([{"role": "user", "content": "hi"}], config=cfg)
        self.assertNotIn("authorization", self.state.requests[-1][1])

    def test_stub_determinism_and_validation(self) -> None:
        with self.assertRaises(ValidationError):
            DeterministicStubBackend(inject_failure="nonexistent_mode")
        with self.assertRaises(ValidationError):
            OllamaBackend(model="   ")

        stub = DeterministicStubBackend()
        cfg = RunConfig()
        msgs = [{"role": "user", "content": "AUTHORITY STACK\ntask_accuracy_01"}]
        r1 = stub.generate(msgs, config=cfg)
        r2 = stub.generate(msgs, config=cfg)
        self.assertEqual(r1.content, r2.content)

        # Protocol marker takes precedence even if task quotes "By thy epithets"
        mixed = [{"role": "user", "content": "AUTHORITY STACK\nBy thy epithets\ntask_accuracy_01"}]
        r_mixed = stub.generate(mixed, config=cfg)
        self.assertEqual(r_mixed.content, r1.content)


if __name__ == "__main__":
    unittest.main()
