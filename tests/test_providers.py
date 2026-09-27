"""Real-provider tests without network access.

httpx.MockTransport stands in for each API: it checks the request we send
(URL, auth header, structured-output settings) and returns a response shaped
like the real API's, which we check is parsed correctly.
"""

import base64
import json
import unittest

import httpx

from contentforge.providers import ProviderError
from contentforge.providers.anthropic_provider import AnthropicTextProvider
from contentforge.providers.gemini_provider import GeminiTextProvider
from contentforge.providers.openai_provider import OpenAIImageProvider, OpenAITextProvider
from contentforge.providers.stability_provider import StabilityImageProvider
from contentforge.schemas import Critique

CRITIQUE = {"score": 7, "issues": ["too long"], "suggestions": ["cut the intro"]}


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class ProviderTests(unittest.TestCase):
    def test_openai_text(self):
        def handler(request: httpx.Request):
            self.assertEqual(str(request.url), "https://api.openai.com/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer sk-test")
            body = json.loads(request.content)
            fmt = body["response_format"]
            self.assertEqual(fmt["type"], "json_schema")
            self.assertTrue(fmt["json_schema"]["strict"])
            self.assertEqual(fmt["json_schema"]["schema"]["required"], ["score", "issues", "suggestions"])
            return httpx.Response(200, json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": json.dumps(CRITIQUE), "refusal": None}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 40},
            })

        result = OpenAITextProvider("sk-test", "gpt-4o-mini", client=client_for(handler)) \
            .generate_structured("sys", "prompt", Critique)
        self.assertEqual(result.data, CRITIQUE)
        self.assertEqual((result.input_tokens, result.output_tokens), (120, 40))

    def test_anthropic_text_uses_forced_tool_call(self):
        def handler(request: httpx.Request):
            self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages")
            self.assertEqual(request.headers["x-api-key"], "ak-test")
            self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
            body = json.loads(request.content)
            self.assertEqual(body["system"], "sys")
            self.assertEqual(body["tool_choice"], {"type": "tool", "name": "emit_output"})
            self.assertIn("properties", body["tools"][0]["input_schema"])
            return httpx.Response(200, json={
                "model": "claude-haiku-4-5",
                "content": [{"type": "tool_use", "id": "t1", "name": "emit_output", "input": CRITIQUE}],
                "usage": {"input_tokens": 200, "output_tokens": 60},
            })

        result = AnthropicTextProvider("ak-test", "claude-haiku-4-5", client=client_for(handler)) \
            .generate_structured("sys", "prompt", Critique)
        self.assertEqual(result.data, CRITIQUE)
        self.assertEqual(result.input_tokens, 200)

    def test_gemini_text(self):
        def handler(request: httpx.Request):
            self.assertEqual(
                str(request.url),
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            )
            self.assertEqual(request.headers["x-goog-api-key"], "g-test")
            config = json.loads(request.content)["generationConfig"]
            self.assertEqual(config["responseMimeType"], "application/json")
            self.assertIn("properties", config["responseJsonSchema"])
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": json.dumps(CRITIQUE)}]}}],
                "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 30},
            })

        result = GeminiTextProvider("g-test", "gemini-2.5-flash", client=client_for(handler)) \
            .generate_structured("sys", "prompt", Critique)
        self.assertEqual(result.data, CRITIQUE)
        self.assertEqual(result.output_tokens, 30)

    def test_openai_image(self):
        png = b"\x89PNG fake"

        def handler(request: httpx.Request):
            self.assertEqual(str(request.url), "https://api.openai.com/v1/images/generations")
            self.assertEqual(json.loads(request.content)["model"], "gpt-image-1")
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}]})

        result = OpenAIImageProvider("sk-test", client=client_for(handler)).generate_image("a cat")
        self.assertEqual(result.image_bytes, png)

    def test_stability_image_uses_multipart(self):
        def handler(request: httpx.Request):
            self.assertIn("stable-image/generate/core", str(request.url))
            self.assertTrue(request.headers["content-type"].startswith("multipart/form-data"))
            self.assertIn(b"16:9", request.content)
            return httpx.Response(200, content=b"\x89PNG stability")

        result = StabilityImageProvider("st-test", client=client_for(handler)).generate_image("a cat", "1792x1024")
        self.assertEqual(result.image_bytes, b"\x89PNG stability")

    def test_http_errors_become_provider_errors(self):
        handler = lambda request: httpx.Response(401, json={"error": "bad key"})  # noqa: E731
        with self.assertRaises(ProviderError) as ctx:
            OpenAITextProvider("bad", "gpt-4o-mini", client=client_for(handler)).generate_structured("s", "p", Critique)
        self.assertIn("401", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
