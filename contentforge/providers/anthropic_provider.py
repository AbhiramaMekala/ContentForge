"""Anthropic: structured output via forced tool use (function calling).

We define one tool whose input_schema is our output schema and force the model
to call it. The tool's input is then exactly the structured object we want.
"""

from pydantic import BaseModel

from ..schema_utils import to_llm_schema
from .base import ProviderError, TextProvider, TextResult, post

URL = "https://api.anthropic.com/v1/messages"
TOOL_NAME = "emit_output"


class AnthropicTextProvider(TextProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int = 4096, **kwargs):
        super().__init__(model=model, **kwargs)
        self.max_tokens = max_tokens
        self.headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    def generate_structured(self, system: str, prompt: str, output_model: type[BaseModel]) -> TextResult:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{
                "name": TOOL_NAME,
                "description": f"Return the final {output_model.__name__} object.",
                "input_schema": to_llm_schema(output_model),
            }],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
        }
        body = post(self.client, URL, self.name, headers=self.headers, json=payload).json()
        tool_calls = [b for b in body.get("content", []) if b.get("type") == "tool_use"]
        if not tool_calls:
            raise ProviderError(f"anthropic: no tool call in response: {str(body)[:300]}")
        usage = body.get("usage", {})
        return TextResult(
            data=tool_calls[0]["input"], provider=self.name, model=body.get("model", self.model),
            input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
        )
