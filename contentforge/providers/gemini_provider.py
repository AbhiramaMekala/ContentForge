"""Google Gemini: JSON mode with a response JSON schema."""

import json

from pydantic import BaseModel

from ..schema_utils import to_llm_schema
from .base import ProviderError, TextProvider, TextResult, post

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiTextProvider(TextProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.headers = {"x-goog-api-key": api_key}

    def generate_structured(self, system: str, prompt: str, output_model: type[BaseModel]) -> TextResult:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": to_llm_schema(output_model),
            },
        }
        url = f"{BASE_URL}/{self.model}:generateContent"
        body = post(self.client, url, self.name, headers=self.headers, json=payload).json()
        try:
            text = "".join(p.get("text", "") for p in body["candidates"][0]["content"]["parts"])
            data = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"gemini: unexpected response: {str(body)[:300]}") from exc
        usage = body.get("usageMetadata", {})
        return TextResult(
            data=data, provider=self.name, model=self.model,
            input_tokens=usage.get("promptTokenCount", 0), output_tokens=usage.get("candidatesTokenCount", 0),
        )
