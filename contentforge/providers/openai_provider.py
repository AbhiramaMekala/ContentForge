"""OpenAI: Chat Completions with JSON-schema structured outputs, and Images."""

import base64
import json

from pydantic import BaseModel

from ..schema_utils import to_llm_schema
from .base import ImageProvider, ImageResult, ProviderError, TextProvider, TextResult, post

BASE_URL = "https://api.openai.com/v1"


class OpenAITextProvider(TextProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def generate_structured(self, system: str, prompt: str, output_model: type[BaseModel]) -> TextResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": output_model.__name__, "strict": True, "schema": to_llm_schema(output_model)},
            },
        }
        body = post(self.client, f"{BASE_URL}/chat/completions", self.name, headers=self.headers, json=payload).json()
        try:
            message = body["choices"][0]["message"]
            if message.get("refusal"):
                raise ProviderError(f"openai: model refused: {message['refusal']}")
            data = json.loads(message["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"openai: unexpected response: {str(body)[:300]}") from exc
        usage = body.get("usage", {})
        return TextResult(
            data=data, provider=self.name, model=body.get("model", self.model),
            input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0),
        )


class OpenAIImageProvider(ImageProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-image-1", quality: str = "medium", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.quality = quality
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def generate_image(self, prompt: str, size: str = "1024x1024") -> ImageResult:
        payload = {"model": self.model, "prompt": prompt, "size": size, "n": 1}
        if self.model.startswith("dall-e"):
            payload["response_format"] = "b64_json"  # gpt-image models always return base64
        else:
            payload["quality"] = self.quality
        body = post(self.client, f"{BASE_URL}/images/generations", self.name, headers=self.headers, json=payload).json()
        try:
            image_bytes = base64.b64decode(body["data"][0]["b64_json"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"openai: unexpected image response: {str(body)[:300]}") from exc
        return ImageResult(image_bytes=image_bytes, provider=self.name, model=self.model)
