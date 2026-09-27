"""Offline providers — no API key needed.

MockTextProvider builds a valid response directly from the JSON schema, so the
whole pipeline (validation, critique, refine, guardrails, metrics) runs end to
end without network access. MockImageProvider draws a placeholder PNG.
"""

import io
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from ..schema_utils import inline_refs
from .base import ImageProvider, ImageResult, TextProvider, TextResult


def _topic_from(prompt: str) -> str:
    match = re.search(r"Topic:\s*(.+)", prompt)
    return match.group(1).strip() if match else "your topic"


def _fake_value(name: str, schema: dict, topic: str, refined: bool):
    kind = schema.get("type")
    if kind == "object":
        return {k: _fake_value(k, v, topic, refined) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        count = max(schema.get("minItems", 0), min(3, schema.get("maxItems", 3)))
        return [_fake_value(name.rstrip("s"), schema["items"], topic, refined) for _ in range(count)]
    if kind == "integer":
        low, high = schema.get("minimum", 1), schema.get("maximum", 10)
        return 9 if refined else (low + high) // 2
    if kind == "number":
        return 0.5
    if kind == "boolean":
        return True
    # string
    label = name.replace("_", " ")
    text = f"{'Refined' if refined else 'Draft'} {label} about {topic}."
    min_len, max_len = schema.get("minLength", 0), schema.get("maxLength", 10_000)
    while len(text) < min_len:
        text += f" More detail on {topic}."
    return text[:max_len]


class MockTextProvider(TextProvider):
    name = "mock"

    def __init__(self, model: str = "mock", **kwargs):
        super().__init__(model=model, **kwargs)

    def generate_structured(self, system: str, prompt: str, output_model: type[BaseModel]) -> TextResult:
        schema = inline_refs(output_model.model_json_schema())
        refined = "EDITOR FEEDBACK" in prompt  # the refine step includes the critique
        data = _fake_value("root", schema, _topic_from(prompt), refined)
        # Rough token estimate (~4 chars per token) so metrics look realistic.
        return TextResult(
            data=data, provider=self.name, model=self.model,
            input_tokens=(len(system) + len(prompt)) // 4, output_tokens=len(str(data)) // 4,
        )


class MockImageProvider(ImageProvider):
    name = "mock"
    model = "mock-image"

    def generate_image(self, prompt: str, size: str = "1024x1024") -> ImageResult:
        width, height = (int(n) for n in size.lower().split("x"))
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        for y in range(height):  # simple vertical gradient
            shade = int(40 + 120 * y / height)
            draw.line([(0, y), (width, y)], fill=(shade // 2, shade // 3, shade))
        font_size = max(12, width // 28)
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:  # Pillow < 10.1
            font = ImageFont.load_default()
        caption = "\n".join(textwrap.wrap(f"[mock image] {prompt}", width=max(20, int(width / font_size * 1.7)))[:10])
        draw.multiline_text((width // 16, height // 3), caption, fill=(255, 255, 255), font=font, spacing=font_size // 3)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return ImageResult(image_bytes=buffer.getvalue(), provider=self.name, model=self.model)
