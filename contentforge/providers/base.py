"""Common interface every provider implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from pydantic import BaseModel


class ProviderError(RuntimeError):
    """A provider call failed (missing key, HTTP error, unparseable response)."""


@dataclass
class TextResult:
    data: dict            # parsed JSON returned by the model (not yet validated)
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ImageResult:
    image_bytes: bytes    # PNG bytes
    provider: str
    model: str


class TextProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, client: httpx.Client | None = None, timeout: float = 90):
        self.model = model
        self.client = client or httpx.Client(timeout=timeout)

    @abstractmethod
    def generate_structured(self, system: str, prompt: str, output_model: type[BaseModel]) -> TextResult:
        """Ask the model for JSON that matches `output_model`'s schema."""


class ImageProvider(ABC):
    name: str = "base"
    model: str = "base"

    def __init__(self, client: httpx.Client | None = None, timeout: float = 120):
        self.client = client or httpx.Client(timeout=timeout)

    @abstractmethod
    def generate_image(self, prompt: str, size: str = "1024x1024") -> ImageResult:
        """Generate one image and return it as PNG bytes."""


def post(client: httpx.Client, url: str, provider: str, **kwargs) -> httpx.Response:
    """POST with uniform error handling."""
    try:
        response = client.post(url, **kwargs)
    except httpx.HTTPError as exc:
        raise ProviderError(f"{provider}: request failed: {exc}") from exc
    if response.status_code >= 400:
        raise ProviderError(f"{provider}: HTTP {response.status_code}: {response.text[:500]}")
    return response
