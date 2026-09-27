"""Provider factory: pick a provider by name, using keys from settings."""

from ..config import settings
from .anthropic_provider import AnthropicTextProvider
from .base import ImageProvider, ImageResult, ProviderError, TextProvider, TextResult
from .gemini_provider import GeminiTextProvider
from .mock import MockImageProvider, MockTextProvider
from .openai_provider import OpenAIImageProvider, OpenAITextProvider
from .stability_provider import StabilityImageProvider

__all__ = [
    "get_text_provider", "get_image_provider", "ProviderError",
    "TextProvider", "ImageProvider", "TextResult", "ImageResult",
]


def _require(key: str, env_name: str) -> str:
    if not key:
        raise ProviderError(f"{env_name} is not set. Add it to your .env file or use provider 'mock'.")
    return key


def get_text_provider(name: str | None = None, model: str | None = None) -> TextProvider:
    name = name or settings.default_text_provider
    if name == "auto":
        name = "openai" if settings.openai_api_key else "mock"
    timeout = settings.request_timeout
    if name == "mock":
        return MockTextProvider()
    if name == "openai":
        return OpenAITextProvider(_require(settings.openai_api_key, "OPENAI_API_KEY"),
                                  model or settings.openai_model, timeout=timeout)
    if name == "anthropic":
        return AnthropicTextProvider(_require(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
                                     model or settings.anthropic_model, timeout=timeout)
    if name == "gemini":
        return GeminiTextProvider(_require(settings.gemini_api_key, "GEMINI_API_KEY"),
                                  model or settings.gemini_model, timeout=timeout)
    raise ProviderError(f"Unknown text provider '{name}'")


def get_image_provider(name: str | None = None) -> ImageProvider:
    name = name or settings.default_image_provider
    if name == "auto":
        name = "openai" if settings.openai_api_key else "mock"
    if name == "mock":
        return MockImageProvider()
    if name == "openai":
        return OpenAIImageProvider(_require(settings.openai_api_key, "OPENAI_API_KEY"), settings.openai_image_model,
                                   quality=settings.openai_image_quality)
    if name == "stability":
        return StabilityImageProvider(_require(settings.stability_api_key, "STABILITY_API_KEY"))
    raise ProviderError(f"Unknown image provider '{name}'")
