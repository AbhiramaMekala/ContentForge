"""All configuration comes from environment variables (or a .env file)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


class Settings:
    def __init__(self) -> None:
        # API keys — leave empty to use the built-in mock providers.
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.stability_api_key = os.getenv("STABILITY_API_KEY", "")

        # Which providers to use when a request doesn't specify one.
        # "auto" = OpenAI if OPENAI_API_KEY is set, otherwise the offline mock.
        self.default_text_provider = os.getenv("DEFAULT_TEXT_PROVIDER", "auto")
        self.default_image_provider = os.getenv("DEFAULT_IMAGE_PROVIDER", "auto")

        # Default model per provider.
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.openai_image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        self.openai_image_quality = os.getenv("OPENAI_IMAGE_QUALITY", "medium")  # gpt-image: low | medium | high

        # Pipeline behaviour.
        self.refine_threshold = int(os.getenv("REFINE_THRESHOLD", "8"))  # skip refine if critique score >= this
        self.max_repair_attempts = int(os.getenv("MAX_REPAIR_ATTEMPTS", "1"))
        self.request_timeout = float(os.getenv("REQUEST_TIMEOUT", "90"))

        # Guardrails.
        self.max_input_chars = int(os.getenv("MAX_INPUT_CHARS", "2000"))
        self.blocked_terms = _csv(os.getenv("BLOCKED_TERMS", ""))

        # Where generated images are written.
        self.image_dir = Path(os.getenv("IMAGE_DIR", "outputs/images"))


settings = Settings()
