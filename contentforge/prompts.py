"""Versioned prompt templates.

Each template has a name and a version ("v1", "v2", ...). To change a prompt,
add a new version instead of editing the old one — past outputs stay
reproducible, and every response reports the exact version + fingerprint used.
"""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    user: str

    def render(self, **variables: str) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) with {placeholders} filled in."""
        return self.system, self.user.format(**variables)

    @property
    def fingerprint(self) -> str:
        """Short hash of the prompt text, so silent edits are detectable."""
        return hashlib.sha256((self.system + self.user).encode()).hexdigest()[:12]

    def info(self) -> dict:
        return {"name": self.name, "version": self.version, "fingerprint": self.fingerprint}


_WRITER_SYSTEM = (
    "You are a senior content writer. You always respond with JSON that matches "
    "the provided schema exactly. Write original, accurate, specific content. "
    "Never include personal data such as email addresses or phone numbers."
)

_BRIEF = (
    "Topic: {topic}\n"
    "Audience: {audience}\n"
    "Tone: {tone}\n"
    "Extra instructions: {extra_instructions}\n"
)

_TEMPLATES = [
    # --- Draft templates, one per content type -----------------------------
    PromptTemplate(
        "blog_post", "v1", _WRITER_SYSTEM,
        "Write a blog post.\n" + _BRIEF,
    ),
    PromptTemplate(
        "blog_post", "v2", _WRITER_SYSTEM,
        "Write a blog post with a strong hook in the summary, 3-5 sections that "
        "each make one clear point, and concrete examples.\n" + _BRIEF
        + "The image_prompt should describe an editorial illustration, not text.",
    ),
    PromptTemplate(
        "social_post", "v1", _WRITER_SYSTEM,
        "Write one social media post (max 280 characters of text) that stops the "
        "scroll. Hashtags go in the hashtags field, without the # symbol, not in the text.\n" + _BRIEF,
    ),
    PromptTemplate(
        "product_description", "v1", _WRITER_SYSTEM,
        "Write e-commerce product copy. Lead with the main benefit, keep features "
        "scannable, and avoid unverifiable claims.\n" + _BRIEF
        + "The image_prompt should describe a clean studio product photo.",
    ),
    # --- Shared self-critique and refine templates --------------------------
    PromptTemplate(
        "critique", "v1",
        "You are a demanding editor. Respond only with JSON matching the schema.",
        "Review this {content_type} draft against the brief. Be specific and strict: "
        "a 10 means publish-ready with nothing to improve.\n\n"
        "BRIEF\n" + _BRIEF + "\nDRAFT (JSON)\n{draft}",
    ),
    PromptTemplate(
        "refine", "v1", _WRITER_SYSTEM,
        "Rewrite the {content_type} draft below, fixing every issue the editor "
        "raised while keeping what already works.\n\n"
        "BRIEF\n" + _BRIEF + "\nDRAFT (JSON)\n{draft}\n\nEDITOR FEEDBACK (JSON)\n{critique}",
    ),
]

TEMPLATES: dict[str, dict[str, PromptTemplate]] = {}
for _t in _TEMPLATES:
    TEMPLATES.setdefault(_t.name, {})[_t.version] = _t


def get_template(name: str, version: str | None = None) -> PromptTemplate:
    """Fetch a template; `version=None` means the latest one."""
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template '{name}'")
    versions = TEMPLATES[name]
    if version is None:
        version = max(versions, key=lambda v: int(v.lstrip("v")))
    if version not in versions:
        raise KeyError(f"Template '{name}' has no version '{version}'. Available: {sorted(versions)}")
    return versions[version]


def list_templates() -> list[dict]:
    return [t.info() for versions in TEMPLATES.values() for t in versions.values()]
