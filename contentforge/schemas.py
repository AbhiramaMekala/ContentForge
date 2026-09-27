"""Pydantic models.

1. Output schemas — the structured content the LLM must return. Every LLM
   response is validated against one of these before it leaves the pipeline.
2. API request models — what clients send to the FastAPI service.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Output schemas (what the model must produce)
# ---------------------------------------------------------------------------


class Section(BaseModel):
    heading: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=40, description="One or two paragraphs of prose.")


class BlogPost(BaseModel):
    title: str = Field(min_length=5, max_length=120)
    summary: str = Field(min_length=20, max_length=400, description="Two-sentence overview.")
    sections: list[Section] = Field(min_length=2, max_length=6)
    tags: list[str] = Field(min_length=1, max_length=8)
    image_prompt: str = Field(min_length=10, max_length=400, description="Visual description for a header image.")


class SocialPost(BaseModel):
    platform: str = Field(min_length=2, max_length=30)
    text: str = Field(min_length=10, max_length=280)
    hashtags: list[str] = Field(min_length=1, max_length=5)
    call_to_action: str = Field(min_length=3, max_length=100)
    image_prompt: str = Field(min_length=10, max_length=400, description="Visual description for an attached image.")


class ProductDescription(BaseModel):
    product_name: str = Field(min_length=2, max_length=80)
    headline: str = Field(min_length=5, max_length=100)
    description: str = Field(min_length=50, max_length=1200)
    key_features: list[str] = Field(min_length=3, max_length=6)
    image_prompt: str = Field(min_length=10, max_length=400, description="Visual description for a product shot.")


class Critique(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall quality from 1 (poor) to 10 (excellent).")
    issues: list[str] = Field(description="Concrete problems in the draft.")
    suggestions: list[str] = Field(description="Specific, actionable improvements.")


# Content types the service can generate: name -> output schema.
CONTENT_TYPES: dict[str, type[BaseModel]] = {
    "blog_post": BlogPost,
    "social_post": SocialPost,
    "product_description": ProductDescription,
}

ContentType = Literal["blog_post", "social_post", "product_description"]
TextProviderName = Literal["auto", "mock", "openai", "anthropic", "gemini"]
ImageProviderName = Literal["auto", "mock", "openai", "stability"]

# ---------------------------------------------------------------------------
# 2. API request models
# ---------------------------------------------------------------------------


class TextRequest(BaseModel):
    content_type: ContentType = "blog_post"
    topic: str = Field(min_length=3, description="What the content is about.")
    audience: str = "general readers"
    tone: str = "clear and friendly"
    extra_instructions: str = ""
    provider: TextProviderName | None = None
    model: str | None = None
    template_version: str | None = Field(default=None, description="e.g. 'v1'. Defaults to the latest.")
    refine: bool = Field(default=True, description="Run the critique -> refine steps.")
    refine_threshold: int | None = Field(
        default=None, ge=1, le=11,
        description="Refine when the critique score is below this. 11 = always refine. Defaults to REFINE_THRESHOLD.",
    )


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=3)
    provider: ImageProviderName | None = None
    size: str = "1024x1024"


class ContentRequest(TextRequest):
    """Text generation followed by an image built from the text's image_prompt."""

    generate_image: bool = True
    image_provider: ImageProviderName | None = None
    image_size: str = "1024x1024"
