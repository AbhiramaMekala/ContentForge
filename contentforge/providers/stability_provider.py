"""Stability AI: diffusion image generation (Stable Image Core)."""

from .base import ImageProvider, ImageResult, post

URL = "https://api.stability.ai/v2beta/stable-image/generate/core"

# Stability takes an aspect ratio rather than a pixel size.
_ASPECT = {"1024x1024": "1:1", "1792x1024": "16:9", "1536x1024": "3:2", "1024x1792": "9:16", "1024x1536": "2:3"}


class StabilityImageProvider(ImageProvider):
    name = "stability"
    model = "stable-image-core"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}

    def generate_image(self, prompt: str, size: str = "1024x1024") -> ImageResult:
        form = {"prompt": prompt, "output_format": "png", "aspect_ratio": _ASPECT.get(size, "1:1")}
        # The API requires multipart/form-data; the empty file part forces that encoding.
        response = post(self.client, URL, self.name, headers=self.headers, data=form, files={"none": ""})
        return ImageResult(image_bytes=response.content, provider=self.name, model=self.model)
