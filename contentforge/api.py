"""FastAPI service. Run with:  uvicorn contentforge.api:app --reload

Demo page at http://127.0.0.1:8000  ·  API docs at http://127.0.0.1:8000/docs
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import settings
from .guardrails import GuardrailError
from .pipeline import ContentValidationError, generate_content, generate_image, generate_text
from .prompts import list_templates
from .providers import ProviderError
from .schemas import CONTENT_TYPES, ContentRequest, ImageRequest, TextRequest
from .tracking import metrics_store

app = FastAPI(
    title="ContentForge",
    version=__version__,
    description="Multimodal generative content pipeline: draft -> critique -> refine, "
                "schema-validated JSON, plus image generation.",
)

settings.image_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=settings.image_dir), name="images")


# --- Error handling: map pipeline errors to HTTP status codes ---------------

@app.exception_handler(GuardrailError)
def _guardrail(_: Request, exc: GuardrailError):
    return JSONResponse(status_code=400, content={"error": "guardrail", "detail": str(exc)})


@app.exception_handler(ContentValidationError)
def _invalid(_: Request, exc: ContentValidationError):
    return JSONResponse(status_code=422, content={"error": "invalid_model_output", "detail": str(exc)})


@app.exception_handler(ProviderError)
def _provider(_: Request, exc: ProviderError):
    return JSONResponse(status_code=502, content={"error": "provider", "detail": str(exc)})


# --- Endpoints ---------------------------------------------------------------

@app.get("/", include_in_schema=False)
def demo_page():
    """The interactive demo UI."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "default_text_provider": settings.default_text_provider,
        "default_image_provider": settings.default_image_provider,
        "configured_keys": {
            "openai": bool(settings.openai_api_key),
            "anthropic": bool(settings.anthropic_api_key),
            "gemini": bool(settings.gemini_api_key),
            "stability": bool(settings.stability_api_key),
        },
    }


@app.get("/templates")
def templates():
    """All prompt templates with their versions and fingerprints."""
    return {"content_types": list(CONTENT_TYPES), "templates": list_templates()}


@app.get("/schemas/{content_type}")
def schema(content_type: str):
    """The JSON schema a content type's output is validated against."""
    if content_type not in CONTENT_TYPES:
        return JSONResponse(status_code=404, content={"error": f"Unknown content type '{content_type}'"})
    return CONTENT_TYPES[content_type].model_json_schema()


@app.post("/generate/text")
def text(req: TextRequest):
    """Structured text: draft -> self-critique -> refine."""
    return generate_text(req)


@app.post("/generate/image")
def image(req: ImageRequest):
    """A single image from a prompt."""
    return generate_image(req)


@app.post("/generate")
def content(req: ContentRequest):
    """Text plus a matching image generated from the text's image_prompt."""
    return generate_content(req)


@app.get("/metrics")
def metrics():
    """Cost and latency totals plus the most recent runs."""
    return metrics_store.snapshot()
