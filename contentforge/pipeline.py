"""The generation pipeline.

Text:   draft -> self-critique -> refine, every step schema-validated.
Image:  prompt -> guardrails -> image provider -> PNG saved to disk.
Both:   text first, then an image from the text's `image_prompt` field.

Each model call goes through `_structured_call`, which validates the JSON
against the Pydantic schema and, if it fails, sends the validation errors back
to the model for a repair attempt.
"""

import json
import time
import uuid

from pydantic import BaseModel, ValidationError

from .config import settings
from .guardrails import check_input, check_output
from .prompts import get_template
from .providers import ImageProvider, ProviderError, TextProvider, get_image_provider, get_text_provider
from .schemas import CONTENT_TYPES, ContentRequest, Critique, ImageRequest, TextRequest
from .tracking import IMAGE_PRICES, RunTracker, StepMetric, metrics_store, text_cost


class ContentValidationError(RuntimeError):
    """The model kept returning output that doesn't match the schema."""


def _structured_call(provider: TextProvider, system: str, prompt: str,
                     output_model: type[BaseModel], step: str, tracker: RunTracker) -> BaseModel:
    """One model call that is guaranteed to return a valid `output_model` (or raise)."""
    for attempt in range(settings.max_repair_attempts + 1):
        started = time.perf_counter()
        result = provider.generate_structured(system, prompt, output_model)
        tracker.add(StepMetric(
            step=step if attempt == 0 else f"{step}:repair{attempt}",
            provider=result.provider, model=result.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            cost_usd=round(text_cost(result.model, result.input_tokens, result.output_tokens), 6),
        ))
        try:
            return output_model.model_validate(result.data)
        except ValidationError as exc:
            error = exc
            # Ask the model to fix its own output on the next attempt.
            prompt = (
                f"{prompt}\n\nYour previous answer was invalid:\n{json.dumps(result.data)[:2000]}\n"
                f"Validation errors:\n{exc}\nReturn a corrected JSON object."
            )
    raise ContentValidationError(f"{step}: output failed schema validation: {error}")


def generate_text(req: TextRequest, provider: TextProvider | None = None, record: bool = True) -> dict:
    """Run draft -> critique -> refine for one content request."""
    for text in (req.topic, req.audience, req.tone, req.extra_instructions):
        check_input(text, settings.max_input_chars, settings.blocked_terms)

    provider = provider or get_text_provider(req.provider, req.model)
    schema = CONTENT_TYPES[req.content_type]
    brief = {"topic": req.topic, "audience": req.audience, "tone": req.tone,
             "extra_instructions": req.extra_instructions or "none"}
    tracker = RunTracker(kind="text")

    # 1. Draft
    draft_template = get_template(req.content_type, req.template_version)
    content = _structured_call(provider, *draft_template.render(**brief), schema, "draft", tracker)
    draft = content
    templates_used = [draft_template.info()]
    critique: Critique | None = None
    refined = False

    if req.refine:
        # 2. Self-critique
        critique_template = get_template("critique")
        critique = _structured_call(
            provider,
            *critique_template.render(**brief, content_type=req.content_type, draft=content.model_dump_json()),
            Critique, "critique", tracker,
        )
        templates_used.append(critique_template.info())

        # 3. Refine (skipped when the draft is already good enough)
        threshold = req.refine_threshold or settings.refine_threshold
        if critique.score < threshold:
            refine_template = get_template("refine")
            content = _structured_call(
                provider,
                *refine_template.render(**brief, content_type=req.content_type,
                                        draft=content.model_dump_json(), critique=critique.model_dump_json()),
                schema, "refine", tracker,
            )
            templates_used.append(refine_template.info())
            refined = True

    # 4. Output guardrails (redaction), then re-validate the cleaned result.
    clean, flags = check_output(content.model_dump(), settings.blocked_terms)
    content = schema.model_validate(clean)

    metrics = tracker.summary()
    if record:
        metrics_store.record(metrics)
    return {
        "content_type": req.content_type,
        "content": content.model_dump(),
        "draft": draft.model_dump(),
        "critique": critique.model_dump() if critique else None,
        "refined": refined,
        "templates": templates_used,
        "guardrail_flags": flags,
        "metrics": metrics,
    }


def generate_image(req: ImageRequest, provider: ImageProvider | None = None, record: bool = True) -> dict:
    """Generate one image, save it as a PNG and return where it lives."""
    check_input(req.prompt, settings.max_input_chars, settings.blocked_terms)
    provider = provider or get_image_provider(req.provider)
    tracker = RunTracker(kind="image")

    started = time.perf_counter()
    result = provider.generate_image(req.prompt, req.size)
    tracker.add(StepMetric(
        step="image", provider=result.provider, model=result.model,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        cost_usd=IMAGE_PRICES.get(result.model, 0.0),
    ))

    settings.image_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    (settings.image_dir / filename).write_bytes(result.image_bytes)

    metrics = tracker.summary()
    if record:
        metrics_store.record(metrics)
    return {
        "image_url": f"/images/{filename}",
        "file_path": str(settings.image_dir / filename),
        "prompt": req.prompt,
        "provider": result.provider,
        "model": result.model,
        "metrics": metrics,
    }


def generate_content(req: ContentRequest, text_provider: TextProvider | None = None,
                     image_provider: ImageProvider | None = None) -> dict:
    """Text + image in one call: the image is generated from the text's image_prompt."""
    text = generate_text(req, provider=text_provider, record=False)
    image, image_error = None, None
    if req.generate_image:
        image_req = ImageRequest(prompt=text["content"]["image_prompt"],
                                 provider=req.image_provider, size=req.image_size)
        try:
            image = generate_image(image_req, provider=image_provider, record=False)
        except ProviderError as exc:  # keep the text even if the image fails
            image_error = str(exc)

    # Combine both runs into one metrics record.
    steps = text["metrics"]["steps"] + (image["metrics"]["steps"] if image else [])
    metrics = {
        "kind": "content",
        "total_latency_ms": round(text["metrics"]["total_latency_ms"] + (image["metrics"]["total_latency_ms"] if image else 0), 1),
        "total_input_tokens": text["metrics"]["total_input_tokens"],
        "total_output_tokens": text["metrics"]["total_output_tokens"],
        "total_cost_usd": round(sum(s["cost_usd"] for s in steps), 6),
        "steps": steps,
    }
    metrics_store.record(metrics)
    return {"text": text, "image": image, "image_error": image_error, "metrics": metrics}
