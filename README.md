# ContentForge

A multimodal generative content pipeline behind one FastAPI service.

- **Draft → self-critique → refine.** The model writes a draft, then scores it against the brief. If the score is below a threshold, it rewrites the draft using that feedback.
- **Schema-validated structured output.** Every model response is JSON checked against a Pydantic schema. OpenAI uses JSON-schema structured outputs, Anthropic uses forced tool calling and Gemini uses JSON mode with a schema. If validation fails, the errors go back to the model so it can fix its answer.
- **Text + image in one call.** Each text output includes an `image_prompt`, which goes to an image model: OpenAI `gpt-image-1` or Stability AI's diffusion model.
- **Production basics.** Prompt templates are versioned, input and output have guardrails, and each model call records its tokens, latency and estimated cost.

The provider defaults to `auto`: it uses OpenAI when `OPENAI_API_KEY` is set and offline mock providers otherwise, so it runs with no keys at all.

## Run it

Requires Python 3.10+.

```bash
git clone <this repo> && cd ContentForge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your OPENAI_API_KEY in .env
uvicorn contentforge.api:app --reload
```

Open **http://127.0.0.1:8000** for the demo page, or **http://127.0.0.1:8000/docs** for the API explorer. Press Ctrl+C to stop the server. Without an API key, the demo runs in offline mock mode.

On the demo page, pick a content type and topic and press **Generate** (real models take about 20–60 seconds):
- the **Pipeline** row fills in each step (draft, critique score, refine, guardrails, image) with its time, tokens and cost
- **Final content** shows the finished piece and its generated image
- **Draft vs final** puts the first draft next to the version rewritten after self-critique. Set **Refine when** to **Always refine** to guarantee a rewrite.
- **Critique**, **Cost & latency** and **Raw JSON** show what happened at each step

Call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "Why cities are adding bike lanes", "content_type": "blog_post", "refine_threshold": 11}'
```

Or use the command line, without the server:

```bash
python -m contentforge.cli "Noise-cancelling headphones" --type product_description --image
```

### Troubleshooting

| You see | What to do |
|---|---|
| "Draft vs final" says no refine ran | The critique scored the draft 8 or higher. Set **Refine when** to **Always refine**. |
| "Offline mock mode" badge | `.env` has no `OPENAI_API_KEY`, or `DEFAULT_TEXT_PROVIDER` is set to `mock`. Use `auto` or `openai`, then restart. |
| "Model provider error … 401" | The key is wrong or was revoked. |
| "… 429" or "insufficient_quota" | Add billing or credit to your OpenAI account. |
| Text works, image fails with "verified" | `gpt-image-1` needs organization verification in the OpenAI dashboard. Complete it, or set `OPENAI_IMAGE_MODEL=dall-e-3`. |

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | `/generate/text` | Structured text: draft → critique → refine |
| POST | `/generate/image` | One image from a prompt |
| POST | `/generate` | Text, plus an image from the text's `image_prompt` |
| GET | `/templates` | All prompt templates, with version and fingerprint |
| GET | `/schemas/{content_type}` | The JSON schema an output is validated against |
| GET | `/metrics` | Total cost, average latency and the most recent runs |
| GET | `/` | The interactive demo page |
| GET | `/health` | Status and which API keys are configured |
| GET | `/images/{file}` | Serves generated images |

Main request fields: `refine_threshold` (refine when the critique score is below this; `11` = always), `content_type` (`blog_post`, `social_post`, `product_description`), `topic`, `audience`, `tone`, `extra_instructions`, `provider` (`auto`, `mock`, `openai`, `anthropic`, `gemini`), `model`, `template_version` and `refine`. For `/generate`, you can also set `generate_image`, `image_provider` (`auto`, `mock`, `openai`, `stability`) and `image_size`.

A response includes the validated `content`, the `critique`, whether it was `refined`, the `templates` used (name, version and fingerprint), any `guardrail_flags`, and `metrics`. The metrics give tokens, latency and cost for each step, plus totals.

## How it's organised

```
contentforge/
├── config.py         settings from .env
├── schemas.py        output schemas (BlogPost, SocialPost, ...) + API request models
├── schema_utils.py   Pydantic model → strict JSON schema for LLM APIs
├── prompts.py        versioned prompt templates
├── guardrails.py     input checks + output PII/blocked-term redaction
├── tracking.py       per-call cost/latency + running metrics
├── pipeline.py       draft → critique → refine, image generation, text+image
├── api.py            FastAPI app
├── cli.py            command-line entry point
├── static/index.html the demo page
└── providers/
    ├── base.py               the TextProvider / ImageProvider interfaces
    ├── mock.py               offline providers (no keys needed)
    ├── openai_provider.py    chat completions + images
    ├── anthropic_provider.py messages API with forced tool use
    ├── gemini_provider.py    generateContent with a JSON schema
    └── stability_provider.py Stable Image Core (diffusion)
```

The providers call each API directly over HTTP with `httpx`, so you don't need to install any vendor SDKs.

## Common changes

- **New content type:** add a Pydantic model in `schemas.py`, register it in `CONTENT_TYPES` and the `ContentType` literal, then add a draft template to `prompts.py`.
- **New prompt version:** add a `PromptTemplate("blog_post", "v3", ...)` to `prompts.py`. The latest version is used by default, and you can pin an older one with `template_version`.
- **New provider:** subclass `TextProvider` or `ImageProvider`, then add it to `providers/__init__.py`.
- **Prices:** `tracking.py` has per-model price tables. The numbers are estimates, so check them against each provider's pricing page.

## Tests

```bash
python -m unittest discover -s tests -t .      # or: pytest
```

The tests run offline. The provider tests use `httpx.MockTransport` to check the exact requests sent to OpenAI, Anthropic, Gemini and Stability, and how their responses are parsed.
