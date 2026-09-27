"""Run the pipeline from the terminal, without starting the server.

    python -m contentforge.cli "Why cities are adding bike lanes"
    python -m contentforge.cli "Noise-cancelling headphones" --type product_description --image
    python -m contentforge.cli "Launch day" --type social_post --provider anthropic
"""

import argparse
import json

from .pipeline import generate_content, generate_text
from .schemas import CONTENT_TYPES, ContentRequest, TextRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="ContentForge command line")
    parser.add_argument("topic")
    parser.add_argument("--type", default="blog_post", choices=list(CONTENT_TYPES))
    parser.add_argument("--provider", default=None, help="mock | openai | anthropic | gemini")
    parser.add_argument("--audience", default="general readers")
    parser.add_argument("--tone", default="clear and friendly")
    parser.add_argument("--no-refine", action="store_true", help="skip critique and refine")
    parser.add_argument("--image", action="store_true", help="also generate an image")
    parser.add_argument("--image-provider", default=None, help="mock | openai | stability")
    args = parser.parse_args()

    common = dict(content_type=args.type, topic=args.topic, audience=args.audience,
                  tone=args.tone, provider=args.provider, refine=not args.no_refine)
    if args.image:
        result = generate_content(ContentRequest(**common, image_provider=args.image_provider))
    else:
        result = generate_text(TextRequest(**common))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
