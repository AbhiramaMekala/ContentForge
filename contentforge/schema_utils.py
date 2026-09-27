"""Turn Pydantic models into JSON schemas that LLM APIs accept.

Pydantic emits `$ref`/`$defs` and validation keywords (maxLength, minItems...)
that some structured-output APIs reject. `to_llm_schema` inlines the refs,
moves the constraints into the field description (so the model still sees
them), and makes every object strict: all fields required, no extra keys.
Pydantic re-checks the real constraints after the model responds.
"""

import copy

from pydantic import BaseModel

# Keywords moved out of the schema into the description text.
_CONSTRAINTS = ("minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum")
# Keywords dropped entirely.
_DROP = {"title", "default", "format", "pattern", "exclusiveMinimum", "exclusiveMaximum"}


def inline_refs(schema: dict) -> dict:
    """Replace every {"$ref": "#/$defs/X"} with the definition of X."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, list):
            return [resolve(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return resolve(defs[node["$ref"].split("/")[-1]])
        return {k: resolve(v) for k, v in node.items()}

    return resolve(schema)


def to_llm_schema(model: type[BaseModel]) -> dict:
    """Strict, self-contained JSON schema for structured-output APIs."""

    def clean(node):
        if isinstance(node, list):
            return [clean(n) for n in node]
        if not isinstance(node, dict):
            return node

        out: dict = {}
        hints = []
        for key, value in node.items():
            if key in _CONSTRAINTS:
                hints.append(f"{key}={value}")
            elif key in _DROP:
                continue
            elif key == "properties":  # keys here are field names, not keywords
                out[key] = {name: clean(sub) for name, sub in value.items()}
            else:
                out[key] = clean(value)

        if hints:
            desc = out.get("description", "")
            out["description"] = f"{desc} (constraints: {', '.join(hints)})".strip()
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}))
        return out

    return clean(inline_refs(model.model_json_schema()))
