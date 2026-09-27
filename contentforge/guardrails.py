"""Input and output guardrails.

Input:  reject over-long text, prompt-injection attempts and blocked terms.
Output: redact personal data (emails, phone numbers) and blocked terms from
        the generated content, and report what was changed as "flags".
"""

import re


class GuardrailError(ValueError):
    """Raised when a request is rejected by a guardrail."""


_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) instructions",
    r"disregard (all |any )?(the )?(previous|prior|above)",
    r"reveal (your|the) (system )?prompt",
    r"you are now (a|an|in) ",
]
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")


def check_input(text: str, max_chars: int, blocked_terms: list[str]) -> None:
    """Raise GuardrailError if the text should not be sent to a model."""
    if len(text) > max_chars:
        raise GuardrailError(f"Input is {len(text)} characters; the limit is {max_chars}.")
    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            raise GuardrailError("Input looks like a prompt-injection attempt.")
    for term in blocked_terms:
        if term and term in lowered:
            raise GuardrailError(f"Input contains a blocked term: '{term}'.")


def check_output(data, blocked_terms: list[str]) -> tuple[object, list[str]]:
    """Redact PII and blocked terms in every string. Returns (clean_data, flags)."""
    flags: list[str] = []

    def clean(value, path):
        if isinstance(value, dict):
            return {k: clean(v, f"{path}.{k}" if path else k) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if not isinstance(value, str):
            return value
        text = value
        if _EMAIL.search(text):
            text = _EMAIL.sub("[email removed]", text)
            flags.append(f"{path}: email address redacted")
        if _PHONE.search(text):
            text = _PHONE.sub("[phone removed]", text)
            flags.append(f"{path}: phone number redacted")
        for term in blocked_terms:
            if term and term in text.lower():
                text = re.sub(re.escape(term), "[removed]", text, flags=re.IGNORECASE)
                flags.append(f"{path}: blocked term '{term}' removed")
        return text

    return clean(data, ""), flags
