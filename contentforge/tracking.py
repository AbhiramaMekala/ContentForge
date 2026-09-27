"""Cost and latency tracking.

Each request gets a RunTracker that records every model call (tokens, latency,
estimated cost). Finished runs are added to a process-wide MetricsStore that
backs the GET /metrics endpoint.

Prices are ESTIMATES in USD — check your provider's pricing page and edit
these tables. Unknown models are tracked with a cost of 0.
"""

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

# USD per 1M tokens: (input, output)
TEXT_PRICES: dict[str, tuple[float, float]] = {
    "mock": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}

# USD per image
IMAGE_PRICES: dict[str, float] = {
    "mock-image": 0.0,
    "gpt-image-1": 0.04,
    "dall-e-3": 0.04,
    "stable-image-core": 0.03,
}


def text_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = TEXT_PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class StepMetric:
    step: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class RunTracker:
    """Collects the metrics for one request."""

    def __init__(self, kind: str):
        self.kind = kind
        self.steps: list[StepMetric] = []
        self._started = time.perf_counter()

    def add(self, metric: StepMetric) -> None:
        self.steps.append(metric)

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "total_latency_ms": round((time.perf_counter() - self._started) * 1000, 1),
            "total_input_tokens": sum(s.input_tokens for s in self.steps),
            "total_output_tokens": sum(s.output_tokens for s in self.steps),
            "total_cost_usd": round(sum(s.cost_usd for s in self.steps), 6),
            "steps": [asdict(s) for s in self.steps],
        }


class MetricsStore:
    """Thread-safe running totals plus the most recent runs."""

    def __init__(self, keep_last: int = 50):
        self._lock = threading.Lock()
        self._recent: deque = deque(maxlen=keep_last)
        self._totals = {"runs": 0, "model_calls": 0, "cost_usd": 0.0, "latency_ms": 0.0}

    def record(self, summary: dict) -> None:
        with self._lock:
            self._recent.append(summary)
            self._totals["runs"] += 1
            self._totals["model_calls"] += len(summary["steps"])
            self._totals["cost_usd"] += summary["total_cost_usd"]
            self._totals["latency_ms"] += summary["total_latency_ms"]

    def snapshot(self) -> dict:
        with self._lock:
            runs = self._totals["runs"]
            return {
                "runs": runs,
                "model_calls": self._totals["model_calls"],
                "total_cost_usd": round(self._totals["cost_usd"], 6),
                "avg_latency_ms": round(self._totals["latency_ms"] / runs, 1) if runs else 0.0,
                "recent_runs": list(self._recent)[-10:],
            }

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._totals = {"runs": 0, "model_calls": 0, "cost_usd": 0.0, "latency_ms": 0.0}


metrics_store = MetricsStore()
