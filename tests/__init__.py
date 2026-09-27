"""Tests always run offline on the mock providers, even if .env has real API keys."""

import os

os.environ["DEFAULT_TEXT_PROVIDER"] = "mock"
os.environ["DEFAULT_IMAGE_PROVIDER"] = "mock"
