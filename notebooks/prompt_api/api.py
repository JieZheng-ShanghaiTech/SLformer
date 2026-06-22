"""Small high-level wrapper around the configured chat client."""

from __future__ import annotations

from pathlib import Path

from .client import AigcBestChatClient, LLMResponse


class SLformerAPI:
    def __init__(self, *, config_path: str | Path | None = None) -> None:
        self.client = AigcBestChatClient(config_path=config_path)

    def complete(self, prompt: str) -> LLMResponse:
        return self.client.complete(prompt)
