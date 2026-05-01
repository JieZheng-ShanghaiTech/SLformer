"""LLM client and prompt processing for OpenAI-compatible chat APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol
from urllib.request import Request, urlopen

from . import config
from .generator import generate_prompt
from .models import GenePair


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage
    raw: Any


class LLMClientProtocol(Protocol):
    def complete(self, prompt: str) -> LLMResponse:
        ...


def _usage_from_response(obj: dict[str, Any]) -> LLMUsage:
    usage = obj["usage"]
    return LLMUsage(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
    )


class AigcBestChatClient:
    """OpenAI-compatible `/chat/completions` client configured by JSON."""

    def __init__(self, *, config_path: str | Path | None = None) -> None:
        self.settings = config.load_config(config_path) if config_path is not None else config.CONFIG

    def complete(self, prompt: str) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.settings.model.split("/")[-1],
            "messages": [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": str(prompt)},
            ],
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
        }
        if self.settings.max_tokens is not None:
            payload["max_tokens"] = self.settings.max_tokens

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.settings.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.settings.request_timeout_s) as response:
            obj = json.loads(response.read().decode("utf-8"))

        return LLMResponse(
            text=str(obj["choices"][0]["message"]["content"]),
            model=str(payload["model"]),
            usage=_usage_from_response(obj),
            raw=obj,
        )


class PromptProcessor:
    """Generate prompts, query the LLM client, and optionally save outputs."""

    def __init__(self, client: LLMClientProtocol | None = None, *, config_path: str | Path | None = None) -> None:
        self.client = client if client is not None else AigcBestChatClient(config_path=config_path)
        self.settings = self.client.settings

    def generate_explanation(
        self,
        gene_pair: GenePair,
        context: str,
        save_output: bool = True,
        output_dir: Path | str | None = None,
        score_override: float | None = None,
        strategy: str = "baseline",
    ) -> Dict[str, Any]:
        prompt = generate_prompt(gene_pair, context=context, score_override=score_override)
        strategy_key = str(strategy).replace("-", "_").strip().lower()
        trace = self._run_strategy(strategy_key, prompt)
        response = trace.final.text if trace.final is not None else trace.initial.text
        result: Dict[str, Any] = {"prompt": prompt, "strategy": strategy_key, "response": response}
        if trace.name != "baseline":
            result["trace"] = trace.to_dict()
        if save_output:
            result["output_path"] = str(save_response(response, gene_pair, context=context, output_dir=output_dir, settings=self.settings))
        return result

    def _run_strategy(self, strategy_key: str, prompt: str):
        from .strategies import run_baseline, run_cove, run_self_refine

        runners = {
            "baseline": lambda: run_baseline(self.client, prompt),
            "self_refine": lambda: run_self_refine(self.client, prompt),
            "cove": lambda: run_cove(self.client, prompt),
        }
        return runners[strategy_key]()


def save_response(
    response: str,
    gene_pair: GenePair,
    context: str | None = None,
    output_dir: Path | str | None = None,
    settings: config.PromptAPIConfig = config.CONFIG,
) -> Path:
    base_dir = Path(output_dir) if output_dir else settings.output_dir
    out_dir = base_dir / settings.cross_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    context_str = context.lower() if context else "unknown"
    out_path = out_dir / f"{gene_pair.primary}-{gene_pair.partner}_{context_str}.txt"
    out_path.write_text(response, encoding="utf-8")
    return out_path
