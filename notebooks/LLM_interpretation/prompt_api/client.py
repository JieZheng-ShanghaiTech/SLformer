"""LLM client and prompt processing (AIGC Best / OpenAI-compatible)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from . import config
from .generator import generate_prompt
from .models import GenePair


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage = LLMUsage()
    raw: Any = None


class LLMClientProtocol(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        ...


class AigcBestChatClient:
    """OpenAI-compatible /chat/completions client using stdlib urllib."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        import os

        self.base_url = (base_url or config.AIGC_BEST_BASE_URL).rstrip("/")
        self.api_key = (api_key or config.AIGC_API_KEY).strip()
        self.model = model or config.MODEL
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        import json
        from urllib.request import Request, urlopen

        model_id = (model or self.model).split("/")[-1]
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": str(prompt or "")},
            ],
            "temperature": temperature if temperature is not None else config.TEMPERATURE,
            "top_p": top_p if top_p is not None else config.TOP_P,
        }
        effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        body_bytes = json.dumps(payload).encode("utf-8")
        with urlopen(
            Request(url, data=body_bytes, headers=headers, method="POST"),
            timeout=float(config.LLM_REQUEST_TIMEOUT_S),
        ) as resp:
            obj = json.loads(resp.read().decode("utf-8"))

        content = str(obj["choices"][0]["message"]["content"])
        usage_obj = obj.get("usage", {}) or {}
        return LLMResponse(
            text=content,
            model=str(payload["model"]),
            usage=LLMUsage(
                prompt_tokens=usage_obj.get("prompt_tokens"),
                completion_tokens=usage_obj.get("completion_tokens"),
                total_tokens=usage_obj.get("total_tokens"),
            ),
            raw=obj,
        )


class PromptProcessor:
    """Generates prompts, queries GPT, and optionally saves outputs."""

    def __init__(self, client: Optional[LLMClientProtocol] = None) -> None:
        self.client = client or AigcBestChatClient()

    def generate_explanation(
        self,
        gene_pair: GenePair,
        context: str,
        save_output: bool = True,
        output_dir: Optional[Path | str] = None,
        score_override: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strategy: str = "baseline",
        self_refine_rounds: Optional[int] = None,
        cove_questions: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt = generate_prompt(gene_pair, context=context, score_override=score_override)
        if prompt.startswith("Error:"):
            return {"error": prompt}

        strategy_norm = str(strategy or "baseline").strip().lower()
        result: Dict[str, Any] = {"prompt": prompt, "strategy": strategy_norm}

        try:
            if strategy_norm == "baseline":
                final = self.client.complete(
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                result["response"] = final.text
            elif strategy_norm in {"self_refine", "self-refine"}:
                from .strategies import run_self_refine

                trace = run_self_refine(
                    self.client,
                    prompt,
                    rounds=self_refine_rounds,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                result["response"] = trace.final.text if trace.final else trace.initial.text
                result["trace"] = {
                    "initial": trace.initial.text,
                    "feedback": trace.feedback.text if trace.feedback else None,
                    "refined": trace.refined.text if trace.refined else None,
                }
            elif strategy_norm == "cove":
                from .strategies import run_cove

                trace = run_cove(
                    self.client,
                    prompt,
                    n_questions=cove_questions,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                result["response"] = trace.final.text if trace.final else trace.initial.text
                result["trace"] = {
                    "initial": trace.initial.text,
                    "questions": trace.questions.text if trace.questions else None,
                    "answers": trace.answers.text if trace.answers else None,
                }
            else:
                return {"error": f"Unknown strategy: {strategy_norm}", "prompt": prompt}
        except Exception as e:
            result["error"] = f"LLM pipeline failed (strategy={strategy_norm}): {e}"
            return result

        if save_output:
            result["output_path"] = str(
                save_response(
                    result.get("response", ""),
                    gene_pair,
                    context=context,
                    output_dir=output_dir,
                )
            )
        return result


def save_response(
    response: str,
    gene_pair: GenePair,
    context: Optional[str] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    base_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    out_dir = base_dir / config.CROSS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    context_str = context.lower() if context else "unknown"
    filename = f"{gene_pair.primary}-{gene_pair.partner}_{context_str}.txt"
    out_path = out_dir / filename
    out_path.write_text(response, encoding="utf-8")
    return out_path
