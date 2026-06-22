"""LLM strategy helpers for SAE interpretation notebooks."""

from __future__ import annotations

import re
from typing import Any

from prompt_api.strategies import StrategyTrace, run_baseline, run_cove, run_self_refine


def run_llm_strategy(client: Any, prompt: str, strategy: str) -> StrategyTrace:
    strategy_key = str(strategy).replace("-", "_").strip().lower()
    runners = {
        "baseline": lambda: run_baseline(client, prompt),
        "self_refine": lambda: run_self_refine(client, prompt),
        "cove": lambda: run_cove(client, prompt),
    }
    return runners[strategy_key]()


def clean_markdown_bullets(text: str) -> str:
    return re.sub(r"(?m)^(\s*)[\*•]\s+", r"\1- ", str(text))


def final_strategy_text(trace: StrategyTrace) -> str:
    return clean_markdown_bullets(trace.final.text if trace.final is not None else trace.initial.text)


def strategy_report_text(trace: StrategyTrace) -> str:
    if trace.name == "baseline":
        return final_strategy_text(trace)

    sections = [final_strategy_text(trace), "", "# === TRACE (for debugging) ==="]
    sections.extend(["[Initial draft]", clean_markdown_bullets(trace.initial.text).strip(), ""])
    for round_index, feedback in enumerate(trace.feedback, start=1):
        label = "[Self-refine feedback]" if len(trace.feedback) == 1 else f"[Self-refine feedback round {round_index}]"
        sections.extend([label, clean_markdown_bullets(feedback.text).strip(), ""])
    for round_index, refined in enumerate(trace.refined, start=1):
        label = "[Self-refine revised draft]" if len(trace.refined) == 1 else f"[Self-refine revised draft round {round_index}]"
        sections.extend([label, clean_markdown_bullets(refined.text).strip(), ""])
    if trace.questions is not None:
        sections.extend(["[CoVe questions]", clean_markdown_bullets(trace.questions.text).strip(), ""])
    if trace.answers is not None:
        sections.extend(["[CoVe answers]", clean_markdown_bullets(trace.answers.text).strip(), ""])
    return "\n".join(sections).rstrip() + "\n"


def strategy_total_tokens(trace: StrategyTrace) -> int:
    responses = [trace.initial, trace.questions, trace.answers, trace.final, *trace.feedback, *trace.refined]
    return int(sum(response.usage.total_tokens for response in responses if response is not None))


def strategy_call_count(strategy: str, *, self_refine_rounds: int, cove_num_questions: int) -> int:
    strategy_key = str(strategy).replace("-", "_").strip().lower()
    if strategy_key == "baseline":
        return 1
    if strategy_key == "self_refine":
        return 1 + 2 * int(self_refine_rounds)
    if strategy_key == "cove":
        return 4 + 2 * int(self_refine_rounds)
    raise KeyError(strategy_key)
