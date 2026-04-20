from __future__ import annotations

from dataclasses import dataclass

from . import config
from .client import LLMClientProtocol, LLMResponse
from .strategy_prompts import (
    cove_answers_prompt,
    cove_questions_prompt,
    cove_revise_prompt,
    self_refine_feedback_prompt,
    self_refine_rewrite_prompt,
)


@dataclass
class StrategyTrace:
    name: str
    initial: LLMResponse
    feedback: LLMResponse | None = None
    refined: LLMResponse | None = None
    questions: LLMResponse | None = None
    answers: LLMResponse | None = None
    final: LLMResponse | None = None


def _build_prompt(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def run_baseline(
    client: LLMClientProtocol,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> StrategyTrace:
    initial = client.complete(
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return StrategyTrace(name="baseline", initial=initial, final=initial)


def run_self_refine(
    client: LLMClientProtocol,
    prompt: str,
    *,
    draft: LLMResponse | str | None = None,
    system_prompt: str | None = None,
    rounds: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> StrategyTrace:
    """Self-refine loop tailored to the current embedding-explanation prompt.

    Goal: minimal semantic drift while improving compliance with the *existing* prompt constraints.
    """

    rounds = max(1, int(config.SELF_REFINE_ROUNDS if rounds is None else rounds))
    initial = (
        draft
        if isinstance(draft, LLMResponse)
        else (LLMResponse(text=str(draft), model="draft") if draft is not None else None)
    )
    if initial is None:
        initial = client.complete(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    trace = StrategyTrace(name="self_refine", initial=initial)

    current = trace.initial
    for _ in range(rounds):
        feedback_prompt = self_refine_feedback_prompt(
            original_prompt=str(prompt or ""),
            model_response=str(current.text or ""),
        )
        feedback = client.complete(
            feedback_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        rewrite_prompt = self_refine_rewrite_prompt(
            original_prompt=str(prompt or ""),
            model_response=str(current.text or ""),
            feedback=str(feedback.text or ""),
        )
        refined = client.complete(
            rewrite_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        trace.feedback = feedback
        trace.refined = refined
        current = refined

    trace.final = current
    return trace


def run_cove(
    client: LLMClientProtocol,
    prompt: str,
    *,
    draft: LLMResponse | str | None = None,
    system_prompt: str | None = None,
    n_questions: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> StrategyTrace:
    """Chain-of-Verification (CoVe) tailored to the current embedding-explanation prompt."""

    n_questions = max(1, int(config.COVE_NUM_QUESTIONS if n_questions is None else n_questions))
    initial = (
        draft
        if isinstance(draft, LLMResponse)
        else (LLMResponse(text=str(draft), model="draft") if draft is not None else None)
    )
    if initial is None:
        initial = client.complete(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    trace = StrategyTrace(name="cove", initial=initial)

    q_prompt = cove_questions_prompt(
        original_prompt=str(prompt or ""),
        draft_response=str(trace.initial.text or ""),
        n_questions=int(n_questions),
    )
    questions = client.complete(
        q_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    a_prompt = cove_answers_prompt(
        original_prompt=str(prompt or ""),
        draft_response=str(trace.initial.text or ""),
        questions=str(questions.text or ""),
    )
    answers = client.complete(
        a_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    revise_prompt = cove_revise_prompt(
        original_prompt=str(prompt or ""),
        draft_response=str(trace.initial.text or ""),
        verification_answers=str(answers.text or ""),
    )
    final = client.complete(
        revise_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    trace.questions = questions
    trace.answers = answers
    trace.final = final
    return trace
