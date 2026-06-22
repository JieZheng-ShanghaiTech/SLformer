"""Prompting strategies built on one configured LLM client."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

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
    feedback: list[LLMResponse] = field(default_factory=list)
    refined: list[LLMResponse] = field(default_factory=list)
    questions: LLMResponse | None = None
    answers: LLMResponse | None = None
    final: LLMResponse | None = None

    def to_dict(self) -> dict[str, str | list[str] | None]:
        return {
            "initial": self.initial.text,
            "feedback": [response.text for response in self.feedback],
            "refined": [response.text for response in self.refined],
            "questions": self.questions.text if self.questions else None,
            "answers": self.answers.text if self.answers else None,
            "final": self.final.text if self.final else None,
        }


def run_baseline(client: LLMClientProtocol, prompt: str) -> StrategyTrace:
    initial = client.complete(prompt)
    return StrategyTrace(name="baseline", initial=initial, final=initial)


def _wait_between_calls(wait_s: float) -> None:
    if wait_s > 0:
        print(f"Waiting {wait_s:.0f}s before next strategy API call.")
        time.sleep(float(wait_s))


def run_self_refine(client: LLMClientProtocol, prompt: str) -> StrategyTrace:
    trace = StrategyTrace(name="self_refine", initial=client.complete(prompt))
    current = trace.initial
    for _ in range(int(client.settings.self_refine_rounds)):
        _wait_between_calls(float(client.settings.self_refine_wait_s))
        feedback = client.complete(
            self_refine_feedback_prompt(
                prompt_dir=client.settings.prompt_dir,
                prompt_file=client.settings.post_handling_prompt,
                original_prompt=prompt,
                model_response=current.text,
            )
        )
        _wait_between_calls(float(client.settings.self_refine_wait_s))
        refined = client.complete(
            self_refine_rewrite_prompt(
                prompt_dir=client.settings.prompt_dir,
                prompt_file=client.settings.post_handling_prompt,
                original_prompt=prompt,
                model_response=current.text,
                feedback=feedback.text,
            )
        )
        trace.feedback.append(feedback)
        trace.refined.append(refined)
        current = refined
    trace.final = current
    return trace


def run_cove(client: LLMClientProtocol, prompt: str) -> StrategyTrace:
    trace = StrategyTrace(name="cove", initial=client.complete(prompt))
    current = trace.initial
    for _ in range(int(client.settings.self_refine_rounds)):
        _wait_between_calls(float(client.settings.self_refine_wait_s))
        feedback = client.complete(
            self_refine_feedback_prompt(
                prompt_dir=client.settings.prompt_dir,
                prompt_file=client.settings.post_handling_prompt,
                original_prompt=prompt,
                model_response=current.text,
            )
        )
        _wait_between_calls(float(client.settings.self_refine_wait_s))
        refined = client.complete(
            self_refine_rewrite_prompt(
                prompt_dir=client.settings.prompt_dir,
                prompt_file=client.settings.post_handling_prompt,
                original_prompt=prompt,
                model_response=current.text,
                feedback=feedback.text,
            )
        )
        trace.feedback.append(feedback)
        trace.refined.append(refined)
        current = refined

    _wait_between_calls(float(client.settings.cove_wait_s))
    trace.questions = client.complete(
        cove_questions_prompt(
            prompt_dir=client.settings.prompt_dir,
            prompt_file=client.settings.post_handling_prompt,
            original_prompt=prompt,
            draft_response=current.text,
            n_questions=int(client.settings.cove_num_questions),
        )
    )
    _wait_between_calls(float(client.settings.cove_wait_s))
    trace.answers = client.complete(
        cove_answers_prompt(
            prompt_dir=client.settings.prompt_dir,
            prompt_file=client.settings.post_handling_prompt,
            original_prompt=prompt,
            draft_response=current.text,
            questions=trace.questions.text,
        )
    )
    _wait_between_calls(float(client.settings.cove_wait_s))
    trace.final = client.complete(
        cove_revise_prompt(
            prompt_dir=client.settings.prompt_dir,
            prompt_file=client.settings.post_handling_prompt,
            original_prompt=prompt,
            draft_response=current.text,
            verification_answers=trace.answers.text,
        )
    )
    return trace
