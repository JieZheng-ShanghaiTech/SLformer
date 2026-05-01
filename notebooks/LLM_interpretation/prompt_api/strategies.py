"""Prompting strategies built on a single configured LLM client."""

from __future__ import annotations

from dataclasses import dataclass

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

    def to_dict(self) -> dict[str, str | None]:
        return {
            "initial": self.initial.text,
            "feedback": self.feedback.text if self.feedback else None,
            "refined": self.refined.text if self.refined else None,
            "questions": self.questions.text if self.questions else None,
            "answers": self.answers.text if self.answers else None,
            "final": self.final.text if self.final else None,
        }


def run_baseline(client: LLMClientProtocol, prompt: str) -> StrategyTrace:
    initial = client.complete(prompt)
    return StrategyTrace(name="baseline", initial=initial, final=initial)


def run_self_refine(client: LLMClientProtocol, prompt: str) -> StrategyTrace:
    n_rounds = int(client.settings.self_refine_rounds)
    initial = client.complete(prompt)
    trace = StrategyTrace(name="self_refine", initial=initial)
    current = trace.initial

    for _ in range(n_rounds):
        feedback = client.complete(
            self_refine_feedback_prompt(
                original_prompt=str(prompt),
                model_response=str(current.text),
            )
        )
        refined = client.complete(
            self_refine_rewrite_prompt(
                original_prompt=str(prompt),
                model_response=str(current.text),
                feedback=str(feedback.text),
            )
        )
        trace.feedback = feedback
        trace.refined = refined
        current = refined

    trace.final = current
    return trace


def run_cove(client: LLMClientProtocol, prompt: str) -> StrategyTrace:
    question_count = int(client.settings.cove_num_questions)
    initial = client.complete(prompt)
    trace = StrategyTrace(name="cove", initial=initial)

    questions = client.complete(
        cove_questions_prompt(
            original_prompt=str(prompt),
            draft_response=str(trace.initial.text),
            n_questions=question_count,
        )
    )
    answers = client.complete(
        cove_answers_prompt(
            original_prompt=str(prompt),
            draft_response=str(trace.initial.text),
            questions=str(questions.text),
        )
    )
    final = client.complete(
        cove_revise_prompt(
            original_prompt=str(prompt),
            draft_response=str(trace.initial.text),
            verification_answers=str(answers.text),
        )
    )

    trace.questions = questions
    trace.answers = answers
    trace.final = final
    return trace
