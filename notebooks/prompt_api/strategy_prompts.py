"""Text-file prompt templates for post-handling strategies."""

from __future__ import annotations

from pathlib import Path


def _sections(path: Path) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            sections[current] = []
        else:
            sections[current].append(line)
    return {key: "".join(value).strip() for key, value in sections.items() if key}


def _render(prompt_dir: str | Path, prompt_file: str, label: str, **values: object) -> str:
    template = _sections(Path(prompt_dir) / prompt_file)[label]
    return template.format(**{key: str(value or "") for key, value in values.items()}).strip()


def self_refine_feedback_prompt(*, prompt_dir: str | Path, prompt_file: str, original_prompt: str, model_response: str) -> str:
    return _render(prompt_dir, prompt_file, "self_refine_feedback", original_prompt=original_prompt, model_response=model_response)


def self_refine_rewrite_prompt(*, prompt_dir: str | Path, prompt_file: str, original_prompt: str, model_response: str, feedback: str) -> str:
    return _render(prompt_dir, prompt_file, "self_refine_rewrite", original_prompt=original_prompt, model_response=model_response, feedback=feedback)


def cove_questions_prompt(*, prompt_dir: str | Path, prompt_file: str, original_prompt: str, draft_response: str, n_questions: int) -> str:
    return _render(prompt_dir, prompt_file, "cove_questions", original_prompt=original_prompt, draft_response=draft_response, n_questions=n_questions)


def cove_answers_prompt(*, prompt_dir: str | Path, prompt_file: str, original_prompt: str, draft_response: str, questions: str) -> str:
    return _render(prompt_dir, prompt_file, "cove_answers", original_prompt=original_prompt, draft_response=draft_response, questions=questions)


def cove_revise_prompt(*, prompt_dir: str | Path, prompt_file: str, original_prompt: str, draft_response: str, verification_answers: str) -> str:
    return _render(prompt_dir, prompt_file, "cove_revise", original_prompt=original_prompt, draft_response=draft_response, verification_answers=verification_answers)
