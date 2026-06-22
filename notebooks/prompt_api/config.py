"""YAML configuration for the prompt API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_CLIENT_CONFIG_PATH = Path(__file__).with_name("client_config.yaml")


@dataclass(frozen=True)
class PromptAPIConfig:
    base_url: str
    pinned_ip: str | None
    api_key: str
    api_mode: str
    user_agent: str
    content_type: str
    model: str
    stream: bool
    system_prompt: str
    include_system_prompt: bool
    temperature: float
    top_p: float
    reasoning_effort: str
    max_tokens: int | None
    request_timeout_s: float
    request_retries: int
    retry_wait_s: float
    retry_status_codes: tuple[int, ...]
    self_refine_wait_s: float
    cove_wait_s: float
    prompt_dir: Path
    post_handling_prompt: str
    self_refine_rounds: int
    cove_num_questions: int


def _config_path(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def load_config(config_path: str | Path = DEFAULT_CLIENT_CONFIG_PATH) -> PromptAPIConfig:
    client_path = Path(config_path).expanduser().resolve()
    client_data = yaml.safe_load(client_path.read_text(encoding="utf-8"))
    client_dir = client_path.parent
    client = client_data["client"]
    request = client_data["request"]
    strategy_wait = client_data["strategy_wait"]

    model_config_path = _config_path(client_dir, str(client["model_config_path"]))
    model_data = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    model_dir = model_config_path.parent
    model = model_data["model"]
    strategy = model_data["strategy"]
    prompts = model_data["prompts"]
    max_tokens = model["max_tokens"]

    return PromptAPIConfig(
        base_url=str(model["base_url"]).rstrip("/"),
        pinned_ip=None if model["pinned_ip"] is None else str(model["pinned_ip"]),
        api_key=str(model["api_key"]).strip(),
        api_mode=str(model["api_mode"]).strip().lower(),
        user_agent=str(model["user_agent"]),
        content_type=str(client_data["headers"]["content_type"]),
        model=str(model["name"]),
        stream=bool(model["stream"]),
        system_prompt=str(model["system_prompt"]),
        include_system_prompt=bool(model["include_system_prompt"]),
        temperature=float(model["temperature"]),
        top_p=float(model["top_p"]),
        reasoning_effort=str(model["reasoning_effort"]),
        max_tokens=None if max_tokens is None else int(max_tokens),
        request_timeout_s=float(request["timeout_s"]),
        request_retries=int(request["retries"]),
        retry_wait_s=float(request["retry_wait_s"]),
        retry_status_codes=tuple(int(code) for code in request["retry_status_codes"]),
        self_refine_wait_s=float(strategy_wait["self_refine_wait_s"]),
        cove_wait_s=float(strategy_wait["cove_wait_s"]),
        prompt_dir=_config_path(model_dir, str(prompts["prompt_dir"])),
        post_handling_prompt=str(prompts["post_handling_prompt"]),
        self_refine_rounds=int(strategy["self_refine_rounds"]),
        cove_num_questions=int(strategy["cove_num_questions"]),
    )


CONFIG: PromptAPIConfig | None = None
