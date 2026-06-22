"""Small OpenAI-compatible chat/response client."""

from __future__ import annotations

import io
import json
import socket
import ssl
import time
from dataclasses import dataclass
from http.client import HTTPSConnection, RemoteDisconnected
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.response import addinfourl
from urllib.parse import urlparse

from . import config


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
    settings: config.PromptAPIConfig

    def complete(self, prompt: str) -> LLMResponse:
        ...


def _usage_from_chat(obj: dict[str, Any]) -> LLMUsage:
    usage = obj["usage"]
    return LLMUsage(
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        total_tokens=int(usage["total_tokens"]),
    )


def _usage_from_response(obj: dict[str, Any]) -> LLMUsage:
    usage = obj["usage"]
    prompt_tokens = int(usage["input_tokens"])
    completion_tokens = int(usage["output_tokens"])
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(usage["total_tokens"]),
    )


def _response_text(obj: dict[str, Any]) -> str:
    if "output_text" in obj:
        return str(obj["output_text"])
    texts = []
    for item in obj["output"]:
        for content in item["content"]:
            if content["type"] == "output_text":
                texts.append(str(content["text"]))
    return "\n".join(texts)


def _http_error_detail(error: HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace") if error.fp is not None else ""
    content_type = str(error.headers.get("Content-Type", "")).lower()
    body_lower = body.lower().lstrip()
    if not body or "html" in content_type or body_lower.startswith("<"):
        return ""
    try:
        obj = json.loads(body)
        return json.dumps(obj, ensure_ascii=False)[:2000]
    except json.JSONDecodeError:
        return body[:2000]


def _json_from_sse(data: bytes) -> dict[str, Any]:
    final_obj = None
    output_text_parts = []
    event_name = ""
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
            continue
        if not line.startswith("data:"):
            continue
        event_data = line[len("data:"):].strip()
        if not event_data or event_data == "[DONE]":
            continue
        obj = json.loads(event_data)
        if event_name == "error" or "error" in obj:
            error = obj["error"]
            error_type = str(error.get("type", "")) if isinstance(error, dict) else ""
            error_code = str(error.get("code", "")) if isinstance(error, dict) else ""
            if error_type == "service_unavailable_error" or error_code == "server_is_overloaded":
                raise RuntimeError(f"retriable_streaming_error: {error}")
            raise RuntimeError(f"Streaming LLM error: {error}")
        event_type = str(obj.get("type", event_name))
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            output_text_parts.append(str(obj["delta"]))
        elif event_type in {"response.completed", "response.done"}:
            final_obj = obj["response"]
        elif "output" in obj and "usage" in obj:
            final_obj = obj
    if final_obj is not None:
        if not _response_text(final_obj) and output_text_parts:
            final_obj["output_text"] = "".join(output_text_parts)
        return final_obj
    if output_text_parts:
        return {
            "output_text": "".join(output_text_parts),
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "stream_incomplete": True,
        }
    return None


def _read_response_body(data: bytes) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace").lstrip()
    if text.startswith("data:") or "\ndata:" in text:
        obj = _json_from_sse(data)
        if obj is None:
            raise ValueError("Streaming response ended without a completed response event")
        return obj
    return json.loads(text)


class SNIHTTPSConnection(HTTPSConnection):
    def __init__(self, ip: str, host: str, port: int, timeout: float) -> None:
        super().__init__(ip, port=port, timeout=timeout, context=ssl.create_default_context())
        self._sni_host = host

    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), self.timeout, self.source_address)
        self.sock = self._context.wrap_socket(sock, server_hostname=self._sni_host)


class AigcBestChatClient:
    def __init__(self, *, config_path: str | Path | None = None) -> None:
        self.settings = config.load_config(config_path) if config_path is not None else config.load_config()

    def complete(self, prompt: str) -> LLMResponse:
        if self.settings.api_mode == "response":
            payload = self._response_payload(prompt)
            obj = self._post_json(self.settings.base_url, payload)
            return LLMResponse(
                text=_response_text(obj),
                model=str(payload["model"]),
                usage=_usage_from_response(obj),
                raw=obj,
            )

        payload = self._completion_payload(prompt)
        obj = self._post_json(self.settings.base_url, payload)
        return LLMResponse(
            text=str(obj["choices"][0]["message"]["content"]),
            model=str(payload["model"]),
            usage=_usage_from_chat(obj),
            raw=obj,
        )

    def _completion_payload(self, prompt: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": str(prompt)}]
        if self.settings.include_system_prompt:
            messages.insert(0, {"role": "system", "content": self.settings.system_prompt})
        payload: dict[str, Any] = {
            "model": self.settings.model.split("/")[-1],
            "messages": messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
        }
        if self.settings.max_tokens is not None:
            payload["max_tokens"] = self.settings.max_tokens
        return payload

    def _response_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model.split("/")[-1],
            "input": str(prompt),
            "stream": self.settings.stream,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "reasoning": {"effort": self.settings.reasoning_effort},
        }
        if self.settings.include_system_prompt:
            payload["instructions"] = self.settings.system_prompt
        if self.settings.max_tokens is not None:
            payload["max_output_tokens"] = self.settings.max_tokens
        return payload

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, int(self.settings.request_retries) + 1):
            started = time.monotonic()
            try:
                return self._post_json_once(url, payload)
            except HTTPError as error:
                detail = _http_error_detail(error)
                message = (
                    f"LLM HTTP {error.code} {error.reason} on attempt {attempt}/{self.settings.request_retries} "
                    f"after {time.monotonic() - started:.1f}s; model={self.settings.model}"
                )
                if detail:
                    message += f"; detail={detail}"
                if error.code not in self.settings.retry_status_codes or attempt == int(self.settings.request_retries):
                    raise RuntimeError(message) from error
                print(f"{message}; retrying same model in {self.settings.retry_wait_s:.0f}s.")
                time.sleep(float(self.settings.retry_wait_s))
            except RuntimeError as error:
                if ("rate_limit" not in str(error) and "retriable_streaming_error" not in str(error)) or attempt == int(self.settings.request_retries):
                    raise
                print(
                    f"LLM streaming retriable error on attempt {attempt}/{self.settings.request_retries} "
                    f"after {time.monotonic() - started:.1f}s; model={self.settings.model}; "
                    f"retrying same model in {self.settings.retry_wait_s:.0f}s."
                )
                time.sleep(float(self.settings.retry_wait_s))
            except ValueError as error:
                if "Streaming response ended without a completed response event" not in str(error) or attempt == int(self.settings.request_retries):
                    raise
                print(
                    f"LLM streaming response incomplete on attempt {attempt}/{self.settings.request_retries} "
                    f"after {time.monotonic() - started:.1f}s; model={self.settings.model}; "
                    f"retrying same model in {self.settings.retry_wait_s:.0f}s."
                )
                time.sleep(float(self.settings.retry_wait_s))
            except (TimeoutError, URLError, OSError, RemoteDisconnected) as error:
                if attempt == int(self.settings.request_retries):
                    raise
                print(
                    f"LLM transport error on attempt {attempt}/{self.settings.request_retries} "
                    f"after {time.monotonic() - started:.1f}s: {error}; retrying same model "
                    f"{self.settings.model} in {self.settings.retry_wait_s:.0f}s."
                )
                time.sleep(float(self.settings.retry_wait_s))
        raise RuntimeError("unreachable LLM retry state")

    def _post_json_once(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = urlparse(url)
        host = str(parsed.hostname)
        port = int(parsed.port or 443)
        connect_host = self.settings.pinned_ip or host
        body = json.dumps(payload).encode("utf-8")
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        conn = SNIHTTPSConnection(connect_host, host, port, float(self.settings.request_timeout_s))
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Host": host,
                "Content-Type": self.settings.content_type,
                "User-Agent": self.settings.user_agent,
                "Authorization": f"Bearer {self.settings.api_key}",
            },
        )
        response = conn.getresponse()
        data = response.read()
        conn.close()
        if response.status >= 400:
            error_body = addinfourl(io.BytesIO(data), response.headers, url)
            raise HTTPError(url, response.status, response.reason, response.headers, error_body)
        return _read_response_body(data)
