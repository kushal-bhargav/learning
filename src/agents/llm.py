from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"


class StructuredLLM(Protocol):
    provider: LLMProvider

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        temperature: float,
        model: str,
    ) -> dict[str, Any]: ...


class ProviderUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class HTTPStructuredLLM:
    provider: LLMProvider
    timeout_seconds: float = 120.0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        temperature: float,
        model: str,
    ) -> dict[str, Any]:
        handlers = {
            LLMProvider.OLLAMA: self._ollama,
            LLMProvider.AZURE_OPENAI: self._azure_openai,
            LLMProvider.OPENAI: self._openai,
            LLMProvider.GEMINI: self._gemini,
            LLMProvider.CLAUDE: self._claude,
        }
        return handlers[self.provider](
            system_prompt, user_prompt, dict(schema), temperature, model
        )

    def _ollama(
        self, system: str, user: str, schema: dict[str, Any], temperature: float, model: str
    ) -> dict[str, Any]:
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        payload = {
            "model": os.getenv("OLLAMA_MODEL", model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature},
        }
        response = self._request(f"{host}/api/chat", payload)
        return self._parse_json(response["message"]["content"])

    def _openai(
        self, system: str, user: str, schema: dict[str, Any], temperature: float, model: str
    ) -> dict[str, Any]:
        key = self._require("OPENAI_API_KEY")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        payload = self._openai_payload(system, user, schema, temperature, model)
        response = self._request(
            f"{base}/chat/completions", payload, {"Authorization": f"Bearer {key}"}
        )
        return self._parse_json(response["choices"][0]["message"]["content"])

    def _azure_openai(
        self, system: str, user: str, schema: dict[str, Any], temperature: float, model: str
    ) -> dict[str, Any]:
        key = self._require("AZURE_OPENAI_API_KEY")
        endpoint = self._require("AZURE_OPENAI_ENDPOINT").rstrip("/")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", model)
        version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        url = (
            f"{endpoint}/openai/deployments/{quote(deployment)}/chat/completions?"
            + urlencode({"api-version": version})
        )
        payload = self._openai_payload(system, user, schema, temperature, model)
        payload.pop("model", None)
        response = self._request(url, payload, {"api-key": key})
        return self._parse_json(response["choices"][0]["message"]["content"])

    def _gemini(
        self, system: str, user: str, schema: dict[str, Any], temperature: float, model: str
    ) -> dict[str, Any]:
        key = os.getenv("GEMINI_API_KEY") or self._require("GOOGLE_API_KEY")
        model = os.getenv("GEMINI_MODEL", model)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model)}:"
            f"generateContent?{urlencode({'key': key})}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        response = self._request(url, payload)
        return self._parse_json(response["candidates"][0]["content"]["parts"][0]["text"])

    def _claude(
        self, system: str, user: str, schema: dict[str, Any], temperature: float, model: str
    ) -> dict[str, Any]:
        key = self._require("ANTHROPIC_API_KEY")
        payload = {
            "model": os.getenv("ANTHROPIC_MODEL", model),
            "max_tokens": 2048,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{"name": "submit_result", "description": "Return the structured agent result.", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "submit_result"},
        }
        response = self._request(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        for block in response["content"]:
            if block.get("type") == "tool_use" and block.get("name") == "submit_result":
                return dict(block["input"])
        raise ValueError("Claude response did not contain submit_result tool input")

    @staticmethod
    def _openai_payload(
        system: str, user: str, schema: dict[str, Any], temperature: float, model: str
    ) -> dict[str, Any]:
        return {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "agent_result", "strict": True, "schema": schema},
            },
        }

    def _request(
        self, url: str, payload: Mapping[str, Any], headers: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise ProviderUnavailableError(f"{self.provider.value} request failed: {exc}") from exc

    @staticmethod
    def _parse_json(value: str) -> dict[str, Any]:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Structured model response must be a JSON object")
        return parsed

    @staticmethod
    def _require(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise ProviderUnavailableError(f"Missing required environment variable: {name}")
        return value


def select_provider(explicit: str | LLMProvider | None = None) -> LLMProvider:
    requested = explicit or os.getenv("GMGI_LLM_PROVIDER")
    if requested:
        return LLMProvider(requested)
    if all(os.getenv(name) for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")):
        return LLMProvider.AZURE_OPENAI
    if os.getenv("OPENAI_API_KEY"):
        return LLMProvider.OPENAI
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return LLMProvider.GEMINI
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLMProvider.CLAUDE
    return LLMProvider.OLLAMA


def create_llm(provider: str | LLMProvider | None = None) -> HTTPStructuredLLM:
    return HTTPStructuredLLM(select_provider(provider))
