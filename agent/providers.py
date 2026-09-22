

from __future__ import annotations

import json
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, messages: list, tools: list, max_tokens: int = 1024) -> dict:
        
        raise NotImplementedError


class AnthropicProvider(LLMProvider):

    def __init__(self, api_key: str, model: str):
        import anthropic

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system, messages, tools, max_tokens=1024):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        return {"stop_reason": response.stop_reason, "content": content}


class OpenAICompatibleProvider(LLMProvider):

    def __init__(self, api_key: str, model: str, base_url: str):
        from openai import OpenAI

        # Ollama ignores the API key entirely; Groq/OpenAI require a real one.
        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self.model = model

    @staticmethod
    def _tools_to_wire(tools: list) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _messages_to_wire(system: str, messages: list) -> list:
        wire = [{"role": "system", "content": system}]
        for msg in messages:
            role, content = msg["role"], msg["content"]

            if role == "user" and isinstance(content, str):
                wire.append({"role": "user", "content": content})

            elif role == "user" and isinstance(content, list):
                # Our canonical tool-result messages -> one OpenAI "tool" message each.
                for block in content:
                    if block.get("type") == "tool_result":
                        wire.append(
                            {
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": block["content"],
                            }
                        )

            elif role == "assistant":
                text = " ".join(b["text"] for b in content if b["type"] == "text").strip()
                tool_calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    }
                    for b in content
                    if b["type"] == "tool_use"
                ]
                assistant_msg = {"role": "assistant", "content": text or None}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                wire.append(assistant_msg)

        return wire

    def complete(self, system, messages, tools, max_tokens=1024):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages_to_wire(system, messages),
            tools=self._tools_to_wire(tools),
            max_tokens=max_tokens,
        )
        message = response.choices[0].message

        content = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        if message.tool_calls:
            for call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.function.name,
                        "input": json.loads(call.function.arguments),
                    }
                )

        stop_reason = "tool_use" if message.tool_calls else "end_turn"
        return {"stop_reason": stop_reason, "content": content}


_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    "openai": "https://api.openai.com/v1",
}


def get_provider(config) -> LLMProvider:
    if config.provider == "anthropic":
        return AnthropicProvider(api_key=config.anthropic_api_key, model=config.model)

    if config.provider in _BASE_URLS:
        base_url = config.openai_compatible_base_url or _BASE_URLS[config.provider]
        api_key = {"groq": config.groq_api_key, "openai": config.openai_api_key}.get(config.provider, "")
        return OpenAICompatibleProvider(api_key=api_key, model=config.model, base_url=base_url)

    raise ValueError(f"Unknown provider: {config.provider!r}. Use 'groq', 'ollama', 'openai', or 'anthropic'.")
