from __future__ import annotations

from dataclasses import dataclass, field

from .client import ChatResult, ModelServerClient


@dataclass
class Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    last_result: ChatResult | None = None
    enable_thinking: bool = True

    def reset(self, system_prompt: str = "", *, enable_thinking: bool = True) -> None:
        self.messages.clear()
        if system_prompt.strip():
            self.messages.append({"role": "system", "content": system_prompt.strip()})
        self.last_result = None
        self.enable_thinking = enable_thinking

    def send(self, client: ModelServerClient, text: str) -> ChatResult:
        content = text.strip()
        if not content:
            raise ValueError("消息不能为空")

        user_message = {"role": "user", "content": content}
        self.messages.append(user_message)
        try:
            result = client.chat(self.messages, enable_thinking=self.enable_thinking)
        except Exception:
            self.messages.pop()
            raise

        self.messages.append({"role": "assistant", "content": result.content})
        self.last_result = result
        return result
