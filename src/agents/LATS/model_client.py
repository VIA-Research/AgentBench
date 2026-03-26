import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence

from langchain_openai import ChatOpenAI


Message = Dict[str, str]


class OpenAIChatClient:
    """Small ChatOpenAI wrapper used by LATS."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.llm = ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url,
            temperature=temperature,
            stream_usage=True,
        )
        self.model = model
        self.temperature = temperature
        self.completion_tokens = 0
        self.prompt_tokens = 0

    def _temperature(self, temperature: Optional[float]) -> float:
        if temperature is None:
            return self.temperature
        return temperature

    def _accumulate_usage(self, usage: Any) -> None:
        if usage is None:
            return
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
            self.prompt_tokens += int(prompt or 0)
            self.completion_tokens += int(completion or 0)
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    @staticmethod
    def _message_to_text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    chunks.append(item)
            return "".join(chunks)
        return str(content)

    def _accumulate_message_usage(self, message: Any) -> None:
        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            self._accumulate_usage(usage_metadata)
            return
        response_metadata = getattr(message, "response_metadata", None)
        if isinstance(response_metadata, dict):
            self._accumulate_usage(response_metadata.get("token_usage"))

    def _build_call_kwargs(
        self,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> Dict[str, Any]:
        call_kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            call_kwargs["temperature"] = self._temperature(temperature)
        return call_kwargs

    def chat(
        self,
        messages: Sequence[Message],
        n: int = 1,
        stop: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        base_kwargs = self._build_call_kwargs(max_tokens=max_tokens, temperature=temperature)
        llm = self.llm.bind(**base_kwargs) if base_kwargs else self.llm
        outputs: List[str] = []
        for _ in range(n):
            message = llm.invoke(list(messages), stop=stop)
            self._accumulate_message_usage(message)
            outputs.append(self._message_to_text(message))
        return outputs

    async def achat(
        self,
        messages: Sequence[Message],
        n: int = 1,
        stop: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        base_kwargs = self._build_call_kwargs(max_tokens=max_tokens, temperature=temperature)
        llm = self.llm.bind(**base_kwargs) if base_kwargs else self.llm
        tasks = [asyncio.create_task(llm.ainvoke(list(messages), stop=stop)) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        outputs: List[str] = []
        for message in results:
            self._accumulate_message_usage(message)
            outputs.append(self._message_to_text(message))
        return outputs

    async def achat_batch(
        self,
        batch_messages: Sequence[Sequence[Message]],
        n: int = 1,
        stop: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[List[str]]:
        tasks = [
            asyncio.create_task(
                self.achat(
                    messages=messages,
                    n=n,
                    stop=stop,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            for messages in batch_messages
        ]
        return await asyncio.gather(*tasks)

    def chat_batch(
        self,
        batch_messages: Sequence[Sequence[Message]],
        n: int = 1,
        stop: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[List[str]]:
        return asyncio.run(
            self.achat_batch(
                batch_messages=batch_messages,
                n=n,
                stop=stop,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )

    def text_chat(
        self,
        prompt: str,
        n: int = 1,
        stop: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages=messages,
            n=n,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def text_chat_batch(
        self,
        prompts: Sequence[str],
        n: int = 1,
        stop: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[List[str]]:
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        return self.chat_batch(
            batch_messages=batch_messages,
            n=n,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def get_usage(self) -> Dict[str, float]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
