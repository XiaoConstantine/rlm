from collections import defaultdict
from typing import Any

import anthropic

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class AnthropicClient(BaseLM):
    """
    LM Client for running models with the Anthropic API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str | None = None,
        max_tokens: int = 32768,
        enable_caching: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.enable_caching = enable_caching

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.model_cache_creation_tokens: dict[str, int] = defaultdict(int)
        self.model_cache_read_tokens: dict[str, int] = defaultdict(int)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        messages, system = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")

        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        # Use streaming to handle long responses (required for max_tokens > 21333)
        parts: list[str] = []
        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                parts.append(text)
            response = stream.get_final_message()

        full_response = "".join(parts)

        self._track_cost(response, model)
        return full_response

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        messages, system = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")

        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        # Use streaming to handle long responses (required for max_tokens > 21333)
        parts: list[str] = []
        async with self.async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                parts.append(text)
            response = await stream.get_final_message()

        full_response = "".join(parts)

        self._track_cost(response, model)
        return full_response

    def _prepare_messages(
        self, prompt: str | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Prepare messages and extract system prompt for Anthropic API.

        When caching is enabled, adds cache_control breakpoints to:
        1. System prompt (cached across all calls)
        2. Last assistant message in conversation history (cached across iterations)
        """
        system = None

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            # Extract system message if present (Anthropic handles system separately)
            messages = []
            for msg in prompt:
                if msg.get("role") == "system":
                    content = msg.get("content")
                    if self.enable_caching:
                        # Enable caching on system prompt with cache_control
                        system = [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    else:
                        system = content
                else:
                    messages.append(msg)

            # Add cache breakpoint to last assistant message for conversation history caching
            if self.enable_caching and messages:
                self._add_cache_breakpoint_to_last_assistant(messages)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        return messages, system

    def _add_cache_breakpoint_to_last_assistant(self, messages: list[dict[str, Any]]) -> None:
        """Add cache_control breakpoint to the last assistant message in the conversation.

        This enables prefix caching for multi-turn conversations, so that the
        conversation history up to this point can be cached and reused.

        Note: Anthropic limits cache_control to 4 blocks max. We first remove any
        existing cache_control from assistant messages, then add to only the last one.
        This ensures we stay within the limit (1 for system + 1 for last assistant = 2).
        """
        # First pass: remove any existing cache_control from all messages
        # This prevents accumulation as conversation history grows
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        del block["cache_control"]

        # Second pass: add cache_control only to the last assistant message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "")
                # Convert string content to structured format with cache_control
                if isinstance(content, str):
                    messages[i]["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(content, list):
                    # Content is already structured, add cache_control to last text block
                    for j in range(len(content) - 1, -1, -1):
                        if content[j].get("type") == "text":
                            content[j]["cache_control"] = {"type": "ephemeral"}
                            break
                break

    def _track_cost(self, response: anthropic.types.Message, model: str):
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += response.usage.input_tokens
        self.model_output_tokens[model] += response.usage.output_tokens
        self.model_total_tokens[model] += response.usage.input_tokens + response.usage.output_tokens

        # Track cache metrics if available
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        self.model_cache_creation_tokens[model] += cache_creation
        self.model_cache_read_tokens[model] += cache_read

        # Track last call for handler to read
        self.last_prompt_tokens = response.usage.input_tokens
        self.last_completion_tokens = response.usage.output_tokens
        self.last_cache_creation_tokens = cache_creation
        self.last_cache_read_tokens = cache_read

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                cache_creation_tokens=self.model_cache_creation_tokens[model],
                cache_read_tokens=self.model_cache_read_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            cache_creation_tokens=getattr(self, "last_cache_creation_tokens", 0),
            cache_read_tokens=getattr(self, "last_cache_read_tokens", 0),
        )
