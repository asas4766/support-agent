"""
The agent loop: send messages with tools attached, and if the model asks to
use a tool, run it, feed the result back, and let the model continue —
repeating until it produces a plain text answer or we hit a turn limit.

"""

from __future__ import annotations

from .providers import get_provider
from .tools import TOOL_SCHEMAS, dispatch_tool

SYSTEM_PROMPT = """You are a customer support agent for NimbusCart, an online retailer.

You have three tools:
- search_knowledge_base: for questions about policy (shipping, returns, exchanges, payments, cancellations, accounts, loyalty program).
- check_order_status: for questions about a specific order, given an order ID.
- create_support_ticket: to escalate to a human when you can't resolve something yourself.

Rules:
- Only use information returned by your tools when answering policy or order questions. Do not guess or invent policy details.
- If you need an order ID and the customer hasn't given one, ask for it before calling check_order_status.
- If a request is outside what your tools can resolve (fraud, disputes, anything policy doesn't cover), create a support ticket rather than guessing, and tell the customer you've done so.
- Be concise and warm. You're talking to a real customer, not writing documentation.
"""


class SupportAgent:
    def __init__(self, config, retriever):
        self.config = config
        self.retriever = retriever
        self.provider = get_provider(config)
        self.history: list[dict] = []

    def reset(self):
        self.history = []

    def send(self, user_message: str, on_tool_call=None) -> str:
        """
        Send one user message, run the tool loop to completion, and return
        the model's final text reply. `on_tool_call(name, input, result)` is
        an optional callback so callers (like the CLI demo) can print what
        the agent is doing under the hood.
        """
        self.history.append({"role": "user", "content": user_message})

        for _ in range(self.config.max_agent_turns):
            result = self.provider.complete(
                system=SYSTEM_PROMPT,
                messages=self.history,
                tools=TOOL_SCHEMAS,
                max_tokens=1024,
            )
            self.history.append({"role": "assistant", "content": result["content"]})

            if result["stop_reason"] != "tool_use":
                return "".join(block["text"] for block in result["content"] if block["type"] == "text")

            tool_results = []
            for block in result["content"]:
                if block["type"] != "tool_use":
                    continue
                tool_result = dispatch_tool(block["name"], block["input"], self.retriever, self.config.tickets_path)
                if on_tool_call:
                    on_tool_call(block["name"], block["input"], tool_result)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block["id"], "content": str(tool_result)}
                )

            self.history.append({"role": "user", "content": tool_results})

        return "I'm having trouble resolving this — let me escalate it to a human agent."
