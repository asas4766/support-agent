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
- When escalating, call create_support_ticket immediately with the best information you have. Do not wait to collect the customer's email first — omit customer_email if you don't have it yet, and ask for it in the same reply where you confirm the ticket was filed.
- If an order is significantly delayed past its ETA, or has no tracking number after
  several days, treat this as a service failure: after checking status, escalate by
  creating a support ticket rather than only reporting the raw status back to the customer.
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
                try:
                    tool_result = dispatch_tool(
                        block["name"], block["input"], self.retriever, self.config.tickets_path
                    )
                except Exception as e:
                    # Don't let a bad tool call (bad input, retriever error, etc.)
                    # crash the whole conversation — feed the error back so the
                    # model can recover or decide to escalate itself.
                    tool_result = {"error": f"{type(e).__name__}: {e}", "tool": block["name"]}
                if on_tool_call:
                    on_tool_call(block["name"], block["input"], tool_result)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block["id"], "content": str(tool_result)}
                )

            self.history.append({"role": "user", "content": tool_results})

        return self._escalate_after_max_turns()

    def _escalate_after_max_turns(self) -> str:
        """
        Called when the loop exhausts its turn budget without producing a
        plain-text answer. Actually files a ticket rather than just telling
        the customer we've escalated — so what we say and what we do match.
        """
        try:
            ticket = dispatch_tool(
                "create_support_ticket",
                {
                    "customer_email": "unknown@nimbuscart.example",  # no email collected yet; a
                                                                       # real deployment should ask
                                                                       # for one earlier in the flow
                    "subject": "Agent could not resolve within turn limit",
                    "description": (
                        "The automated agent hit its turn limit without resolving the "
                        f"customer's issue. Conversation history:\n{self.history}"
                    ),
                },
                self.retriever,
                self.config.tickets_path,
            )
            return (
                "I wasn't able to resolve this on my own, so I've escalated it to a "
                f"human agent (ticket {ticket.get('ticket_id', 'N/A')}). "
                "They'll follow up within 1 business day."
            )
        except Exception:
            # Even escalation failed — be honest, don't claim a ticket exists.
            return (
                "I'm sorry — I couldn't resolve this and I wasn't able to file a "
                "ticket automatically. Please contact support@nimbuscart.example directly."
            )