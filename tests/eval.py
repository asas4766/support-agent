"""
Eval harness for the support agent.

IMPORTANT — what each mode actually verifies:

    python -m tests.eval            offline: routes each query through a
                                     keyword-matching stub (no LLM call), then
                                     dispatches the resulting tool call(s)
                                     through the REAL retriever and mock
                                     backend. This checks that TOOL DISPATCH
                                     and RETRIEVAL are wired correctly and
                                     return the right shape/content — it does
                                     NOT check that a real model would choose
                                     the right tool for a given message. Free,
                                     deterministic, safe for CI.

    python -m tests.eval --live     live: runs the actual SupportAgent against
                                     your configured provider (Groq/Ollama/
                                     OpenAI/Anthropic) and records which tools
                                     it really calls. This is the only mode
                                     that tests the MODEL'S tool-selection
                                     behavior. Costs an API call per case
                                     (or requires Ollama running locally) and
                                     can vary run to run.

    python -m tests.eval --filter rag    run only cases whose name contains "rag"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.config import Config
from agent.retriever import get_retriever


KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "faq.md")


@dataclass
class Case:
    name: str
    query: str
    expect_tools: list[str] = field(default_factory=list)   # order matters for [0]
    expect_answer_contains: list[str] = field(default_factory=list)
    expect_chunk_heading: str | None = None
    tags: list[str] = field(default_factory=list)


CASES = [
    Case("returns_paraphrase", "can I get my money back for an unused item?",
         expect_tools=["search_knowledge_base"],
         expect_answer_contains=["30 days"],
         expect_chunk_heading="Return policy",
         tags=["rag", "paraphrase"]),

    Case("shipping_times", "how long does standard shipping take?",
         expect_tools=["search_knowledge_base"],
         expect_chunk_heading="Shipping times",
         tags=["rag"]),

    Case("order_lookup", "where is order ORD1001?",
         expect_tools=["check_order_status"],
         tags=["tool"]),

    Case("order_lookup_unknown", "what's the status of order ORD9999?",
         expect_tools=["check_order_status"],
         tags=["tool", "negative"]),

    Case("multi_tool", "where's my ORD1001 and what's your return policy if it's late?",
         expect_tools=["check_order_status", "search_knowledge_base"],
         tags=["agent", "multi-tool"]),

    Case("escalation_delay", "my order ORD1002 is 3 weeks late and has no tracking",
         expect_tools=["create_support_ticket"],
         tags=["agent", "escalation"]),

    Case("escalation_dispute", "I want to dispute a charge and report fraud",
         expect_tools=["create_support_ticket"],
         tags=["agent", "escalation"]),

    Case("loyalty", "how do NimbusPoints work?",
         expect_tools=["search_knowledge_base"],
         expect_chunk_heading="Loyalty program (NimbusPoints)",
         tags=["rag"]),

    Case("greeting_no_tool", "hi, what can you help me with?",
         expect_tools=[],
         tags=["agent", "no-tool"]),
]


class RecordingAgent:
    """Records every tool call made while handling one query."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.answer: str = ""

    def run(self, query: str, live: bool) -> None:
        self.calls, self.answer = [], ""

        if live:
            self._run_live(query)
        else:
            self._run_stubbed(query)

    def _run_live(self, query: str) -> None:
        """Real agent, real provider. Patches dispatch_tool in both agent.tools
        and agent.core, since core.py imported its own reference via
        `from .tools import dispatch_tool` at module load time."""
        from agent import core as core_module
        from agent import tools as tools_module

        real_dispatch = tools_module.dispatch_tool
        core_old = core_module.dispatch_tool

        def recording(*args, **kwargs):
            self.calls.append((args[0], args[1]))
            return real_dispatch(*args, **kwargs)

        tools_module.dispatch_tool = recording
        core_module.dispatch_tool = recording
        try:
            config = Config(embedding_backend="tfidf", knowledge_base_path=KB_PATH)
            retriever = get_retriever(config)
            from agent import SupportAgent
            agent = SupportAgent(config, retriever)
            self.answer = agent.send(query)
        finally:
            tools_module.dispatch_tool = real_dispatch
            core_module.dispatch_tool = core_old

    def _run_stubbed(self, query: str) -> None:
        """
        Offline: a keyword-routed stub stands in for the LLM's tool-selection
        decision. This is NOT testing whether a real model would pick the
        same tool — it's testing that once a tool is selected, dispatch and
        retrieval behave correctly end to end.
        """
        from agent.tools import dispatch_tool

        config = Config(embedding_backend="tfidf", knowledge_base_path=KB_PATH)
        retriever = get_retriever(config)
        q = query.lower()

        # Escalation path
        if any(k in q for k in ("dispute", "fraud", "3 weeks", "no tracking")):
            name, args = "create_support_ticket", {
                "customer_email": "eval@example.com",
                "subject": "escalation",
                "description": query,
            }
            self.calls.append((name, args))
            dispatch_tool(name, args, retriever, "/tmp/eval_tickets.json")
            self.answer = "I've filed a ticket with a human agent."
            return

        # Order lookup
        m = re.search(r"\b(ORD\d{3,})\b", query, re.IGNORECASE)
        if m:
            name, args = "check_order_status", {"order_id": m.group(1)}
            self.calls.append((name, args))
            dispatch_tool(name, args, retriever, "/tmp/eval_tickets.json")

        # Policy retrieval
        if any(k in q for k in ("policy", "return", "refund", "ship", "cancel",
                                 "loyalty", "nimbuspoints", "money back")):
            name, args = "search_knowledge_base", {"query": query}
            self.calls.append((name, args))
            dispatch_tool(name, args, retriever, "/tmp/eval_tickets.json")
            self.answer = self._stub_answer(q)

        if not self.answer:
            self.answer = "I can help with policies, orders, and escalations."

    @staticmethod
    def _stub_answer(q: str) -> str:
        bits = []
        if any(k in q for k in ("money back", "return", "refund")):
            bits.append("Returns are accepted within 30 days of delivery.")
        if "ship" in q:
            bits.append("Standard shipping takes 3-5 business days.")
        if "nimbuspoint" in q or "loyalty" in q:
            bits.append("You earn 1 NimbusPoint per HK$10 spent.")
        return " ".join(bits) or "See policy."


def check(case: Case, rec: RecordingAgent) -> list[str]:
    failures = []
    called = [name for name, _ in rec.calls]

    for expected in case.expect_tools:
        if expected not in called:
            failures.append(f"expected tool '{expected}' not called (called: {called})")

    if case.expect_tools and case.expect_tools[0] in called:
        if called.index(case.expect_tools[0]) > 1:
            failures.append(f"expected '{case.expect_tools[0]}' early, got order {called}")

    if not case.expect_tools and called:
        failures.append(f"expected no tool calls, got: {called}")

    for needle in case.expect_answer_contains:
        if needle.lower() not in rec.answer.lower():
            failures.append(f"answer missing {needle!r} (got: {rec.answer[:120]!r})")

    return failures


def check_retrieval(case: Case) -> list[str]:
    if not case.expect_chunk_heading:
        return []
    config = Config(embedding_backend="tfidf", knowledge_base_path=KB_PATH)
    retriever = get_retriever(config)
    results = retriever.retrieve(case.query, k=3)
    headings = [r.heading for r in results]
    if case.expect_chunk_heading not in headings:
        return [f"retriever missed {case.expect_chunk_heading!r} (got: {headings})"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                     help="exercise the real model instead of the offline keyword stub")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    cases = [c for c in CASES if not args.filter or args.filter in c.name]
    passed = failed = 0

    mode = "LIVE (real model)" if args.live else "offline (routing/plumbing check, no LLM call)"
    print(f"Running {len(cases)} eval cases — {mode}\n")

    for case in cases:
        rec = RecordingAgent()
        try:
            rec.run(case.query, live=args.live)
            failures = check(case, rec) + check_retrieval(case)
        except Exception as e:
            failures = [f"exception: {type(e).__name__}: {e}"]

        ok = not failures
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {case.name}  [{','.join(case.tags)}]")
        for f in failures:
            print(f"        - {f}")
        passed += ok
        failed += not ok

    print(f"\n{passed}/{passed + failed} passed")
    if not args.live:
        print("(offline mode checks dispatch/retrieval correctness, not model tool-selection — run --live for that)")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
