

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.config import Config
from agent.retriever import get_retriever, load_chunks
from agent.mock_backend import check_order_status, create_support_ticket
from agent.tools import dispatch_tool, TOOL_SCHEMAS
from agent.providers import OpenAICompatibleProvider


KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "faq.md")


class TestChunking(unittest.TestCase):
    def test_loads_all_sections(self):
        chunks = load_chunks(KB_PATH)
        self.assertGreaterEqual(len(chunks), 8)
        headings = [c.heading for c in chunks]
        self.assertIn("Return policy", headings)
        self.assertIn("Shipping times", headings)


class TestTfidfRetriever(unittest.TestCase):
    def setUp(self):
        config = Config(embedding_backend="tfidf", knowledge_base_path=KB_PATH)
        self.retriever = get_retriever(config)

    def test_retrieves_relevant_chunk_for_returns(self):
        results = self.retriever.retrieve("can I get my money back for an unused item", k=3)
        headings = [r.heading for r in results]
        self.assertIn("Return policy", headings)

    def test_retrieves_relevant_chunk_for_shipping(self):
        results = self.retriever.retrieve("how long until my package arrives", k=3)
        headings = [r.heading for r in results]
        self.assertIn("Shipping times", headings)

    def test_returns_k_results(self):
        results = self.retriever.retrieve("password reset", k=2)
        self.assertEqual(len(results), 2)


class TestMockBackend(unittest.TestCase):
    def test_known_order_found(self):
        result = check_order_status("ord1001")  # lowercase on purpose
        self.assertTrue(result["found"])
        self.assertEqual(result["item"], "Wireless Mouse")

    def test_unknown_order_not_found(self):
        result = check_order_status("ORD9999")
        self.assertFalse(result["found"])

    def test_create_ticket_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tickets_path = os.path.join(tmp, "tickets.json")
            ticket = create_support_ticket(
                customer_email="test@example.com",
                subject="Test issue",
                description="Something went wrong.",
                tickets_path=tickets_path,
            )
            self.assertTrue(ticket["ticket_id"].startswith("TCK-"))
            with open(tickets_path) as f:
                saved = json.load(f)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["customer_email"], "test@example.com")


class TestToolDispatch(unittest.TestCase):
    def setUp(self):
        config = Config(embedding_backend="tfidf", knowledge_base_path=KB_PATH)
        self.retriever = get_retriever(config)

    def test_dispatch_search_knowledge_base(self):
        result = dispatch_tool("search_knowledge_base", {"query": "exchange for a different size"}, self.retriever, "/tmp/unused.json")
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)

    def test_dispatch_check_order_status(self):
        result = dispatch_tool("check_order_status", {"order_id": "ORD1003"}, self.retriever, "/tmp/unused.json")
        self.assertEqual(result["status"], "Delivered")

    def test_dispatch_unknown_tool(self):
        result = dispatch_tool("not_a_real_tool", {}, self.retriever, "/tmp/unused.json")
        self.assertIn("error", result)


class TestOpenAICompatibleWireFormat(unittest.TestCase):
    """
    Tests the message/tool translation for Groq/Ollama/OpenAI — pure
    functions, no network access or API key needed.
    """

    def test_tools_to_wire_shape(self):
        wire_tools = OpenAICompatibleProvider._tools_to_wire(TOOL_SCHEMAS)
        self.assertEqual(len(wire_tools), len(TOOL_SCHEMAS))
        first = wire_tools[0]
        self.assertEqual(first["type"], "function")
        self.assertIn("name", first["function"])
        self.assertIn("parameters", first["function"])

    def test_plain_user_message(self):
        wire = OpenAICompatibleProvider._messages_to_wire(
            "system prompt", [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(wire[0], {"role": "system", "content": "system prompt"})
        self.assertEqual(wire[1], {"role": "user", "content": "hello"})

    def test_assistant_tool_use_becomes_tool_calls(self):
        history = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "check_order_status", "input": {"order_id": "ORD1001"}}],
            }
        ]
        wire = OpenAICompatibleProvider._messages_to_wire("sys", history)
        assistant_msg = wire[1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertEqual(len(assistant_msg["tool_calls"]), 1)
        call = assistant_msg["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "check_order_status")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"order_id": "ORD1001"})

    def test_tool_result_becomes_tool_role_message(self):
        history = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "some result"}],
            }
        ]
        wire = OpenAICompatibleProvider._messages_to_wire("sys", history)
        self.assertEqual(wire[1], {"role": "tool", "tool_call_id": "call_1", "content": "some result"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
