"""
Tool definitions the agent can call, in Anthropic's tool-use schema.

"""

from . import mock_backend

TOOL_SCHEMAS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search NimbusCart's policy knowledge base (shipping, returns, exchanges, "
            "payments, cancellations, account issues, loyalty program). Use this whenever "
            "a question is about company policy rather than a specific customer's order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language description of what to look up, e.g. 'return window for unused items'.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_order_status",
        "description": "Look up the live status, ETA, and tracking number for a specific order by its order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. 'ORD1001'. Ask the customer for this if they haven't given it.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Escalate to a human support agent by filing a ticket. Use this only when you "
            "cannot resolve the issue yourself with search_knowledge_base or "
            "check_order_status — e.g. fraud reports, disputes, or requests outside policy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_email": {"type": "string", "description": "The customer's email address."},
                "subject": {"type": "string", "description": "A short one-line summary of the issue."},
                "description": {"type": "string", "description": "Full details a human agent would need to follow up."},
            },
            "required": ["customer_email", "subject", "description"],
        },
    },
]


def dispatch_tool(name: str, tool_input: dict, retriever, tickets_path: str) -> dict:
    #Run the requested tool and return a JSON-serializable result.
    if name == "search_knowledge_base":
        chunks = retriever.retrieve(tool_input["query"], k=3)
        return {
            "results": [
                {"heading": c.heading, "text": c.text, "relevance_score": round(c.score, 3)}
                for c in chunks
            ]
        }

    if name == "check_order_status":
        return mock_backend.check_order_status(tool_input["order_id"])

    if name == "create_support_ticket":
        return mock_backend.create_support_ticket(
            customer_email=tool_input["customer_email"],
            subject=tool_input["subject"],
            description=tool_input["description"],
            tickets_path=tickets_path,
        )

    return {"error": f"Unknown tool: {name}"}
