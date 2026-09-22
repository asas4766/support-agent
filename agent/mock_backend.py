"""
Stand-ins for NimbusCart's real order-management and ticketing systems.

"""

import json
import os
import uuid
from datetime import datetime, timedelta

_ORDERS = {
    "ORD1001": {
        "status": "Shipped",
        "item": "Wireless Mouse",
        "eta": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
        "tracking_number": "HK1234567890",
    },
    "ORD1002": {
        "status": "Processing",
        "item": "Mechanical Keyboard",
        "eta": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
        "tracking_number": None,
    },
    "ORD1003": {
        "status": "Delivered",
        "item": "USB-C Hub",
        "eta": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "tracking_number": "HK9876543210",
    },
    "ORD1004": {
        "status": "Cancelled",
        "item": "Laptop Stand",
        "eta": None,
        "tracking_number": None,
    },
    "ORD1005": {
        # Genuinely overdue: still "Processing" past its ETA, with no tracking
        # number ever assigned. Used by the escalation_delay eval case — unlike
        # ORD1002, this one's data actually backs up a "this is late" complaint.
        "status": "Processing",
        "item": "Desk Lamp",
        "eta": (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d"),
        "tracking_number": None,
    },
}


def check_order_status(order_id: str) -> dict:
    order_id = order_id.strip().upper()
    order = _ORDERS.get(order_id)
    if order is None:
        return {"found": False, "order_id": order_id, "message": "No order found with this ID."}
    return {"found": True, "order_id": order_id, **order}


def create_support_ticket(
    customer_email: str,
    subject: str,
    description: str,
    tickets_path: str,
) -> dict:
    ticket = {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "customer_email": customer_email,
        "subject": subject,
        "description": description,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "open",
    }

    os.makedirs(os.path.dirname(tickets_path), exist_ok=True)
    tickets = []
    if os.path.exists(tickets_path):
        with open(tickets_path, "r", encoding="utf-8") as f:
            try:
                tickets = json.load(f)
            except json.JSONDecodeError:
                tickets = []

    tickets.append(ticket)
    with open(tickets_path, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2)

    return ticket