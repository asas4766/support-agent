"""
Run this locally to see what the model actually said, not just which
tools it called. Usage:

    python debug_escalation.py
"""
import sys
sys.path.insert(0, ".")
from tests.eval import RecordingAgent

QUERIES = [
    "my order ORD1005 is 3 weeks late and has no tracking",
    "I want to dispute a charge and report fraud",
]

for q in QUERIES:
    rec = RecordingAgent()
    rec.run(q, live=True)
    print("QUERY: ", q)
    print("CALLS: ", rec.calls)
    print("ANSWER:", rec.answer)
    print("-" * 60)
