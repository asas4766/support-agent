#!/usr/bin/env python3
"""
Talk to the NimbusCart support agent.

    python demo.py                 interactive chat
    python demo.py --scripted      plays a fixed set of example questions
                                
"""

import argparse
import sys
import time

from agent import config, SupportAgent, get_retriever

SCRIPT = [
    "Hi, what's your return policy on unused items?",
    "Great — and can you check the status of order ORD1001?",
    "I never received order ORD1002 and it's been three weeks, I think something's wrong and I want a refund now.",
]


def print_tool_call(name, tool_input, result):
    print(f"    \033[2m→ tool call: {name}({tool_input})\033[0m")
    print(f"    \033[2m← result: {result}\033[0m")


def run_turn(agent: SupportAgent, message: str):
    print(f"\n\033[1mYou:\033[0m {message}")
    reply = agent.send(message, on_tool_call=print_tool_call)
    print(f"\033[1mAgent:\033[0m {reply}")


def check_api_key():
    if config.provider == "ollama":
        return  # no key needed for a local model
    if config.api_key_for_provider():
        return

    hints = {
        "groq": "Get a free key (no credit card) at https://console.groq.com, then:\n  export GROQ_API_KEY=gsk-...",
        "openai": "Get a key at https://platform.openai.com, then:\n  export OPENAI_API_KEY=sk-...",
        "anthropic": "Get a key at https://console.anthropic.com, then:\n  export ANTHROPIC_API_KEY=sk-ant-...",
    }
    hint = hints.get(config.provider, f"No API key found for provider '{config.provider}'.")
    sys.exit(f"No API key set for provider '{config.provider}'.\n{hint}\nOr copy .env.example to .env and fill it in.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted", action="store_true", help="run the built-in demo script instead of interactive input")
    parser.add_argument("--delay", type=float, default=1.2, help="seconds between scripted turns (for recording)")
    args = parser.parse_args()

    check_api_key()

    print(f"Provider: {config.provider} | Model: {config.model} | Embeddings: {config.embedding_backend}")
    print("Loading knowledge base...")
    retriever = get_retriever(config)
    agent = SupportAgent(config, retriever)
    print("Ready. (Ctrl+C to quit)\n")

    if args.scripted:
        for message in SCRIPT:
            run_turn(agent, message)
            time.sleep(args.delay)
        return

    try:
        while True:
            message = input("\n\033[1mYou:\033[0m ")
            if not message.strip():
                continue
            reply = agent.send(message, on_tool_call=print_tool_call)
            print(f"\033[1mAgent:\033[0m {reply}")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
