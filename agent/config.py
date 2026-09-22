"""
Central configuration for the support agent.

"""
# Load .env file manually
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed

import os
from dataclasses import dataclass

_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",
    "ollama": "llama3.1",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5-20250929",
}


@dataclass
class Config:
    
    provider: str = os.environ.get("AGENT_PROVIDER", "groq")

    # Only the key matching your chosen provider needs to be set.
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # Override if you're pointing at a self-hosted or third-party
    # OpenAI-compatible endpoint (Together.ai, Fireworks, a remote Ollama, etc).
    openai_compatible_base_url: str = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "")

    model: str = os.environ.get("AGENT_MODEL", "")

    embedding_backend: str = os.environ.get("EMBEDDING_BACKEND", "tfidf")

    knowledge_base_path: str = os.environ.get(
        "KB_PATH", os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "faq.md")
    )
    tickets_path: str = os.environ.get(
        "TICKETS_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "tickets.json")
    )

    top_k: int = int(os.environ.get("RETRIEVAL_TOP_K", "3"))
    max_agent_turns: int = int(os.environ.get("MAX_AGENT_TURNS", "6"))

    def __post_init__(self):
        if not self.model:
            self.model = _DEFAULT_MODELS.get(self.provider, "")

    def api_key_for_provider(self) -> str:
        return {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "ollama": "not-needed",  # Ollama runs locally with no key
        }.get(self.provider, "")


config = Config()
