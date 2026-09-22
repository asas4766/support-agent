"""
Retrieval-augmented generation, minus the framework.

We split the markdown knowledge base into chunks (one per "## " heading),
embed each chunk, and at query time return the top-k most similar chunks.

Two backends are provided behind the same interface:

- TfidfRetriever: scikit-learn TF-IDF + cosine similarity. No network
  access, no model download. Good enough to demo the agent's *behavior*
  (deciding when to retrieve, citing the right chunk) even with no internet.
- SentenceTransformerRetriever: real sentence embeddings via
  `sentence-transformers`, stored in a local Chroma collection. This is
  what you'd actually want in production — it understands paraphrases and
  synonyms that TF-IDF misses (e.g. "money back" retrieving the refund
  policy chunk even though the chunk never says "money back").

Both implement `.retrieve(query, k) -> list[Chunk]`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    heading: str
    text: str
    score: float = 0.0

    def as_context(self) -> str:
        return f"[{self.heading}]\n{self.text.strip()}"


def load_chunks(markdown_path: str) -> list[Chunk]:
    """Split a markdown file into chunks on '## ' headings."""
    with open(markdown_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on level-2 headings, keep the heading with its body.
    parts = re.split(r"\n(?=## )", text)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part.startswith("## "):
            continue  # skip the H1 title block before the first heading
        heading, _, body = part.partition("\n")
        heading = heading.lstrip("# ").strip()
        chunks.append(Chunk(id=f"chunk_{i}", heading=heading, text=body.strip()))
    return chunks


class TfidfRetriever:
    def __init__(self, knowledge_base_path: str):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = load_chunks(knowledge_base_path)
        corpus = [f"{c.heading}. {c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        results = []
        for chunk, score in ranked[:k]:
            results.append(Chunk(id=chunk.id, heading=chunk.heading, text=chunk.text, score=float(score)))
        return results


class SentenceTransformerRetriever:
    """Real semantic embeddings via sentence-transformers + a local Chroma collection."""

    def __init__(self, knowledge_base_path: str, model_name: str = "all-MiniLM-L6-v2"):
        import chromadb
        from chromadb.utils import embedding_functions

        self.chunks = load_chunks(knowledge_base_path)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        client = chromadb.Client()  # in-memory, rebuilt each run
        self.collection = client.create_collection(name="support_kb", embedding_function=embed_fn)
        self.collection.add(
            ids=[c.id for c in self.chunks],
            documents=[f"{c.heading}. {c.text}" for c in self.chunks],
            metadatas=[{"heading": c.heading} for c in self.chunks],
        )
        self._by_id = {c.id: c for c in self.chunks}

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        result = self.collection.query(query_texts=[query], n_results=k)
        chunks = []
        for doc_id, distance in zip(result["ids"][0], result["distances"][0]):
            base = self._by_id[doc_id]
            # Chroma returns a distance (lower = closer); flip to a similarity-ish score for display.
            chunks.append(Chunk(id=base.id, heading=base.heading, text=base.text, score=1.0 - distance))
        return chunks


def get_retriever(config):
    if config.embedding_backend == "sentence-transformers":
        return SentenceTransformerRetriever(config.knowledge_base_path)
    return TfidfRetriever(config.knowledge_base_path)
