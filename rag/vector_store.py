import os
import json
import openai
import numpy as np


def embed_texts(texts, model="text-embedding-3-small"):
    """Create embeddings for a list of texts using OpenAI embeddings API."""
    if len(texts) == 0:
        return []
    resp = openai.Embedding.create(input=texts, model=model)
    return [item["embedding"] for item in resp["data"]]


class SimpleVectorStore:
    """A tiny file-backed vector store (JSON) suitable for demos.

    Stores items as list of {id, text, embedding}.
    Querying computes cosine similarity over all vectors (works for small corpora).
    """

    def __init__(self, path="vector_store.json"):
        self.path = path
        self.data = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f)

    def add(self, docs, embeddings):
        for doc, emb in zip(docs, embeddings):
            self.data.append({"id": doc["id"], "text": doc["text"], "embedding": emb})
        self._save()

    def query(self, query_text, k=3, embed_model="text-embedding-3-small"):
        if len(self.data) == 0:
            return []
        q_emb = embed_texts([query_text], model=embed_model)[0]
        qv = np.array(q_emb, dtype=float)
        results = []
        for item in self.data:
            try:
                v = np.array(item["embedding"], dtype=float)
                score = float(np.dot(qv, v) / ((np.linalg.norm(qv) * np.linalg.norm(v)) + 1e-10))
            except Exception:
                score = -1.0
            results.append((score, item))
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:k]]
