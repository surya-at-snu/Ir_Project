"""Dense (embedding) retrieval over plot passages.

Every film is split into overlapping passages of a few sentences.  A bi-encoder
(BAAI/bge-small-en-v1.5, a 33M-parameter retrieval-tuned sentence encoder) embeds passages and
queries into the same 384-d space; the film's score is its best passage (MaxP, Dai & Callan 2019):

    dense(q, d) = max_{p in passages(d)} cos(E(q), E(p))

Dense retrieval matches paraphrases that BM25 misses ("pretends to be blind" ~ "feigns
blindness") but is known to ignore negation (Weller et al., NevIR, EACL 2024), which is why
exclusions are handled lexically in this project.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import DENSE_MODEL, DENSE_QUERY_PREFIX

_SENT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RX.split(text or "") if len(s.strip()) > 3]


def make_passages(lead: str, plot: str, max_words: int = 90, stride: int = 2) -> list[str]:
    """Lead summary + sliding windows of ~3 plot sentences (<= max_words)."""
    out = []
    lead_s = sentences(lead)
    if lead_s:
        out.append(" ".join(lead_s[:3]))
    sents = sentences(plot)
    i = 0
    while i < len(sents):
        chunk, n, j = [], 0, i
        while j < len(sents) and (n == 0 or n + len(sents[j].split()) <= max_words):
            chunk.append(sents[j])
            n += len(sents[j].split())
            j += 1
        out.append(" ".join(chunk))
        if j >= len(sents):
            break
        i += max(1, min(stride, j - i))
    return out or [lead[:500] or "(no text)"]


class DenseIndex:
    def __init__(self, model_name: str = DENSE_MODEL, query_prefix: str = DENSE_QUERY_PREFIX):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._model = None
        self.emb: np.ndarray | None = None          # (P, D), L2-normalised
        self.starts: np.ndarray | None = None       # first passage of each doc (passages are grouped by doc)
        self.passages: list[str] = []
        self.N = 0
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def build(self, doc_passages: list[list[str]], batch_size: int = 64,
              cache: dict[str, np.ndarray] | None = None) -> "DenseIndex":
        """Embed every passage; passages found in `cache` (text -> vector from a previous build) are reused."""
        self.N = len(doc_passages)
        self.passages = [p for ps in doc_passages for p in ps]
        self.starts = np.cumsum([0] + [len(ps) for ps in doc_passages[:-1]]).astype(np.int64)
        cache = cache or {}
        todo = [p for p in dict.fromkeys(self.passages) if p not in cache]
        if todo:
            vecs = self.model.encode(todo, batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            cache = {**cache, **dict(zip(todo, vecs))}
        self.emb = np.stack([cache[p] for p in self.passages]).astype(np.float32)
        self.reused = len(self.passages) - len(todo)
        return self

    @staticmethod
    def load_cache(folder: Path) -> dict[str, np.ndarray]:
        """text -> vector for every passage of a previously saved index (empty if there is none)."""
        folder = Path(folder)
        try:
            emb = np.load(folder / "dense_emb.npy")
            texts = json.loads((folder / "dense_passages.json").read_text(encoding="utf-8"))
            meta = json.loads((folder / "dense_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if meta.get("model") != DENSE_MODEL or len(texts) != len(emb):
            return {}
        return dict(zip(texts, emb))

    def encode(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            vecs = self.model.encode([self.query_prefix + t for t in todo], batch_size=32,
                                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            for t, v in zip(todo, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.emb.shape[1]), np.float32)

    def passage_sims(self, qvec: np.ndarray) -> np.ndarray:
        return self.emb @ qvec

    def doc_scores(self, qvec: np.ndarray) -> np.ndarray:
        """MaxP score for every document."""
        return np.maximum.reduceat(self.passage_sims(qvec), self.starts)

    def best_passage(self, qvec: np.ndarray, doc: int) -> tuple[int, float]:
        a = int(self.starts[doc])
        b = int(self.starts[doc + 1]) if doc + 1 < self.N else len(self.passages)
        sims = self.emb[a:b] @ qvec
        j = int(np.argmax(sims))
        return a + j, float(sims[j])

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        np.save(folder / "dense_emb.npy", self.emb)
        np.save(folder / "dense_starts.npy", self.starts)
        (folder / "dense_passages.json").write_text(json.dumps(self.passages, ensure_ascii=False), encoding="utf-8")
        (folder / "dense_meta.json").write_text(json.dumps({"model": self.model_name, "prefix": self.query_prefix,
                                                             "N": self.N}), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "DenseIndex":
        folder = Path(folder)
        meta = json.loads((folder / "dense_meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["model"], meta["prefix"])
        obj.N = meta["N"]
        obj.emb = np.load(folder / "dense_emb.npy")
        obj.starts = np.load(folder / "dense_starts.npy")
        obj.passages = json.loads((folder / "dense_passages.json").read_text(encoding="utf-8"))
        return obj
