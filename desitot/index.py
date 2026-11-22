"""Inverted index with BM25F and TF-IDF (vector space model) scoring.

Each film document has several fields (title, lead, plot, cast, people, genre, music, other).
For every term the index keeps a posting list: the ids of the documents containing it and,
per posting, the pre-computed BM25F "saturated tf" and the TF-IDF weight.

BM25F (Robertson, Zaragoza & Taylor, CIKM 2004; Robertson & Zaragoza, 2009):

    tf~(t,d) = sum_f  w_f * tf_f(t,d) / ((1 - b_f) + b_f * len_f(d) / avglen_f)
    BM25F(q,d) = sum_{t in q}  q_t * idf(t) * tf~(t,d) * (k1 + 1) / (k1 + tf~(t,d))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

TF-IDF cosine (lecture 6):  w(t,d) = (1 + ln tf(t,d)) * ln(N / df(t)),
    cos(q,d) = sum_t w(t,q) w(t,d) / (|q| |d|)
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .text import analyze


@dataclass
class Posting:
    ids: np.ndarray        # int32 doc ids, ascending
    bm25: np.ndarray       # float32 tf~(k1+1)/(k1+tf~)   (idf applied at query time)
    tfidf: np.ndarray      # float32 (1 + ln tf) * ln(N/df), already divided by the doc norm
    tf: np.ndarray         # float32 field-weighted raw term frequency


class InvertedIndex:
    def __init__(self, field_weights: dict, field_b: dict, k1: float = 1.2):
        self.field_weights = dict(field_weights)
        self.field_b = dict(field_b)
        self.k1 = k1
        self.N = 0
        self.postings: dict[str, Posting] = {}
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.field_len: dict[str, np.ndarray] = {}
        self.avg_len: dict[str, float] = {}
        self.doc_terms: list[Counter] = []        # per-document field-weighted term counts (for NB, PRF, snippets)

    # ------------------------------------------------------------------ building
    def build(self, docs: list[dict[str, str]]) -> "InvertedIndex":
        fields = list(self.field_weights)
        self.N = N = len(docs)
        per_doc: list[dict[str, Counter]] = []
        self.field_len = {f: np.zeros(N, dtype=np.float32) for f in fields}
        for i, doc in enumerate(docs):
            counts = {}
            for f in fields:
                toks = analyze(doc.get(f, "") or "")
                counts[f] = Counter(toks)
                self.field_len[f][i] = len(toks)
            per_doc.append(counts)
        self.avg_len = {f: float(max(self.field_len[f].mean(), 1e-6)) for f in fields}

        raw: dict[str, list] = defaultdict(list)
        self.doc_terms = []
        for i, counts in enumerate(per_doc):
            norm = {f: (1 - self.field_b[f]) + self.field_b[f] * self.field_len[f][i] / self.avg_len[f] for f in fields}
            merged: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for f in fields:
                w = self.field_weights[f]
                for t, c in counts[f].items():
                    m = merged[t]
                    m[0] += w * c / norm[f]      # BM25F pseudo-frequency
                    m[1] += w * c                 # weighted raw tf for TF-IDF
            self.doc_terms.append(Counter({t: v[1] for t, v in merged.items()}))
            for t, (tft, tfw) in merged.items():
                raw[t].append((i, tft, tfw))

        self.df = {t: len(p) for t, p in raw.items()}
        self.idf = {t: math.log(1 + (N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}
        idf_vsm = {t: math.log(N / df) for t, df in self.df.items()}

        doc_norm = np.zeros(N, dtype=np.float64)
        for t, plist in raw.items():
            for i, _, tfw in plist:
                doc_norm[i] += ((1 + math.log(tfw)) * idf_vsm[t]) ** 2 if tfw >= 1 else (tfw * idf_vsm[t]) ** 2
        doc_norm = np.sqrt(np.maximum(doc_norm, 1e-12))

        k1 = self.k1
        for t, plist in raw.items():
            ids = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
            tft = np.fromiter((p[1] for p in plist), dtype=np.float64, count=len(plist))
            tfw = np.fromiter((p[2] for p in plist), dtype=np.float64, count=len(plist))
            logtf = np.where(tfw >= 1, 1 + np.log(np.maximum(tfw, 1e-9)), tfw)
            self.postings[t] = Posting(
                ids=ids,
                bm25=(tft * (k1 + 1) / (k1 + tft)).astype(np.float32),
                tfidf=(logtf * idf_vsm[t] / doc_norm[ids]).astype(np.float32),
                tf=tfw.astype(np.float32))
        self._idf_vsm = idf_vsm
        return self

    # ------------------------------------------------------------------ scoring
    def bm25(self, qweights: dict[str, float]) -> np.ndarray:
        """BM25F score of every document for a weighted bag of query stems."""
        scores = np.zeros(self.N, dtype=np.float32)
        for t, w in qweights.items():
            p = self.postings.get(t)
            if p is not None and w:
                scores[p.ids] += np.float32(w * self.idf[t]) * p.bm25
        return scores

    def tfidf_cosine(self, qweights: dict[str, float]) -> np.ndarray:
        """Cosine similarity between the TF-IDF query vector and every document vector."""
        scores = np.zeros(self.N, dtype=np.float32)
        qnorm = 0.0
        for t, w in qweights.items():
            p = self.postings.get(t)
            if p is None or w <= 0:
                continue
            qt = (1 + math.log(w)) * self._idf_vsm[t] if w >= 1 else w * self._idf_vsm[t]
            scores[p.ids] += np.float32(qt) * p.tfidf
            qnorm += qt * qt
        return scores / np.float32(math.sqrt(qnorm) + 1e-9)

    def term_contributions(self, qweights: dict[str, float], doc: int) -> dict[str, float]:
        """Per-term BM25F contribution for one document (used by the explanation panel)."""
        out = {}
        for t, w in qweights.items():
            p = self.postings.get(t)
            if p is None:
                continue
            j = np.searchsorted(p.ids, doc)
            if j < len(p.ids) and p.ids[j] == doc:
                out[t] = float(w * self.idf[t] * p.bm25[j])
        return out

    def has_term(self, t: str) -> bool:
        return t in self.postings

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "InvertedIndex":
        obj = cls.__new__(cls)
        with open(path, "rb") as fh:
            obj.__dict__.update(pickle.load(fh))
        return obj
