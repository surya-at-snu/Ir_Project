"""Multinomial Naive Bayes for predicting a film's genre and decade from text (lecture on text classification).

    P(c | x)  ∝  P(c) * prod_t P(t | c)^{n_t(x)}
    P(t | c) = (N_tc + alpha) / (N_c + alpha * |V|)          (Laplace / Lidstone smoothing)

Genres are multi-label, so each (document, genre) pair becomes one training example
("copy transformation"); the model then outputs a distribution over genres.

Naive Bayes posteriors are notoriously over-confident because the independence assumption
multiplies many correlated likelihoods.  `predict_proba` therefore tempers the log-likelihood by
a temperature T (power-scaling), which we choose on held-out data.  The engine uses the tempered
posteriors as *soft* filters: films whose genre/decade the query predicts get a small bonus, nothing
is ever removed.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class MultinomialNB:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.classes_: list = []
        self.log_prior = None
        self.log_lik = None       # (C, V)
        self.vocab: dict[str, int] = {}

    def fit(self, docs: list[list[str]], labels: list[list], vocab: dict[str, int] | None = None) -> "MultinomialNB":
        """docs: token lists; labels: list of label lists (one or more labels per doc)."""
        if vocab is None:
            df = {}
            for toks in docs:
                for t in set(toks):
                    df[t] = df.get(t, 0) + 1
            n = len(docs)
            keep = sorted(t for t, c in df.items() if c >= 3 and c <= 0.6 * n)
            vocab = {t: i for i, t in enumerate(keep)}
        self.vocab = vocab
        self.classes_ = sorted({l for ls in labels for l in ls})
        cidx = {c: i for i, c in enumerate(self.classes_)}
        C, V = len(self.classes_), len(vocab)
        counts = np.zeros((C, V), dtype=np.float64)
        prior = np.zeros(C, dtype=np.float64)
        X = self._vectorize(docs)
        for i, ls in enumerate(labels):
            row = X.getrow(i)
            for l in ls:
                counts[cidx[l], row.indices] += row.data
                prior[cidx[l]] += 1.0
        self.log_prior = np.log(prior / prior.sum())
        sm = counts + self.alpha
        self.log_lik = np.log(sm / sm.sum(axis=1, keepdims=True))
        return self

    def _vectorize(self, docs: list[list[str]]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for i, toks in enumerate(docs):
            cnt = {}
            for t in toks:
                j = self.vocab.get(t)
                if j is not None:
                    cnt[j] = cnt.get(j, 0) + 1
            for j, c in cnt.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float64)

    def joint_log_likelihood(self, docs: list[list[str]]) -> np.ndarray:
        X = self._vectorize(docs)
        return np.asarray(X @ self.log_lik.T) + self.log_prior

    def predict_proba(self, docs: list[list[str]], temperature: float | None = None) -> np.ndarray:
        """Tempered posterior.  temperature=None uses T = max(1, sqrt(#known tokens))."""
        X = self._vectorize(docs)
        ll = np.asarray(X @ self.log_lik.T)
        if temperature is None:
            n = np.asarray(X.sum(axis=1)).ravel()
            T = np.maximum(1.0, np.sqrt(np.maximum(n, 1.0)))[:, None]
        else:
            T = temperature
        z = ll / T + self.log_prior
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, docs: list[list[str]]) -> list:
        jll = self.joint_log_likelihood(docs)
        return [self.classes_[i] for i in jll.argmax(axis=1)]

    def top_features(self, cls, k: int = 15) -> list[str]:
        """Most indicative words for a class: largest log P(t|c) - mean_c' log P(t|c')."""
        i = self.classes_.index(cls)
        score = self.log_lik[i] - self.log_lik.mean(axis=0)
        inv = {j: t for t, j in self.vocab.items()}
        return [inv[j] for j in np.argsort(-score)[:k]]
