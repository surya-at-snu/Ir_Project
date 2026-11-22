"""Learning to rank: three ways of combining the ranking features into one score.

1. CoordinateAscent (Metzler & Croft, 2007): a linear model whose weights are tuned one at a
   time by line search to directly maximise MRR on the training queries.
2. PairwiseLogistic (RankNet with a linear scorer; Burges et al., 2005): minimise
       L(w) = mean_q mean_{j in neg(q)} log(1 + exp(-(w . (x_pos - x_j)))) + lambda ||w||^2
3. LambdaMART (Burges, 2010) via LightGBM's lambdarank objective (gradient-boosted trees).

Training data: for every query, a candidate list (feature matrix X_q) with label 1 for the
target film and 0 for every other candidate.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize


def _mrr(groups, scores_list) -> float:
    rr = []
    for (X, y), s in zip(groups, scores_list):
        pos = np.flatnonzero(y > 0)
        if len(pos) == 0:
            rr.append(0.0)
            continue
        rank = 1 + int(np.sum(s > s[pos[0]]))
        rr.append(1.0 / rank)
    return float(np.mean(rr)) if rr else 0.0


class _StackedGroups:
    """All training groups stacked into one matrix, so MRR for a weight vector is one matrix product plus a
    grouped count instead of a Python loop over queries.  Same rank definition as `_mrr`: 1 + the number of
    candidates scored strictly higher than the (first) relevant one; groups without a relevant film count 0."""

    def __init__(self, groups):
        self.n = len(groups)
        sizes = np.array([len(X) for X, _ in groups], dtype=np.int64)
        keep = sizes > 0
        self.X = np.vstack([X for X, _ in groups if len(X)]) if keep.any() else np.zeros((0, 1))
        starts = np.concatenate([[0], np.cumsum(sizes[keep])[:-1]]).astype(np.int64)
        self.starts = starts
        self.gid = np.repeat(np.arange(int(keep.sum())), sizes[keep])
        pos = []
        for (X, y), s0 in zip((g for g in groups if len(g[0])), starts):
            p = np.flatnonzero(np.asarray(y) > 0)
            pos.append(int(s0 + p[0]) if len(p) else -1)
        self.pos = np.array(pos, dtype=np.int64)
        self.valid = self.pos >= 0

    def mrr(self, w: np.ndarray) -> float:
        if not self.n or not len(self.pos):
            return 0.0
        s = self.X @ w
        target = np.where(self.valid, s[np.maximum(self.pos, 0)], np.inf)
        higher = np.add.reduceat((s > target[self.gid]).astype(np.int64), self.starts)
        return float(np.where(self.valid, 1.0 / (1 + higher), 0.0).sum() / self.n)


class _Standardizer:
    def fit(self, groups):
        allx = np.vstack([X for X, _ in groups])
        self.mu = allx.mean(axis=0)
        self.sd = allx.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class LinearModel:
    """Shared save/load/score for the linear rankers (weights live in standardised feature space)."""
    kind = "linear"

    def __init__(self, names):
        self.names = list(names)
        self.w = np.zeros(len(self.names))
        self.std: _Standardizer | None = None

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.std(X) @ self.w

    def contributions(self, x: np.ndarray) -> dict:
        z = (x - self.std.mu) / self.std.sd
        return {n: float(c) for n, c in zip(self.names, z * self.w)}

    def to_json(self) -> dict:
        return {"kind": self.kind, "names": self.names, "w": self.w.tolist(),
                "mu": self.std.mu.tolist(), "sd": self.std.sd.tolist()}

    @classmethod
    def from_json(cls, d: dict):
        m = cls(d["names"])
        m.w = np.array(d["w"])
        m.std = _Standardizer()
        m.std.mu, m.std.sd = np.array(d["mu"]), np.array(d["sd"])
        return m


class CoordinateAscent(LinearModel):
    kind = "coordinate_ascent"

    def fit(self, groups, init: dict | None = None, passes: int = 4, seed: int = 0, nonpositive=()):
        """nonpositive: features whose weight must stay <= 0 (e.g. exclusion penalties: ruling something out
        may lower a film's score but never raise it)."""
        self.std = _Standardizer().fit(groups)
        Z = [(self.std(X), y) for X, y in groups]
        self.w = np.zeros(len(self.names))
        for n, v in (init or {}).items():
            if n in self.names:
                self.w[self.names.index(n)] = v
        if not self.w.any():
            self.w[0] = 1.0
        capped = np.array([n in set(nonpositive) for n in self.names])
        self.w[capped] = np.minimum(self.w[capped], 0.0)
        stacked = _StackedGroups(Z)                     # vectorised MRR (same values as _mrr, ~10x faster)
        best = stacked.mrr(self.w)
        steps = np.concatenate([-np.logspace(-2, 0.5, 12)[::-1], [0.0], np.logspace(-2, 0.5, 12)])
        rng = np.random.default_rng(seed)
        for _ in range(passes):
            improved = False
            for f in rng.permutation(len(self.names)):
                base = self.w[f]
                for st in steps:
                    self.w[f] = min(base + st, 0.0) if capped[f] else base + st
                    m = stacked.mrr(self.w)
                    if m > best + 1e-6:
                        best, base, improved = m, self.w[f], True
                self.w[f] = base
            if not improved:
                break
        norm = np.abs(self.w).sum() or 1.0
        self.w /= norm
        self.train_mrr = best
        return self


class PairwiseLogistic(LinearModel):
    kind = "pairwise_logistic"

    def fit(self, groups, l2: float = 1e-2):
        self.std = _Standardizer().fit(groups)
        diffs, weights = [], []
        for X, y in groups:
            Z = self.std(X)
            pos, neg = Z[y > 0], Z[y == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            d = (pos[:, None, :] - neg[None, :, :]).reshape(-1, Z.shape[1])
            diffs.append(d)
            weights.append(np.full(len(d), 1.0 / len(d)))
        D = np.vstack(diffs)
        W = np.concatenate(weights) / len(diffs)

        def f(w):
            m = D @ w
            loss = np.sum(W * np.logaddexp(0.0, -m)) + l2 * w @ w
            g = -D.T @ (W / (1.0 + np.exp(np.clip(m, -50, 50)))) + 2 * l2 * w
            return loss, g

        res = minimize(f, np.zeros(D.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.w = res.x
        return self


class LambdaMART:
    kind = "lambdamart"

    def __init__(self, names, **params):
        self.names = list(names)
        self.params = dict(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=1.0, lambdarank_truncation_level=20, verbose=-1, random_state=0)
        self.params.update(params)
        self.model = None

    def fit(self, groups):
        import lightgbm as lgb
        X = np.vstack([g[0] for g in groups])
        y = np.concatenate([g[1] for g in groups]).astype(int)
        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(X, y, group=[len(g[1]) for g in groups])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def contributions(self, x: np.ndarray) -> dict:
        contrib = self.model.predict(x[None, :], pred_contrib=True)[0]
        return {n: float(c) for n, c in zip(self.names, contrib[:-1])}

    def save(self, path) -> None:
        self.model.booster_.save_model(str(path))
        with open(str(path) + ".json", "w") as fh:
            json.dump({"names": self.names}, fh)

    @classmethod
    def load(cls, path):
        import lightgbm as lgb
        with open(str(path) + ".json") as fh:
            names = json.load(fh)["names"]
        obj = cls(names)

        class _Wrap:  # minimal predict() compatible wrapper around a Booster
            def __init__(self, b):
                self.b = b

            def predict(self, X, pred_contrib=False):
                return self.b.predict(X, pred_contrib=pred_contrib)

        obj.model = _Wrap(lgb.Booster(model_file=str(path)))
        return obj
