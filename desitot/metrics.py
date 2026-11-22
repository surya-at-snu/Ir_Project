"""Evaluation measures for known-item search (exactly one relevant film per query).

With a single relevant item at rank r (r = inf when it is not retrieved):
    RR = 1 / r                       MRR = mean RR  (= MAP for one relevant item)
    Success@k = R@k = 1[r <= k]      P@k = 1[r <= k] / k     F1@k = 2 P R / (P + R)
    DCG@k = 1 / log2(r + 1) if r <= k;  IDCG@k = 1   =>  nDCG@k = 1 / log2(r + 1)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def json_safe(obj):
    """Replace NaN / infinity (valid in Python's json, invalid in browsers' JSON.parse) with None."""
    if isinstance(obj, float) or isinstance(obj, np.floating):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def rank_of(ranking: list, target) -> float:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return math.inf


def per_query(rank: float, ks=(1, 5, 10)) -> dict:
    out = {"rr": 0.0 if math.isinf(rank) else 1.0 / rank,
           "ndcg10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
           "ndcg1000": 1.0 / math.log2(rank + 1) if rank <= 1000 else 0.0,
           "rank": rank}
    for k in ks:
        hit = 1.0 if rank <= k else 0.0
        p, r = hit / k, hit
        out[f"s@{k}"] = hit
        out[f"p@{k}"] = p
        out[f"f1@{k}"] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "rank"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    ranks = np.array([r["rank"] for r in rows], dtype=float)
    finite = ranks[np.isfinite(ranks)]
    out["median_rank"] = float(np.median(ranks)) if len(ranks) else math.nan
    out["mean_rank_found"] = float(finite.mean()) if len(finite) else math.nan
    out["n"] = len(rows)
    return out


def paired_permutation_test(a: list[float], b: list[float], n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided paired randomisation test on per-query differences (Smucker et al., CIKM 2007)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))


def bootstrap_ci(values: list[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def kendall_tau(x: list[float], y: list[float]) -> tuple[float, float]:
    tau, p = stats.kendalltau(x, y)
    return float(tau), float(p)


def classification_report(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    """Accuracy plus per-class and macro precision / recall / F1."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    out = {"accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else 0.0, "per_class": {}}
    ps, rs, fs = [], [], []
    for c in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out["per_class"][str(c)] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
        ps.append(p), rs.append(r), fs.append(f)
    out["macro_precision"], out["macro_recall"], out["macro_f1"] = map(lambda v: float(np.mean(v)), (ps, rs, fs))
    return out


def binary_prf(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if not a and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and not b)
    tn = sum(1 for a, b in zip(y_true, y_pred) if not a and not b)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "accuracy": (tp + tn) / max(1, tp + tn + fp + fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
