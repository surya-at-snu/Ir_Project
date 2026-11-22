"""C3 - retrieve scenes in the order they are remembered.

BM25F and TF-IDF are bags of words and the clause scores are summed, so "she leaves him and then he goes blind"
and "he goes blind and then she leaves him" score every film the same.  When a description joins two events with
an order marker ("and then", "later", "after that", "in the end"; "before that" reverses them), this module finds,
for each candidate film, the plot passage that best matches each event and asks whether they occur in that order:

    sim(e, p)    = cos(E(e), E(p)) + 0.5 * idf-weighted share of the event's words found in passage p
    s_order(d)   = max_{i <= j} [ sim(e1, p_i) + sim(e2, p_j) ]          (passages in story order)
    s_free(d)    = max_i sim(e1, p_i) + max_j sim(e2, p_j)
    Delta(d)     = s_order(d) - s_free(d)  <= 0;  0 when the remembered order fits the plot

Delta is added to the candidates' scores (scaled by the spread of the top scores) and the two passages are shown
in sequence as evidence, with the reversed description one click away.
"""
from __future__ import annotations

import re

import numpy as np

from .text import analyze

SEQ_RX = re.compile(r"(?<![a-z])(?:and then|then|after that|afterwards|after which|later on|later|eventually|"
                    r"finally|in the end|at the end|by the end|next|after this)(?![a-z])")
REV_RX = re.compile(r"(?<![a-z])(?:before that|before this|earlier|previously)(?![a-z])")
LAMBDA = 0.5          # score shift per 0.1 of Delta, in units of the top scores' standard deviation
MAX_SHIFT = 1.0       # cap: Delta never moves a film by more than LAMBDA * MAX_SHIFT standard deviations
TOP_K = 20
FIT_TOL = 0.02


def event_pairs(qa) -> list[tuple[int, int]]:
    """Ordered (earlier, later) clause indices joined by an explicit order marker."""
    content = [i for i, c in enumerate(qa.clauses)
               if not c.meta and not c.exclusion and not c.context and sum(1 for t in c.tokens if t.term) >= 2]
    pairs = []
    for a, b in zip(content, content[1:]):
        ca, cb = qa.clauses[a], qa.clauses[b]
        gap = (qa.text[ca.end:cb.start] + " " + cb.text[:24]).lower()
        if REV_RX.search(gap):
            pairs.append((b, a))
        elif SEQ_RX.search(gap):
            pairs.append((a, b))
    return pairs


def _passages(engine, doc: int):
    dense = engine.dense
    a = int(dense.starts[doc])
    b = int(dense.starts[doc + 1]) if doc + 1 < dense.N else len(dense.passages)
    lead = (engine.films[doc].get("lead") or "")[:40]
    first = a + 1 if lead and b - a > 1 and dense.passages[a].startswith(lead[:30]) else a   # skip the lead summary
    return first, b


def _lexical(engine, text: str, passages: list[str]) -> np.ndarray:
    ev = set(analyze(text, query=True))
    if not ev:
        return np.zeros(len(passages))
    idf = engine.index.idf
    total = sum(idf.get(t, 0.0) for t in ev) or 1.0
    out = np.zeros(len(passages))
    for i, p in enumerate(passages):
        ps = set(analyze(p))
        out[i] = sum(idf.get(t, 0.0) for t in ev & ps) / total
    return out


def evidence(engine, qa, scores: np.ndarray, pairs=None, top_k: int = TOP_K, lam: float = LAMBDA):
    """Return (adjusted scores, {doc: order evidence}) for the top candidates; scores unchanged without pairs."""
    pairs = event_pairs(qa) if pairs is None else pairs
    if not pairs or engine.dense is None:
        return scores, {}
    cand = np.argsort(-scores, kind="stable")[:top_k]
    spread = float(np.std(scores[cand])) or 1.0
    texts = list(dict.fromkeys(qa.clauses[i].text for p in pairs for i in p))
    vecs = dict(zip(texts, engine.dense.encode(texts)))
    out, adj = {}, scores.copy()
    for d in cand:
        d = int(d)
        a, b = _passages(engine, d)
        if b - a < 2:
            continue
        P = engine.dense.passages[a:b]
        total_delta, steps = 0.0, []
        for e1, e2 in pairs:
            t1, t2 = qa.clauses[e1].text, qa.clauses[e2].text
            s1 = engine.dense.emb[a:b] @ vecs[t1] + 0.5 * _lexical(engine, t1, P)
            s2 = engine.dense.emb[a:b] @ vecs[t2] + 0.5 * _lexical(engine, t2, P)
            free = float(s1.max() + s2.max())
            best1 = np.maximum.accumulate(s1)              # best e1 at or before position j
            tot = best1 + s2
            j = int(np.argmax(tot))
            i = int(np.argmax(s1[:j + 1]))
            delta = float(tot[j]) - free
            total_delta += delta
            fi, fj = int(np.argmax(s1)), int(np.argmax(s2))
            steps.append({"first": t1, "second": t2, "delta": round(delta, 3), "fits": delta >= -FIT_TOL,
                          "passage_first": a + (i if delta >= -FIT_TOL else fi),
                          "passage_second": a + (j if delta >= -FIT_TOL else fj),
                          "pos_first": round((i if delta >= -FIT_TOL else fi) / max(1, b - a - 1), 2),
                          "pos_second": round((j if delta >= -FIT_TOL else fj) / max(1, b - a - 1), 2)})
        adj[d] = scores[d] + lam * spread * max(total_delta / 0.1, -MAX_SHIFT)
        out[d] = {"delta": round(total_delta, 3), "fits": all(s["fits"] for s in steps), "steps": steps}
    return adj, out


def reversed_query(qa, pair: tuple[int, int]) -> str:
    """The description with the two events swapped (for the "reverse the events" check)."""
    a, b = sorted(pair)
    ca, cb = qa.clauses[a], qa.clauses[b]
    t = qa.text
    return t[:ca.start] + cb.text + t[ca.end:cb.start] + ca.text + t[cb.end:]
