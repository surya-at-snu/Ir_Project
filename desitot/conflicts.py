"""C1 - detect a memory borrowed from another film.

Certainty weights can only scale a clue up or down: "I'm certain Akshay Kumar was in it" is trusted *more* than
"maybe Akshay Kumar was in it", even when Akshay Kumar was in a different film.  Memory research calls this a
source-monitoring error (Johnson, Hashtroudi & Lindsay, 1993): the memory is real, its source is wrong.  This module
checks every clue against the films that support it, in the spirit of claim verification (Thorne et al., 2018):

    clues     each person/character mention is an ACTOR clue; the remaining content clauses of one sentence form
              a PLOT clue (so a borrowed sentence that the clause splitter cut at its commas stays one clue)
    support   sup_j(d) in [0, 1]: actor clue -> 1 if the person is in film d's parsed cast/credits, else 0
              plot clue  -> 0.5 * BM25F_j(d) / max BM25F_j + 0.5 * rescaled dense MaxP similarity
    test      d*_{-j} = the film the OTHER clues agree on (the ranker re-run without clue j)
              kappa_j = max_{d' != d*_{-j}} sup_j(d') - sup_j(d*_{-j})
              the clue is BORROWED when kappa_j >= tau and the film that best explains it (its "source") ranks
              below FAR_RANK without the clue - a clue that only decides between close candidates is kept
    action    a borrowed clue is withdrawn (trust 0) and the ranking recomputed; if the source film is another
              version of the same story (C2) the clue is not wrong at all: it picks that version
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .query import QueryAnalysis

TAU = 0.5
FAR_RANK = 5
CONF_MIN = 0.5              # the other clues must agree on one story with at least this softmax share
RATIO = 1.5                 # a side is withdrawn only when the other side carries 1.5x its evidence
HEDGE_MASS = 0.5            # a hedged detail counts half; certainty never counts more than a plain memory
MIN_REST_TERMS = 3
MIN_CLUE_TERMS = 3          # a plot clue needs a few content words before it can be judged
MAX_ACTOR_TERMS = 6         # longer clauses that mention an actor are plot memories ("Amitabh plays a cop who...")
MAX_CHECKS = 4
NAME_SCORE = 0.85            # entity matches below this (partial one-word names) are not used as actor clues
UNIT = {"hedged": 1.0, "confident": 1.0, "context": 1.0, "meta": 0.0, "mu": 1.0}


@dataclass
class Clue:
    idx: int
    clauses: list            # clause indices
    text: str
    kind: str                # "actor" | "plot"
    certainty: str           # "hedged" | "plain" | "confident"
    names: list = field(default_factory=list)
    docs: set = field(default_factory=set)       # films of the named people (actor clues)
    n_terms: int = 0


@dataclass
class CheckResult:
    clues: list              # Clue
    checks: list             # dicts, one per checked clue
    scores: np.ndarray       # final scores after withdrawing borrowed clues / picking versions
    withdrawn: list          # clause indices withdrawn
    qa_final: QueryAnalysis | None = None


def _sentence_ids(qa: QueryAnalysis) -> list[int]:
    ids, sid, prev = [], 0, None
    for c in qa.clauses:
        if prev is not None and re.search(r"[.!?;\n]", qa.text[prev:c.start]):
            sid += 1
        ids.append(sid)
        prev = c.end
    return ids


def _certainty(clauses) -> str:
    if any(c.hedged for c in clauses):
        return "hedged"
    if any(c.confident for c in clauses):
        return "confident"
    return "plain"


def make_clues(qa: QueryAnalysis) -> list[Clue]:
    """Actor clues (a short clause naming a credited person: "I'm certain Akshay Kumar was in it") and plot clues
    (the content clauses of one sentence; character names stay in the plot clue, where they are plot evidence)."""
    sids = _sentence_ids(qa)
    clues, groups = [], {}
    for i, c in enumerate(qa.clauses):
        if c.meta or c.exclusion:
            continue
        people = [m for m, in_scope in c.entities
                  if m.entity.kind == "person" and not in_scope and m.score >= NAME_SCORE]
        n_terms = sum(1 for t in c.tokens if t.term and t.cue not in ("lang", "genre"))
        if people and n_terms <= MAX_ACTOR_TERMS:
            names = list(dict.fromkeys(m.entity.name for m in people))
            docs = set().union(*(m.entity.docs for m in people))
            clues.append(Clue(len(clues), [i], c.text, "actor", _certainty([c]), names, docs, n_terms))
        elif n_terms and not c.context:            # "I watched it in Hindi" is context, not a plot memory
            groups.setdefault(sids[i], []).append(i)
    for sid, idx in sorted(groups.items()):
        cl = [qa.clauses[i] for i in idx]
        text = qa.text[cl[0].start:cl[-1].end]
        clues.append(Clue(len(clues), idx, text, "plot", _certainty(cl), [], set(),
                          sum(1 for c in cl for t in c.tokens if t.term and t.cue not in ("lang", "genre"))))
    clues.sort(key=lambda k: qa.clauses[k.clauses[0]].start)
    for i, k in enumerate(clues):
        k.idx = i
    return clues


def _without(qa: QueryAnalysis, drop: set[int]) -> QueryAnalysis:
    keep = [c for i, c in enumerate(qa.clauses) if i not in drop]
    text, pos = [], 0
    for i, c in enumerate(qa.clauses):
        if i in drop:
            text.append(qa.text[pos:c.start])
            pos = c.end
    text.append(qa.text[pos:])
    dense_text = re.sub(r"\s*([.,;])(\s*[.,;])+", r"\1", "".join(text)).strip(" .,;") or qa.text
    return QueryAnalysis(qa.text, keep, dense_text=dense_text)


def _with_trust(qa: QueryAnalysis, withdrawn: set[int]) -> QueryAnalysis:
    """Copy of the analysis whose withdrawn clauses have trust 0 (kept, so the display still shows them)."""
    import copy
    clauses = []
    for i, c in enumerate(qa.clauses):
        if i in withdrawn:
            c = copy.copy(c)
            c.trust = 0.0
        clauses.append(c)
    return QueryAnalysis(qa.text, clauses, dense_text=_without(qa, withdrawn).dense_text)


def support(engine, qa: QueryAnalysis, clue: Clue) -> np.ndarray:
    """sup_j(d) for every film."""
    N = engine.N
    if clue.kind == "actor":
        s = np.zeros(N)
        idx = np.fromiter(clue.docs, dtype=np.int64)
        s[idx] = 1.0
        return s
    sub = QueryAnalysis(qa.text, [qa.clauses[i] for i in clue.clauses])
    terms, _ = sub.term_weights(UNIT, use_clauses=False, negation="scope", hinglish=True)
    lex = engine.index.bm25(terms).astype(np.float64) if terms else np.zeros(N)
    lex = lex / lex.max() if lex.max() > 0 else lex
    if engine.dense is None:
        return lex
    v = engine.dense.encode([clue.text])[0]
    d = engine.dense.doc_scores(v).astype(np.float64)
    med, top = float(np.median(d)), float(d.max())
    dn = np.clip((d - med) / (top - med), 0, 1) if top > med else np.zeros(N)
    return 0.5 * lex + 0.5 * dn


def _rank(scores: np.ndarray, d: int) -> int:
    return int(1 + np.sum(scores > scores[d]))


def _source(sup: np.ndarray, answer: int, base_scores: np.ndarray) -> tuple[int, float]:
    """The film that best explains a clue, other than `answer`, and kappa = sup(source) - sup(answer)."""
    others = sup.copy()
    others[answer] = -np.inf
    best = others.max()
    if not np.isfinite(best):
        return answer, 0.0
    cand = np.flatnonzero(others >= best - 1e-9)
    src = int(cand[np.argmax(base_scores[cand])])        # tie-break: the film the full description favours
    return src, float(sup[src] - sup[answer])


def _story_confidence(scores: np.ndarray, win: int, same) -> float:
    """How sure the ranking is of its winner's story: softmax share of the winner and its other versions in the
    top 5 (two versions of one remade story do not compete with each other)."""
    top = np.argsort(-scores, kind="stable")[:5]
    s = scores[top]
    s = s[np.isfinite(s)]
    if len(s) < 2:
        return 1.0
    spread = float(np.std(s)) or 1.0
    p = np.exp((s - s.max()) / spread)
    p /= p.sum()
    return float(sum(p[i] for i, d in enumerate(top[:len(s)]) if same(int(d), win)))


def check(engine, qa: QueryAnalysis, base_scores: np.ndarray, opts, ranker: str = "ltr", w: dict | None = None,
          tau: float = TAU, far_rank: int = FAR_RANK, max_checks: int = MAX_CHECKS, versions=None,
          mask: np.ndarray | None = None, keep_scores: bool = False) -> CheckResult:
    """Run the leave-one-clue-out conflict test and return the final scores."""
    restrict = (lambda s: s) if mask is None else (lambda s: np.where(mask, s, -np.inf))
    same = (lambda a, b: a == b or (versions is not None and versions.same_story(a, b)))
    clues = make_clues(qa)
    content = [k for k in clues]
    total_terms = sum(k.n_terms for k in content)
    checks, flagged = [], []
    sups = {}
    if len(content) >= 2:
        sups = {k.idx: support(engine, qa, k) for k in content}
        # most specific clues first: names, then longer plot clues
        order = sorted(content, key=lambda k: (k.kind != "actor", -k.n_terms))[:max_checks]
        for k in order:
            if total_terms - k.n_terms < MIN_REST_TERMS:
                continue                       # the other clues are too thin to point at any film
            if k.kind == "plot" and k.n_terms < MIN_CLUE_TERMS:
                continue                       # "comedy of confusion": too little to judge
            sup = sups[k.idx]
            qa_minus = _without(qa, set(k.clauses))
            s_minus = restrict(engine.scores(engine.compute(qa_minus, opts, w), ranker))
            win = int(np.argmax(s_minus))
            src, kappa = _source(sup, win, base_scores)
            r_src = _rank(s_minus, src)
            conf = _story_confidence(s_minus, win, same)
            checks.append({"clue": k.idx, "clauses": k.clauses, "text": k.text, "kind": k.kind,
                           "certainty": k.certainty, "names": k.names, "kappa": round(kappa, 3),
                           "support_answer": round(float(sup[win]), 3), "winner_without": win,
                           "source": src, "source_rank_without": r_src, "confidence_without": round(conf, 3),
                           "candidate": bool(kappa >= tau and r_src > far_rank and conf >= CONF_MIN),
                           "flagged": False, "relation": "", "scores_without": s_minus})
    cands = [c for c in checks if c["candidate"]]
    consensus, resolved = None, False
    if cands:
        # several clues can conflict with each other (a true plot memory and a borrowed one each look "borrowed"
        # from the other's side): the description's consensus is the story supported by the most evidence.
        # Evidence = content words x support; a hedged detail counts half, certainty never counts extra.
        films = {c["winner_without"] for c in cands} | {c["source"] for c in cands} | {int(np.argmax(base_scores))}
        weight = {k.idx: k.n_terms * (HEDGE_MASS if k.certainty == "hedged" else 1.0) for k in content}
        mass = {d: sum(weight[k.idx] * float(sups[k.idx][d]) for k in content) for d in films}
        consensus = max(films, key=lambda d: (mass[d], base_scores[d]))
        rivals = [mass[d] for d in films if not same(d, consensus)]
        resolved = not rivals or mass[consensus] >= RATIO * max(rivals)
        for c in cands:
            sup = sups[c["clue"]]
            if sup[consensus] >= 0.5:
                c["kept"] = "agrees with the consensus"
                continue
            src, kappa = _source(sup, consensus, base_scores)
            rel = "other_version" if same(src, consensus) else ("borrowed" if resolved else "conflict")
            c.update({"flagged": True, "consensus": consensus, "source": src, "kappa": round(kappa, 3),
                      "winner_without": consensus, "relation": rel,
                      "evidence_for": round(mass[consensus], 2), "evidence_against": round(max(rivals or [0.0]), 2)})
            flagged.append(c)
    borrowed = [c for c in flagged if c["relation"] == "borrowed"]
    withdrawn = sorted({i for c in borrowed for i in c["clauses"]})
    scores, qa_final = base_scores, None
    if len(borrowed) == 1 and borrowed[0].get("scores_without") is not None:
        scores = borrowed[0]["scores_without"]
        qa_final = _with_trust(qa, set(withdrawn))
    elif len(borrowed) > 1:
        qa_final = _with_trust(qa, set(withdrawn))
        scores = restrict(engine.scores(engine.compute(qa_final, opts, w), ranker))
    # second pass: an actor clue that agreed with a borrowed plot clue (both from the same other film) is
    # checked again against the consensus
    if borrowed:
        top = int(np.argmax(scores))
        for c in checks:
            if c["flagged"] or c["kind"] != "actor":
                continue
            k = clues[c["clue"]]
            if top in k.docs or consensus in k.docs:
                continue
            qa2 = _with_trust(qa, set(withdrawn) | set(k.clauses))
            s2 = restrict(engine.scores(engine.compute(qa2, opts, w), ranker))
            win2 = int(np.argmax(s2))
            if win2 in k.docs or _story_confidence(s2, win2, same) < CONF_MIN:
                continue
            docs = np.fromiter(k.docs, dtype=np.int64)
            src = int(docs[np.argmax(base_scores[docs])])
            if _rank(s2, src) > far_rank:
                rel = "other_version" if versions is not None and versions.same_story(src, win2) else "borrowed"
                c.update({"flagged": True, "relation": rel, "kappa": 1.0, "winner_without": win2, "source": src,
                          "source_rank_without": _rank(s2, src), "second_pass": True})
                if rel == "borrowed":
                    withdrawn = sorted(set(withdrawn) | set(k.clauses))
                    qa_final, scores = qa2, s2
    # an unresolved conflict (both sides carry similar evidence) keeps both films at the top, so the viewer sees
    # them side by side and the one-question step (C4) can settle it
    conflicts_open = [c for c in flagged if c["relation"] == "conflict"]
    if conflicts_open and not borrowed:
        a = consensus
        b = max(conflicts_open, key=lambda c: c["kappa"])["source"]
        if np.isfinite(scores[a]) and np.isfinite(scores[b]):
            top_score = float(np.max(scores))
            scores = scores.copy()
            first, second = (a, b) if scores[a] >= scores[b] else (b, a)
            scores[first] = max(scores[first], top_score + 2e-6)
            scores[second] = max(scores[second], top_score + 1e-6)
    # a clue that fits another version of the winning story picks that version (the version step, C2, ranks it)
    for c in checks:
        if c["flagged"] and c["relation"] == "other_version":
            src, win = c["source"], int(np.argmax(scores))
            if versions is not None and versions.same_story(src, win):
                scores = scores.copy()
                scores[src] = max(scores[src], scores[win] + 1e-6)
    if not keep_scores:
        for c in checks:
            c.pop("scores_without", None)
    return CheckResult(clues, checks, scores, withdrawn, qa_final)
