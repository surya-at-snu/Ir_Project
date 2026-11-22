"""C4 - ask the one question most likely to resolve the search.

When the top films stay close (relative confidence gap below GAP), the system asks ONE question instead of
guessing.  Candidate questions are about details a viewer can know: the decade, the language, which of a few
actors was in it, which version of a remade story they watched (C2), or - for a viewer describing scenes - what
happened next after the last scene they described (C3's passage positions).  Each candidate question splits the
films still in play D (the top 5 with their softmax confidences P(d)) into answer groups, and is scored by its
expected information gain times the chance the viewer can answer (Rao & Daume III, 2018; Aliannejadi et al., 2019):

    q* = argmax_q  P_ans(q) * [ H(D) - sum_a P(a | q) H(D | a) ]

"I don't remember" is always a valid answer and changes nothing.  An answer re-weights every film:
    s'(d) = s(d) + T * ln(likelihood(answer | d)),   likelihood = 1 if d fits the answer, else EPS
P_ans is an assumption (a prior on how often viewers know each kind of detail), higher for the kind of detail the
description already shows the viewer remembers; the evaluation varies it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .dense import sentences
from .text import fold

P_ANSWER = {"language": 0.9, "decade": 0.6, "actor": 0.5, "version": 0.7, "next_event": 0.5}
P_ANSWER_REMEMBERED = {"actor": 0.8, "next_event": 0.7, "decade": 0.75}
GAP = 0.3
MIN_EXPECTED_GAIN = 0.25
EPS = 0.05
META_SENT = re.compile(r"(?<![a-z])(?:film|movie|inspired by|released|directed|starring|box office|cast)(?![a-z])", re.I)
LANGS = {"hi": "Hindi", "te": "Telugu", "en": "English"}


@dataclass
class Question:
    kind: str
    text: str
    options: list                 # [{"value", "label", "films": candidate ids}]
    gain: float                   # bits, if answered truthfully
    p_answer: float
    expected_gain: float
    about: str = ""
    candidates: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {"kind": self.kind, "text": self.text, "options": self.options, "gain_bits": round(self.gain, 3),
                "p_answer": self.p_answer, "expected_gain": round(self.expected_gain, 3), "about": self.about,
                "candidates": self.candidates}


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def information_gain(probs: np.ndarray, groups: list[list[int]]) -> float:
    """H(D) - sum_a P(a) H(D | a) for a partition of the candidate positions into answer groups."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    h = entropy(probs)
    cond = 0.0
    for g in groups:
        pa = float(probs[g].sum()) if g else 0.0
        if pa > 0:
            cond += pa * entropy(probs[g] / pa)
    return h - cond


def _decade(y) -> int:
    return (int(y) // 10) * 10 if y else 0


def _groups(values: list, options: list) -> list[list[int]]:
    groups = [[i for i, v in enumerate(values) if v == o] for o in options]
    rest = [i for i, v in enumerate(values) if v not in options]
    return groups + ([rest] if rest else [])


def _remembered(qa) -> dict:
    """Kinds of detail the description already shows the viewer remembers."""
    kinds = {"actor": False, "next_event": False, "decade": False}
    content = 0
    for c in qa.clauses:
        if c.meta or c.exclusion:
            continue
        if any(m.entity.kind in ("person", "character") for m, _ in c.entities):
            kinds["actor"] = True
        if c.years:
            kinds["decade"] = True
        if sum(1 for t in c.tokens if t.term) >= 3:
            content += 1
    kinds["next_event"] = content >= 2
    return kinds


def _next_scene(engine, doc: int, event_vec) -> str | None:
    """First sentence of the passage after the one that best matches the event, in film `doc`'s plot."""
    dense = engine.dense
    a = int(dense.starts[doc])
    b = int(dense.starts[doc + 1]) if doc + 1 < dense.N else len(dense.passages)
    if b - a < 3:
        return None
    sims = dense.emb[a:b] @ event_vec
    i = int(np.argmax(sims[1:])) + 1                 # skip the lead summary
    if i + 1 >= b - a:
        return None
    nxt = sentences(dense.passages[a + i + 1])
    cur = set(sentences(dense.passages[a + i]))
    new = [s for s in nxt if s not in cur and not META_SENT.search(s) and len(s.split()) >= 6]
    return new[0] if new else None


def _short(s: str, n: int = 120) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def candidate_questions(engine, qa, top: list[int], probs: np.ndarray, versions_info: dict | None = None,
                        p_answer: dict | None = None) -> list[Question]:
    films = engine.films
    pa = dict(P_ANSWER)
    rem = _remembered(qa)
    for kind, flag in rem.items():
        if flag and kind in P_ANSWER_REMEMBERED:
            pa[kind] = P_ANSWER_REMEMBERED[kind]
    if p_answer:
        pa.update(p_answer)
    out = []
    # decade and language
    for kind, attr, text, label in (("decade", lambda f: _decade(f["year"]), "Which decade was it from?",
                                     lambda v: f"{v}s"),
                                    ("language", lambda f: f["lang"], "Which language was the film made in?",
                                     lambda v: LANGS.get(v, v))):
        values = [attr(films[d]) for d in top]
        opts = list(dict.fromkeys(values))
        if len(opts) < 2:
            continue
        g = information_gain(probs, _groups(values, opts))
        out.append(Question(kind, text, [{"value": v, "label": label(v), "films": [top[i] for i, x in enumerate(values)
                                                                                 if x == v]} for v in opts],
                            g, pa[kind], pa[kind] * g, candidates=list(top)))
    # actors: one distinctive first-billed star per candidate
    stars = []
    for d in top:
        others = set().union(*({fold(s) for s in films[o]["stars"] + films[o]["cast"][:6]} for o in top if o != d))
        pick = next((s for s in films[d]["stars"] + films[d]["cast"][:3] if fold(s) not in others), None)
        stars.append(pick)
    opts = [s for s in dict.fromkeys(stars) if s]
    if len(opts) >= 2:
        g = information_gain(probs, _groups(stars, opts))
        out.append(Question("actor", "Do you remember any of these actors in it?",
                            [{"value": s, "label": s, "films": [top[i] for i, x in enumerate(stars) if x == s]}
                             for s in opts], g, pa["actor"], pa["actor"] * g, candidates=list(top)))
    # versions of one story (C2)
    if versions_info and versions_info.get("ambiguous"):
        vs = [v["id"] for v in versions_info["versions"]]
        in_top = [d for d in vs if d in top]
        if len(in_top) >= 2:
            labels = []
            for d in in_top:
                f = films[d]
                star = (f["stars"] or f["cast"] or [""])[0]
                labels.append({"value": d, "label": f"{f['title']} ({LANGS.get(f['lang'], f['lang'])}, {f['year']})"
                                                     + (f" with {star}" if star else ""), "films": [d]})
            values = [d if d in in_top else None for d in top]
            g = information_gain(probs, _groups(values, in_top))
            out.append(Question("version", "Which version did you watch?", labels, g, pa["version"],
                                pa["version"] * g, candidates=list(top)))
    # what happened next (scene descriptions)
    if engine.dense is not None:
        plot = [c for c in qa.clauses if not c.meta and not c.exclusion and sum(1 for t in c.tokens if t.term) >= 3]
        if plot:
            ev = plot[-1].text
            vec = engine.dense.encode([ev])[0]
            scenes = [_next_scene(engine, d, vec) for d in top]
            opts = [s for s in dict.fromkeys(scenes) if s]
            if len(opts) >= 2:
                # versions of one story share their scenes, so a scene question cannot tell them apart
                groups = _groups(scenes, opts)
                vs = getattr(engine, "versions", None)
                if vs is not None:
                    merged = []
                    for g_ in groups:
                        hit = next((m for m in merged if any(vs.same_story(top[i], top[j]) for i in g_ for j in m)), None)
                        if hit is not None:
                            hit.extend(g_)
                        else:
                            merged.append(list(g_))
                    groups = merged
                g = information_gain(probs, groups)
                out.append(Question("next_event", f"In the film you remember, what happened after “{_short(ev, 80)}”?",
                                    [{"value": s, "label": _short(s), "films": [top[i] for i, x in enumerate(scenes)
                                                                               if x == s]} for s in opts],
                                    g, pa["next_event"], pa["next_event"] * g, about=ev, candidates=list(top)))
    return out


def choose(engine, qa, top: list[int], probs: np.ndarray, versions_info=None, gap: float = GAP,
           min_gain: float = MIN_EXPECTED_GAIN, p_answer: dict | None = None, only: str | None = None):
    """The question to ask, or None when the top film is clearly ahead or no question helps enough."""
    if len(top) < 2:
        return None
    probs = np.asarray(probs, dtype=np.float64)
    if probs[0] - probs[1] >= gap:
        return None
    qs = candidate_questions(engine, qa, top, probs, versions_info, p_answer)
    if only:
        qs = [q for q in qs if q.kind == only]
    qs = [q for q in qs if q.expected_gain >= min_gain or only]
    return max(qs, key=lambda q: q.expected_gain) if qs else None


def likelihood(engine, answer: dict) -> np.ndarray:
    """Likelihood of the answer for every film (1 = fits, EPS = does not); ones for "I don't remember"."""
    N, films = engine.N, engine.films
    like = np.ones(N)
    kind, value = answer.get("kind"), answer.get("value")
    if value in (None, "", "dont_know"):
        return like
    offered = answer.get("offered") or []
    if kind == "decade":
        dec = np.array([_decade(f["year"]) for f in films])
        if value == "none":
            like[np.isin(dec, [int(v) for v in offered])] = EPS
        else:
            like[dec != int(value)] = EPS
    elif kind == "language":
        lang = np.array([f["lang"] for f in films])
        if value == "none":
            like[np.isin(lang, offered)] = EPS
        else:
            like[lang != value] = EPS
    elif kind == "actor":
        people = engine.person_docs
        if value == "none":
            for name in offered:
                like[list(people.get(fold(name), ()))] = EPS
        else:
            docs = people.get(fold(value), set())
            mask = np.ones(N, dtype=bool)
            mask[list(docs)] = False
            like[mask] = EPS
    elif kind in ("version", "next_event"):
        chosen = set(answer.get("films") or [])
        cands = set(answer.get("candidates") or [])
        for d in cands:
            if d not in chosen:
                like[d] = EPS
    return like


def apply_answer(engine, scores: np.ndarray, answer: dict, temperature: float) -> np.ndarray:
    like = likelihood(engine, answer)
    return scores + max(temperature, 1e-6) * np.log(like)


def truthful_answer(engine, q: Question, target: int) -> dict:
    """What a viewer who remembers the detail answers (for the simulated evaluation)."""
    f = engine.films[target]
    offered = [o["value"] for o in q.options]
    base = {"kind": q.kind, "offered": offered, "candidates": q.candidates}
    if q.kind == "decade":
        v = _decade(f["year"])
        return {**base, "value": v if v in offered else "none"}
    if q.kind == "language":
        return {**base, "value": f["lang"] if f["lang"] in offered else "none"}
    if q.kind == "actor":
        cast = {fold(s) for s in f["stars"] + f["cast"]}
        hit = next((o for o in offered if fold(o) in cast), None)
        return {**base, "value": hit or "none"}
    hit = next((o for o in q.options if target in o["films"]), None)
    return {**base, "value": hit["value"] if hit else "none", "films": hit["films"] if hit else []}
