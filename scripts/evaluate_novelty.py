#!/usr/bin/env python
"""Evaluate the four evidence capabilities (C1-C4) on top of the deployed ranker.

  C1  borrowed memories   the 830 vague queries plus one injected false memory (the same random draws as the
                          vague+noise set of scripts/evaluate.py), stated hedged / plain / certain, whose source film
                          is known; correct-actor controls; clean query sets for false alarms.  Clue-level detection
                          precision / recall / F1, source attribution, false-alarm rate, MRR and Success@1.
  C2  remakes and dubs    curated films whose story has another version in the index: their vague queries with the
                          version hints removed, plus the language heard or the lead actor; Telugu films that were
                          dubbed into Hindi, plus "I watched it in Hindi".  Story accuracy, exact-version accuracy and
                          credit for keeping the ambiguity when both versions fit.
  C3  event order         two events from each curated plot (one early, one late, names removed) joined by "and
                          then", in plot order and reversed.  Pairwise order accuracy and MRR.
  C4  one question        simulated viewers who know each kind of detail with probability P_ans: expected Success@1
                          after one targeted question vs a fixed "Which decade?" question, and the answer rate.

The base ranker is the deployed web-app model (trained on all 2,490 LLM queries), so absolute scores on the vague
sets are in-sample; every comparison below is base vs base + the capability on the same queries.

Usage:  python scripts/evaluate_novelty.py          (about 25 minutes on a laptop CPU)
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desitot import questions  # noqa: E402
from desitot.config import RESULTS  # noqa: E402
from desitot.dense import sentences  # noqa: E402
from desitot.engine import Engine, Options  # noqa: E402
from desitot.metrics import json_safe, paired_permutation_test  # noqa: E402
from desitot.queries import load_queries  # noqa: E402
from desitot.text import fold  # noqa: E402
from desitot.versions import LANG_NAMES  # noqa: E402

HEDGES = ["I think", "maybe", "I'm not sure but I think", "probably", "if I remember correctly", "I guess"]
CERTAIN = ["I'm certain", "I'm sure", "I clearly remember", "I distinctly remember", "I'm positive", "definitely"]
PLAIN = [""] * len(HEDGES)
OPTS = Options()
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def rank_of(scores: np.ndarray, d: int) -> int:
    s = scores[d]
    return int(1 + np.sum(scores > s) + np.sum((scores == s) & (np.arange(len(scores)) < d)))


def mrr(r) -> float:
    return float(np.mean(1.0 / np.asarray(r, dtype=float))) if len(r) else float("nan")


def s_at(r, k=1) -> float:
    return float(np.mean(np.asarray(r) <= k)) if len(r) else float("nan")


def prf(tp, fp, fn) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if tp and p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def run(eng, text: str, use: tuple, keep_scores: bool = False):
    qa = eng.analyze(text)
    qf = eng.compute(qa, OPTS)
    base = eng.scores(qf, "ltr")
    scores, st = eng.evidence(qf, base, OPTS, "ltr", None, None, use=use, keep_scores=keep_scores)
    return qa, base, scores, st


# ---------------------------------------------------------------------------------------------------- query sets
def make_variant(eng, queries, cues, seed=7):
    """Same random draws as Experiment.make_noisy in scripts/evaluate.py (so the hedged set is the paper's
    vague+noise set), with the cue words swapped; records where the injected text starts and its source film."""
    rng = random.Random(seed)
    films = eng.films
    pool = defaultdict(list)
    for f in films:
        if f["curated"]:
            pool[f["lang"]].append(f)
    out = []
    for q in queries:
        t = films[q["target"]]
        other = rng.choice([f for f in pool[t["lang"]] if f["id"] != t["id"]])
        sents = [s for s in sentences(other["plot"]) if 8 <= len(s.split()) <= 25]
        bits, actor = [], None
        if sents:
            s = rng.choice(sents)
            bits.append(f"{rng.choice(cues)} {s[0].lower() + s[1:]}".rstrip(".").strip())
        actors = [a for a in other["stars"] if a not in t["cast"]]
        if actors and rng.random() < 0.5:
            cue = rng.choice(cues)
            actor = rng.choice(actors)
            bits.append(f"{cue} {actor} was in it".strip())
        head = q["query"].rstrip(". ") + ". "
        out.append({**q, "query": head + ". ".join(bits) + ".", "span": len(head), "source": other["id"],
                    "actor": actor})
    return out


def true_actor_variant(eng, queries, cue):
    out = []
    for q in queries:
        f = eng.films[q["target"]]
        stars = f["stars"] or f["cast"]
        if not stars:
            continue
        head = q["query"].rstrip(". ") + ". "
        out.append({**q, "query": head + f"{cue} {stars[0]} was in it.".strip(), "span": len(head),
                    "true_actor": stars[0]})
    return out


# ---------------------------------------------------------------------------------------------------- C1
def eval_c1(eng, sets: dict) -> dict:
    out, sweep_rows = {}, []
    for name, Q in sets.items():
        base_r, c1_r, rows = [], [], []
        tp = fp = fn = 0
        attrib, flagged_queries, alarm_queries, withdrawn_queries = [], 0, 0, 0
        checks_log = []
        t_name = time.time()
        for q in Q:
            qa, base, scores, st = run(eng, q["query"], ("c1",), keep_scores=True)
            t = q["target"]
            rb, rc = rank_of(base, t), rank_of(scores, t)
            base_r.append(rb)
            c1_r.append(rc)
            chk = st["check"]
            span = q.get("span")
            injected = {k.idx for k in chk.clues if span is not None and qa.clauses[k.clauses[0]].start >= span}
            flagged = {c["clue"] for c in chk.checks if c["flagged"] and c["relation"] in ("borrowed", "conflict")}
            withdrawn_q = any(c["flagged"] and c["relation"] == "borrowed" for c in chk.checks)
            false_mem = name.endswith("_false")
            if false_mem:
                tp += len(flagged & injected)
                fn += len(injected - flagged)
                fp += len(flagged - injected)
                for c in chk.checks:
                    if c["flagged"] and c["clue"] in injected:
                        attrib.append(c["source"] == q["source"])
            else:
                # clean sets: every flag is a false alarm (for the true-actor sets, the injected clue is correct)
                fp += len(flagged)
            withdrawn_queries += withdrawn_q
            if flagged:
                flagged_queries += 1
                if not false_mem or not (flagged & injected):
                    alarm_queries += 1
            for c in chk.checks:
                sw = c.pop("scores_without", None)
                checks_log.append({"set": name, "qid": q["qid"], "kind": c["kind"], "kappa": c["kappa"],
                                   "r_src": c["source_rank_without"], "injected": c["clue"] in injected,
                                   "conf": c.get("confidence_without"), "relation": c["relation"],
                                   "target_rank_without": rank_of(sw, t) if sw is not None else None,
                                   "base_rank": rb, "source_is_true": c["source"] == q.get("source")})
            rows.append({"qid": q["qid"], "target": eng.films[t]["title"], "base_rank": rb, "c1_rank": rc,
                         "n_flagged": len(flagged), "n_injected": len(injected)})
        res = {"n": len(Q), "base_mrr": mrr(base_r), "c1_mrr": mrr(c1_r), "base_s1": s_at(base_r),
               "c1_s1": s_at(c1_r), "base_s5": s_at(base_r, 5), "c1_s5": s_at(c1_r, 5),
               "p": paired_permutation_test([1 / r for r in c1_r], [1 / r for r in base_r]),
               "queries_with_flag": flagged_queries / len(Q), "false_alarm_queries": alarm_queries / len(Q),
               "queries_with_withdrawal": withdrawn_queries / len(Q),
               "seconds_per_query": (time.time() - t_name) / max(1, len(Q))}
        if name.endswith("_false"):
            res.update(prf(tp, fp, fn))
            res["attribution"] = float(np.mean(attrib)) if attrib else float("nan")
            res["n_injected_clues"] = tp + fn
        else:
            res["false_alarm_clues"] = fp
        out[name] = res
        out.setdefault("_rows", {})[name] = rows
        sweep_rows += checks_log
        log(f"C1 {name}: MRR {res['base_mrr']:.3f} -> {res['c1_mrr']:.3f}, S@1 {res['base_s1']:.3f} -> "
            f"{res['c1_s1']:.3f}, flagged queries {res['queries_with_flag']:.3f}"
            + (f", P {res['precision']:.3f} R {res['recall']:.3f} F1 {res['f1']:.3f}, source {res['attribution']:.3f}"
               if name.endswith("_false") else f", false alarms {res['false_alarm_queries']:.3f}"))
    out["_checks"] = sweep_rows
    return out


def c1_sweep(c1: dict) -> list[dict]:
    """Detection F1 and false-alarm rate for other thresholds (flags recomputed from the logged checks)."""
    rows = []
    checks = c1["_checks"]
    by_q = defaultdict(list)
    for c in checks:
        by_q[(c["set"], c["qid"])].append(c)
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for far in (1, 3, 5, 10):
            tp = fp = fn = 0
            clean_alarm, clean_n, new_ranks, base_ranks = 0, 0, [], []
            for (set_name, qid), cs in by_q.items():
                flags = [c for c in cs if c["kappa"] >= tau and c["r_src"] > far]
                if set_name.endswith("_false"):
                    tp += sum(1 for c in flags if c["injected"])
                    fp += sum(1 for c in flags if not c["injected"])
                    fn += sum(1 for c in cs if c["injected"] and c not in flags)
                    base = cs[0]["base_rank"]
                    best = max(flags, key=lambda c: c["kappa"]) if flags else None
                    new_ranks.append(best["target_rank_without"] if best else base)
                    base_ranks.append(base)
                else:
                    clean_n += 1
                    clean_alarm += bool(flags)
            r = prf(tp, fp, fn)
            rows.append({"tau": tau, "far_rank": far, **{k: r[k] for k in ("precision", "recall", "f1")},
                         "false_alarm_rate_checked_clean_queries": clean_alarm / max(1, clean_n),
                         "approx_mrr_false_sets": mrr(new_ranks)})
    return rows


# ---------------------------------------------------------------------------------------------------- C2
HINT_RX = re.compile(r"(?<![a-z])(?:hindi|telugu|english|tamil|version|remake|remade|original|dub|dubbed|bollywood|"
                     r"tollywood|hollywood|south)(?![a-z])", re.I)


def strip_hints(eng, text: str) -> str:
    qa = eng.analyze(text)
    kept = [c.text for c in qa.clauses if not HINT_RX.search(c.text)]
    return ", ".join(kept) if kept else text


def eval_c2(eng, vague: list[dict]) -> dict:
    vi = eng.versions
    by_target = {q["target"]: q for q in vague}
    items = []
    for members in vi.members.values():
        for t in members:
            if t not in by_target or not eng.films[t]["curated"]:
                continue
            others = [o for o in members if o != t]
            o = sorted(others, key=lambda d: eng.films[d]["lang"] == eng.films[t]["lang"])[0]
            ft, fo = eng.films[t], eng.films[o]
            cast_o = {fold(s) for s in fo["stars"] + fo["cast"]}
            star = next((s for s in ft["stars"] + ft["cast"] if fold(s) not in cast_o), None)
            plot = strip_hints(eng, by_target[t]["query"])
            v = {"original": (by_target[t]["query"], "exact"), "plot_only": (plot, "ambiguous_ok"),
                 "plot+language_heard": (plot + f". I watched it in {LANG_NAMES[ft['lang']]}.", "exact")}
            if star:
                v["plot+lead_actor"] = (plot + f". {star} was in it.", "exact")
            if fo["lang"] != ft["lang"]:
                v["plot+other_language_heard"] = (plot + f". I watched it in {LANG_NAMES[fo['lang']]}.", "ambiguous_ok")
            items.append({"t": t, "o": o, "story": set(members), "variants": v})
    systems = {"base": None, "c2": ("c2",), "full": ("c1", "c2", "c3")}
    res = defaultdict(lambda: defaultdict(list))
    for it in items:
        for vname, (text, rule) in it["variants"].items():
            for sname, use in systems.items():
                if use is None:
                    qa = eng.analyze(text)
                    scores = eng.scores(eng.compute(qa, OPTS), "ltr")
                    decision = None
                else:
                    _, _, scores, st = run(eng, text, use)
                    decision = st["versions"]["decision"] if st.get("versions") else None
                top2 = [int(d) for d in np.argsort(-scores, kind="stable")[:2]]
                exact = top2[0] == it["t"]
                story = top2[0] in it["story"]
                both = it["t"] in top2 and it["o"] in top2
                credit = exact or (rule == "ambiguous_ok" and both)
                r = res[vname][sname]
                r.append({"exact": exact, "story": story, "both_top2": both, "credit": credit,
                          "stated_ambiguous": decision == "ambiguous", "rank": rank_of(scores, it["t"])})
    table = {}
    for vname, by_sys in res.items():
        table[vname] = {s: {"n": len(rows), "story@1": np.mean([r["story"] for r in rows]),
                            "exact@1": np.mean([r["exact"] for r in rows]),
                            "both_in_top2": np.mean([r["both_top2"] for r in rows]),
                            "credit": np.mean([r["credit"] for r in rows]),
                            "stated_ambiguous": np.mean([r["stated_ambiguous"] for r in rows]),
                            "mrr": mrr([r["rank"] for r in rows])} for s, rows in by_sys.items()}
        log(f"C2 {vname}: " + "; ".join(f"{s} exact {v['exact@1']:.2f} credit {v['credit']:.2f}"
                                        for s, v in table[vname].items()))
    # dubs: Telugu films with a Hindi dub and no Hindi version in the index
    dub_ids = [q["target"] for q in vague if eng.films[q["target"]]["lang"] == "te"
               and "hi" in vi.dubs[q["target"]] and not vi.versions(q["target"])]
    dub = {"base": [], "c2": []}
    decisions = []
    for t in dub_ids:
        text = by_target[t]["query"].rstrip(". ") + ". I watched it in Hindi."
        qa = eng.analyze(text)
        dub["base"].append(rank_of(eng.scores(eng.compute(qa, OPTS), "ltr"), t))
        _, _, scores, st = run(eng, text, ("c2",))
        dub["c2"].append(rank_of(scores, t))
        decisions.append(st["versions"]["decision"] if st.get("versions") else None)
    table["telugu_dubbed_into_hindi+heard_hindi"] = {
        s: {"n": len(r), "mrr": mrr(r), "exact@1": s_at(r)} for s, r in dub.items()}
    table["telugu_dubbed_into_hindi+heard_hindi"]["c2"]["said_dub"] = float(np.mean([d == "dub" for d in decisions])) \
        if decisions else float("nan")
    log(f"C2 dubs (n={len(dub_ids)}): MRR {mrr(dub['base']):.3f} -> {mrr(dub['c2']):.3f}")
    return {"n_items": len(items), "n_stories": len(vi.members), "table": table}


# ---------------------------------------------------------------------------------------------------- C3
def event_text(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", s)
    words = s.split()
    kept = [w for i, w in enumerate(words) if i == 0 or not w[:1].isupper()]
    text = " ".join(kept[:14]).strip(" ,;:.\"'")
    return (text[:1].lower() + text[1:]) if text else ""


def order_pairs(eng, seed=11) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for f in eng.films:
        if not f["curated"]:
            continue
        sents = sentences(f["plot"] or "")
        if len(sents) < 8:
            continue
        n = len(sents)
        early = [s for s in sents[:int(0.4 * n)] if 10 <= len(s.split()) <= 35]
        late = [s for s in sents[int(0.6 * n):] if 10 <= len(s.split()) <= 35]
        if not early or not late:
            continue
        a, b = event_text(rng.choice(early)), event_text(rng.choice(late))
        if len(a.split()) < 5 or len(b.split()) < 5:
            continue
        out.append({"target": f["id"], "correct": f"{a} and then {b}", "reversed": f"{b} and then {a}"})
    return out


def eval_c3(eng, vague: list[dict]) -> dict:
    pairs = order_pairs(eng)
    rec = []
    for p in pairs:
        row = {}
        for which in ("correct", "reversed"):
            qa = eng.analyze(p[which])
            qf = eng.compute(qa, OPTS)
            base = eng.scores(qf, "ltr")
            scores, st = eng.evidence(qf, base, OPTS, "ltr", None, None, use=("c3",))
            oe = st["order"].get(p["target"])
            row[which] = {"base_rank": rank_of(base, p["target"]), "c3_rank": rank_of(scores, p["target"]),
                          "base_score": float(base[p["target"]]), "c3_score": float(scores[p["target"]]),
                          "delta": oe["delta"] if oe else None, "pairs": len(st["pairs"])}
        rec.append(row)

    def pairwise(key_rank):
        out = []
        for r in rec:
            a, b = r["correct"][key_rank], r["reversed"][key_rank]
            out.append(1.0 if a < b else 0.5 if a == b else 0.0)
        return float(np.mean(out))
    with_delta = [r for r in rec if r["correct"]["delta"] is not None and r["reversed"]["delta"] is not None]
    evid = [1.0 if r["correct"]["delta"] > r["reversed"]["delta"] else 0.5 if r["correct"]["delta"] == r["reversed"]["delta"]
            else 0.0 for r in with_delta]
    res = {"n_pairs": len(rec), "detected_order_marker": float(np.mean([r["correct"]["pairs"] > 0 for r in rec])),
           "pairwise_rank_base": pairwise("base_rank"), "pairwise_rank_c3": pairwise("c3_rank"),
           "pairwise_score_base": float(np.mean([1.0 if r["correct"]["base_score"] > r["reversed"]["base_score"] else 0.5
                                                 if r["correct"]["base_score"] == r["reversed"]["base_score"] else 0.0
                                                 for r in rec])),
           "pairwise_score_c3": float(np.mean([1.0 if r["correct"]["c3_score"] > r["reversed"]["c3_score"] else 0.5
                                               if r["correct"]["c3_score"] == r["reversed"]["c3_score"] else 0.0
                                               for r in rec])),
           "order_evidence_accuracy": float(np.mean(evid)) if evid else float("nan"), "n_with_evidence": len(evid),
           "mrr_correct_base": mrr([r["correct"]["base_rank"] for r in rec]),
           "mrr_correct_c3": mrr([r["correct"]["c3_rank"] for r in rec]),
           "mrr_reversed_base": mrr([r["reversed"]["base_rank"] for r in rec]),
           "mrr_reversed_c3": mrr([r["reversed"]["c3_rank"] for r in rec])}
    # no harm on the ordinary vague queries
    b, c = [], []
    for q in vague:
        qa = eng.analyze(q["query"])
        qf = eng.compute(qa, OPTS)
        base = eng.scores(qf, "ltr")
        scores, st = eng.evidence(qf, base, OPTS, "ltr", None, None, use=("c3",))
        b.append(rank_of(base, q["target"]))
        c.append(rank_of(scores, q["target"]))
    res.update({"vague_mrr_base": mrr(b), "vague_mrr_c3": mrr(c),
                "vague_queries_with_marker": float(np.mean([x != y for x, y in zip(b, c)]))})
    log(f"C3: pairwise rank accuracy {res['pairwise_rank_base']:.3f} -> {res['pairwise_rank_c3']:.3f}; "
        f"evidence {res['order_evidence_accuracy']:.3f}; vague MRR {res['vague_mrr_base']:.3f} -> {res['vague_mrr_c3']:.3f}")
    return res


# ---------------------------------------------------------------------------------------------------- C4
EQUAL = {k: 0.6 for k in ("language", "decade", "actor", "version", "next_event")}


def eval_c4(eng, sets: dict) -> dict:
    out = {}
    for name, Q in sets.items():
        recs = []
        for q in Q:
            qa, base, scores, st = run(eng, q["query"], ("c1", "c2", "c3"))
            order = [int(d) for d in np.argsort(-scores, kind="stable")[:5]]
            probs, spread = eng.confidences(scores, order)
            t = q["target"]
            r0 = rank_of(scores, t)
            rec = {"r0": r0}
            for policy, kw in (("targeted", {}), ("fixed_decade", {"only": "decade"})):
                qq = questions.choose(eng, st["qa_used"], order, probs, st["versions"], **kw)
                if qq is None:
                    rec[policy] = None
                    continue
                ans = questions.truthful_answer(eng, qq, t)
                r1 = rank_of(questions.apply_answer(eng, scores, ans, spread), t)
                rec[policy] = {"kind": qq.kind, "p_prior": qq.p_answer,
                               "p_equal": EQUAL[qq.kind], "r1": r1}
            recs.append(rec)
        res = {"n": len(recs), "s1_no_question": s_at([r["r0"] for r in recs]),
               "mrr_no_question": mrr([r["r0"] for r in recs])}
        for policy in ("targeted", "fixed_decade"):
            for world in ("p_prior", "p_equal"):
                s1, rr, asked, pans, kinds = [], [], 0, [], defaultdict(int)
                for r in recs:
                    q = r[policy]
                    if q is None:
                        s1.append(float(r["r0"] == 1))
                        rr.append(1 / r["r0"])
                        continue
                    p = q[world]
                    asked += 1
                    pans.append(p)
                    kinds[q["kind"]] += 1
                    s1.append(p * (q["r1"] == 1) + (1 - p) * (r["r0"] == 1))
                    rr.append(p / q["r1"] + (1 - p) / r["r0"])
                res[f"{policy}|{world}"] = {"s1": float(np.mean(s1)), "mrr": float(np.mean(rr)),
                                            "ask_rate": asked / len(recs),
                                            "answer_rate": float(np.mean(pans)) if pans else float("nan"),
                                            "kinds": dict(kinds)}
        out[name] = res
        log(f"C4 {name}: S@1 none {res['s1_no_question']:.3f}; targeted {res['targeted|p_prior']['s1']:.3f} "
            f"(asked {res['targeted|p_prior']['ask_rate']:.2f}); fixed decade {res['fixed_decade|p_prior']['s1']:.3f}")
    return out


# ---------------------------------------------------------------------------------------------------- main
def write_md(summary: dict) -> None:
    c1, c2, c3, c4 = summary["c1"], summary["c2"], summary["c3"], summary["c4"]
    L = ["# Evaluation of the evidence capabilities (C1-C4)\n", "Base = the deployed ranker; every row compares the same queries with and without the capability.\n",
         "## C1: borrowed memories\n", "| Query set | n | MRR base | MRR +C1 | S@1 base | S@1 +C1 | p | Flagged queries | Precision | Recall | F1 | Source named correctly |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in c1.items():
        if name.startswith("_"):
            continue
        fm = name.endswith("_false")
        p_str = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        tail = (f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['attribution']:.3f} |" if fm
                else "— | — | — | — |")
        L.append(f"| {name} | {r['n']} | {r['base_mrr']:.3f} | {r['c1_mrr']:.3f} | {r['base_s1']:.3f} | {r['c1_s1']:.3f} | "
                 f"{p_str} | {r['queries_with_flag']:.3f} | " + tail)
    L += ["\n## C2: remakes and dubs\n", "| Variant | System | n | Story@1 | Exact@1 | Both in top 2 | Credit | Says ambiguous | MRR |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v, by in c2["table"].items():
        for s, r in by.items():
            L.append(f"| {v} | {s} | {r.get('n')} | {r.get('story@1', float('nan')):.3f} | {r.get('exact@1', float('nan')):.3f} | "
                     f"{r.get('both_in_top2', float('nan')):.3f} | {r.get('credit', float('nan')):.3f} | "
                     f"{r.get('stated_ambiguous', float('nan')):.3f} | {r.get('mrr', float('nan')):.3f} |")
    L += ["\n## C3: event order\n", "| Measure | Base | +C3 |", "|---|---|---|",
          f"| Pairwise order accuracy (rank) | {c3['pairwise_rank_base']:.3f} | {c3['pairwise_rank_c3']:.3f} |",
          f"| Pairwise order accuracy (score) | {c3['pairwise_score_base']:.3f} | {c3['pairwise_score_c3']:.3f} |",
          f"| MRR, events in plot order | {c3['mrr_correct_base']:.3f} | {c3['mrr_correct_c3']:.3f} |",
          f"| MRR, events reversed | {c3['mrr_reversed_base']:.3f} | {c3['mrr_reversed_c3']:.3f} |",
          f"| MRR, ordinary vague queries | {c3['vague_mrr_base']:.3f} | {c3['vague_mrr_c3']:.3f} |",
          f"\nOrder evidence prefers the plot order for the target in {c3['order_evidence_accuracy']:.3f} of {c3['n_with_evidence']} pairs.\n"]
    L += ["## C4: one question (simulated viewers)\n", "| Query set | Policy | Viewer knowledge | Ask rate | Answer rate | S@1 | MRR |",
          "|---|---|---|---|---|---|---|"]
    for name, r in c4.items():
        L.append(f"| {name} | no question | — | 0 | — | {r['s1_no_question']:.3f} | {r['mrr_no_question']:.3f} |")
        for key in ("fixed_decade|p_prior", "targeted|p_prior", "fixed_decade|p_equal", "targeted|p_equal"):
            pol, world = key.split("|")
            x = r[key]
            L.append(f"| {name} | {pol} | {'priors' if world == 'p_prior' else 'all 0.6'} | {x['ask_rate']:.3f} | "
                     f"{x['answer_rate']:.3f} | {x['s1']:.3f} | {x['mrr']:.3f} |")
    (RESULTS / "novelty_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    eng = Engine()
    if eng.dense is not None:
        eng.dense.encode(["warm up"])
    films = eng.films
    vague = load_queries(films, kinds=("vague",))
    detailed = load_queries(films, kinds=("llm",))
    if "--quick" in sys.argv:                       # smoke test: a few queries per set
        vague, detailed = vague[::40], detailed[::40]
    log(f"{len(vague)} vague, {len(detailed)} detailed queries; {len(eng.versions.members)} stories with versions")
    sets = {"clean_vague": [q for q in vague if "false_memory" not in q["tags"]],
            "clean_detailed": [q for q in detailed if "false_memory" not in q["tags"]],
            "hedged_false": make_variant(eng, vague, HEDGES), "plain_false": make_variant(eng, vague, PLAIN),
            "certain_false": make_variant(eng, vague, CERTAIN),
            "maybe_true_actor": true_actor_variant(eng, vague, "maybe"),
            "certain_true_actor": true_actor_variant(eng, vague, "I'm certain")}
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else {"c1", "c2", "c3", "c4"}
    path = RESULTS / "novelty_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if only != {"c1", "c2", "c3", "c4"} and path.exists() else {}
    summary.update({"n_films": eng.N, "sets": {k: len(v) for k, v in sets.items()}})
    if "c1" in only:
        c1 = eval_c1(eng, sets)
        summary["c1_sweep"] = c1_sweep(c1)
        rows = c1.pop("_rows")
        checks = c1.pop("_checks")
        summary["c1"] = c1
        with open(RESULTS / "novelty_c1_checks.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(checks[0].keys()))
            w.writeheader()
            w.writerows(checks)
        with open(RESULTS / "per_query_novelty_c1.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["set", "qid", "target", "base_rank", "c1_rank", "n_flagged", "n_injected"])
            for name, rs in rows.items():
                for r in rs:
                    w.writerow([name, r["qid"], r["target"], r["base_rank"], r["c1_rank"], r["n_flagged"], r["n_injected"]])
    if "c2" in only:
        summary["c2"] = eval_c2(eng, vague)
    if "c3" in only:
        summary["c3"] = eval_c3(eng, vague)
    if "c4" in only:
        summary["c4"] = eval_c4(eng, {"vague": vague, "certain_false": sets["certain_false"]})
    if "--quick" in sys.argv:
        log("quick run: nothing written")
        print(json.dumps(json_safe({k: summary[k] for k in ("c1", "c3") if k in summary}), indent=1, default=float)[:3000])
        return
    (RESULTS / "novelty_summary.json").write_text(json.dumps(json_safe(summary), indent=1, default=float), encoding="utf-8")
    write_md(summary)
    log("done; results/novelty_summary.json and results/novelty_results.md written")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
