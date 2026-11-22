"""Preliminary probes for the four proposed capabilities, run on the existing Desi ToT system.

1. Certain-but-wrong memories: the paper's vague+noise set injects one HEDGED false memory per query.
   We rebuild it with the same random draws but with CONFIDENT cues ("I'm certain ...") and with no cue,
   and add a CORRECT actor clue to the clean queries. Systems: BM25F, clause-weighted BM25 (CV weights),
   coordinate ascent trained by 5-fold CV on the vague+noise set (replicates the paper's 0.819), and the
   deployed web-app model.
2. Remakes and dubs: verified cross-language remake pairs that are both in the index; Telugu films whose
   article mentions a Hindi dub.
3. Event order: reverse the clause order of each vague query and compare rankings.
4. A worked demo (Drishyam / Drushyam and Arjun Reddy / Kabir Singh) with actor-clue variants.

Usage (from the project folder, about 12 minutes on a laptop CPU):
    python scripts/probe_conflicts.py . results/probe_conflicts.json
Results of the run used in the report: results/probe_conflicts.json (Section 6.6).
"""
import importlib.util
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
from desitot.dense import sentences  # noqa: E402
from desitot.engine import Options  # noqa: E402
from desitot.ltr import CoordinateAscent  # noqa: E402

T0 = time.time()
CERTAIN = ["I'm certain", "I'm sure", "I clearly remember", "I distinctly remember", "I'm positive", "definitely"]
PLAIN = [""] * len(ev.HEDGES)
assert len(CERTAIN) == len(ev.HEDGES)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def make_variant(x, queries, cues, seed=7):
    """Exact replica of Experiment.make_noisy with another cue list (same random draws, same false content)."""
    rng = random.Random(seed)
    films = x.eng.films
    pool = defaultdict(list)
    for f in films:
        if f["curated"]:
            pool[f["lang"]].append(f)
    out = []
    for q in queries:
        t = films[q["target"]]
        other = rng.choice([f for f in pool[t["lang"]] if f["id"] != t["id"]])
        sents = [s for s in sentences(other["plot"]) if 8 <= len(s.split()) <= 25]
        bits = []
        if sents:
            s = rng.choice(sents)
            bits.append(f"{rng.choice(cues)} {s[0].lower() + s[1:]}".rstrip(".").strip())
        actors = [a for a in other["stars"] if a not in t["cast"]]
        actor_added = False
        if actors and rng.random() < 0.5:
            bits.append(f"{rng.choice(cues)} {rng.choice(actors)} was in it".strip())
            actor_added = True
        text = q["query"].rstrip(". ") + ". " + ". ".join(bits) + "."
        nq = {**q, "query": text, "source": other["id"], "actor_added": actor_added}
        nq["qa"] = x.eng.analyze(text)
        out.append(nq)
    return out


def true_actor_variant(x, queries, cue):
    out = []
    for q in queries:
        t = x.eng.films[q["target"]]
        stars = t["stars"] or t["cast"]
        if stars:
            text = q["query"].rstrip(". ") + f". {cue} {stars[0]} was in it."
        else:
            text = q["query"]
        nq = {**q, "query": text, "has_actor": bool(stars)}
        nq["qa"] = x.eng.analyze(text)
        out.append(nq)
    return out


def mrr(r):
    r = np.asarray(r, dtype=float)
    return float(np.mean(1.0 / r))


def s_at(r, k):
    return float(np.mean(np.asarray(r) <= k))


def main():
    x = ev.Experiment()
    eng = x.eng
    N = eng.N
    tie = 1e-9 * np.arange(N)[::-1]
    V0 = x.sets["vague"]
    V1 = x.sets["vague+noise"]
    folds = np.array(x.folds["vague+noise"])
    V1r = make_variant(x, V0, ev.HEDGES)
    same = sum(a["query"] == b["query"] for a, b in zip(V1, V1r))
    log(f"replica check: {same}/{len(V1)} hedged false-memory queries identical to the paper's set")
    variants = {
        "clean": V0,
        "hedged_false": V1r,
        "plain_false": make_variant(x, V0, PLAIN),
        "certain_false": make_variant(x, V0, CERTAIN),
        "hedged_true_actor": true_actor_variant(x, V0, "maybe"),
        "certain_true_actor": true_actor_variant(x, V0, "I'm certain"),
    }
    for name, Q in variants.items():
        x.prefetch_dense(Q)
    log("dense prefetch done")

    # ---------------------------------------------------------------- CV models (paper's vague+noise condition)
    opts_lex = Options(clauses=True, negation="scope", hinglish=True)
    fold_w, w_all, _ = x.learn_weights(V1r, folds, opts_lex)
    log("fold clause weights:", fold_w)
    names = Options().feature_names()
    cache = {}

    def feats(vname, i, w):
        key = (vname, i, tuple(sorted(w.items())))
        if key not in cache:
            cache[key] = eng.compute(variants[vname][i]["qa"], Options(), w)
        return cache[key]

    res = {v: {"bm25": [None] * len(V0), "lex_cv": [None] * len(V0), "ca_cv": [None] * len(V0),
               "src_bm25": [None] * len(V0), "src_lex_cv": [None] * len(V0), "src_ca_cv": [None] * len(V0)}
           for v in variants}
    ca_entity_w = []
    for f in range(ev.K_FOLDS):
        w = fold_w[f]
        tr = np.flatnonzero(folds != f)
        te = np.flatnonzero(folds == f)
        qfs = [feats("hedged_false", i, w) for i in tr]
        groups = ev.training_groups(qfs, [V1r[i]["target"] for i in tr], names, True)
        model = CoordinateAscent(names).fit(groups, init={"bm25_clause": 1.0})
        ca_entity_w.append(dict(zip(names, model.w.tolist())))
        for vname, Q in variants.items():
            for i in te:
                qf = feats(vname, i, w)
                t = Q[i]["target"]
                bm = eng.scores(qf, "bm25")
                lex = qf.lex + tie
                order = ev.ltr_order(qf, model, names, True)
                res[vname]["bm25"][i] = ev.rank_in(bm, t)
                res[vname]["lex_cv"][i] = ev.rank_in(lex, t)
                res[vname]["ca_cv"][i] = ev.pos_of(order, t)
                src = Q[i].get("source")
                if src is not None:
                    res[vname]["src_bm25"][i] = ev.rank_in(bm, src)
                    res[vname]["src_lex_cv"][i] = ev.rank_in(lex, src)
                    res[vname]["src_ca_cv"][i] = ev.pos_of(order, src)
        log(f"fold {f} done")

    # ---------------------------------------------------------------- deployed model (the web app)
    dep = {v: {"rank": [], "src": []} for v in variants}
    for vname, Q in variants.items():
        for q in Q:
            qf = eng.compute(q["qa"], Options(), eng.weights)
            s = eng.scores(qf, "ltr")
            dep[vname]["rank"].append(ev.rank_in(s, q["target"]))
            if q.get("source") is not None:
                dep[vname]["src"].append(ev.rank_in(s, q["source"]))
        log(f"deployed: {vname} done")

    # sanity check against the paper's per-query ranks
    import csv
    paper = []
    with open(ROOT / "results" / "per_query_vague_noise.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            paper.append(int(float(row["Full LTR: coordinate ascent (ours)"])))
    agree = sum(int(a) == b for a, b in zip(res["hedged_false"]["ca_cv"], paper))

    summary = {"replica_identical": same, "n": len(V0), "fold_weights": fold_w,
               "paper_ca_mrr_vague_noise": mrr(paper), "replica_ca_rank_agreement": agree,
               "cv_entity_match_weight_mean": float(np.mean([w["entity_match"] for w in ca_entity_w])),
               "deployed_clause_weights": eng.weights, "variants": {}}
    actor_idx = [i for i, q in enumerate(variants["certain_false"]) if q["actor_added"]]
    has_actor = [i for i, q in enumerate(variants["certain_true_actor"]) if q["has_actor"]]
    for vname in variants:
        row = {}
        for sysname in ("bm25", "lex_cv", "ca_cv"):
            r = res[vname][sysname]
            row[sysname] = {"mrr": mrr(r), "s@1": s_at(r, 1), "s@5": s_at(r, 5), "s@10": s_at(r, 10)}
            if "true_actor" in vname:
                sub = [r[i] for i in has_actor]
                row[sysname]["mrr_subset_with_actor"] = mrr(sub)
            if vname.endswith("_false"):
                src = res[vname]["src_" + sysname]
                row[sysname]["source_above_target"] = float(np.mean([s < t for s, t in zip(src, r)]))
                row[sysname]["source_top1"] = float(np.mean([s == 1 for s in src]))
                row[sysname]["mrr_actor_subset"] = mrr([r[i] for i in actor_idx])
        r = dep[vname]["rank"]
        row["deployed"] = {"mrr": mrr(r), "s@1": s_at(r, 1), "s@5": s_at(r, 5), "s@10": s_at(r, 10)}
        if vname.endswith("_false"):
            src = dep[vname]["src"]
            row["deployed"]["source_above_target"] = float(np.mean([s < t for s, t in zip(src, r)]))
            row["deployed"]["source_top1"] = float(np.mean([s == 1 for s in src]))
            row["deployed"]["mrr_actor_subset"] = mrr([r[i] for i in actor_idx])
        if "true_actor" in vname:
            row["deployed"]["mrr_subset_with_actor"] = mrr([r[i] for i in has_actor])
        summary["variants"][vname] = row
    summary["n_actor_subset"] = len(actor_idx)
    summary["n_has_actor"] = len(has_actor)
    summary["examples"] = {v: variants[v][3]["query"] for v in variants}
    log("variants summarised")

    # ---------------------------------------------------------------- event order (deployed model)
    flips, same_rank, n_ord, rr_o, rr_r, bm_same = 0, 0, 0, [], [], 0
    for q in V0:
        cl = q["qa"].clauses
        if len([c for c in cl if not c.meta]) < 2:
            continue
        rev = ". ".join(c.text for c in reversed(cl)) + "."
        qa_r = eng.analyze(rev)
        qf_o = eng.compute(q["qa"], Options(), eng.weights)
        qf_r = eng.compute(qa_r, Options(), eng.weights)
        s_o, s_r = eng.scores(qf_o, "ltr"), eng.scores(qf_r, "ltr")
        ro, rr = ev.rank_in(s_o, q["target"]), ev.rank_in(s_r, q["target"])
        n_ord += 1
        same_rank += ro == rr
        flips += int(np.argmax(s_o) != np.argmax(s_r))
        rr_o.append(1 / ro)
        rr_r.append(1 / rr)
        bm_same += int(np.allclose(qf_o.F["bm25_plain"], qf_r.F["bm25_plain"]))
    summary["order"] = {"n": n_ord, "target_rank_unchanged": same_rank / n_ord, "top1_changed": flips / n_ord,
                        "mrr_original": float(np.mean(rr_o)), "mrr_reversed": float(np.mean(rr_r)),
                        "bm25_scores_identical": bm_same / n_ord}
    log("order probe done", summary["order"])

    # ---------------------------------------------------------------- remakes and dubs (deployed model)
    films = eng.films
    def fid(title, year):
        c = [f for f in films if (f["title"] == title or f["title"].startswith(title)) and f["year"] == year]
        return c[0]["id"] if c else None
    pairs = [(("Arjun Reddy", 2017), ("Kabir Singh", 2019)), (("Drushyam", 2014), ("Drishyam", 2015)),
             (("Drushyam 2", 2021), ("Drishyam 2", 2022)), (("Pokiri", 2006), ("Wanted", 2009)),
             (("Dabangg", 2010), ("Gabbar Singh", 2012)), (("Munna Bhai M.B.B.S.", 2003), ("Shankar Dada M.B.B.S.", 2004)),
             (("Lage Raho Munna Bhai", 2006), ("Shankar Dada Zindabad", 2007)), (("Pink", 2016), ("Vakeel Saab", 2021)),
             (("Temper", 2015), ("Simmba", 2018)), (("OMG", 2012), ("Gopala Gopala", 2015)),
             (("Hello Brother", 1994), ("Judwaa", 1997)), (("Kalisundam Raa", 2000), ("Kuch Tum Kaho Kuch Hum Kahein", 2002))]
    by_target = {q["target"]: q for q in V0}
    lang_word = {"hi": "Hindi", "te": "Telugu", "en": "English"}
    remake_rows, evidence = [], {}
    for a, b in pairs:
        ia, ib = fid(*a), fid(*b)
        if ia is None or ib is None:
            remake_rows.append({"pair": [a, b], "missing": True})
            continue
        for t_id, o_id in ((ia, ib), (ib, ia)):
            f = films[t_id]
            lead = f.get("lead") or ""
            evidence[f["title"]] = [s for s in re.split(r"(?<=[.!?])\s+", lead) if re.search(r"remake|remade", s, re.I)][:2]
            q = by_target.get(t_id)
            if q is None:
                continue
            row = {"target": f"{f['title']} ({f['year']}, {f['lang']})", "other": f"{films[o_id]['title']} ({films[o_id]['year']}, {films[o_id]['lang']})"}
            for label, text in (("plain", q["query"]),
                                ("watched_in_lang", q["query"].rstrip(". ") + f". I watched it in {lang_word[f['lang']]}.")):
                qf = eng.compute(eng.analyze(text), Options(), eng.weights)
                s = eng.scores(qf, "ltr")
                rt, ro = ev.rank_in(s, t_id), ev.rank_in(s, o_id)
                row[label] = {"target_rank": rt, "other_rank": ro}
            remake_rows.append(row)
    ok = [r for r in remake_rows if not r.get("missing")]
    def agg(label):
        return {"n": len(ok), "story_top1": float(np.mean([min(r[label]["target_rank"], r[label]["other_rank"]) == 1 for r in ok])),
                "story_top5_both": float(np.mean([max(r[label]["target_rank"], r[label]["other_rank"]) <= 5 for r in ok])),
                "exact_top1": float(np.mean([r[label]["target_rank"] == 1 for r in ok])),
                "other_version_above_target": float(np.mean([r[label]["other_rank"] < r[label]["target_rank"] for r in ok]))}
    summary["remakes"] = {"plain": agg("plain"), "watched_in_lang": agg("watched_in_lang"), "rows": remake_rows,
                          "lead_evidence": evidence}
    log("remake probe done")

    dub_ids = [f["id"] for f in films if f["curated"] and f["lang"] == "te" and re.search(
        r"dubbed[^.]{0,80}hindi|hindi[- ]dubbed|hindi[^.]{0,40}dubbed",
        (f.get("lead") or "") + " " + (f.get("other") or "")[:20000], re.I)]
    before, after, hindi_top = [], [], 0
    for t_id in dub_ids:
        q = by_target.get(t_id)
        if q is None:
            continue
        for label, text in (("b", q["query"]), ("a", q["query"].rstrip(". ") + ". I watched it in Hindi.")):
            qf = eng.compute(eng.analyze(text), Options(), eng.weights)
            s = eng.scores(qf, "ltr")
            r = ev.rank_in(s, t_id)
            (before if label == "b" else after).append(r)
            if label == "a" and r > 1 and films[int(np.argmax(s))]["lang"] == "hi":
                hindi_top += 1
    summary["dubs"] = {"n": len(before), "mrr_before": mrr(before), "mrr_after": mrr(after),
                       "s1_before": s_at(before, 1), "s1_after": s_at(after, 1),
                       "worse": float(np.mean([a > b for a, b in zip(after, before)])),
                       "better": float(np.mean([a < b for a, b in zip(after, before)])),
                       "hindi_film_took_first_place": hindi_top}
    log("dub probe done", summary["dubs"])

    # ---------------------------------------------------------------- demo cases (deployed model)
    demos = {
        "drishyam": ("A cable TV operator who is crazy about films protects his family after his daughter accidentally "
                     "kills a boy who was blackmailing her. He uses tricks he learned from movies to build an alibi.",
                     ["", "Maybe Venkatesh was in it.", "I'm certain Venkatesh was in it.", "I'm certain Ajay Devgn was in it.",
                      "Maybe Akshay Kumar was in it.", "I'm certain Akshay Kumar was in it."],
                     [("Drishyam", 2015), ("Drushyam", 2014)]),
        "arjun_reddy": ("A brilliant young surgeon with anger issues destroys himself with alcohol and drugs after the "
                        "girl he loves is forced to marry someone else.",
                        ["", "I'm certain Shahid Kapoor was in it.", "I'm certain Vijay Deverakonda was in it.",
                         "Maybe Ranbir Kapoor was in it.", "I'm certain Ranbir Kapoor was in it."],
                        [("Kabir Singh", 2019), ("Arjun Reddy", 2017)]),
    }
    summary["demos"] = {}
    for key, (base, extras, watch) in demos.items():
        ids = [fid(*w) for w in watch]
        rows = []
        for extra in extras:
            text = (base + " " + extra).strip()
            qf = eng.compute(eng.analyze(text), Options(), eng.weights)
            s = eng.scores(qf, "ltr")
            top = np.argsort(-s, kind="stable")[:3]
            rows.append({"added": extra or "(nothing)",
                         "top3": [f"{films[d]['title']} ({films[d]['year']}, {films[d]['lang']})" for d in top],
                         "ranks": {films[i]["title"]: ev.rank_in(s, i) for i in ids if i is not None}})
        summary["demos"][key] = {"base": base, "rows": rows}
    log("demos done")
    OUT.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    log("wrote", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
