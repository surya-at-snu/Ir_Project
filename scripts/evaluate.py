#!/usr/bin/env python
"""Run every experiment reported in the project and train the final models used by the web app.

Query conditions
  detailed      300 LLM-written queries rich in names, songs and dialogue (an easy, "ceiling" condition)
  vague         300 LLM-written queries in the style of real tip-of-the-tongue posts (PRIMARY condition)
  vague+noise   the vague queries plus one injected *hedged* false memory each (robustness to false memories)
  human         queries your team adds through the web app (scored with models trained on LLM queries only)

Everything that is learned (clause weights c_h, c_conf, c_ctx, the exclusion weight mu, and the
learning-to-rank models) is evaluated with 5-fold cross-validation: weights are chosen on four folds
and tested on the fifth, so no test query influences its own ranking.

Outputs go to results/ (CSV + Markdown tables, summary.json) and the final learned weights and
ranker go to data/index/ for the web app.  Run `python scripts/make_figures.py` afterwards for plots.

Usage:  python scripts/evaluate.py
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desitot.config import INDEX_DIR, RESULTS  # noqa: E402
from desitot.dense import sentences  # noqa: E402
from desitot.engine import GROUPS, Engine, Options  # noqa: E402
from desitot.ltr import CoordinateAscent, LambdaMART, PairwiseLogistic  # noqa: E402
from desitot.metrics import (aggregate, binary_prf, bootstrap_ci, classification_report, json_safe,  # noqa: E402
                             kendall_tau, paired_permutation_test, per_query)
from desitot.nb import MultinomialNB  # noqa: E402
from desitot.queries import load_queries  # noqa: E402
from desitot.text import analyze  # noqa: E402

K_FOLDS = 5
CAND = 100
PRIMARY = "vague"
GRID = {"hedged": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], "confident": [1.0, 1.25, 1.5],
        "context": [0.25, 0.5, 1.0], "mu": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]}
DEFAULT_POINT = {"hedged": 0.3, "confident": 1.2, "context": 0.5, "mu": 0.5}
TREC_2025 = [  # Table 1 of Arguello et al., "Overview of the TREC 2025 Tip-of-the-Tongue track" (arXiv:2601.20671)
    ("TREC 2025: BM25 (PyTerrier)", 0.1223, 0.1114, 0.4855, 0.1700),
    ("TREC 2025: BM25 (Anserini)", 0.0763, 0.0724, 0.2749, 0.1032),
    ("TREC 2025: dense (Lightning IR)", 0.0189, 0.0160, 0.2653, 0.0514),
    ("TREC 2025: best run (scrb-tot-04)", 0.6576, 0.6258, 0.9051, 0.6824),
]
HEDGES = ["I think", "maybe", "I'm not sure but I think", "probably", "if I remember correctly", "I guess"]
LEX_CONFIGS = {
    "Clause-weighted BM25 (learned c_h, c_conf, c_ctx)": Options(clauses=True, negation="none", hinglish=False),
    "+ scope-aware negation (learned mu)": Options(clauses=True, negation="scope", hinglish=False),
    "+ Hinglish spelling handling": Options(clauses=True, negation="scope", hinglish=True),
}
FULL_LEX = "+ Hinglish spelling handling"
LTR_LABELS = {"ca": "Full LTR: coordinate ascent (ours)", "rank": "Full LTR: pairwise logistic / RankNet-linear (ours)",
              "lmart": "Full LTR: LambdaMART (ours)"}
BASE = "BM25F (baseline)"


# --------------------------------------------------------------------------- helpers
def rank_in(scores: np.ndarray, target: int) -> int:
    s = scores[target]
    return int(1 + np.sum(scores > s) + np.sum((scores == s) & (np.arange(len(scores)) < target)))


def make_folds(queries, k=K_FOLDS, seed=13):
    films = sorted({q["target"] for q in queries})
    random.Random(seed).shuffle(films)
    fold_of = {f: i % k for i, f in enumerate(films)}
    return [fold_of[q["target"]] for q in queries]


def summarize(ranks) -> dict:
    return aggregate([per_query(r) for r in ranks])


def best_grid_index(mean_rr: np.ndarray, keys: list[tuple]) -> int:
    """argmax of mean RR; ties (common on easy queries) go to the point closest to the default weights."""
    top = np.flatnonzero(mean_rr >= mean_rr.max() - 1e-9)
    dist = [sum(abs(v - DEFAULT_POINT[n]) for n, v in zip(GRID.keys(), keys[i])) for i in top]
    return int(top[int(np.argmin(dist))])


def ltr_order(qf, model, names, use_dense: bool, k: int = CAND) -> np.ndarray:
    first = (qf.F["rrf"] if use_dense else qf.lex) + 1e-9 * qf.F["bm25_plain"]
    order = np.argsort(-first, kind="stable")
    cand = order[:k]
    s = model.score(qf.matrix(names, cand))
    return np.concatenate([cand[np.argsort(-s, kind="stable")], order[k:]])


def training_groups(qfs, targets, names, use_dense: bool, k: int = CAND):
    groups = []
    for qf, t in zip(qfs, targets):
        first = (qf.F["rrf"] if use_dense else qf.lex) + 1e-9 * qf.F["bm25_plain"]
        cand = np.argsort(-first, kind="stable")[:k]
        if t not in cand:
            cand = np.concatenate([cand[:-1], [t]])
        groups.append((qf.matrix(names, cand), (cand == t).astype(float)))
    return groups


def pos_of(order: np.ndarray, target: int) -> int:
    return int(np.flatnonzero(order == target)[0]) + 1


class Experiment:
    def __init__(self):
        t0 = time.time()
        self.eng = Engine()
        films = self.eng.films
        self.sets = {"detailed": load_queries(films, kinds=("llm",)), "vague": load_queries(films, kinds=("vague",))}
        human = load_queries(films, kinds=("human",))
        for Q in list(self.sets.values()) + [human]:
            for q in Q:
                q["qa"] = self.eng.analyze(q["query"])
        self.sets["vague+noise"] = self.make_noisy(self.sets["vague"])
        self.human = human
        self.folds = {n: make_folds(Q) for n, Q in self.sets.items()}
        self.res: dict[str, dict[str, list[int]]] = {n: {} for n in self.sets}
        self.fold_w, self.w_all = {}, {}
        self.out: dict = {"n_films": self.eng.N, "n_curated": int(self.eng.curated.sum()),
                          "conditions": {n: len(Q) for n, Q in self.sets.items()}, "n_human": len(human)}
        print(f"{ {n: len(Q) for n, Q in self.sets.items()} } queries, {len(human)} human, {self.eng.N} films "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- query sets
    def make_noisy(self, queries, seed=7):
        """Inject one hedged false memory per query: a plot sentence (or a star) of another film of the same
        language, in the spirit of simulated known-item queries (Azzopardi et al., SIGIR 2007)."""
        rng = random.Random(seed)
        films = self.eng.films
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
                bits.append(f"{rng.choice(HEDGES)} {s[0].lower() + s[1:]}".rstrip("."))
            actors = [a for a in other["stars"] if a not in t["cast"]]
            if actors and rng.random() < 0.5:
                bits.append(f"{rng.choice(HEDGES)} {rng.choice(actors)} was in it")
            text = q["query"].rstrip(". ") + ". " + ". ".join(bits) + "."
            nq = {**q, "query": text, "qid": q["qid"] + "-noisy", "tags": sorted(set(q["tags"]) | {"false_memory", "hedge"})}
            nq["qa"] = self.eng.analyze(text)
            out.append(nq)
        return out

    def prefetch_dense(self, queries):
        """Encode every query and clause text in large batches (much faster than one call per query)."""
        if self.eng.dense is None:
            return
        texts = []
        for q in queries:
            texts.append(q["qa"].text)
            texts += [t for t, _ in q["qa"].content_text(self.eng.weights, True)]
        todo = [t for t in dict.fromkeys(texts) if t not in self.eng.dense._cache]
        for i in range(0, len(todo), 256):
            self.eng.dense.encode(todo[i:i + 256])

    # ---------------------------------------------------------------- systems
    def baselines(self, queries):
        opts = Options(clauses=False, negation="none", hinglish=False, dense=True, nb=False, cues=False, popularity=False)
        labels = [("bm25", BASE), ("tfidf", "TF-IDF cosine (VSM)"),
                  ("dense", "Dense bi-encoder (bge-small, MaxP)"), ("hybrid", "BM25 + dense (RRF)")]
        res = {label: [] for _, label in labels}
        for q in queries:
            qf = self.eng.compute(q["qa"], opts)
            for ranker, label in labels:
                res[label].append(rank_in(self.eng.scores(qf, ranker), q["target"]))
        return res

    def grid_rr(self, queries, opts: Options):
        """Reciprocal rank of every query at every point of the clause-weight grid."""
        keys = list(itertools.product(*GRID.values()))
        if opts.negation in ("none", "drop"):
            keys = [k for k in keys if k[3] == DEFAULT_POINT["mu"]]
        if not opts.clauses:
            keys = [k for k in keys if k[:3] == (1.0, 1.0, 1.0)] or [(1.0, 1.0, 1.0, k[3]) for k in keys[:1]]
        rr = np.zeros((len(queries), len(keys)))
        tie = 1e-9 * np.arange(self.eng.N)[::-1]
        for i, q in enumerate(queries):
            P, Nn = self.eng.component_scores(q["qa"], opts)
            for j, k in enumerate(keys):
                w = dict(zip(GRID.keys(), k), meta=0.0 if opts.clauses else 1.0)
                rr[i, j] = 1.0 / rank_in(self.eng.combine_components(P, Nn, w, self.eng.N) + tie, q["target"])
        return keys, rr

    def learn_weights(self, queries, folds, opts: Options):
        keys, rr = self.grid_rr(queries, opts)
        folds = np.array(folds)
        fold_w, ranks = {}, [0] * len(queries)
        for f in range(K_FOLDS):
            best = best_grid_index(rr[folds != f].mean(axis=0), keys)
            fold_w[f] = dict(zip(GRID.keys(), keys[best]), meta=0.0)
            for i in np.flatnonzero(folds == f):
                ranks[i] = int(round(1.0 / rr[i, best]))
        w_all = dict(zip(GRID.keys(), keys[best_grid_index(rr.mean(axis=0), keys)]), meta=0.0)
        return fold_w, w_all, ranks

    def ltr_cv(self, queries, folds, opts: Options, fold_w: dict, kinds=("ca",)):
        names = opts.feature_names()
        folds = np.array(folds)
        ranks = {k: [0] * len(queries) for k in kinds}
        weights = []
        for f in range(K_FOLDS):
            qfs = [self.eng.compute(q["qa"], opts, fold_w[f]) for q in queries]
            tr = np.flatnonzero(folds != f)
            groups = training_groups([qfs[i] for i in tr], [queries[i]["target"] for i in tr], names, opts.dense)
            models = {}
            if "ca" in kinds:
                models["ca"] = CoordinateAscent(names).fit(groups, init={"bm25_clause": 1.0})
                weights.append(dict(zip(names, models["ca"].w.tolist())))
            if "rank" in kinds:
                models["rank"] = PairwiseLogistic(names).fit(groups)
            if "lmart" in kinds:
                models["lmart"] = LambdaMART(names).fit(groups)
            for i in np.flatnonzero(folds == f):
                for k, m in models.items():
                    ranks[k][i] = pos_of(ltr_order(qfs[i], m, names, opts.dense), queries[i]["target"])
        return ranks, weights

    def main_systems(self, name):
        t0 = time.time()
        Q, folds = self.sets[name], self.folds[name]
        systems = self.baselines(Q)
        learned = {}
        for label, opts in LEX_CONFIGS.items():
            fw, wa, ranks = self.learn_weights(Q, folds, opts)
            systems[label] = ranks
            learned[label] = {"all": wa, "folds": fw}
        self.fold_w[name], self.w_all[name] = learned[FULL_LEX]["folds"], learned[FULL_LEX]["all"]
        ranks, ca_w = self.ltr_cv(Q, folds, Options(), self.fold_w[name], kinds=tuple(LTR_LABELS))
        for k, label in LTR_LABELS.items():
            systems[label] = ranks[k]
        self.res[name] = systems
        self.out.setdefault("learned_weights", {})[name] = learned
        self.out.setdefault("ca_feature_weights_folds", {})[name] = ca_w
        print(f"[{name}] systems done in {time.time() - t0:.0f}s: "
              + ", ".join(f"{s.split(':')[0][:22]}={summarize(r)['rr']:.3f}" for s, r in systems.items()), flush=True)

    # ---------------------------------------------------------------- experiments
    def run(self):
        RESULTS.mkdir(exist_ok=True)
        t0 = time.time()
        for Q in list(self.sets.values()) + [self.human]:
            self.prefetch_dense(Q)
        print(f"dense pre-encoding done ({time.time() - t0:.0f}s)", flush=True)
        for name in self.sets:
            self.main_systems(name)
        self.report_main()
        for name in (PRIMARY, "vague+noise"):
            self.ablation(name)
        print(f"ablations done ({time.time() - t0:.0f}s)", flush=True)
        for name in (PRIMARY, "detailed"):
            self.negation_modes(name)
        for name in (PRIMARY, "vague+noise"):
            self.hedge_sweep(name)
        self.phenomena()
        self.index_size(PRIMARY)
        self.nb_eval()
        self.cue_detection()
        self.train_final()
        self.human_eval()
        self.trec_comparison()
        self.examples(PRIMARY)
        self.write_per_query()
        (RESULTS / "summary.json").write_text(json.dumps(json_safe(self.out), indent=1, default=float), encoding="utf-8")
        print(f"all done in {time.time() - t0:.0f}s; results in {RESULTS}", flush=True)

    def report_main(self):
        self.out["main"], overview = {}, {}
        for name, systems in self.res.items():
            base_rr = [1 / r for r in systems[BASE]]
            rows = []
            for label, ranks in systems.items():
                m = summarize(ranks)
                rr = [1 / r for r in ranks]
                lo, hi = bootstrap_ci(rr)
                p = paired_permutation_test(rr, base_rr) if label != BASE else float("nan")
                rows.append({"system": label, **{k: m[k] for k in ("rr", "s@1", "s@5", "s@10", "ndcg10", "ndcg1000",
                                                                    "p@10", "f1@10", "median_rank")},
                             "mrr_ci_low": lo, "mrr_ci_high": hi, "p_vs_bm25": p})
                overview.setdefault(label, {})[name] = {"mrr": m["rr"], "s@1": m["s@1"], "s@10": m["s@10"]}
            self.out["main"][name] = rows
            self._csv(f"main_results_{name.replace('+', '_')}.csv", rows)
            lines = [f"### Condition: {name} ({len(self.sets[name])} queries, {self.eng.N} films)\n",
                     "| System | MRR (95% CI) | S@1 | S@5 | S@10 | nDCG@10 | nDCG@all | P@10 | F1@10 | p vs BM25 |",
                     "|---|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                p = "—" if math.isnan(r["p_vs_bm25"]) else ("<0.001" if r["p_vs_bm25"] < 0.001 else f"{r['p_vs_bm25']:.3f}")
                lines.append(f"| {r['system']} | {r['rr']:.3f} ({r['mrr_ci_low']:.3f}–{r['mrr_ci_high']:.3f}) | "
                             f"{r['s@1']:.3f} | {r['s@5']:.3f} | {r['s@10']:.3f} | {r['ndcg10']:.3f} | "
                             f"{r['ndcg1000']:.3f} | {r['p@10']:.3f} | {r['f1@10']:.3f} | {p} |")
            (RESULTS / f"main_results_{name.replace('+', '_')}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print("\n".join(lines), flush=True)
        self.out["overview"] = overview
        conds = list(self.res)
        lines = ["| System | " + " | ".join(f"MRR {c}" for c in conds) + " | " + " | ".join(f"S@1 {c}" for c in conds) + " |",
                 "|---|" + "---|" * (2 * len(conds))]
        for label, d in overview.items():
            lines.append(f"| {label} | " + " | ".join(f"{d[c]['mrr']:.3f}" for c in conds) + " | "
                         + " | ".join(f"{d[c]['s@1']:.3f}" for c in conds) + " |")
        (RESULTS / "main_results_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def ablation(self, name):
        Q, folds = self.sets[name], self.folds[name]
        variants = {"Full model": Options(),
                    "– clause weighting": Options(clauses=False),
                    "– negation handling": Options(negation="none"),
                    "– Hinglish handling": Options(hinglish=False),
                    "– dense retrieval": Options(dense=False),
                    "– Naive Bayes soft filters": Options(nb=False),
                    "– explicit cues (title/person/year/language/genre)": Options(cues=False),
                    "– PageRank / HITS / page views": Options(popularity=False)}
        rows, full_rr = [], None
        for label, opts in variants.items():
            ranks, _ = self.ltr_cv(Q, folds, opts, self.fold_w[name], kinds=("ca",))
            m = summarize(ranks["ca"])
            rr = [1 / r for r in ranks["ca"]]
            full_rr = full_rr or rr
            rows.append({"variant": label, "mrr": m["rr"], "s@1": m["s@1"], "s@10": m["s@10"], "ndcg10": m["ndcg10"],
                         "delta_mrr": m["rr"] - float(np.mean(full_rr)),
                         "p_vs_full": paired_permutation_test(rr, full_rr) if label != "Full model" else float("nan")})
        self.out.setdefault("ablation", {})[name] = rows
        self._csv(f"ablation_{name.replace('+', '_')}.csv", rows)
        for r in rows:
            print(f"  [{name}] {r['variant']:52s} MRR {r['mrr']:.3f}  Δ {r['delta_mrr']:+.3f}", flush=True)

    def negation_modes(self, name):
        Q, folds = self.sets[name], self.folds[name]
        gold = [i for i, q in enumerate(Q) if "negation" in q["tags"]]
        rows = []
        for mode in ("none", "drop", "penalize_all", "scope"):
            _, wa, ranks = self.learn_weights(Q, folds, Options(clauses=True, negation=mode, hinglish=True))
            sub = [ranks[i] for i in gold]
            rows.append({"mode": mode, "mrr_all": summarize(ranks)["rr"], "mrr_negation_queries": summarize(sub)["rr"],
                         "s@1_negation_queries": summarize(sub)["s@1"], "n_negation_queries": len(sub),
                         "learned_mu": wa["mu"]})
        self.out.setdefault("negation_modes", {})[name] = rows
        self._csv(f"negation_modes_{name.replace('+', '_')}.csv", rows)

    def hedge_sweep(self, name):
        Q = self.sets[name]
        opts = Options(clauses=True, negation="scope", hinglish=True)
        comps = [self.eng.component_scores(q["qa"], opts) for q in Q]
        hedged = [i for i, q in enumerate(Q) if any(c.hedged for c in q["qa"].clauses)]
        tie = 1e-9 * np.arange(self.eng.N)[::-1]
        rows = []
        for param, values in (("hedged", [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]),
                              ("mu", [0, .1, .25, .5, .75, 1.0, 1.5, 2.0])):
            for v in values:
                w = dict(self.w_all[name], **{param: v})
                ranks = [rank_in(self.eng.combine_components(P, Nn, w, self.eng.N) + tie, q["target"])
                         for (P, Nn), q in zip(comps, Q)]
                rows.append({"param": param, "value": v, "mrr_all": summarize(ranks)["rr"],
                             "mrr_hedged_queries": summarize([ranks[i] for i in hedged])["rr"] if hedged else None,
                             "n_hedged": len(hedged)})
        self.out.setdefault("hedge_sweep", {})[name] = rows
        self._csv(f"hedge_sweep_{name.replace('+', '_')}.csv", rows)

    def phenomena(self):
        tags = ["hedge", "negation", "false_memory", "hinglish", "decade", "name", "social", "lang"]
        systems = [BASE, "Dense bi-encoder (bge-small, MaxP)", "BM25 + dense (RRF)", FULL_LEX, LTR_LABELS["ca"]]
        self.out["phenomena"] = {}
        for name in ("detailed", PRIMARY, "vague+noise"):
            Q, rows = self.sets[name], []
            for tag in tags + ["(none of hedge/negation/false memory)"]:
                if tag.startswith("("):
                    idx = [i for i, q in enumerate(Q) if not set(q["tags"]) & {"hedge", "negation", "false_memory"}]
                else:
                    idx = [i for i, q in enumerate(Q) if tag in q["tags"]]
                if len(idx) < 3:
                    continue
                row = {"phenomenon": tag, "n": len(idx)}
                for s in systems:
                    row[s] = summarize([self.res[name][s][i] for i in idx])["rr"]
                rows.append(row)
            self.out["phenomena"][name] = rows
            self._csv(f"phenomena_{name.replace('+', '_')}.csv", rows)

    def _fit_ca(self, queries, w):
        opts = Options()
        names = opts.feature_names()
        qfs = [self.eng.compute(q["qa"], opts, w) for q in queries]
        groups = training_groups(qfs, [q["target"] for q in queries], names, True)
        return CoordinateAscent(names).fit(groups, init={"bm25_clause": 1.0})

    def index_size(self, name):
        Q, w = self.sets[name], self.w_all[name]
        model = self._fit_ca(Q, w)
        names = Options().feature_names()
        base_opts = Options(clauses=False, negation="none", hinglish=False, nb=False, cues=False, popularity=False)
        rows = []
        for scope in ("curated only (300 films)", f"full index ({self.eng.N} films)"):
            r_b, r_f = [], []
            for q in Q:
                s = self.eng.scores(self.eng.compute(q["qa"], base_opts), "bm25")
                order = ltr_order(self.eng.compute(q["qa"], Options(), w), model, names, True)
                if scope.startswith("curated"):
                    s = np.where(self.eng.curated, s, -np.inf)
                    order = np.array([d for d in order if self.eng.curated[d]])
                r_b.append(rank_in(s, q["target"]))
                r_f.append(pos_of(order, q["target"]))
            for label, r in ((BASE, r_b), (LTR_LABELS["ca"] + " [in-sample]", r_f)):
                m = summarize(r)
                rows.append({"index": scope, "system": label, "mrr": m["rr"], "s@1": m["s@1"], "s@10": m["s@10"]})
        self.out["index_size"] = rows
        self._csv("index_size.csv", rows)

    def nb_eval(self):
        films = self.eng.films
        toks = [[t for t in analyze(f["plot"]) if not t.isdigit()] for f in films]
        decade = lambda y: max(1950, min(2020, (y or 2000) // 10 * 10))
        genres = self.eng.nb_genres
        idx = [i for i, f in enumerate(films) if toks[i] and f["year"]]
        random.Random(3).shuffle(idx)
        fold = {d: j % K_FOLDS for j, d in enumerate(idx)}
        y_true_d, y_pred_d, maj, g_hit, g_true, g_pred = [], [], [], [], [], []
        for f in range(K_FOLDS):
            tr = [i for i in idx if fold[i] != f]
            te = [i for i in idx if fold[i] == f]
            nbd = MultinomialNB(alpha=0.5).fit([toks[i] for i in tr], [[decade(films[i]["year"])] for i in tr])
            y_true_d += [decade(films[i]["year"]) for i in te]
            y_pred_d += nbd.predict([toks[i] for i in te])
            maj += [Counter(decade(films[i]["year"]) for i in tr).most_common(1)[0][0]] * len(te)
            trg = [i for i in tr if set(films[i]["genres"]) & set(genres)]
            nbg = MultinomialNB(alpha=0.5).fit([toks[i] for i in trg],
                                               [sorted(set(films[i]["genres"]) & set(genres)) for i in trg], vocab=nbd.vocab)
            teg = [i for i in te if set(films[i]["genres"]) & set(genres)]
            for i, p in zip(teg, nbg.predict_proba([toks[i] for i in teg], temperature=1.0)):
                order = np.argsort(-p)
                true = set(films[i]["genres"]) & set(genres)
                g_hit.append(nbg.classes_[order[0]] in true)
                g_true.append(true)
                g_pred.append({nbg.classes_[j] for j in order[:2]})
        dec, dec_major = classification_report(y_true_d, y_pred_d), classification_report(y_true_d, maj)
        per_genre = {g: binary_prf([g in s for s in g_true], [g in s for s in g_pred]) for g in genres}
        macro = {k: float(np.mean([v[k] for v in per_genre.values()])) for k in ("precision", "recall", "f1")}
        tp, fp, fn = (sum(v[k] for v in per_genre.values()) for k in ("tp", "fp", "fn"))
        mp, mr = tp / max(1, tp + fp), tp / max(1, tp + fn)
        q_true, q_pred, q_hit = [], [], []
        for q in self.sets[PRIMARY]:
            pos, _ = q["qa"].term_weights(self.eng.weights)
            tk = [t for t in pos if not t.isdigit()]
            f = films[q["target"]]
            if not tk:
                continue
            q_true.append(decade(f["year"]))
            q_pred.append(self.eng.nb_decade.classes_[int(np.argmax(self.eng.nb_decade.predict_proba([tk])[0]))])
            if set(f["genres"]) & set(genres):
                q_hit.append(self.eng.nb_genre.classes_[int(np.argmax(self.eng.nb_genre.predict_proba([tk])[0]))] in f["genres"])
        labels = sorted(set(y_true_d) | set(y_pred_d))
        self.out["nb"] = {
            "decade_docs_cv": {k: dec[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")},
            "decade_docs_majority_baseline": {k: dec_major[k] for k in ("accuracy", "macro_f1")},
            "decade_per_class": dec["per_class"],
            "decade_confusion": {"labels": labels, "matrix": [[sum(1 for a, b in zip(y_true_d, y_pred_d) if a == r and b == c)
                                                                for c in labels] for r in labels]},
            "genre_docs_cv": {"top1_accuracy": float(np.mean(g_hit)), "macro_precision@2": macro["precision"],
                              "macro_recall@2": macro["recall"], "macro_f1@2": macro["f1"], "micro_precision@2": mp,
                              "micro_recall@2": mr, "micro_f1@2": 2 * mp * mr / max(1e-9, mp + mr)},
            "genre_per_class": per_genre,
            "queries_" + PRIMARY: {"decade_accuracy": float(np.mean([a == b for a, b in zip(q_true, q_pred)])),
                                   "genre_top1_accuracy": float(np.mean(q_hit)) if q_hit else None, "n": len(q_true)},
            "top_words": {g: self.eng.nb_genre.top_features(g, 10) for g in self.eng.nb_genre.classes_},
        }
        (RESULTS / "nb_eval.json").write_text(json.dumps(self.out["nb"], indent=1), encoding="utf-8")

    def cue_detection(self):
        Q = self.sets["detailed"] + self.sets[PRIMARY]
        res, errors = {}, {}
        for name, tag, pred in (("hedge", "hedge", lambda qa: any(c.hedged for c in qa.clauses)),
                                ("exclusion", "negation", lambda qa: any(c.exclusion for c in qa.clauses))):
            t = [tag in q["tags"] for q in Q]
            p = [pred(q["qa"]) for q in Q]
            res[name] = binary_prf(t, p)
            errors[name] = {"false_positives": [q["query"] for q, a, b in zip(Q, t, p) if b and not a][:8],
                            "false_negatives": [q["query"] for q, a, b in zip(Q, t, p) if a and not b][:8]}
        kinds = Counter(c.kind for q in Q for c in q["qa"].clauses)
        self.out["cue_detection"] = {**res, "n_queries": len(Q), "errors": errors, "clause_kinds": dict(kinds),
                                     "clauses_per_query": float(np.mean([len(q["qa"].clauses) for q in Q]))}
        (RESULTS / "cue_detection.json").write_text(json.dumps(self.out["cue_detection"], indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- final models
    def fit_all(self, train, min_hedge: float = 0.0, nonpositive=()):
        """Learn clause weights and the three rankers on `train`.

        min_hedge bounds c_h from below: the injected false-memory set makes *every* hedged clause wrong, which
        pushes the unconstrained optimum to c_h = 0 (hedges ignored).  For the deployed model we keep hedged
        memories down-weighted but never discarded (c_h >= 0.3, the design value of the original plan).
        nonpositive keeps the coordinate-ascent weights of those features <= 0.  The deployed model uses it for the
        exclusion features: many LLM queries rule out a sequel or remake whose title words also appear in the
        target's own article ("not Karthikeya 2"), which can teach the ranker a small *positive* weight on the
        BM25 exclusion feature, so that "definitely not a comedy" would favour comedies."""
        opts = Options(clauses=True, negation="scope", hinglish=True)
        keys, rr = self.grid_rr(train, opts)
        allowed = [i for i, k in enumerate(keys) if k[0] >= min_hedge]
        keys_a, rr_a = [keys[i] for i in allowed], rr[:, allowed]
        w = dict(zip(GRID.keys(), keys_a[best_grid_index(rr_a.mean(axis=0), keys_a)]), meta=0.0)
        lex_w = {}
        for label, o in LEX_CONFIGS.items():
            k2, r2 = self.grid_rr(train, o)
            lex_w[label] = dict(zip(GRID.keys(), k2[best_grid_index(r2.mean(axis=0), k2)]), meta=0.0)
        names = Options().feature_names()
        qfs = [self.eng.compute(q["qa"], Options(), w) for q in train]
        groups = training_groups(qfs, [q["target"] for q in train], names, True)
        models = {"ca": CoordinateAscent(names).fit(groups, init={"bm25_clause": 1.0}, nonpositive=nonpositive),
                  "rank": PairwiseLogistic(names).fit(groups), "lmart": LambdaMART(names).fit(groups)}
        return w, lex_w, models

    def train_final(self):
        # all LLM queries, including the false-memory variants, so the deployed model learns to discount
        # hedged statements (real ToT requests contain false memories; TREC ToT 2025)
        train = self.sets["detailed"] + self.sets[PRIMARY] + self.sets["vague+noise"]
        w, lex_w, models = self.fit_all(train, min_hedge=0.3, nonpositive=GROUPS["negation"])
        self.final = (w, lex_w, models)
        for path in (INDEX_DIR / "learned_weights.json", RESULTS / "learned_weights.json"):
            path.write_text(json.dumps(w, indent=1), encoding="utf-8")
        (INDEX_DIR / "ltr_model.json").write_text(json.dumps(models["ca"].to_json(), indent=1), encoding="utf-8")
        self.eng.weights.update(w)
        self.eng.model = models["ca"]
        self.out["final_model"] = {"trained_on": len(train), "clause_weights": w, "kind": models["ca"].kind,
                                   "weights": dict(zip(models["ca"].names, models["ca"].w.tolist())),
                                   "train_mrr": getattr(models["ca"], "train_mrr", None)}

    def human_eval(self):
        H = self.human
        if not H:
            self.out["human"] = {"status": "no human queries yet: add some with the web app or data/queries/human_queries.jsonl"}
            return
        w, lex_w, models = self.final
        res = self.baselines(H)
        for label, o in LEX_CONFIGS.items():
            res[label] = [rank_in(self.eng.compute(q["qa"], o, lex_w[label]).lex + 1e-9 * np.arange(self.eng.N)[::-1],
                                  q["target"]) for q in H]
        names = Options().feature_names()
        for k, label in LTR_LABELS.items():
            res[label] = [pos_of(ltr_order(self.eng.compute(q["qa"], Options(), w), models[k], names, True), q["target"])
                          for q in H]
        self.res["human"] = res
        rows = [{"system": s, **{k: summarize(r)[k] for k in ("rr", "s@1", "s@10", "ndcg10")}} for s, r in res.items()]
        self.out["human"] = {"n": len(H), "rows": rows}
        self._csv("main_results_human.csv", rows)
        if len(H) >= 10:
            systems = list(res)
            llm = [summarize(self.res[PRIMARY][s])["rr"] for s in systems]
            hum = [summarize(res[s])["rr"] for s in systems]
            tau, p = kendall_tau(llm, hum)
            self.out["human"]["kendall"] = {"systems": systems, "llm_mrr": llm, "human_mrr": hum, "tau": tau, "p": p}
        else:
            self.out["human"]["kendall"] = {"status": f"{len(H)} human queries; at least 10 are needed for Kendall's tau"}

    # ---------------------------------------------------------------- reporting
    def trec_comparison(self):
        rows = [{"system": n, "ndcg10": a, "mrr": b, "recall1000": c, "ndcg1000": d, "queries": "622 (2025 test)",
                 "corpus": "6,407,814 Wikipedia articles"} for n, a, b, c, d in TREC_2025]
        for cond in ("detailed", PRIMARY, "vague+noise"):
            for label in (BASE, "Dense bi-encoder (bge-small, MaxP)", LTR_LABELS["ca"]):
                m = summarize(self.res[cond][label])
                rows.append({"system": f"Desi ToT [{cond}]: {label}", "ndcg10": m["ndcg10"], "mrr": m["rr"],
                             "recall1000": 1.0, "ndcg1000": m["ndcg1000"], "queries": str(len(self.sets[cond])),
                             "corpus": f"{self.eng.N} film articles"})
        self.out["trec"] = rows
        lines = ["| System | Queries | Corpus | nDCG@10 | MRR | nDCG@1000 |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['system']} | {r['queries']} | {r['corpus']} | {r['ndcg10']:.3f} | {r['mrr']:.3f} | {r['ndcg1000']:.3f} |"
                  for r in rows]
        (RESULTS / "trec_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def examples(self, name):
        Q = self.sets[name]
        bm, full = self.res[name][BASE], self.res[name][LTR_LABELS["ca"]]
        gains = sorted(range(len(Q)), key=lambda i: (1 / full[i] - 1 / bm[i]), reverse=True)
        lines = [f"# Qualitative examples ({name} queries)\n",
                 "Largest improvements of the full model over BM25 first, then the largest losses.\n"]
        for i in gains[:8] + gains[-4:]:
            q, t = Q[i], self.eng.films[Q[i]["target"]]
            res = self.eng.search(q["query"], k=5)
            lines.append(f"## {t['title']} ({t['year']}): BM25 rank {bm[i]}, full model rank {full[i]}\n")
            lines.append(f"> {q['query']}\n")
            lines.append("Clauses: " + "; ".join(f"`{c['text']}` → {c['kind']} (w={c['weight']})"
                                                for c in res["query"]["clauses"]) + "\n")
            lines.append("Top 5 now: " + ", ".join(f"{r['title']} ({r['year']})" for r in res["results"]) + "\n")
        (RESULTS / "examples.md").write_text("\n".join(lines), encoding="utf-8")

    def write_per_query(self):
        for name, systems in self.res.items():
            Q = self.sets.get(name, self.human)
            labels = list(systems)
            with open(RESULTS / f"per_query_{name.replace('+', '_')}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["qid", "target", "tags", "query"] + labels)
                for i, q in enumerate(Q):
                    w.writerow([q["qid"], self.eng.films[q["target"]]["title"], "|".join(q["tags"]), q["query"]]
                               + [systems[s][i] for s in labels])
        curves = {}
        for name in ("detailed", PRIMARY, "vague+noise"):
            curves[name] = {}
            for label in (BASE, "TF-IDF cosine (VSM)", "Dense bi-encoder (bge-small, MaxP)", "BM25 + dense (RRF)",
                          FULL_LEX, LTR_LABELS["ca"]):
                r = np.array(self.res[name][label])
                curves[name][label] = [float(np.mean(r <= k)) for k in range(1, 51)]
        self.out["success_curves"] = curves

    def _csv(self, name, rows):
        if not rows:
            return
        with open(RESULTS / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def final_only():
    """Retrain only the web app's model (and the examples) and update results/summary.json in place."""
    x = Experiment()
    for Q in x.sets.values():
        x.prefetch_dense(Q)
    summary_path = RESULTS / "summary.json"
    old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    x.train_final()
    x.res[PRIMARY] = {}
    for label in (BASE, LTR_LABELS["ca"]):                 # per-query ranks for the examples come from the last full run
        col = []
        with open(RESULTS / f"per_query_{PRIMARY}.csv", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                col.append(int(float(row[label])))
        x.res[PRIMARY][label] = col
    x.examples(PRIMARY)
    old["final_model"] = x.out["final_model"]
    summary_path.write_text(json.dumps(json_safe(old), indent=1, default=float), encoding="utf-8")
    print("final model:", x.out["final_model"]["clause_weights"], "trained on", x.out["final_model"]["trained_on"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--final-only" in sys.argv:
        final_only()
    else:
        Experiment().run()
