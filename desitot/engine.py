"""The Desi ToT search engine: query analysis -> features -> ranking -> explanation.

Rankers
  bm25     plain BM25F over every query word (TREC-style lexical baseline)
  tfidf    TF-IDF cosine (vector space model baseline)
  dense    bge-small bi-encoder, MaxP over plot passages
  hybrid   reciprocal rank fusion of bm25 and dense (Cormack et al., 2009)
  lex      clause-weighted BM25F minus the exclusion penalty:
               s(d) = sum_j c_j BM25(clause_j, d) - mu * sum_j c'_j BM25(neg_j, d)
  ltr      a learned model over all features (coordinate ascent / RankNet-linear / LambdaMART)
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import conflicts, order as order_mod, questions
from .config import CORPUS, DEFAULT_WEIGHTS, INDEX_DIR
from .dense import DenseIndex
from .index import InvertedIndex
from .ltr import CoordinateAscent, LambdaMART, LinearModel, PairwiseLogistic
from .query import QueryAnalysis, parse_query
from .text import fold, stem, word_spans
from .versions import LANG_NAMES, VersionIndex, resolve as resolve_versions

FEATURES = ["bm25_plain", "bm25_clause", "bm25_excl", "tfidf", "dense_query", "dense_clause", "rrf",
            "title_match", "entity_match", "entity_excl", "year_pref", "year_excl", "lang_pref", "lang_excl",
            "genre_pref", "genre_excl", "nb_genre", "nb_decade", "pagerank", "hits_auth", "pageviews", "doc_len"]
GROUPS = {
    "lexical": ["bm25_plain", "bm25_clause", "tfidf"],
    "negation": ["bm25_excl", "entity_excl", "year_excl", "lang_excl", "genre_excl"],
    "dense": ["dense_query", "dense_clause", "rrf"],
    "cues": ["title_match", "entity_match", "year_pref", "lang_pref", "genre_pref"],
    "nb": ["nb_genre", "nb_decade"],
    "popularity": ["pagerank", "hits_auth", "pageviews"],
    "prior": ["doc_len"],
}
FEATURE_LABELS = {
    "bm25_plain": "BM25 (all words)", "bm25_clause": "Clause-weighted BM25", "bm25_excl": "Exclusion penalty (BM25)",
    "tfidf": "TF-IDF cosine", "dense_query": "Dense match (whole query)", "dense_clause": "Dense match (per clause)",
    "rrf": "Hybrid rank fusion", "title_match": "Title match", "entity_match": "Cast / character match",
    "entity_excl": "Excluded person", "year_pref": "Year cue", "year_excl": "Excluded decade",
    "lang_pref": "Language cue", "lang_excl": "Excluded language", "genre_pref": "Genre cue",
    "genre_excl": "Excluded genre", "nb_genre": "Naive Bayes genre", "nb_decade": "Naive Bayes decade",
    "pagerank": "PageRank", "hits_auth": "HITS authority", "pageviews": "Page views", "doc_len": "Plot length",
}
RRF_K = 60


@dataclass
class Options:
    clauses: bool = True          # hedge / confidence / context / meta clause weights
    negation: str = "scope"       # none | drop | penalize_all | scope
    hinglish: bool = True         # entity rewriting, phonetic spelling correction, Hinglish lexicon
    dense: bool = True
    nb: bool = True
    cues: bool = True             # explicit title / person / year / language / genre cues
    popularity: bool = True

    def feature_names(self) -> list[str]:
        drop = set()
        if not self.dense:
            drop |= set(GROUPS["dense"])
        if not self.nb:
            drop |= set(GROUPS["nb"])
        if not self.cues:
            drop |= set(GROUPS["cues"])
        if not self.popularity:
            drop |= set(GROUPS["popularity"])
        if self.negation in ("none", "drop"):
            drop |= set(GROUPS["negation"])
        return [f for f in FEATURES if f not in drop]

    def key(self) -> tuple:
        return (self.clauses, self.negation, self.hinglish, self.dense, self.nb, self.cues, self.popularity)


@dataclass
class QueryFeatures:
    qa: QueryAnalysis
    opts: Options
    F: dict
    lex: np.ndarray
    pos: dict
    neg: dict
    qvec: np.ndarray | None = None
    genre_probs: dict = field(default_factory=dict)
    decade_probs: dict = field(default_factory=dict)
    tfidf_plain: np.ndarray | None = None

    def matrix(self, names: list[str], idx: np.ndarray | None = None) -> np.ndarray:
        cols = [self.F[n] if idx is None else self.F[n][idx] for n in names]
        return np.stack(cols, axis=1).astype(np.float64)


def _short(s: str, n: int = 120) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.float64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def _maxnorm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x)) if len(x) else 0.0
    return x / m if m > 1e-12 else np.zeros_like(x)


class Engine:
    def __init__(self, index_dir: Path = INDEX_DIR, corpus: Path = CORPUS, load_dense: bool = True):
        self.index_dir = Path(index_dir)
        self.films = json.loads(Path(corpus).read_text(encoding="utf-8"))
        self.N = len(self.films)
        self.index = InvertedIndex.load(self.index_dir / "bm25f.pkl")
        with open(self.index_dir / "artifacts.pkl", "rb") as fh:
            art = pickle.load(fh)
        self.matcher, self.corrector, self.vocab = art["matcher"], art["corrector"], art["vocab"]
        self.nb_genre, self.nb_decade, self.nb_genres = art["nb_genre"], art["nb_decade"], art["nb_genres"]
        self.static = art["static"]
        self.years = np.array([f["year"] or -100 for f in self.films], dtype=np.float64)
        self.langs = np.array([f["lang"] for f in self.films])
        self.genres = [set(f["genres"]) for f in self.films]
        self.decades = np.array([max(1950, min(2020, int(y) // 10 * 10)) if y > 0 else 0 for y in self.years])
        self.curated = np.array([f["curated"] for f in self.films])
        self.title_to_id = {f["wiki_title"]: f["id"] for f in self.films}
        self.dense = DenseIndex.load(self.index_dir) if load_dense and (self.index_dir / "dense_emb.npy").exists() else None
        self.weights = dict(DEFAULT_WEIGHTS)
        wpath = self.index_dir / "learned_weights.json"
        if wpath.exists():
            self.weights.update(json.loads(wpath.read_text(encoding="utf-8")))
        self.model = self._load_model()
        # genre membership matrix for the NB feature
        self.genre_matrix = np.array([[g in gs for g in self.nb_genre.classes_] for gs in self.genres], dtype=np.float64)
        # C2: versions of the same story (remakes) and dub languages, built from the articles once and cached
        self.versions = VersionIndex.load_or_build(self.films, self.index_dir / "stories.json")
        # films of every credited person (for actor questions and actor clues)
        self.person_docs: dict[str, set] = {}
        for e in self.matcher.entities:
            if e.kind == "person":
                self.person_docs.setdefault(fold(e.name), set()).update(e.docs)
        for f in self.films:
            for s in f["stars"] + f["cast"]:
                self.person_docs.setdefault(fold(s), set()).add(f["id"])

    # ------------------------------------------------------------------ models
    def _load_model(self):
        p = self.index_dir / "ltr_model.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["kind"] == "lambdamart":
            return LambdaMART.load(self.index_dir / "ltr_lambdamart.txt")
        return {"coordinate_ascent": CoordinateAscent, "pairwise_logistic": PairwiseLogistic}.get(
            d["kind"], LinearModel).from_json(d)

    # ------------------------------------------------------------------ analysis
    def analyze(self, query: str) -> QueryAnalysis:
        return parse_query(query, self.matcher, self.corrector, self.vocab)

    def component_scores(self, qa: QueryAnalysis, opts: Options):
        """BM25 vectors per clause signature, so clause weights can be grid-searched cheaply.

        lex(w) = sum_sig coef(sig, w) * P_sig  -  mu * sum_h coef_h(w) * N_h, which is exact because
        BM25 is linear in the query-term weights.
        """
        pos_parts, neg_parts = {}, {}
        unit = {"hedged": 1.0, "confident": 1.0, "context": 1.0, "meta": 1.0, "mu": 1.0}
        for c in qa.clauses:
            single = QueryAnalysis(qa.text, [c])
            p, n = single.term_weights(unit, use_clauses=False, negation=opts.negation, hinglish=opts.hinglish)
            sig = ("meta" if c.meta else "ctx" if c.context else "-",
                   "hedged" if c.hedged else "confident" if c.confident else "plain")
            for s, v in p.items():
                pos_parts.setdefault(sig, {})[s] = pos_parts.setdefault(sig, {}).get(s, 0.0) + v
            hs = "hedged" if c.hedged else "confident" if c.confident else "plain"
            for s, v in n.items():
                neg_parts.setdefault(hs, {})[s] = neg_parts.setdefault(hs, {}).get(s, 0.0) + v
        P = {sig: self.index.bm25(tw) for sig, tw in pos_parts.items()}
        Nn = {h: self.index.bm25(tw) for h, tw in neg_parts.items()}
        return P, Nn

    @staticmethod
    def combine_components(P: dict, Nn: dict, w: dict, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        for (ctx, h), v in P.items():
            if ctx == "meta":
                coef = w.get("meta", 0.0)
            else:
                coef = (w.get("context", 0.5) if ctx == "ctx" else 1.0) * \
                       (w.get("hedged", 0.3) if h == "hedged" else w.get("confident", 1.2) if h == "confident" else 1.0)
            out += coef * v
        for h, v in Nn.items():
            coef = w.get("hedged", 0.3) if h == "hedged" else w.get("confident", 1.2) if h == "confident" else 1.0
            out -= w.get("mu", 0.5) * coef * v
        return out

    # ------------------------------------------------------------------ features
    def compute(self, query, opts: Options | None = None, w: dict | None = None) -> QueryFeatures:
        opts = opts or Options()
        w = dict(w or self.weights)
        qa = query if isinstance(query, QueryAnalysis) else self.analyze(query)
        N = self.N
        F = {}
        pos, neg = qa.term_weights(dict(w, mu=1.0), use_clauses=opts.clauses, negation=opts.negation,
                                   hinglish=opts.hinglish)
        flat, _ = qa.term_weights(w, use_clauses=False, negation="none", hinglish=False)
        bm_flat = self.index.bm25(flat).astype(np.float64)
        bm_pos = self.index.bm25(pos).astype(np.float64)
        bm_neg = self.index.bm25(neg).astype(np.float64) if neg else np.zeros(N)
        F["bm25_plain"] = _maxnorm(bm_flat)
        mx = max(float(bm_pos.max()), 1e-9)
        F["bm25_clause"] = bm_pos / mx
        F["bm25_excl"] = bm_neg / mx
        F["tfidf"] = _maxnorm(self.index.tfidf_cosine(pos).astype(np.float64))
        lex = bm_pos - w.get("mu", 0.5) * bm_neg
        tfidf_plain = self.index.tfidf_cosine(flat).astype(np.float64)

        qvec = None
        if opts.dense and self.dense is not None:
            qvec = self.dense.encode([qa.dense_text or qa.text])[0]
            d_q = self.dense.doc_scores(qvec).astype(np.float64)
            parts = qa.content_text(w, opts.clauses)
            if parts:
                V = self.dense.encode([t for t, _ in parts])
                ws = np.array([max(x, 0.0) for _, x in parts])
                S = np.stack([self.dense.doc_scores(v) for v in V])
                d_c = (ws[:, None] * S).sum(0) / ws.sum() if ws.sum() > 0 else d_q
            else:
                d_c = d_q
            F["dense_query"], F["dense_clause"] = d_q, d_c
            F["rrf"] = (1.0 / (RRF_K + _ranks(lex)) + 1.0 / (RRF_K + _ranks(d_c))) * (RRF_K + 1) / 2
        else:
            for n in GROUPS["dense"]:
                F[n] = np.zeros(N)

        # explicit cues ------------------------------------------------------------------
        for n in GROUPS["cues"] + GROUPS["negation"][1:]:
            F[n] = np.zeros(N)
        yc = qa.year_cues(w, opts.clauses, opts.negation)
        prefs = [(y, wt) for y, wt in yc if y.mode in ("prefer", "upper")]
        if prefs:
            s, tot = np.zeros(N), 0.0
            for y, wt in prefs:
                dist = np.maximum(y.lo - self.years, 0) + np.maximum(self.years - y.hi, 0) if y.mode == "prefer" \
                    else np.maximum(self.years - y.hi, 0)
                s += wt * np.exp(-dist / 3.0)
                tot += wt
            F["year_pref"] = s / tot
        for y, wt in yc:
            if y.mode == "exclude":
                F["year_excl"] = np.maximum(F["year_excl"], wt * ((self.years >= y.lo) & (self.years <= y.hi)))
        lp, le = qa.lang_prefs(w, opts.clauses, opts.negation)
        for lang, wt in lp.items():
            F["lang_pref"] = np.maximum(F["lang_pref"], wt * (self.langs == lang))
        for lang, wt in le.items():
            F["lang_excl"] = np.maximum(F["lang_excl"], wt * (self.langs == lang))
        gp, ge = qa.genre_prefs(w, opts.clauses, opts.negation)
        for i, gs in enumerate(self.genres):
            if gp:
                F["genre_pref"][i] = max((gp.get(g, 0.0) for g in gs), default=0.0)
            if ge:
                F["genre_excl"][i] = max((ge.get(g, 0.0) for g in gs), default=0.0)
        ep, en = qa.entity_prefs(w, opts.clauses, opts.negation, hinglish=opts.hinglish)
        for target, store in ((ep, None), (en, "entity_excl")):
            by_surface: dict[tuple, np.ndarray] = {}
            for (name, kind), (val, docs, how, surface) in target.items():
                fname = store or ("title_match" if kind == "title" else "entity_match")
                arr = by_surface.setdefault((fname, surface), np.zeros(N))
                idx = np.fromiter(docs, dtype=np.int64)
                arr[idx] = np.maximum(arr[idx], val)
            for (fname, _), arr in by_surface.items():
                F[fname] = F[fname] + arr
        for n in ("entity_match", "title_match", "entity_excl"):
            F[n] = np.minimum(F[n], 3.0)

        # Naive Bayes soft filters -------------------------------------------------------------
        genre_probs, decade_probs = {}, {}
        toks = [t for t in pos if not t.isdigit()]
        if opts.nb and toks:
            pg = self.nb_genre.predict_proba([toks])[0]
            pdec = self.nb_decade.predict_proba([toks])[0]
            genre_probs = {g: float(p) for g, p in zip(self.nb_genre.classes_, pg)}
            decade_probs = {int(d): float(p) for d, p in zip(self.nb_decade.classes_, pdec)}
            F["nb_genre"] = self.genre_matrix @ pg
            F["nb_decade"] = np.array([decade_probs.get(int(d), 0.0) for d in self.decades])
        else:
            F["nb_genre"], F["nb_decade"] = np.zeros(N), np.zeros(N)

        for n in GROUPS["popularity"] + GROUPS["prior"]:
            F[n] = self.static[n] if (opts.popularity or n == "doc_len") else np.zeros(N)
        return QueryFeatures(qa, opts, F, lex, pos, neg, qvec, genre_probs, decade_probs, tfidf_plain)

    # ------------------------------------------------------------------ ranking
    def scores(self, qf: QueryFeatures, ranker: str = "ltr", model=None) -> np.ndarray:
        F = qf.F
        tie = 1e-6 * F["bm25_plain"]
        if ranker == "bm25":
            return F["bm25_plain"] + 1e-9 * F["pageviews"]
        if ranker == "tfidf":
            return qf.tfidf_plain + 1e-9 * F["pageviews"]
        if ranker == "dense":
            return F["dense_query"] + tie
        if ranker == "hybrid":
            d = F["dense_query"]
            return 1.0 / (RRF_K + _ranks(F["bm25_plain"])) + 1.0 / (RRF_K + _ranks(d)) + tie
        if ranker == "lex":
            return qf.lex + tie
        model = model or self.model
        if model is None:
            return qf.lex + tie
        return model.score(qf.matrix(model.names)) + tie

    def rank(self, qf: QueryFeatures, ranker: str = "ltr", model=None, curated_only: bool = False) -> np.ndarray:
        s = self.scores(qf, ranker, model)
        if curated_only:
            s = np.where(self.curated, s, -np.inf)
        return np.argsort(-s, kind="stable")

    # ------------------------------------------------------------------ evidence layer (C1-C4)
    @staticmethod
    def confidences(scores: np.ndarray, order: list) -> tuple[np.ndarray, float]:
        """Relative confidence: softmax over the top-k scores, scaled by their spread so that it does not depend on
        the ranker's score scale (BM25 sums vs. learned-model scores).  Returns (probabilities, spread)."""
        top = scores[order]
        spread = float(np.std(top)) if len(top) > 1 else 0.0
        z = (top - top.max()) / spread if spread > 1e-12 else np.zeros_like(top)
        probs = np.exp(z) / np.exp(z).sum() if len(top) else top
        return probs, spread

    def evidence(self, qf: QueryFeatures, base: np.ndarray, opts: Options, ranker: str = "ltr",
                 mask: np.ndarray | None = None, answer: dict | None = None, use: tuple = ("c1", "c2", "c3"),
                 keep_scores: bool = False):
        """Check the ranking against the evidence: C1 withdraws clues borrowed from another film, C3 prefers films
        whose plot has the remembered events in the remembered order, C2 picks (or keeps open) the version of a
        remade story, and an answer to C4's question re-weights every film.  Returns (scores, state)."""
        mask = np.ones(self.N, dtype=bool) if mask is None else mask
        qa = qf.qa
        st = {"base_top": int(np.argmax(base)), "pairs": [], "order": {}, "versions": None, "check": None}
        scores, qa_used = base, qa
        if "c1" in use:
            chk = conflicts.check(self, qa, base, opts, ranker, versions=self.versions, mask=mask,
                                  keep_scores=keep_scores)
            st["check"] = chk
            scores = np.where(mask, chk.scores, -np.inf)
            qa_used = chk.qa_final or qa
        if "c3" in use:
            st["pairs"] = order_mod.event_pairs(qa_used)
            scores, st["order"] = order_mod.evidence(self, qa_used, scores, st["pairs"])
            scores = np.where(mask, scores, -np.inf)
        if "c2" in use:
            top = [int(d) for d in np.argsort(-scores, kind="stable")[:5] if np.isfinite(scores[d])]
            _, spread = self.confidences(scores, top)
            withdrawn = set(st["check"].withdrawn) if st["check"] else set()
            clues = st["check"].clues if st["check"] else conflicts.make_clues(qa_used)
            actor_clues = [{"docs": k.docs, "hedged": k.certainty == "hedged", "name": ", ".join(k.names)}
                           for k in clues if k.kind == "actor" and not set(k.clauses) & withdrawn]
            vinfo = resolve_versions(self.films, self.versions, qa_used, scores, top, actor_clues, spread)
            if vinfo and vinfo["decision"] in ("resolved", "ambiguous"):
                members = [d for d in vinfo["story"] if np.isfinite(scores[d])]
                if members:
                    hi = max(scores[d] for d in members)
                    scores = scores.copy()
                    scores[vinfo["chosen"]] = hi + 1e-6
                    if vinfo["ambiguous"]:
                        second = vinfo["versions"][1]["id"]
                        if np.isfinite(scores[second]):
                            scores[second] = hi                    # keep both versions at the top
            st["versions"] = vinfo
        if answer:
            top = [int(d) for d in np.argsort(-scores, kind="stable")[:5] if np.isfinite(scores[d])]
            _, spread = self.confidences(scores, top)
            scores = np.where(mask, questions.apply_answer(self, scores, answer, spread), -np.inf)
        st["qa_used"] = qa_used
        return scores, st

    def _brief(self, d: int) -> dict:
        f = self.films[d]
        return {"id": int(d), "title": f["title"], "year": f["year"], "lang": f["lang"], "pageid": f["pageid"]}

    def _clue_message(self, c: dict) -> str:
        if not c["flagged"]:
            return ""
        src, win = self.films[c["source"]], self.films[c["winner_without"]]
        what = ", ".join(c["names"]) if c["kind"] == "actor" else f"“{_short(c['text'], 90)}”"
        if c["relation"] == "other_version":
            return (f"{what} fits {src['title']} ({LANG_NAMES.get(src['lang'], src['lang'])}, {src['year']}), "
                    f"another version of the same story.")
        if c["relation"] == "conflict":
            return (f"Parts of your description point to different films: {what} fits {src['title']} "
                    f"({src['year']}), the rest fits {win['title']} ({win['year']}). Both are kept until you say which.")
        if c["kind"] == "actor":
            return (f"{what} is not in the cast of {win['title']}. This memory may come from "
                    f"{src['title']} ({src['year']}).")
        return (f"{what} fits {src['title']} ({src['year']}) much better than {win['title']}: it may be a scene "
                f"from that film.")

    def _versions_message(self, v: dict) -> str:
        name = lambda d: f"{self.films[d]['title']} ({LANG_NAMES.get(self.films[d]['lang'])}, {self.films[d]['year']})"
        heard = ", ".join(LANG_NAMES.get(l, l) for l in v.get("heard", []))
        if v["decision"] == "dub":
            return f"You heard it in {heard}: {name(v['chosen'])} was dubbed into {heard}, so you may have seen the dub."
        if v["decision"] == "language-mismatch":
            return f"You heard it in {heard}, but {name(v['chosen'])} was made in {LANG_NAMES.get(self.films[v['chosen']]['lang'])}."
        if v["ambiguous"]:
            a, b = v["versions"][0]["id"], v["versions"][1]["id"]
            return f"Two versions of this story fit your description: {name(a)} and {name(b)}."
        why = "; ".join(v["versions"][0].get("evidence") or [])
        return f"{name(v['chosen'])} is the version that fits" + (f" ({why})." if why else ".")

    def _evidence_json(self, st: dict, order: list, probs: np.ndarray, answer: dict | None) -> dict:
        chk = st["check"]
        checks = []
        for c in (chk.checks if chk else []):
            item = {k: c[k] for k in ("clue", "clauses", "text", "kind", "certainty", "names", "kappa", "flagged",
                                      "relation", "support_answer", "source_rank_without")}
            item.update(source=self._brief(c["source"]), winner_without=self._brief(c["winner_without"]),
                        message=self._clue_message(c))
            checks.append(item)
        v = st["versions"]
        versions = None
        if v:
            versions = {"decision": v["decision"], "ambiguous": v["ambiguous"], "chosen": self._brief(v["chosen"]),
                        "heard": [LANG_NAMES.get(l, l) for l in v["heard"]], "message": self._versions_message(v),
                        "versions": [{**self._brief(x["id"]), "relation": x["relation"], "post": x["post"],
                                      "evidence": x.get("evidence", []),
                                      "dubbed_into": [LANG_NAMES.get(l, l) for l in x["dubbed_into"]]}
                                     for x in v["versions"]]}
        qa = st["qa_used"]
        pairs = st["pairs"]
        order_json = {"pairs": [[qa.clauses[a].text, qa.clauses[b].text] for a, b in pairs],
                      "reversed_query": order_mod.reversed_query(qa, pairs[0]) if pairs else None}
        question = None
        if answer is None and len(order) >= 2:
            q = questions.choose(self, qa, [int(d) for d in order[:5]], probs[:5], v)
            question = q.to_json() if q else None
        return {"checks": checks, "withdrawn": chk.withdrawn if chk else [], "versions": versions,
                "order": order_json, "question": question, "answer": answer,
                "base_top": self._brief(st["base_top"]),
                "changed": bool(order) and int(order[0]) != st["base_top"]}

    def _result_evidence(self, r: dict, st: dict, top: int) -> None:
        d = r["id"]
        r["warnings"] = []
        chk = st["check"]
        for c in (chk.checks if chk else []):
            if not c["flagged"]:
                continue
            if d == top and c["relation"] in ("borrowed", "conflict"):
                r["warnings"].append(self._clue_message(c))
            elif d == c["source"]:
                what = ", ".join(c["names"]) if c["kind"] == "actor" else f"“{_short(c['text'], 70)}”"
                r["warnings"].append(f"Your clue {what} fits this film.")
        r["versions"] = [{**self._brief(o), "relation": self.versions.relation(d, o)} for o in self.versions.versions(d)]
        r["dubbed_into"] = [LANG_NAMES.get(l, l) for l in self.versions.dubs[d]]
        oe = st["order"].get(d)
        if oe:
            steps = []
            for s in oe["steps"]:
                steps.append({**s, "text_first": _short(self.dense.passages[s["passage_first"]], 260),
                              "text_second": _short(self.dense.passages[s["passage_second"]], 260)})
            r["order"] = {"fits": oe["fits"], "delta": oe["delta"], "steps": steps}

    # ------------------------------------------------------------------ explanation
    def search(self, query: str, k: int = 5, opts: Options | None = None, ranker: str = "ltr",
               curated_only: bool = False, lang: str | None = None, evidence: bool = True,
               answer: dict | None = None) -> dict:
        opts = opts or Options()
        qf = self.compute(query, opts)
        mask = np.ones(self.N, dtype=bool)
        if curated_only:
            mask &= self.curated
        if lang in ("hi", "te", "en"):                         # the search bar's language menu: a hard filter
            mask &= self.langs == lang
        base = np.where(mask, self.scores(qf, ranker), -np.inf)
        scores, st = base, None
        if evidence:
            scores, st = self.evidence(qf, base, opts, ranker, mask, answer)
            if st["check"] is not None and st["check"].qa_final is not None:
                qf = self.compute(st["check"].qa_final, opts)      # explanations follow the clues that were kept
        order = [d for d in np.argsort(-scores, kind="stable")[:k] if np.isfinite(scores[d])]
        probs, _ = self.confidences(scores, order)
        results = [self.explain(qf, int(d), float(scores[d]), float(p)) for d, p in zip(order, probs)]
        out = {"query": qf.qa.to_json(self.weights), "results": results,
               "nb": {"genres": sorted(qf.genre_probs.items(), key=lambda x: -x[1])[:6],
                      "decades": sorted(qf.decade_probs.items())},
               "rewrites": self._rewrites(qf.qa), "ranker": ranker if (ranker != "ltr" or self.model) else "lex"}
        if st is not None:
            out["evidence"] = self._evidence_json(st, order, probs, answer)
            for r in results:
                self._result_evidence(r, st, int(order[0]))
        return out

    def _rewrites(self, qa: QueryAnalysis) -> list[dict]:
        out = []
        for c in qa.clauses:
            for q in c.cues:
                if q.kind in ("entity", "correction", "lexicon"):
                    out.append({"kind": q.kind, "text": q.text, **{k: v for k, v in q.info.items()}})
        return out

    def explain(self, qf: QueryFeatures, doc: int, score: float, prob: float) -> dict:
        f = self.films[doc]
        terms = self.index.doc_terms[doc]
        clause_hits = []
        stem_owner: dict[str, tuple[int, str]] = {}
        for ci, c in enumerate(qf.qa.clauses):
            surface = {}                                   # stem -> the word the user actually typed
            for t in c.tokens:
                if t.term:
                    surface.setdefault(stem(t.tok), t.tok)
            positive = {stem(t.tok) for t in c.tokens if t.term and not (t.in_scope and c.exclusion)}
            positive |= {s for s, _, sc, _ in c.extra if not (sc and c.exclusion)} if qf.opts.hinglish else set()
            negative = {stem(t.tok) for t in c.tokens if t.term and t.in_scope and c.exclusion}
            hit = sorted(s for s in positive if s in terms)
            bad = sorted(s for s in negative if s in terms)
            for s in hit:
                stem_owner.setdefault(s, (ci, c.kind))
            for s in bad:
                stem_owner[s] = (ci, "excluded")
            clause_hits.append({"clause": ci, "kind": c.kind, "matched": [surface.get(s, s) for s in hit],
                                "excluded": [surface.get(s, s) for s in bad], "n_terms": len(positive)})
        # best passage: dense if available, otherwise the plot window with most matched stems
        passage = None
        if qf.qvec is not None and self.dense is not None:
            j, sim = self.dense.best_passage(qf.qvec, doc)
            passage = self.dense.passages[j]
        if passage is None:
            sents = [s for s in (f["plot"] or f["lead"]).split(". ") if s]
            best, best_n = 0, -1
            for i in range(len(sents)):
                window = ". ".join(sents[i:i + 3])
                n = sum(stem(t) in stem_owner for _, _, t in word_spans(window))
                if n > best_n:
                    best, best_n = i, n
            passage = ". ".join(sents[best:best + 3])[:700]
        spans = []
        for a, b, tok in word_spans(passage):
            s = stem(tok)
            if s in stem_owner:
                ci, kind = stem_owner[s]
                spans.append({"start": a, "end": b, "clause": ci, "kind": kind})
        contrib = {}
        model = self.model
        if model is not None and hasattr(model, "contributions"):
            x = qf.matrix(model.names, np.array([doc]))[0]
            contrib = model.contributions(x)
        top_contrib = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:7]
        reasons = self._reasons(qf, doc)
        return {
            "id": doc, "pageid": f["pageid"], "title": f["title"], "year": f["year"], "lang": f["lang"],
            "curated": f["curated"],
            "url": f["url"], "poster": f["poster"], "poster_remote": f["poster_remote"],
            "directors": f["directors"][:2], "stars": (f["stars"] or f["cast"])[:4], "genres": f["genres"][:4],
            "cast_roles": f.get("cast_roles", [])[:4],
            "imdb_id": f.get("imdb_id"),
            "description": f["description"], "score": round(score, 4), "confidence": round(prob, 3),
            "passage": passage, "spans": spans, "clauses": clause_hits, "reasons": reasons,
            "features": {n: round(float(qf.F[n][doc]), 4) for n in FEATURES},
            "contributions": [{"feature": n, "label": FEATURE_LABELS.get(n, n), "value": round(v, 4)}
                              for n, v in top_contrib],
        }

    @staticmethod
    def _with_role(f: dict, name: str, kind: str) -> str:
        """'Amitabh Bachchan (as Jaidev "Jai")' / 'Gabbar Singh (played by Amjad Khan)' from the film's cast list."""
        key = fold(name)
        for r in f.get("cast_roles", []):
            if kind == "person" and fold(r["actor"]) == key and r["character"]:
                return f"{name} (as {r['character']})"
            if kind == "character" and r["character"] and (key in fold(r["character"]) or fold(r["character"]) in key):
                return f"{name} (played by {r['actor']})"
        return name

    def _reasons(self, qf: QueryFeatures, doc: int) -> list[str]:
        F, f, out = qf.F, self.films[doc], []
        if F["title_match"][doc] > 0:
            out.append("A word in your description matches the title")
        if F["entity_match"][doc] > 0:
            hits = [(n, kind) for (n, kind), (v, docs, how, surf) in
                    qf.qa.entity_prefs(self.weights, qf.opts.clauses, qf.opts.negation)[0].items()
                    if doc in docs and kind != "title"]
            if hits:
                out.append("Mentions " + ", ".join(dict.fromkeys(self._with_role(f, n, kind) for n, kind in hits[:3])))
        if F["year_pref"][doc] > 0.6:
            out.append(f"Released in {f['year']}, close to the period you described")
        if F["lang_pref"][doc] > 0:
            out.append({"hi": "Hindi", "te": "Telugu", "en": "English"}[f["lang"]] + " film, as you said")
        if F["genre_pref"][doc] > 0:
            out.append("Genre matches: " + ", ".join(sorted(set(f["genres"]) & set(
                qf.qa.genre_prefs(self.weights, qf.opts.clauses, qf.opts.negation)[0]))))
        if F["nb_genre"][doc] > 0.35:
            out.append("Naive Bayes thinks the story sounds like " +
                       ", ".join(g for g, p in sorted(qf.genre_probs.items(), key=lambda x: -x[1])[:2] if g in f["genres"]
                                 ) or "this genre")
        for n, msg in (("bm25_excl", "Contains words you ruled out"), ("genre_excl", "Its genre is one you ruled out"),
                       ("year_excl", "From a period you ruled out"), ("lang_excl", "In a language you ruled out"),
                       ("entity_excl", "Features someone you ruled out")):
            if F[n][doc] > 0.05:
                out.append("⚠ " + msg)
        if F["pagerank"][doc] > 0.85:
            out.append("Highly central in the Wikipedia link graph (PageRank)")
        return [r for r in out if r and not r.endswith(": ")]
