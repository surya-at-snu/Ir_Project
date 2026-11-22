#!/usr/bin/env python
"""Build every search artifact from data/corpus/films.json into data/index/.

  bm25f.pkl          inverted index (posting lists with BM25F and TF-IDF weights)
  artifacts.pkl      entity matcher, spelling corrector, vocabulary, Naive Bayes models,
                     static per-film features (PageRank, HITS authority, page views, length)
  dense_*.npy/json   passage embeddings for dense retrieval
  graph_stats.json   link-graph summary and the top films by PageRank / HITS (for the report)

Usage:  python scripts/build_index.py [--no-dense]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desitot.config import BM25_K1, CORPUS, FIELD_B, FIELD_WEIGHTS, INDEX_DIR, RAW  # noqa: E402
from desitot.dense import DenseIndex, make_passages  # noqa: E402
from desitot.graph import adjacency, hits, pagerank  # noqa: E402
from desitot.hinglish import EntityMatcher, SpellingCorrector  # noqa: E402
from desitot.index import InvertedIndex  # noqa: E402
from desitot.nb import MultinomialNB  # noqa: E402
from desitot.text import analyze, tokenize  # noqa: E402

NB_MIN_DOCS = 25


def doc_fields(f: dict) -> dict:
    people = f["directors"] + f["composers"] + f["writers"] + f["producers"] + f["stars"]
    title = f["title"] if f["title"] == f["wiki_title"] else f"{f['title']} {f['wiki_title']}"
    return {
        "title": title,
        "lead": f["lead"],
        "plot": f["plot"],
        "cast": f["cast_text"] + "\n" + " ; ".join(f["characters"]),
        "people": " ; ".join(dict.fromkeys(people)),
        "genre": " ".join(f["genres"]) + " ; " + " ; ".join(c for c in f["categories"] if "film" in c.lower()),
        "music": f["music"],
        "other": f["other"],
    }


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def build_graph(films: list[dict]) -> tuple[dict, dict]:
    title_to_id = {f["wiki_title"]: f["id"] for f in films}
    people_index = json.loads((RAW / "people_index.json").read_text(encoding="utf-8")) \
        if (RAW / "people_index.json").exists() else {}
    person_titles = sorted({v["title"] for v in people_index.values() if v})
    nodes = {t: i for i, t in enumerate([f["wiki_title"] for f in films] + person_titles)}
    edges = []
    for f in films:
        src = nodes[f["wiki_title"]]
        for l in f["links"]:
            if l in title_to_id and l != f["wiki_title"]:
                edges.append((src, nodes[l]))                                    # film -> film
        for l in f["people_links"]:
            p = people_index.get(l)
            if p and p["title"] in nodes:
                edges.append((src, nodes[p["title"]]))                           # film -> person (credit)
    link_rx = re.compile(r"\[\[([^\]\|#]+)")
    n_person_edges = 0
    for path in (RAW / "people").glob("*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["title"] not in nodes:
            continue
        src = nodes[rec["title"]]
        for m in set(link_rx.findall(rec["wikitext"])):
            t = m.strip()
            t = t[:1].upper() + t[1:]
            if t in title_to_id:
                edges.append((src, nodes[t]))                                    # person -> film
                n_person_edges += 1
    A = adjacency(len(nodes), edges)
    pr = pagerank(A)
    hub, auth = hits(A)
    n_f = len(films)
    stats = {"nodes": len(nodes), "film_nodes": n_f, "person_nodes": len(person_titles),
             "edges": int(A.nnz), "person_to_film_edges": n_person_edges,
             "top_pagerank": [films[i]["title"] for i in np.argsort(-pr[:n_f])[:15]],
             "top_authority": [films[i]["title"] for i in np.argsort(-auth[:n_f])[:15]],
             "top_hub_people": [person_titles[i - n_f] for i in np.argsort(-hub)[:15] if i >= n_f][:10]}
    return {"pagerank": pr[:n_f], "hits_auth": auth[:n_f], "hits_hub": hub[:n_f]}, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    args = ap.parse_args()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    films = json.loads(CORPUS.read_text(encoding="utf-8"))
    N = len(films)
    t0 = time.time()

    # 1. inverted index -----------------------------------------------------------------
    docs = [doc_fields(f) for f in films]
    index = InvertedIndex(FIELD_WEIGHTS, FIELD_B, BM25_K1).build(docs)
    index.save(INDEX_DIR / "bm25f.pkl")
    print(f"index: {N} docs, {len(index.postings)} terms, "
          f"avg plot length {index.avg_len['plot']:.0f} terms ({time.time() - t0:.0f}s)")

    # 2. vocabulary, spelling corrector and entity matcher --------------------------------
    surface = Counter()
    for d in docs:
        for v in d.values():
            surface.update(tokenize(v))
    vocab = set(surface)
    # proper-noun ratio: how often a word is capitalised when it is NOT the first word of a sentence
    cap, seen = Counter(), Counter()
    for f in films:
        for sent in re.split(r"(?<=[.!?])\s+|\n", f["lead"] + "\n" + f["plot"]):     # running prose only
            words = re.findall(r"[A-Za-z][A-Za-z'-]*", sent)
            for w in words[1:]:
                t = tokenize(w)
                if t:
                    seen[t[0]] += 1
                    cap[t[0]] += w[0].isupper()
    plot_df = Counter()
    for d in docs:
        plot_df.update(set(tokenize(d["plot"] + " " + d["lead"])))
    proper = {t for t, n in seen.items() if n >= 2 and cap[t] / n >= 0.6}
    # ordinary words: mostly lower-case in running text, or so frequent that they cannot identify a film
    # (frequent first names such as "amitabh" or "salman" are covered by the alias table instead)
    common = {t for t, n in seen.items() if n >= 2 and cap[t] / n < 0.4} | {t for t, c in plot_df.items() if c >= 15}
    # whole English words from the sentence encoder's WordPiece vocabulary (an "is this English?" test)
    try:
        from transformers import AutoTokenizer
        from desitot.config import DENSE_MODEL
        english = {w for w in AutoTokenizer.from_pretrained(DENSE_MODEL).get_vocab()
                   if w.isalpha() and w.isascii() and len(w) >= 3}
    except Exception as exc:                               # offline / no model: fall back to no English test
        print("warning: could not load the WordPiece vocabulary:", exc)
        english = set()
    corrector = SpellingCorrector(surface, proper=proper, english=english)
    matcher = EntityMatcher()
    for f in films:
        for name in dict.fromkeys(f["directors"] + f["composers"] + f["writers"] + f["stars"] + f["cast"]):
            matcher.add(name, "person", f["id"])
        for ch in f["characters"]:
            matcher.add(ch, "character", f["id"])
        matcher.add(f["title"], "title", f["id"])
        base = re.split(r"\s*[:–(]\s*", f["title"])[0].strip()
        if base and base != f["title"]:
            matcher.add(base, "title", f["id"])
    matcher.finalize(common, proper=proper, english=english)
    kinds = Counter(e.kind for i, e in enumerate(matcher.entities) if not (i in matcher.generic and e.kind == "character"))
    print(f"entities: {dict(kinds)} ({len(matcher.generic)} generic); vocabulary {len(vocab)} surface forms, "
          f"{len(proper)} proper nouns, {len(common)} ordinary words")

    # 3. Naive Bayes genre / decade models (trained on plot text only, digits removed) -----
    toks = [[t for t in analyze(f["plot"]) if not t.isdigit()] for f in films]
    gcount = Counter(g for f in films for g in f["genres"])
    nb_genres = sorted(g for g, c in gcount.items() if c >= NB_MIN_DOCS)
    gi = [i for i, f in enumerate(films) if set(f["genres"]) & set(nb_genres) and toks[i]]
    nb_genre = MultinomialNB(alpha=0.5).fit([toks[i] for i in gi],
                                            [sorted(set(films[i]["genres"]) & set(nb_genres)) for i in gi])
    decade = lambda y: max(1950, min(2020, (y or 2000) // 10 * 10))
    di = [i for i, f in enumerate(films) if f["year"] and toks[i]]
    nb_decade = MultinomialNB(alpha=0.5).fit([toks[i] for i in di], [[decade(films[i]["year"])] for i in di],
                                             vocab=nb_genre.vocab)
    print(f"naive bayes: genres {nb_genres}; decades {nb_decade.classes_}")

    # 4. link analysis ----------------------------------------------------------------------
    graph, gstats = build_graph(films)
    print(f"graph: {gstats['nodes']} nodes, {gstats['edges']} edges; top PageRank: {gstats['top_pagerank'][:5]}")
    pv = np.array([f["pageviews_60d"] for f in films], dtype=float)
    static = {
        "pagerank": minmax(np.log(graph["pagerank"] + 1e-12)),
        "hits_auth": minmax(np.log(graph["hits_auth"] + 1e-9)),
        "pageviews": minmax(np.log1p(pv)),
        "doc_len": minmax(np.log1p(np.array([f["plot_words"] for f in films], dtype=float))),
    }
    gstats["raw"] = {f["title"]: {"pagerank": float(graph["pagerank"][i]), "authority": float(graph["hits_auth"][i]),
                                  "pageviews_60d": int(pv[i])} for i, f in enumerate(films) if f["curated"]}
    (INDEX_DIR / "graph_stats.json").write_text(json.dumps(gstats, ensure_ascii=False, indent=1), encoding="utf-8")

    with open(INDEX_DIR / "artifacts.pkl", "wb") as fh:
        pickle.dump({"matcher": matcher, "corrector": corrector, "vocab": vocab, "nb_genre": nb_genre,
                     "nb_decade": nb_decade, "nb_genres": nb_genres, "static": static}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # 5. dense passages ---------------------------------------------------------------------
    if not args.no_dense:
        t1 = time.time()
        passages = [make_passages(f["lead"], f["plot"]) for f in films]
        dense = DenseIndex().build(passages, cache=DenseIndex.load_cache(INDEX_DIR))
        dense.save(INDEX_DIR)
        print(f"dense: {len(dense.passages)} passages ({dense.reused} reused from the previous build) "
              f"in {time.time() - t1:.0f}s")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
