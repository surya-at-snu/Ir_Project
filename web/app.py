#!/usr/bin/env python
"""Desi ToT web application.

    python web/app.py            then open http://127.0.0.1:5000

Endpoints
  GET  /                      single-page app (web/static)
  POST /api/search            {q, k, ranker, curated_only, lang, options:{clauses, negation, hinglish, dense, nb, cues, popularity},
                               evidence (C1-C4 checks, default true), answer (the viewer's answer to the C4 question)}
  GET  /api/titles?q=         title autocomplete (for labelling the right film)
  POST /api/label             {query, id, author}  -> appends to data/queries/human_queries.jsonl
  GET  /api/films             curated films for the Browse tab and the home-page shelves
  GET  /api/film/<id>         full record of one film (incl. the cast list: actor, character, role)
  GET  /api/featured          home page: "Can you name it?" slides (vague test descriptions) + example queries
  GET  /api/stats             corpus statistics + evaluation summary (results/summary.json)
  GET  /posters/<pageid>      local poster, or a redirect to the Wikipedia thumbnail
  GET  /figures/<name>        evaluation figures (results/figures)
"""
from __future__ import annotations

import json
import random
import re
import sys
import threading
import time
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desitot.config import POSTERS, QUERIES, RESULTS  # noqa: E402
from desitot.engine import FEATURE_LABELS, Engine, Options  # noqa: E402
from desitot.metrics import json_safe  # noqa: E402
from desitot.queries import load_queries  # noqa: E402
from desitot.text import fold  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
app.json.ensure_ascii = False
app.json.sort_keys = False          # keep systems in the order the evaluation reports them

print("Loading the search engine ...", flush=True)
t0 = time.time()
ENGINE = Engine()
if ENGINE.dense is not None:
    try:
        ENGINE.dense.encode(["warm up the sentence encoder"])      # loads the model once at start-up
    except Exception as exc:                                        # offline and model not cached yet
        print(f"WARNING: could not load the sentence encoder ({exc}); dense retrieval is switched off.", flush=True)
        ENGINE.dense = None
print(f"Engine ready: {ENGINE.N} films, model={'yes' if ENGINE.model else 'no (run scripts/evaluate.py)'} "
      f"({time.time() - t0:.0f}s)", flush=True)
LOCK = threading.Lock()
POSTER_FILES = {}


def poster_index() -> dict:
    if not POSTER_FILES and POSTERS.exists():
        for p in POSTERS.iterdir():
            POSTER_FILES[p.stem] = p.name
    return POSTER_FILES


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.post("/api/search")
def api_search():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("q") or "").strip()
    if not q:
        return jsonify({"error": "empty query"}), 400
    o = body.get("options") or {}
    opts = Options(clauses=bool(o.get("clauses", True)),
                   negation=o.get("negation", "scope") if o.get("negation", "scope") in ("none", "drop", "penalize_all", "scope") else "scope",
                   hinglish=bool(o.get("hinglish", True)), dense=bool(o.get("dense", True)), nb=bool(o.get("nb", True)),
                   cues=bool(o.get("cues", True)), popularity=bool(o.get("popularity", True)))
    ranker = body.get("ranker", "ltr")
    if ranker not in ("ltr", "lex", "bm25", "tfidf", "dense", "hybrid"):
        ranker = "ltr"
    k = max(1, min(int(body.get("k", 5)), 20))
    lang = body.get("lang") if body.get("lang") in ("hi", "te", "en") else None
    answer = clean_answer(body.get("answer"))
    t = time.time()
    with LOCK:                                   # the engine caches embeddings; keep requests serial
        res = ENGINE.search(q, k=k, opts=opts, ranker=ranker, curated_only=bool(body.get("curated_only")), lang=lang,
                            evidence=bool(body.get("evidence", True)), answer=answer)
    res["took_ms"] = round((time.time() - t) * 1000)
    res["feature_labels"] = FEATURE_LABELS
    for r in res["results"]:
        r["poster_url"] = f"/posters/{r['pageid']}"
    return jsonify(res)


def clean_answer(a) -> dict | None:
    """An answer to the one-question step (C4), as the page sends it back; anything malformed is ignored."""
    if not isinstance(a, dict) or a.get("kind") not in ("decade", "language", "actor", "version", "next_event"):
        return None
    ids = lambda xs: [int(x) for x in (xs or []) if isinstance(x, int) and 0 <= x < ENGINE.N][:20]
    value = a.get("value")
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    offered = [o for o in (a.get("offered") or []) if isinstance(o, (str, int)) and not isinstance(o, bool)][:20]
    return {"kind": a["kind"], "value": value if isinstance(value, int) else str(value)[:200],
            "films": ids(a.get("films")), "candidates": ids(a.get("candidates")), "offered": offered,
            "label": str(a.get("label") or value)[:200]}


@app.get("/api/titles")
def api_titles():
    q = fold(request.args.get("q", "")).strip()
    if len(q) < 2:
        return jsonify([])
    hits = []
    for f in ENGINE.films:
        t = fold(f["title"])
        if q in t:
            hits.append((not t.startswith(q), not f["curated"], f["title"], f))
    hits.sort(key=lambda x: x[:3])
    return jsonify([{"id": f["id"], "pageid": f["pageid"], "title": f["title"], "year": f["year"], "lang": f["lang"]}
                    for *_, f in hits[:12]])


@app.post("/api/label")
def api_label():
    body = request.get_json(force=True, silent=True) or {}
    query, fid = (body.get("query") or "").strip(), body.get("id")
    if not query or not isinstance(fid, int) or not (0 <= fid < ENGINE.N):
        return jsonify({"error": "need a query and a film id"}), 400
    f = ENGINE.films[fid]
    author = re.sub(r"[^\w .-]", "", str(body.get("author") or "anonymous"))[:40]
    rec = {"qid": f"hum-{int(time.time() * 1000)}", "pageid": f["pageid"], "film": f["title"], "year": f["year"],
           "author": author, "tags": [], "query": query}
    QUERIES.mkdir(parents=True, exist_ok=True)
    with LOCK, open(QUERIES / "human_queries.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = sum(1 for line in (QUERIES / "human_queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    return jsonify({"ok": True, "saved": rec, "total_human_queries": n})


@app.get("/api/films")
def api_films():
    keep = ("id", "pageid", "title", "year", "lang", "genres", "directors", "stars", "tags", "description",
            "pageviews_60d")
    order = {"hi": 0, "te": 1, "en": 2}
    films = sorted((f for f in ENGINE.films if f["curated"]), key=lambda f: (order.get(f["lang"], 3), f["year"] or 0))
    return jsonify([{**{k: f[k] for k in keep}, "cast_roles": f.get("cast_roles", [])[:3],
                     "poster_url": f"/posters/{f['pageid']}"} for f in films])


@app.get("/api/film/<int:fid>")
def api_film(fid: int):
    if not 0 <= fid < ENGINE.N:
        abort(404)
    f = ENGINE.films[fid]
    keep = ("id", "pageid", "title", "wiki_title", "url", "year", "lang", "curated", "tags", "description", "lead",
            "plot", "directors", "stars", "composers", "writers", "producers", "genres", "characters", "pageviews_60d")
    keep += ("imdb_id",) if "imdb_id" in f else ()
    return jsonify({**{k: f[k] for k in keep}, "cast_roles": f.get("cast_roles", []),
                    "poster_url": f"/posters/{f['pageid']}"})


# example descriptions for the home page ("Try these"); the film is what the description is meant to find
EXAMPLES = [
    ("two small time thieves are hired by an old cop with no arms to catch a bandit who terrorizes a village. maybe 80s?", "Sholay"),
    ("it had amitab bachan and dharmendar, a professor pretends to be a driver who only speaks shudh hindi", "Chupke Chupke"),
    ("a telugu movie where a man is killed and comes back as a fly to take revenge, not the hindi dubbed one", "Eega"),
    ("please help!! watched it on doordarshan as a kid. a woman in a big haveli is possessed by a dancer's ghost, mogembo? no, manjulika", "Bhool Bhulaiyaa"),
    ("a man who forgets everything every few minutes tattoos clues on his body. not the hollywood one", "Ghajini"),
    ("sharukh khan as a hockey coach for the women's team, i don't think it was cricket", "Chak De! India"),
    ("a pianist pretends to be blind and witnesses a murder, i think around 2015", "Andhadhun"),
    ("an indian spy goes undercover in the gang wars of karachi's lyari area, end of 2025", "Dhurandhar"),
    ("Old Hindi film, maybe the 90s, the hero pretends to be blind, I think the heroine was a doctor, definitely not a comedy.", None),
]
FEATURED: dict = {}


def film_card(f: dict) -> dict:
    return {"id": f["id"], "pageid": f["pageid"], "title": f["title"], "year": f["year"], "lang": f["lang"],
            "genres": f["genres"][:3], "tags": f["tags"], "poster_url": f"/posters/{f['pageid']}"}


@app.get("/api/featured")
def api_featured():
    if not FEATURED:
        by_title = {}
        for f in ENGINE.films:
            if f["curated"]:
                by_title.setdefault(fold(f["title"]), f)
        FEATURED["examples"] = [{"query": q, "film": film_card(by_title[fold(t)]) if t and fold(t) in by_title else None}
                                for q, t in EXAMPLES]
        vague = [q for q in load_queries(ENGINE.films, kinds=("vague",)) if len(q["query"]) < 230]
        FEATURED["vague"] = [{"query": q["query"], "tags": q["tags"], "film": film_card(ENGINE.films[q["target"]])}
                             for q in vague]
    rnd = random.Random(request.args.get("seed"))
    pool = FEATURED["vague"]
    return jsonify({"slides": rnd.sample(pool, min(7, len(pool))), "examples": FEATURED["examples"]})


@app.get("/api/stats")
def api_stats():
    films = ENGINE.films
    cur = [f for f in films if f["curated"]]
    stats = {"films": len(films), "curated": len(cur),
             "by_lang": {l: sum(f["lang"] == l for f in cur) for l in ("hi", "te", "en")},
             "distractors": len(films) - len(cur), "terms": len(ENGINE.index.postings),
             "passages": len(ENGINE.dense.passages) if ENGINE.dense else 0,
             "weights": ENGINE.weights, "has_model": ENGINE.model is not None,
             "model_weights": dict(zip(ENGINE.model.names, map(float, ENGINE.model.w)))
             if ENGINE.model is not None and hasattr(ENGINE.model, "w") else {}}
    summary = RESULTS / "summary.json"
    if summary.exists():
        stats["evaluation"] = json_safe(json.loads(summary.read_text(encoding="utf-8")))
    novelty = RESULTS / "novelty_summary.json"
    if novelty.exists():
        nv = json.loads(novelty.read_text(encoding="utf-8"))
        stats["novelty"] = json_safe({k: nv.get(k) for k in ("c1", "c2", "c3", "c4", "sets")})
    stats["stories"] = len(ENGINE.versions.members)
    figs = RESULTS / "figures"
    stats["figures"] = sorted(p.name for p in figs.glob("*.png")) if figs.exists() else []
    return jsonify(stats)


@app.get("/posters/<int:pageid>")
def poster(pageid: int):
    name = poster_index().get(str(pageid))
    if name is None and POSTERS.exists():                     # posters may still be downloading
        POSTER_FILES.clear()
        name = poster_index().get(str(pageid))
    if name:
        resp = send_from_directory(POSTERS, name)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    for f in ENGINE.films:
        if f["pageid"] == pageid and f.get("poster_remote"):
            return redirect(f["poster_remote"])
    return send_from_directory(STATIC, "no-poster.svg")


@app.get("/figures/<path:name>")
def figures(name: str):
    return send_from_directory(RESULTS / "figures", name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    print(f"Open http://{a.host}:{a.port}", flush=True)
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
