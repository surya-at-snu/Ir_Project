"""Loading evaluation queries.

Query files live in data/queries/:
  llm_queries_*.jsonl   "detailed" set: written by an LLM (Claude), one per curated film, rich in names and quotes
  llm_vague_*.jsonl     "vague" set: written by an LLM in the style of real tip-of-the-tongue posts
                        (few names, everyday words, hedges, exclusions, false memories)
  human_queries.jsonl   written by people (the web app's "Add to test set" button appends here)

Each line is a JSON object with a `query` and either a `pageid` (Wikipedia page id) or a
`film` + `year` pair naming the target film, plus optional `tags` describing ToT phenomena
(hedge, negation, false_memory, hinglish, decade, name, social, lang).
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import QUERIES
from .text import fold


def _norm(t: str) -> str:
    return " ".join(fold(t).replace("–", " ").replace(":", " ").split())


def load_queries(films: list[dict], kinds=("llm", "human"), folder: Path = QUERIES) -> list[dict]:
    by_pageid = {f["pageid"]: f["id"] for f in films}
    by_title: dict[str, list[dict]] = {}
    for f in films:
        for t in {f["title"], f["wiki_title"]}:
            by_title.setdefault(_norm(t), []).append(f)
    out, missing = [], []
    for kind in kinds:
        pattern = {"llm": "llm_queries_*.jsonl", "vague": "llm_vague_*.jsonl"}.get(kind, "human_queries.jsonl")
        for path in sorted(Path(folder).glob(pattern)):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                q = json.loads(line)
                target = None
                if q.get("pageid") in by_pageid:
                    target = by_pageid[q["pageid"]]
                elif q.get("film"):
                    cands = by_title.get(_norm(q["film"]), [])
                    if cands:
                        target = min(cands, key=lambda f: (not f["curated"], abs((f["year"] or 0) - q.get("year", 0))))["id"]
                if target is None:
                    missing.append(q.get("film") or q.get("pageid"))
                    continue
                out.append({"qid": q.get("qid") or f"{kind}-{path.stem}-{i}", "set": kind, "query": q["query"],
                            "target": target, "tags": q.get("tags", []), "author": q.get("author")})
    if missing:
        print(f"warning: {len(missing)} queries name films that are not in the corpus: {missing[:10]}")
    return out
