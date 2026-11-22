#!/usr/bin/env python
"""Wikipedia crawler for the Desi Tip-of-the-Tongue film corpus.

Wikimedia's 2026 API rate limits allow 10 requests/minute for clients whose User-Agent carries
no contact details and 200/minute for a compliant User-Agent (one with an e-mail or URL; see
https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits).  This crawler is designed for the
strict limit: everything is fetched in batches of 50 pages per request (article wikitext, lead
image, page props and 60-day page views in a single call) and parsed locally, so the whole
corpus needs only ~150 API requests.  Set the environment variable DESITOT_CONTACT to an e-mail
or URL of your own to identify the crawler and get the higher limit.

Every stage caches its output under data/raw/ and is safe to re-run.

Stages
  resolve      data/seed_films.csv -> raw/resolved.json     seed title -> Wikipedia article
  distractors  year categories     -> raw/distractors.json  extra films that make ranking harder
  fetch        wikitext + lead image + page views, 50 pages per request -> raw/pages/<pageid>.json
  people       wikitext of cast/crew articles (for the PageRank/HITS link graph) -> raw/people/*.json
  posters      poster thumbnails (upload.wikimedia.org)     -> data/posters/<pageid>.<ext>
  all          every stage above, in order

Usage:  python scripts/crawl.py all          (or name several stages: crawl.py fetch posters)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
RAW = DATA / "raw"
PAGES = RAW / "pages"
PEOPLE = RAW / "people"
POSTERS = DATA / "posters"
for _d in (RAW, PAGES, PEOPLE, POSTERS):
    _d.mkdir(parents=True, exist_ok=True)

WIKI_API = "https://en.wikipedia.org/w/api.php"
CONTACT = os.environ.get("DESITOT_CONTACT", "").strip()
UA = ("DesiToT-IR-CourseProject/1.1 (student information-retrieval coursework"
      + (f"; {CONTACT}" if CONTACT else "") + ") python-requests")
# seconds between requests per host: unidentified clients get 10 req/min on the API
INTERVAL = {"en.wikipedia.org": 1.0 if CONTACT else 6.5, "upload.wikimedia.org": 0.3 if CONTACT else 1.1}

LANG_NAME = {"hi": "Hindi", "te": "Telugu", "en": "English"}
# distractors: three sample years per decade, k films per year category
DISTRACTOR_YEARS = [y for y in range(1951, 2026) if y % 10 in (1, 4, 7)]
DISTRACTORS_PER_YEAR = {"hi": 15, "te": 6, "en": 6}

_local = threading.local()
_gates: dict[str, list] = {}
_gates_lock = threading.Lock()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update({"User-Agent": UA})
    return _local.s


def _gate(host: str) -> list:
    with _gates_lock:
        return _gates.setdefault(host, [threading.Lock(), 0.0])


def _wait(host: str) -> None:
    lock_next = _gate(host)
    with lock_next[0]:
        start = max(time.monotonic(), lock_next[1])
        lock_next[1] = start + INTERVAL.get(host, 1.0)
    time.sleep(max(0.0, start - time.monotonic()))


def _back_off(host: str, seconds: float) -> None:
    lock_next = _gate(host)
    with lock_next[0]:
        lock_next[1] = max(lock_next[1], time.monotonic() + seconds)


def http_get(url: str, params: dict | None = None, tries: int = 8, as_json: bool = True):
    host = urlparse(url).netloc
    for attempt in range(tries):
        _wait(host)
        try:
            r = session().get(url, params=params, timeout=90)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                retry = r.headers.get("Retry-After", "")
                wait = float(retry) if retry.isdigit() else min(120, 5 * 2 ** attempt)
                print(f"  HTTP {r.status_code} from {host}; waiting {wait:.0f}s", flush=True)
                _back_off(host, wait + random.random())
                continue
            r.raise_for_status()
            return r.json() if as_json else r.content
        except (requests.RequestException, ValueError) as e:
            print(f"  {type(e).__name__} on {host}; retrying", flush=True)
            _back_off(host, min(60, 2 ** attempt))
    raise RuntimeError(f"giving up on {url} {params}")


def wiki(params: dict) -> dict:
    data = http_get(WIKI_API, {"format": "json", "formatversion": 2, **params})
    if data and "error" in data:
        raise RuntimeError(data["error"])
    return data


def wiki_all(params: dict):
    cont: dict = {}
    while True:
        data = wiki({**params, **cont})
        yield data
        if "continue" not in data:
            break
        cont = data["continue"]


def chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


# --------------------------------------------------------------------------- stage: resolve
YEAR_RE = re.compile(r"\b(?:18|19|20)\d\d\b")


def lookup_titles(titles) -> dict:
    """{requested title: page or None}, following normalisation and redirects."""
    out = {}
    for batch in chunks(dict.fromkeys(titles), 50):
        data = wiki({"action": "query", "titles": "|".join(batch), "redirects": 1,
                     "prop": "pageprops|description", "ppprop": "disambiguation|wikibase_item"})
        q = data.get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for t in batch:
            cur = norm.get(t, t)
            for _ in range(3):
                cur = redir.get(cur, cur)
            p = pages.get(cur)
            out[t] = p if p and not p.get("missing") and not p.get("invalid") else None
    return out


def film_check(page, year: int, hinted: bool = False) -> tuple[bool, str]:
    """Accept a page if it is a film released within a year of the seed's year (two for hints)."""
    if not page:
        return False, "missing"
    if "disambiguation" in page.get("pageprops", {}):
        return False, "disambiguation"
    if re.match(r"(?i)(list|index|outline) of\b", page.get("title", "")):
        return False, "list page"                     # "List of films about pianists" has no description at all
    desc = (page.get("description") or "").lower()
    film_words = r"\bfilm\b|\bmovie\b" + (r"|drama|comedy|thriller|romance|musical|biopic|biographical|animated|"
                                          r"\bepic\b|western|horror|action|adventure|fantasy|\bwar\b|crime|"
                                          r"mystery|superhero|science fiction" if hinted else "")
    if desc and not re.search(film_words, desc):
        return False, f"not a film: {desc}"
    years = [int(y) for y in YEAR_RE.findall(desc)]
    if years and not any(abs(y - year) <= (2 if hinted else 1) for y in years):
        return False, f"wrong year: {desc}"
    return True, desc


def candidates(seed: dict) -> list[str]:
    t, y, lang = seed["title"], seed["year"], seed["lang"]
    name = LANG_NAME[lang]
    c = [seed["wiki_hint"]] if seed.get("wiki_hint") else []
    c.append(f"{t} ({y} film)")
    if lang != "en":
        c += [f"{t} ({y} {name} film)", f"{t} ({name} film)"]
    c += [f"{t} (film)", t]
    return list(dict.fromkeys(c))


def search_fallback(seed: dict):
    data = wiki({"action": "query", "list": "search", "srsearch": f'{seed["title"]} {seed["year"]} film',
                 "srlimit": 10, "srnamespace": 0, "srprop": ""})
    titles = [h["title"] for h in data["query"]["search"]]
    pages = lookup_titles(titles)
    for t in titles:
        ok, _ = film_check(pages.get(t), int(seed["year"]))
        if ok and fuzz.partial_ratio(fold(seed["title"]), fold(t)) >= 75:
            return pages[t]
    return None


def stage_resolve() -> None:
    """Resolve seeds to Wikipedia articles.  Incremental: seeds resolved before are kept (their tags and
    priority are refreshed from the CSV) and only new seeds are looked up."""
    seeds = list(csv.DictReader((DATA / "seed_films.csv").open(encoding="utf-8")))
    key = lambda s: (s["lang"], s["title"], int(s["year"]))
    previous = {key(r): r for r in load(RAW / "resolved.json", [])}
    resolved, todo = [], []
    for s in seeds:
        old = previous.get(key(s))
        if old:
            resolved.append({**old, "priority": int(s["priority"]), "tags": s["tags"].split("|")})
        else:
            todo.append(s)
    print(f"resolve: {len(resolved)} seeds already resolved, {len(todo)} new")
    seeds = todo
    cand = {i: candidates(s) for i, s in enumerate(seeds)}
    pages = lookup_titles(t for cs in cand.values() for t in cs)
    failed = []
    for i, s in enumerate(seeds):
        chosen, why = None, []
        for t in cand[i]:
            ok, msg = film_check(pages.get(t), int(s["year"]), hinted=(t == s.get("wiki_hint")))
            if ok:
                chosen = pages[t]
                break
            why.append(f"{t!r}: {msg}")
        if chosen is None:
            chosen = search_fallback(s)
        if chosen is None:
            failed.append((s["title"], s["year"], why))
            continue
        resolved.append({**s, "year": int(s["year"]), "priority": int(s["priority"]),
                         "tags": s["tags"].split("|"), "wiki_title": chosen["title"],
                         "pageid": chosen["pageid"], "description": chosen.get("description", ""),
                         "qid": chosen.get("pageprops", {}).get("wikibase_item")})
    dup = [k for k, v in Counter(r["pageid"] for r in resolved).items() if v > 1]
    save(RAW / "resolved.json", resolved)
    print(f"resolved {len(seeds) - len(failed)}/{len(seeds)} new seeds ({len(resolved)} in total); duplicates: {dup}")
    for f in failed:
        print("  FAILED:", f[0], f[1], "|", "; ".join(f[2])[:300])


# --------------------------------------------------------------------------- stage: distractors
def stage_distractors() -> None:
    out_path = RAW / "distractors.json"
    if out_path.exists():
        print("distractors already sampled:", len(load(out_path, [])))
        return
    curated = {r["pageid"] for r in load(RAW / "resolved.json", [])}
    picked = []
    jobs = [(lang, year) for lang in DISTRACTORS_PER_YEAR for year in DISTRACTOR_YEARS]
    for n, (lang, year) in enumerate(jobs, 1):
        cat = f"Category:{year} {LANG_NAME[lang]}-language films"
        members = []
        for data in wiki_all({"action": "query", "list": "categorymembers", "cmtitle": cat,
                              "cmtype": "page", "cmnamespace": 0, "cmlimit": 500}):
            members += data.get("query", {}).get("categorymembers", [])
        members = sorted((m for m in members if m["pageid"] not in curated and not re.search(
            r"\((franchise|soundtrack|film series|TV series|miniseries)\)|^List of", m["title"])),
            key=lambda m: m["pageid"])
        rng = random.Random(f"{lang}-{year}")             # reproducible sample per category
        k = DISTRACTORS_PER_YEAR[lang]
        picked += [{"pageid": m["pageid"], "wiki_title": m["title"], "lang": lang, "year": year}
                   for m in rng.sample(members, min(k, len(members)))]
        if n % 10 == 0:
            print(f"  categories {n}/{len(jobs)}", flush=True)
    picked = list({d["pageid"]: d for d in picked}.values())
    for lang in DISTRACTORS_PER_YEAR:
        print(f"  {lang}: {sum(p['lang'] == lang for p in picked)} sampled")
    save(out_path, picked)


# --------------------------------------------------------------------------- stage: fetch
def all_pageids() -> list[int]:
    ids = [r["pageid"] for r in load(RAW / "resolved.json", [])]
    ids += [d["pageid"] for d in load(RAW / "distractors.json", [])]
    return list(dict.fromkeys(ids))


def _fetch_batch(pageids: list[int], folder: Path, with_meta: bool) -> None:
    """One logical request for up to 50 pages (the API may split it with continuation)."""
    params = {"action": "query", "pageids": "|".join(map(str, pageids)),
              "prop": "revisions|info|description" + ("|pageimages|pageprops|pageviews" if with_meta else ""),
              "rvprop": "content", "rvslots": "main", "inprop": "url"}
    if with_meta:
        params.update({"piprop": "thumbnail|name", "pithumbsize": 400, "pilicense": "any", "pilimit": 50,
                       "ppprop": "wikibase_item", "pvipdays": 60})
    merged: dict[int, dict] = {}
    for data in wiki_all(params):
        for p in data.get("query", {}).get("pages", []):
            cur = merged.setdefault(p["pageid"], {})
            for k, v in p.items():
                if k == "pageviews" and isinstance(v, dict):
                    cur.setdefault("pageviews", {}).update(v)
                elif v not in (None, [], {}) or k not in cur:
                    cur[k] = v
    for pid, page in merged.items():
        rev = (page.get("revisions") or [{}])[0]
        page["wikitext"] = rev.get("slots", {}).get("main", {}).get("content", "")
        page.pop("revisions", None)
        if page["wikitext"]:
            save(folder / f"{pid}.json", page)


def stage_fetch() -> None:
    ids = all_pageids()
    todo = [i for i in ids if not (PAGES / f"{i}.json").exists()]
    print(f"fetch: {len(ids)} pages, {len(todo)} to download in {len(todo) // 25 + 1} batches")
    for n, batch in enumerate(chunks(todo, 25), 1):     # 25 keeps big articles under the 8 MB response cap
        _fetch_batch(batch, PAGES, with_meta=True)
        print(f"  batch {n}: {sum((PAGES / f'{i}.json').exists() for i in ids)}/{len(ids)} pages on disk", flush=True)


# --------------------------------------------------------------------------- stage: people
def _infobox_people(wikitext: str) -> set[str]:
    import mwparserfromhell as mw
    code = mw.parse(wikitext)
    out = set()
    for tpl in code.filter_templates(recursive=False):
        if not str(tpl.name).strip().lower().startswith("infobox"):
            continue
        for param in tpl.params:
            if str(param.name).strip().lower() in {"director", "starring", "music", "writer", "screenplay",
                                                   "story", "producer", "lyrics", "cinematography"}:
                for link in param.value.filter_wikilinks():
                    t = str(link.title).strip()
                    if t and not t.lower().startswith(("file:", "image:", "category:")):
                        out.add(t[0].upper() + t[1:])
        break
    return out


def stage_people() -> None:
    credits = Counter()
    for f in PAGES.glob("*.json"):
        credits.update(_infobox_people(load(f, {}).get("wikitext", "")))
    people = sorted(t for t, c in credits.items() if c >= 2)
    done = load(RAW / "people_index.json", {})
    todo = [t for t in people if t not in done]
    print(f"people: {len(people)} credited on >= 2 films, {len(todo)} to fetch")
    for n, batch in enumerate(chunks(todo, 25), 1):
        merged = {}
        for data in wiki_all({"action": "query", "titles": "|".join(batch), "redirects": 1,
                              "prop": "revisions", "rvprop": "content", "rvslots": "main"}):
            q = data.get("query", {})
            redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
            norm = {x["from"]: x["to"] for x in q.get("normalized", [])}
            for p in q.get("pages", []):
                if p.get("missing"):
                    continue
                rev = (p.get("revisions") or [{}])[0]
                txt = rev.get("slots", {}).get("main", {}).get("content")
                if txt:
                    merged[p["title"]] = (p["pageid"], txt)
            for t in batch:
                cur = redir.get(norm.get(t, t), norm.get(t, t))
                if cur in merged:
                    done[t] = {"title": cur, "pageid": merged[cur][0]}
        for title, (pid, txt) in merged.items():
            save(PEOPLE / f"{pid}.json", {"title": title, "pageid": pid, "wikitext": txt})
        for t in batch:
            done.setdefault(t, None)
        save(RAW / "people_index.json", done)
        print(f"  batch {n}: {len(done)}/{len(people)} people", flush=True)


# --------------------------------------------------------------------------- stage: posters
def stage_posters() -> None:
    done = {p.stem for p in POSTERS.iterdir()}
    pages = [load(f, {}) for f in PAGES.glob("*.json")]
    todo = [p for p in pages if str(p.get("pageid")) not in done and (p.get("thumbnail") or {}).get("source")]
    curated = {r["pageid"] for r in load(RAW / "resolved.json", [])}
    todo.sort(key=lambda p: p["pageid"] not in curated)          # curated films first
    print(f"posters: {len(todo)} to download")
    for n, page in enumerate(todo, 1):
        src = page["thumbnail"]["source"]
        ext = Path(src.split("?")[0]).suffix.lower() or ".jpg"
        ext = ".jpg" if ext == ".jpeg" else ext
        try:
            content = http_get(src, as_json=False, tries=5)
        except RuntimeError:
            content = None
        if content:
            (POSTERS / f"{page['pageid']}{ext}").write_bytes(content)
        if n % 100 == 0:
            print(f"  posters {n}/{len(todo)}", flush=True)


WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
INTERVAL["query.wikidata.org"] = 2.0                     # be gentle with the query service


def stage_imdb() -> None:
    """IMDb ids of the seed films and the downloaded distractor films from Wikidata (property P345), for the
    app's 'IMDb' links. One SPARQL request per 250 films; cached in data/raw/imdb_ids.json (None = no IMDb id)."""
    resolved = load(RAW / "resolved.json", [])
    out = load(RAW / "imdb_ids.json", {})
    qids = {r["qid"] for r in resolved if r.get("qid")}
    for f in PAGES.glob("*.json"):
        qids.add((load(f, {}).get("pageprops") or {}).get("wikibase_item"))
    todo = sorted(q for q in qids - set(out) if q and re.fullmatch(r"Q\d+", q))
    print(f"imdb: {len(todo)} films to look up on Wikidata", flush=True)
    for batch in chunks(todo, 250):
        query = ("SELECT ?item ?imdb WHERE { VALUES ?item { " + " ".join(f"wd:{q}" for q in batch)
                 + " } OPTIONAL { ?item wdt:P345 ?imdb } }")
        data = http_get(WIKIDATA_SPARQL, {"query": query, "format": "json"}) or {}
        found = {}
        for b in data.get("results", {}).get("bindings", []):
            imdb = b.get("imdb", {}).get("value", "")
            if re.fullmatch(r"tt\d+", imdb):
                found.setdefault(b["item"]["value"].rsplit("/", 1)[-1], imdb)
        for q in batch:
            out[q] = found.get(q)
        save(RAW / "imdb_ids.json", out)
    print(f"imdb: {sum(1 for v in out.values() if v)} of {len(out)} films have an IMDb id", flush=True)


STAGES = {"resolve": stage_resolve, "distractors": stage_distractors, "fetch": stage_fetch,
          "people": stage_people, "posters": stage_posters, "imdb": stage_imdb}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    args = ap.parse_args()
    names = list(STAGES) if "all" in args.stages else args.stages
    print("User-Agent:", UA)
    for name in names:
        t0 = time.time()
        print(f"== {name}", flush=True)
        STAGES[name]()
        print(f"== {name} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
