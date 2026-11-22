#!/usr/bin/env python
"""Turn the raw Wikipedia wikitext into clean film records (data/corpus/films.json).

For every article this extracts
  lead, plot, cast, music (incl. song titles), other sections   -> text fields for BM25F
  infobox: directors, stars, music, writers, producers, release year, language
  "Actor as Character, role" pairs from the cast section (lists, -> character names + a cast list with
   tables, dual roles; else "Character (Actor)" in the plot)          roles for the UI
  coarse genres from the lead sentence and article categories     -> NB labels / exclusion features
  outgoing article links                                          -> link graph for PageRank/HITS

Curated films: 330 Hindi, 250 Telugu and 250 English films are chosen from the seed list (the original
200/50/50, 50 famous 1990-2026 films added later, 50 more 1980-2026 Hindi films, 140 Telugu + 140 English
1990-2026 films and 50 + 50 + 50 films from 2000-2026), priority-1 seeds first, priority-2 seeds filling gaps.
IMDb ids (Wikidata P345, fetched by `crawl.py imdb`) are attached to the seed films for the app's IMDb links.  Everything else that has a real plot becomes a distractor, so the search space looks more like a
real collection.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import mwparserfromhell as mw
from mwparserfromhell.nodes import (Argument, Comment, ExternalLink, Heading, HTMLEntity, Tag, Template, Text,
                                    Wikilink)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "corpus" / "films.json"
POSTERS = ROOT / "data" / "posters"
# original 200 / 50 / 50, plus 50 famous 1990-2026 films added later (30 Hindi, 10 Telugu, 10 English; tag "added"),
# plus 50 more famous 1980-2026 Hindi films (tags "added" and "hindi50"), plus 140 Telugu and 140 English
# popular, highly rated 1990-2026 films (tags "added" and "te140" / "en140"), plus 50 Hindi + 50 Telugu + 50 English
# 2000-2026 films (tags "added" and "add150")
TARGET = {"hi": 330, "te": 250, "en": 250}

DROP_TAGS = {"ref", "references", "gallery", "timeline", "math", "score", "imagemap", "syntaxhighlight",
             "templatedata", "noinclude", "chem", "graph", "mapframe", "maplink", "indicator", "sup"}
LIST_TEMPLATES = {"plainlist", "plain list", "flatlist", "flat list", "ubl", "unbulleted list", "ublist",
                  "hlist", "bulleted list", "blist", "pagelist", "unbulleted", "cast listing", "columns-list",
                  "col-list", "colist", "div col list", "multicol", "column-list", "collist",
                  # cast sections are often wrapped in these; dropping them lost whole cast lists
                  "castlist", "cast list", "cast-list", "castlisting", "div col", "columns list"}
HEAD_RX = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
# category names that describe a genre ("Indian action thriller films"), not a topic or place
NON_GENRE_CAT = re.compile(r"\b(?:in|about|set|shot|based|scored|directed|produced|featuring|by|of|with|from|for|on|"
                           r"using|adapted|remade|distributed|written|filmed|remakes|controvers\w*|debut|lost)\b", re.I)
LAST_PARAM = {"nowrap", "small", "smaller", "big", "lang", "langx", "transliteration", "transl", "translit",
              "abbr", "nobr", "em", "sic", "script", "noitalic", "nobold", "midsize", "resize", "font",
              "color", "colour", "linktext", "native name", "lang-hi", "lang-te", "lang-ur", "lang-ta",
              "noflag", "tooltip", "title case"}
FIRST_PARAM = {"ill", "interlanguage link", "illm", "ill-wd", "le", "vanchor", "visible anchor", "lit",
               "literal translation", "nihongo", "quote", "blockquote", "cquote", "quote box", "quotation",
               "rquote", "poemquote", "pull quote"}
JOIN_PARAMS = {"sortname", "nts"}

FIELD_RX = [
    ("drop", re.compile(r"references|notes|external links|see also|further reading|bibliography|citations|"
                        r"sources|footnotes|gallery|works cited")),
    ("plot", re.compile(r"plot|synopsis|storyline|\bstory\b|premise|summary|narrative")),
    ("cast", re.compile(r"\bcast\b|characters|starring")),
    ("music", re.compile(r"soundtrack|music|songs|track ?listing|\bscore\b|album")),
]
GENRE_KEYWORDS = [
    ("science fiction", "science fiction"), ("sci-fi", "science fiction"), ("superhero", "superhero"),
    ("coming-of-age", "coming-of-age"), ("coming of age", "coming-of-age"), ("black comedy", "comedy"),
    ("comedy", "comedy"), ("comic", "comedy"), ("satir", "comedy"), ("farce", "comedy"), ("parody", "comedy"),
    ("action", "action"), ("martial arts", "action"), ("masala", "masala"), ("adventure", "adventure"),
    ("swashbuckl", "adventure"), ("animated", "animation"), ("animation", "animation"),
    ("biographical", "biographical"), ("biopic", "biographical"), ("crime", "crime"), ("gangster", "crime"),
    ("heist", "crime"), ("mafia", "crime"), ("noir", "crime"), ("drama", "drama"), ("melodrama", "drama"),
    ("family", "family"), ("fantasy", "fantasy"), ("mythological", "mythological"), ("devotional", "mythological"),
    ("hindu mythology", "mythological"), ("historical", "historical"), ("period drama", "historical"),
    ("period film", "historical"), ("period musical", "historical"), ("period action", "historical"),
    ("period romance", "historical"), ("historical epic", "historical"), ("horror", "horror"), ("supernatural", "horror"), ("musical", "musical"),
    ("dance", "musical"), ("mystery", "mystery"), ("whodunit", "mystery"), ("romance", "romance"),
    ("romantic", "romance"), ("love", "romance"), ("sports", "sports"), ("sport", "sports"), ("spy", "spy"),
    ("espionage", "spy"), ("thriller", "thriller"), ("suspense", "thriller"), ("psychological", "thriller"),
    ("war", "war"), ("western", "western"), ("dacoit", "western"), ("disaster", "action"), ("legal", "drama"),
    ("courtroom", "drama"), ("political", "drama"), ("social", "drama"), ("teen", "coming-of-age"),
]


# --------------------------------------------------------------------------- wikitext rendering
def render(code, out: list[str]) -> None:
    for node in code.nodes:
        if isinstance(node, Text):
            out.append(str(node.value))
        elif isinstance(node, Wikilink):
            title = str(node.title).strip()
            if re.match(r"(?i)(file|image|category|media|wikt|wiktionary|s|q|commons):", title):
                continue
            if node.text is not None:
                render(node.text, out)
            else:
                out.append(title.split("#")[0].split("|")[0])
        elif isinstance(node, Template):
            name = str(node.name).strip().lower().replace("_", " ")
            pos = [p for p in node.params if not p.showkey]
            if name in LIST_TEMPLATES:
                for p in node.params:
                    pname = str(p.name).strip()
                    # {{div col}} keeps its list in content=; its positional params are layout options
                    if pname == "content" or (name != "div col" and (not p.showkey or pname == "1")):
                        out.append("\n")
                        render(p.value, out)
                out.append("\n")
            elif name in LAST_PARAM and pos:
                render(pos[-1].value, out)
            elif name in FIRST_PARAM:
                text = node.get("text").value if node.has("text") else (node.get("quote").value if node.has("quote")
                                                                        else (pos[0].value if pos else None))
                if text is not None:
                    render(text, out)
            elif name in JOIN_PARAMS:
                for p in pos[:2]:
                    render(p.value, out)
                    out.append(" ")
            elif name in ("film date", "start date", "release date", "start date and age"):
                yrs = [str(p.value).strip() for p in pos if re.fullmatch(r"\s*(19|20)\d\d\s*", str(p.value))]
                if yrs:
                    out.append(yrs[0])
            elif name in ("'", "' \""):
                out.append("'")
            elif name in ("-", "en dash", "ndash", "snd", "spaced ndash", "spaced en dash"):
                out.append(" – ")
        elif isinstance(node, Tag):
            tag = str(node.tag).lower()
            if tag in DROP_TAGS:
                continue
            if tag in ("br", "li", "dd", "dt", "tr", "hr"):
                out.append("\n")
            elif tag in ("td", "th"):
                out.append(" ; ")
            if node.contents is not None and not node.self_closing:
                render(node.contents, out)
            if tag in ("p", "div", "table", "tr", "blockquote"):
                out.append("\n")
        elif isinstance(node, HTMLEntity):
            out.append(node.normalize())
        elif isinstance(node, ExternalLink):
            if node.title is not None:
                render(node.title, out)
        elif isinstance(node, Heading):
            out.append("\n")
            render(node.title, out)
            out.append("\n")
        elif isinstance(node, (Comment, Argument)):
            continue


def to_text(code) -> str:
    buf: list[str] = []
    render(code if not isinstance(code, str) else mw.parse(code), buf)
    return tidy("".join(buf))


def tidy(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"\(\s*[;,:]?\s*\)", "", text)
    text = re.sub(r"\(\s*[;,]\s*", "(", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" ([,.;:!?)])", r"\1", text)
    text = re.sub(r"\( ", "(", text)
    return text.strip()


# --------------------------------------------------------------------------- article parsing
def field_for(heads: list[str]) -> str:
    for h in reversed(heads):
        hl = h.lower()
        for name, rx in FIELD_RX:
            if rx.search(hl):
                if name == "plot" and re.search(r"development|writing|background|origin|production", hl):
                    continue
                return name
    return "other" if heads else "lead"


def infobox(code) -> dict:
    for tpl in code.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower()
        if not name.startswith("infobox"):
            continue
        out = {}
        for p in tpl.params:
            key = str(p.name).strip().lower().replace(" ", "_")
            links = []
            for l in p.value.filter_wikilinks():
                t = str(l.title).strip().split("#")[0]
                if t and not re.match(r"(?i)(file|image|category):", t):
                    links.append(t[0].upper() + t[1:])
            text = to_text(p.value)
            items = [x.strip(" *•,;") for x in re.split(r"\n|,(?![^()]*\))|;| and ", text) if x.strip(" *•,;")]
            out[key] = {"raw": str(p.value), "text": text, "items": items, "links": links}
        return out
    return {}


def track_titles(code) -> list[str]:
    songs = []
    for tpl in code.filter_templates(recursive=True):
        name = str(tpl.name).strip().lower()
        if "track listing" in name or "tracklist" in name or name == "track list":
            for p in tpl.params:
                k = str(p.name).strip().lower()
                if re.fullmatch(r"(title|extra|note)\d+", k):
                    t = to_text(p.value)
                    if t:
                        songs.append(t)
    return songs


# --------------------------------------------------------------------------- cast: actor, character, role
ABBREV = {"dr", "mr", "mrs", "ms", "jr", "sr", "st", "lt", "col", "gen", "capt", "prof", "insp", "sgt", "maj", "smt",
          "shri", "sri", "no", "vs", "mt", "ft", "rev", "hon", "adv", "dcp", "acp", "co", "brig", "cdr", "maj.gen"}
NOT_NAME = re.compile(r"\b(?:the|was|were|is|are|has|had|who|which|that|played|plays|appears?|appeared|include[sd]?|"
                      r"including|also|cast|members?|roles?|cameos?|voiced|credited|uncredited|with|from|for|by|"
                      r"and|film|films|actor|actress)\b", re.I)
PAIR_RX = re.compile(r"^(.{2,80}?)\s+(as|–|—|-)\s+(.+)$")
COLON_RX = re.compile(r"^([^:]{2,60}):\s+(.+)$")
# "Shah Rukh Khan in a dual role as" with the roles on the following (sub-bullet) lines
DANGLING_RX = re.compile(r"^(.{2,80}?)\s+(?:as|in (?:a |the )?(?:dual|double|triple) roles?(?: as)?)\s*:?$", re.I)
# "... include Tom Rosqui as Rocco Lampone, ..." inside prose
PROSE_RX = re.compile(r"\b([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){1,3})\s+as\s+((?:the\s+)?[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})")


def first_sentence(text: str) -> str:
    """Text up to the first full stop that ends a sentence ('Dr. Salma Ali' and 'R. N. Shukla' do not)."""
    for m in re.finditer(r"\.\s+", text):
        prev = re.search(r"([\w.]+)$", text[:m.start()])
        w = prev.group(1).lower().strip(".") if prev else ""
        if w in ABBREV or len(w) <= 1:
            continue
        return text[:m.start()]
    return text


def clean_actor(s: str) -> str | None:
    """'Shah Rukh Khan in a dual role' -> 'Shah Rukh Khan'; prose fragments -> None."""
    s = re.sub(r"\s*\([^)]*\)", "", s)                                    # (voice), (uncredited), (credited as X)
    s = re.sub(r"\s+in an? (?:dual|double|triple|special|guest|cameo|extended|brief|short|uncredited|friendly)\b.*$",
               "", s, flags=re.I)                                         # "... in a dual role of twin sisters"
    s = re.sub(r"\s+in (?:his|her|their) (?:debut|first|last|final)\b.*$", "", s, flags=re.I)
    s = s.strip(" ,.:;-–—*•")
    toks = s.split()
    if not 1 <= len(toks) <= 5 or NOT_NAME.search(s) or re.search(r"\d", s) or re.match(r"(?i)(a|an)\s", s):
        return None                                                       # "A Fly as a reincarnation of Nani"
    if sum(t[:1].isupper() for t in toks) < max(1, len(toks) - 1):
        return None
    return s


def cast_pairs(cast_text: str) -> list[tuple[str, str]]:
    """('Amjad Khan', 'Gabbar Singh, a dacoit') pairs from a rendered cast section.  Handles long role
    descriptions, dual roles listed on sub-bullets, ';'-separated lists and 'X as Y' inside prose."""
    pairs, pending, waited, limit = [], None, 0, 0
    for raw in cast_text.split("\n"):
        line = raw.strip(" *•\t")
        if not line:
            continue
        m = DANGLING_RX.match(line)
        if m and (actor := clean_actor(m.group(1))):                        # "Sridevi in a dual role ... as"
            pending, waited = actor, 0
            limit = 3 if re.search(r"(?i)triple", line) else 2 if re.search(r"(?i)dual|double", line) else 3
            continue
        m = PAIR_RX.match(line)
        if pending and waited < limit and (not m or m.group(2) != "as") and len(line.split()) <= 25:
            # a role under the dangling actor: "Ravi Anand – Indu's husband" / "Mark Donald aka Don"
            pairs.append((pending, f"{m.group(1)}, {m.group(3)}" if m else line))
            waited += 1
            continue
        hit = False
        for part in re.split(r";\s+(?=[A-Z])", line):
            part = part.strip(" ;,")
            m = PAIR_RX.match(part)
            actor, role = None, None
            if m:
                actor, role = clean_actor(m.group(1)), m.group(3)
            elif (m := COLON_RX.match(part)):
                actor, role = clean_actor(m.group(1)), m.group(2)
            if actor:
                pairs.append((actor, role.strip()[:400]))
                pending, hit = None, True
                continue
            for pm in PROSE_RX.finditer(part):
                if (actor := clean_actor(pm.group(1))):
                    pairs.append((actor, pm.group(2)))
                    hit = True
        if not hit:
            pending = None
    return list(dict.fromkeys(pairs))


def table_pairs(code) -> list[tuple[str, str]]:
    """Cast tables ('Actor | Role | Notes') -> pairs; the header row says which column is which."""
    cell = lambda n: isinstance(n, Tag) and str(n.tag).lower() in ("td", "th")
    pairs = []
    for tbl in code.ifilter_tags():
        if str(tbl.tag).lower() != "table" or tbl.contents is None:
            continue
        rows, loose = [], []
        for node in tbl.contents.nodes:                       # cells before the first "|-" are not wrapped in <tr>
            if cell(node):
                loose.append(node)
            elif isinstance(node, Tag) and str(node.tag).lower() == "tr" and node.contents is not None:
                if loose:
                    rows.append(loose)
                    loose = []
                rows.append([c for c in node.contents.nodes if cell(c)])
        if loose:
            rows.append(loose)
        cols = None
        for cells in rows:
            texts = [to_text(c.contents).replace("\n", " ").strip(" ,;") if c.contents is not None else "" for c in cells]
            if cols is None:
                texts = [t for t in texts if t != "+"]                       # "|+" caption parsed as a cell
                a = next((i for i, t in enumerate(texts) if re.fullmatch(
                    r"(?i)(actors?|actress(es)?|cast|artists?|names?|played by|performers?)", t)), None)
                r = next((i for i, t in enumerate(texts) if re.fullmatch(
                    r"(?i)(roles?|characters?|parts?)(\s*names?)?|role played|character played", t)), None)
                d = next((i for i, t in enumerate(texts) if re.fullmatch(r"(?i)(notes?|description|summary|details)", t)), None)
                if a is not None and r is not None:
                    cols = (a, r, d)
                continue
            a, r, d = cols
            if len(texts) > max(a, r) and (actor := clean_actor(texts[a])) and texts[r]:
                role = texts[r] + (f", {texts[d]}" if d is not None and d < len(texts) and texts[d] else "")
                pairs.append((actor, role))
    return pairs


def role_names(role: str) -> list[str]:
    """'Jaidev "Jai"' -> ['Jaidev', 'Jai'];  'Thakur Baldev Singh, a retired police officer' -> ['Thakur Baldev Singh'];
    'Mark Donald aka Don' -> ['Mark Donald', 'Don'];  'twins Robert and Albert' -> ['Robert', 'Albert']."""
    main = first_sentence(role)
    main = re.sub(r"\s+(?:a\.k\.a\.?|aka|alias|also known as)\s+", " / ", main, flags=re.I)
    main = re.split(r",|\(|;|:|\s[–—-]\s|\bwho\b|\bwhose\b|\bwhich\b|\ba\b|\ban\b|\bthe\b|\bhis\b|\bher\b|\btheir\b|"
                    r"\bin\b|\bwith\b", main)[0]
    names = re.findall(r"[\"“]([^\"”]{2,30})[\"”]", main)                 # nicknames in double quotes
    main = re.sub(r"\s*[\"“][^\"”]{2,30}[\"”]\s*", " ", main)              # 'Jaidev "Jai"' -> 'Jaidev' + 'Jai'
    for piece in re.split(r"\s+/\s+|/|\s+and\s+|\s+&\s+", main):
        toks = re.sub(r"[\"“”]", "", piece).strip(" .'").split()
        while toks and not toks[0][:1].isupper():                          # "twins Robert" -> "Robert"
            toks.pop(0)
        if 1 <= len(toks) <= 4:
            names.append(" ".join(toks))
    return list(dict.fromkeys(n.strip() for n in names if n.strip() and n.strip().lower() not in {"himself", "herself"}))


def role_display(role: str) -> tuple[str, str]:
    """('Gabbar Singh', 'a dacoit') for showing a cast list: the character and a short description."""
    s = first_sentence(role).strip()
    m = re.match(r"^(.+?)(?:\s*[,(;:]\s*|\s+[–—-]\s+)(.*)$", s)
    name, desc = (m.group(1), m.group(2) or "") if m else (s, "")
    if name.count('"') % 2:                                               # 'Devdhar "Dev' -> 'Devdhar Dev'
        name = name.replace('"', "")
    desc = desc.strip(" .").lstrip("\"”“'–—- ")
    if desc.endswith(")") and desc.count(")") > desc.count("("):         # 'Bharati (Amar's mother)' -> "Amar's mother"
        desc = desc[:-1]
    return name.strip(" .")[:70], (desc[:117] + "…" if len(desc) > 120 else desc)


def plot_pairs(plot: str, known: list[str]) -> list[tuple[str, str]]:
    """'Karthik (Naga Chaitanya) falls for Jessie (Samantha)' -> [('Naga Chaitanya', 'Karthik'), ...] when the
    name in brackets is one of the film's known actors (articles without a cast section often do this)."""
    keys = {re.sub(r"[^a-z]", "", k.lower()): k for k in known if k}
    out = []
    for m in re.finditer(r"\b([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,3})\s+\(([^()]{3,60})\)", plot):
        actor = m.group(2).strip()
        k = re.sub(r"[^a-z]", "", actor.lower())
        hit = keys.get(k) or next((v for kk, v in keys.items() if len(k) > 4 and (k in kk or kk in k)), None)
        toks = m.group(1).split()
        while toks and toks[0].lower() in SENTENCE_STARTERS:                 # "After Karthik (Naga ...)" -> Karthik
            toks.pop(0)
        if hit and toks and not NOT_NAME.search(" ".join(toks)):
            out.append((hit, " ".join(toks)))
    return list(dict.fromkeys(out))


SENTENCE_STARTERS = {"after", "when", "while", "meanwhile", "then", "in", "as", "his", "her", "their", "the", "a", "an",
                     "one", "later", "soon", "there", "once", "now", "at", "on", "with", "but", "and", "so", "since",
                     "before", "during", "upon", "its", "this", "that", "young", "elder", "younger", "however", "however,",
                     "eventually", "finally", "years", "back", "today", "here", "both", "meanwhile,", "unknown", "to"}


def coarse_genres(text: str) -> set[str]:
    t = text.lower()
    return {g for kw, g in GENRE_KEYWORDS if re.search(r"(?<![a-z])" + re.escape(kw), t)}


def display_title(title: str) -> str:
    return re.sub(r"\s*\((?:\d{4} )?(?:[A-Z][a-z]+(?:-language)? )*(?:film|movie)\)$", "", title).strip()


def year_from(*texts) -> int | None:
    for t in texts:
        m = re.search(r"\b(19[2-9]\d|20[0-2]\d)\b", t or "")
        if m:
            return int(m.group(1))
    return None


def split_sections(wikitext: str) -> list[tuple[list[str], str]]:
    """Cut the raw wikitext at heading lines.  Parsing each chunk separately keeps one unclosed
    tag or template from swallowing the rest of the article."""
    out, last, heads = [], 0, []
    for m in HEAD_RX.finditer(wikitext):
        out.append((list(heads), wikitext[last:m.start()]))
        level = len(m.group(1))
        heads = heads[:max(0, level - 2)] + [to_text(m.group(2))]
        last = m.end()
    out.append((heads, wikitext[last:]))
    return out


def parse_article(page: dict) -> dict:
    wikitext = page.get("wikitext", "")
    code = mw.parse(wikitext)
    sections = split_sections(wikitext)
    lead_code = mw.parse(sections[0][1])
    box = infobox(lead_code) or infobox(code)
    categories = [str(l.title).split(":", 1)[1].strip() for l in code.filter_wikilinks()
                  if str(l.title).strip().lower().startswith("category:")]
    links = sorted({(str(l.title).strip().split("#")[0][:1].upper() + str(l.title).strip().split("#")[0][1:])
                    for l in code.filter_wikilinks()
                    if str(l.title).strip() and not re.match(r"(?i)(file|image|category|wikt|s|q|commons|media):",
                                                              str(l.title).strip())})
    buckets: dict[str, list[str]] = {"lead": [], "plot": [], "cast": [], "music": [], "other": []}
    cast_table: list[tuple[str, str]] = []
    for heads, chunk in sections:
        f = field_for(heads)
        if f == "drop":
            continue
        sub = mw.parse(chunk)
        if f == "cast":
            cast_table += table_pairs(sub)
        if f == "lead":
            for tpl in sub.filter_templates(recursive=True):
                if str(tpl.name).strip().lower().startswith("infobox"):
                    try:
                        sub.remove(tpl)
                    except ValueError:
                        pass
        buf: list[str] = []
        render(sub, buf)
        buckets[f].append("".join(buf) + "\n")
    fields = {k: tidy("".join(v)) for k, v in buckets.items()}
    songs = track_titles(code)
    if songs:
        fields["music"] = (fields["music"] + "\nSongs: " + "; ".join(songs)).strip()
    return {"box": box, "categories": categories, "links": links, "cast_table": cast_table, **fields}


def people(box: dict, *keys) -> list[str]:
    out = []
    for k in keys:
        v = box.get(k)
        if not v:
            continue
        names = v["items"] or [v["text"]]
        for n in names:
            n = re.sub(r"\s*\(.*?\)$", "", n)
            n = re.sub(r"[()\[\]\"]", "", n).strip(" .,;:-")
            if (1 <= len(n.split()) <= 6 and n[:1].isupper() and re.search(r"[A-Za-z]{2}", n)
                    and n.lower() not in {"songs", "score", "background score", "lyrics", "music", "original score"}):
                out.append(n)
    return list(dict.fromkeys(n for n in out if not n.lower().startswith(("see ", "uncredited"))))


def main() -> None:
    resolved = json.loads((RAW / "resolved.json").read_text(encoding="utf-8"))
    distractors = json.loads((RAW / "distractors.json").read_text(encoding="utf-8"))
    posters = {p.stem: p.name for p in POSTERS.iterdir()} if POSTERS.exists() else {}
    imdb_path = RAW / "imdb_ids.json"
    imdb = json.loads(imdb_path.read_text(encoding="utf-8")) if imdb_path.exists() else {}
    seeds = {r["pageid"]: r for r in resolved}
    for d in distractors:
        seeds.setdefault(d["pageid"], {**d, "priority": 3, "tags": [], "title": None})

    # choose the curated set: priority 1 first, then priority 2, in seed-file order
    curated = set()
    for lang, k in TARGET.items():
        pool = [r for r in resolved if r["lang"] == lang]
        pool.sort(key=lambda r: r["priority"])          # stable: keeps file order inside a priority
        curated |= {r["pageid"] for r in pool[:k]}

    films, skipped = [], Counter()
    for pid, seed in seeds.items():
        path = RAW / "pages" / f"{pid}.json"
        if not path.exists():
            skipped["not fetched"] += 1
            continue
        page = json.loads(path.read_text(encoding="utf-8"))
        art = parse_article(page)
        box = art["box"]
        lead_first = re.split(r"(?<=[.!?])\s", art["lead"], maxsplit=1)[0]
        is_curated = pid in curated
        plot_words = len(art["plot"].split())
        if not is_curated:
            if plot_words < 60 or not re.search(r"\bfilm\b|\bmovie\b", art["lead"][:400].lower()):
                skipped["distractor without plot"] += 1
                continue
            if re.search(r"documentary|short film|television film|tv film|anthology", lead_first.lower()):
                skipped["documentary/short/tv"] += 1
                continue
        year = year_from(box.get("released", {}).get("raw", ""), page.get("description", ""), lead_first) \
            or seed.get("year")
        if is_curated and seed.get("year") and year and abs(year - seed["year"]) > 2:
            year = seed["year"]
        # primary genres come from the lead sentence ("is a 1975 Indian Hindi-language action-adventure
        # film"); genre-like categories ("Indian heist films") only fill in when the lead names none
        m = re.search(r"\b(?:is|was) (?:an?|the) (.{0,160}?)\bfilm", lead_first)
        genres = coarse_genres(m.group(1) if m else "")
        genre_cats = [c for c in art["categories"] if c.lower().endswith("films") and not NON_GENRE_CAT.search(c)]
        cat_genres = coarse_genres(" ; ".join(genre_cats))
        if not genres:
            genres = cat_genres
        pv = page.get("pageviews") or {}
        title = page["title"]
        stars = people(box, "starring")
        pairs = cast_pairs(art["cast"])
        pairs += [p for p in art["cast_table"] if p not in pairs]
        if len(pairs) < 3:                      # no usable cast section: "Karthik (Naga Chaitanya)" in the plot
            pairs += [p for p in plot_pairs(art["plot"], stars + [a for a, _ in pairs]) if p not in pairs]
        cast = list(dict.fromkeys(stars + [a for a, _ in pairs]))
        chars = list(dict.fromkeys(n for _, r in pairs for n in role_names(r)))
        cast_roles = []
        for a, r in pairs:
            ch, desc = role_display(r)
            cast_roles.append({"actor": a, "character": ch, "role": desc})
        thumb = (page.get("thumbnail") or {}).get("source")
        films.append({
            "pageid": pid, "title": seed.get("title") if is_curated and seed.get("title") else display_title(title),
            "wiki_title": title, "url": page.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "year": year, "lang": seed["lang"], "curated": is_curated, "tags": seed.get("tags", []),
            "description": page.get("description", ""), "lead": art["lead"], "plot": art["plot"],
            "cast_text": art["cast"], "music": art["music"], "other": art["other"][:12000],
            "directors": people(box, "director", "directed_by"), "stars": stars[:12], "cast": cast[:40],
            "characters": chars[:40], "cast_roles": cast_roles[:40], "composers": people(box, "music", "music_by"),
            "writers": people(box, "writer", "screenplay", "story", "written_by"),
            "producers": people(box, "producer", "produced_by"),
            "genres": sorted(genres), "genres_ext": sorted(cat_genres | genres), "categories": art["categories"],
            "people_links": sorted({l for k in ("director", "starring", "music", "writer", "screenplay", "story",
                                                "producer", "lyrics", "cinematography")
                                    for l in box.get(k, {}).get("links", [])}),
            "links": art["links"], "poster": f"posters/{posters[str(pid)]}" if str(pid) in posters else None,
            "poster_remote": thumb, "pageviews_60d": int(sum(v or 0 for v in pv.values())),
            "imdb_id": imdb.get(seed.get("qid") or (page.get("pageprops") or {}).get("wikibase_item") or "") or None,
            "plot_words": plot_words,
        })
    films.sort(key=lambda f: (not f["curated"], f["lang"], f["year"] or 0, f["title"]))
    for i, f in enumerate(films):
        f["id"] = i
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(films, ensure_ascii=False, indent=1), encoding="utf-8")

    cur = [f for f in films if f["curated"]]
    print(f"{len(films)} films written ({len(cur)} curated: "
          + ", ".join(f"{l}={sum(f['lang'] == l for f in cur)}" for l in TARGET)
          + f"; {len(films) - len(cur)} distractors: "
          + ", ".join(f"{l}={sum(f['lang'] == l and not f['curated'] for f in films)}" for l in TARGET) + ")")
    print("skipped:", dict(skipped))
    print("curated without plot:", [f["title"] for f in cur if f["plot_words"] < 50])
    print("curated without poster:", [f["title"] for f in cur if not f["poster"]])
    lang_word = {"hi": "hindi", "te": "telugu", "en": "english"}
    bad_lang = [f["title"] for f in cur if f["lang"] != "en" and lang_word[f["lang"]] not in f["lead"][:600].lower()]
    print("curated whose lead does not mention the expected language:", bad_lang)
    print("genre counts:", Counter(g for f in films for g in f["genres"]).most_common())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
