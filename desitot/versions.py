"""C2 - versions of the same story: remakes and dubbed releases.

A viewer who says "I watched Drishyam in Hindi" may have seen the Hindi remake (Drishyam, 2015) or a Hindi-dubbed
copy of the Telugu film (Drushyam, 2014).  Retrieval that treats every article as an unrelated document cannot
tell "which story" from "which version", so this module

1. builds STORY CLUSTERS from the remake sentences in Wikipedia articles ("It is a remake of the 2016 Hindi film
   Pink", "The film was remade in Hindi as Kabir Singh (2019)"), joining every film whose title is named in such a
   sentence with the article's own film (union-find, so a chain of remakes becomes one story);
2. records the DUB LANGUAGES of every film ("dubbed into Hindi", "a Hindi-dubbed version", "released in Hindi as
   Makkhi");
3. at query time treats the language the viewer heard as evidence for BOTH hypotheses (a version made in that
   language, or a version dubbed into it), weighs actor clues against each version's cast, and either picks the
   version or keeps the ambiguity (both versions shown, and C4 asks a question that separates them):

       score(S) = max_{d in S} s(d)
       P(l | d) = 1 if l = lang(d);  delta if l in dubs(d);  epsilon otherwise
       post(v) ~ softmax(s(v)/T) * P(l | v) * prod_{actor clues a} (1 if a in cast(v) else eta)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .text import fold

LANG_CODES = {"hindi": "hi", "telugu": "te", "english": "en", "tamil": "ta", "kannada": "kn", "malayalam": "ml",
              "bengali": "bn", "marathi": "mr", "punjabi": "pa", "gujarati": "gu", "odia": "or", "bhojpuri": "bh"}
LANG_NAMES = {"hi": "Hindi", "te": "Telugu", "en": "English", "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam",
              "bn": "Bengali", "mr": "Marathi", "pa": "Punjabi", "gu": "Gujarati", "or": "Odia", "bh": "Bhojpuri"}
_L = "|".join(LANG_CODES)
_LLIST = rf"(?:{_L})(?:\s*(?:,|and|&)\s*(?:{_L}))*"

REMAKE_RX = re.compile(r"\bremake of\b|\bremade\b|\bremakes?\b|\bre-made\b|\breboot\b", re.I)
SENT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
DUB_RXS = [
    re.compile(rf"\bdubbed(?:\s+and\s+released)?(?:\s+versions?)?\s+(?:in|into|to)\s+(?:the\s+)?({_LLIST})", re.I),
    re.compile(rf"\b({_LLIST})[\s-]+dubbed\b", re.I),
    re.compile(rf"\bdubbed\s+(?:versions?\s+)?(?:was|were)\s+(?:also\s+)?released\s+in\s+({_LLIST})", re.I),
    re.compile(rf"\bdubbed\b[^.]{{0,60}}?\b(?:in|into)\s+({_LLIST})\b", re.I),
]
# a remake sentence names the other film within a short window after the keyword
WINDOW_TOKENS = 18
YEAR_RX = re.compile(r"(?<![0-9])(19[3-9]\d|20[0-3]\d)(?![0-9])")
DELTA, EPSILON, ETA_SURE, ETA_HEDGED = 0.5, 0.05, 0.2, 0.5
PRIOR_FLATTEN = 3.0
OTHER_VERSION = 0.2           # a version in a language the viewer says belongs to the OTHER version


def _norm(text: str) -> str:
    text = text.replace("&nbsp;", " ").replace(" ", " ")
    return " " + re.sub(r"[^a-z0-9]+", " ", fold(text)).strip() + " "


def _about_this_film(words: list[str], k: int, own: str) -> bool:
    """Is the remake keyword at token k said about the article's own film?  ("It was remade…", "The film is a
    remake…", "A remake of…", "<Title> is a remake…"; not "The first planned collaboration was a remake of…")."""
    before = " " + " ".join(words[:k]) + " "
    return k <= 3 or own in before or any(f" {w} " in before for w in ("it", "film", "movie", "this", "its"))


def _other_remake(words: list[str], k: int) -> bool:
    """ "the 2012 remake of Agneepath", "until the remake of The Lion King": another film's remake, unless the
    sentence says this film *is* that remake ("It is the official remake of …")."""
    prev = words[max(0, k - 3):k]
    return "the" in prev and not any(w in ("is", "was", "be", "being") for w in words[max(0, k - 5):k])


def _sequel_pair(a: str, b: str) -> bool:
    """True when one normalised title is the other plus a sequel number ("drishyam" / "drishyam 2")."""
    a, b = a.strip(), b.strip()
    if a == b:
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    rest = long_[len(short):].strip() if long_.startswith(short + " ") else None
    return bool(rest) and bool(re.fullmatch(r"(?:\d{1,4}|ii|iii|iv|v|part \w+)(?: .*)?", rest))


def dub_languages(text: str) -> list[str]:
    """Language codes a film was dubbed into, from its article text."""
    found = []
    for rx in DUB_RXS:
        for m in rx.finditer(text):
            for name in re.findall(_L, m.group(1).lower()):
                code = LANG_CODES[name]
                if code not in found:
                    found.append(code)
    return found


@dataclass
class Link:
    src: int            # the article that has the sentence
    dst: int            # the film it names
    sentence: str


@dataclass
class VersionIndex:
    story_of: list = field(default_factory=list)       # film id -> story id (its own id when it has no versions)
    members: dict = field(default_factory=dict)        # story id -> sorted film ids (stories with >= 2 films)
    dubs: list = field(default_factory=list)           # film id -> dub language codes
    links: list = field(default_factory=list)          # Link

    # ------------------------------------------------------------------ building
    @classmethod
    def build(cls, films: list[dict]) -> "VersionIndex":
        n = len(films)
        gl = {f["id"]: f["lang"] for f in films}
        titles = []
        for f in films:
            t = _norm(f["title"])
            if len(t.strip()) >= 3:
                titles.append((t, f["id"], int(f["year"] or 0), len(t.split())))
        links = []
        for f in films:
            own = _norm(f["title"])
            text = (f.get("lead") or "") + "\n" + (f.get("other") or "")
            for sent in SENT_RX.split(text):
                if not REMAKE_RX.search(sent):
                    continue
                low = _norm(sent)
                words = low.split()
                # "remake of [up to 8 words] TITLE"  |  "remade / remakes / reboot ... as TITLE"
                of_pos = [i + 1 for i in range(len(words) - 1) if words[i] == "remake" and words[i + 1] == "of"
                          and _about_this_film(words, i, own) and not _other_remake(words, i)]
                as_pos = [i for i, w in enumerate(words) if (w in ("remade", "remakes", "reboot")
                          or (w == "made" and i > 0 and words[i - 1] == "re")) and _about_this_film(words, i, own)]
                if not of_pos and not as_pos:
                    continue
                hits = []                                   # (start token, end token, film id)
                for t, gid, gyear, ntok in titles:
                    if gid == f["id"] or t not in low or _sequel_pair(own, t):
                        continue
                    for m in re.finditer(re.escape(t), low):
                        pos = len(low[:m.start()].split())
                        after_of = any(0 < pos - s <= 8 for s in of_pos)
                        after_as = pos > 0 and words[pos - 1] == "as" and any(0 < pos - s <= 14 for s in as_pos)
                        if not (after_of or after_as):
                            continue
                        if after_as and not after_of and pos >= 3 and words[pos - 3] == "in" \
                                and LANG_CODES.get(words[pos - 2], gl[gid]) != gl[gid]:
                            continue                         # "remade in Bengali as Aashiqui": not our Hindi film
                        tail = words[pos + ntok:pos + ntok + 4]
                        if any(tail[i] == "in" and LANG_CODES.get(tail[i + 1], gl[gid]) != gl[gid]
                               for i in range(len(tail) - 1)):
                            continue                         # "remade as Arjun (2005) in Odia"
                        if any(w in ("sequel", "prequel") for w in words[:min(of_pos + as_pos)]):
                            continue                         # "A sequel titled Drushyam 2, a remake of Drishyam 2"
                        later = [int(y) for y in YEAR_RX.findall(" ".join(words[pos + ntok:pos + ntok + 3]))]
                        if later and not any(abs(y - gyear) <= 1 for y in later):
                            continue                         # "Badri (2001)" is not the Badri of 2000
                        if any(w in ("sequel", "prequel", "sequels") for w in words[max(0, pos - 5):pos]):
                            continue                         # "sequel to the 2015 film Drishyam"
                        near = " ".join(words[max(0, pos - 5):pos + ntok + 3])
                        years = [int(y) for y in YEAR_RX.findall(near)]
                        short = ntok == 1 and len(t.strip()) <= 7
                        if short and not any(abs(y - gyear) <= 1 for y in years):
                            continue      # "Love", "King", "Siva": a one-word title needs its year beside it
                        hits.append((pos, pos + ntok, gid))
                # the longest title wins where matches overlap ("The Godfather" over "Godfather")
                hits.sort(key=lambda h: -(h[1] - h[0]))
                kept = []
                for h in hits:
                    if not any(h[0] < k[1] and k[0] < h[1] for k in kept):
                        kept.append(h)
                for _, _, gid in kept:
                    if not any(l.src == f["id"] and l.dst == gid for l in links):
                        links.append(Link(f["id"], gid, sent.strip()[:400]))
        # union-find over the links
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for l in links:
            a, b = find(l.src), find(l.dst)
            if a != b:
                parent[max(a, b)] = min(a, b)
        story_of = [find(i) for i in range(n)]
        members: dict[int, list[int]] = {}
        for i, s in enumerate(story_of):
            members.setdefault(s, []).append(i)
        members = {s: sorted(m) for s, m in members.items() if len(m) > 1}
        dubs = [dub_languages((f.get("lead") or "") + "\n" + (f.get("other") or "")) for f in films]
        dubs = [[c for c in d if c != films[i]["lang"]] for i, d in enumerate(dubs)]
        return cls(story_of, members, dubs, links)

    def save(self, path: Path, films: list[dict]) -> None:
        name = lambda i: f"{films[i]['title']} ({films[i]['year']}, {films[i]['lang']})"
        data = {"n": len(self.story_of), "story_of": self.story_of, "dubs": self.dubs,
                "stories": [[name(i) for i in m] for m in self.members.values()],
                "links": [{"src": l.src, "dst": l.dst, "src_title": name(l.src), "dst_title": name(l.dst),
                           "sentence": l.sentence} for l in self.links]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VersionIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        story_of = data["story_of"]
        members: dict[int, list[int]] = {}
        for i, s in enumerate(story_of):
            members.setdefault(s, []).append(i)
        members = {s: sorted(m) for s, m in members.items() if len(m) > 1}
        links = [Link(l["src"], l["dst"], l["sentence"]) for l in data["links"]]
        return cls(story_of, members, data["dubs"], links)

    @classmethod
    def load_or_build(cls, films: list[dict], path: Path) -> "VersionIndex":
        path = Path(path)
        if path.exists():
            try:
                vi = cls.load(path)
                if len(vi.story_of) == len(films):
                    return vi
            except (OSError, ValueError, KeyError):
                pass
        vi = cls.build(films)
        try:
            vi.save(path, films)
        except OSError:
            pass
        return vi

    # ------------------------------------------------------------------ queries
    def versions(self, doc: int) -> list[int]:
        """Other films of the same story (empty when the film has no other version in the index)."""
        s = self.story_of[doc]
        return [d for d in self.members.get(s, []) if d != doc]

    def same_story(self, a: int, b: int) -> bool:
        return a != b and self.story_of[a] == self.story_of[b]

    def relation(self, a: int, b: int) -> str:
        """How film b relates to film a: "remake" when an article names the other as its remake or original,
        "same story" when they are only joined through other versions (both remakes of a third film)."""
        for l in self.links:
            if {l.src, l.dst} == {a, b}:
                return "remake"
        return "same story"


# ---------------------------------------------------------------------- query side
OTHER_BEFORE = re.compile(r"(?:remade\s+(?:in|into)|remake\s+of(?:\s+(?:a|an|the))?|copy\s+of(?:\s+(?:a|an|the))?|"
                          r"based\s+on(?:\s+(?:a|an|the))?)\s*$")
OTHER_AFTER = re.compile(r"^\s*(?:original|one\s+it\s+was\s+remade\s+from)(?![a-z])")


def language_mentions(qa) -> tuple[list[str], list[str]]:
    """(heard, other): languages the viewer watched or heard the film in, and languages that name ANOTHER version
    ("remade in Hindi", "a remake of a Telugu hit", "the Hindi original").  Ruled-out languages are neither."""
    heard, other = [], []
    for c in qa.clauses:
        if c.meta:
            continue
        for cue in c.cues:
            if cue.kind != "lang":
                continue
            lang = cue.info.get("lang")
            in_scope = next((sc for l, sc in c.langs if l == lang), False)
            if c.exclusion and in_scope:
                continue
            before = fold(qa.text[max(c.start, cue.start - 40):cue.start])
            after = fold(qa.text[cue.end:cue.end + 40])
            target = other if (OTHER_BEFORE.search(before) or OTHER_AFTER.search(after)) else heard
            if lang not in target:
                target.append(lang)
    return heard, [l for l in other if l not in heard]


def heard_languages(qa) -> list[str]:
    """Languages the viewer says they watched or heard the film in (not ruled out, not another version's)."""
    return language_mentions(qa)[0]


def says_dubbed(qa) -> bool:
    return bool(re.search(r"\bdub(?:bed|bing)?\b", fold(qa.text)))


def resolve(films: list[dict], vindex: VersionIndex, qa, scores: np.ndarray, top: list[int],
            actor_clues: list[dict], temperature: float, margin: float = 2.0) -> dict | None:
    """Decide which version of the top film's story the viewer means.

    actor_clues: [{"docs": set of film ids with that person, "hedged": bool, "name": str}] from the query.
    Returns None when the top film has no other version and no language was heard, otherwise a dict with the
    story's versions, the evidence for each and the decision ("resolved" version or "ambiguous").
    """
    if not top:
        return None
    d1 = top[0]
    vers = [d1] + vindex.versions(d1)
    heard, other = language_mentions(qa)
    if len(vers) == 1:
        # one version in the index: a heard language that is not the film's own may be a dub
        f = films[d1]
        if heard and f["lang"] not in heard:
            dub = [l for l in heard if l in vindex.dubs[d1]]
            return {"story": [d1], "heard": heard, "decision": "dub" if dub else "language-mismatch",
                    "versions": [{"id": d1, "lang": f["lang"], "dubbed_into": vindex.dubs[d1], "post": 1.0,
                                  "relation": "this film"}], "ambiguous": False, "chosen": d1}
        return None
    # versions of one story share the plot, so the ranker's small score differences between them are a weak prior
    T = max(temperature, 1e-6) * PRIOR_FLATTEN
    base = np.array([scores[v] for v in vers], dtype=np.float64)
    prior = np.exp((base - base.max()) / T)
    lang_like, actor_like, why = [], [], []
    for v in vers:
        f = films[v]
        reasons = []
        if heard:
            if f["lang"] in heard:
                p = 1.0
                reasons.append(f"made in {LANG_NAMES.get(f['lang'], f['lang'])}")
            elif any(l in vindex.dubs[v] for l in heard):
                p = DELTA
                reasons.append(f"dubbed into {', '.join(LANG_NAMES.get(l, l) for l in heard if l in vindex.dubs[v])}")
            else:
                p = EPSILON
            lang_like.append(p)
        else:
            lang_like.append(1.0)
        if f["lang"] in other:                       # "it was remade in Hindi": the Hindi film is the other one
            lang_like[-1] *= OTHER_VERSION
        a = 1.0
        for clue in actor_clues:
            if v in clue["docs"]:
                reasons.append(f"{clue['name']} is in its cast")
            else:
                a *= ETA_HEDGED if clue.get("hedged") else ETA_SURE
        actor_like.append(a)
        why.append(reasons)
    post = prior * np.array(lang_like) * np.array(actor_like)
    post = post / post.sum() if post.sum() > 0 else np.full(len(vers), 1 / len(vers))
    order = np.argsort(-post, kind="stable")
    best, second = int(order[0]), int(order[1])
    ambiguous = post[best] < margin * post[second]
    out = {"story": vers, "heard": heard, "says_dubbed": says_dubbed(qa), "ambiguous": bool(ambiguous),
           "chosen": vers[best], "decision": "ambiguous" if ambiguous else "resolved",
           "versions": [{"id": v, "lang": films[v]["lang"], "year": films[v]["year"],
                         "dubbed_into": vindex.dubs[v], "post": round(float(p), 3), "evidence": r,
                         "relation": "this film" if v == d1 else vindex.relation(d1, v)}
                        for v, p, r in sorted(zip(vers, post, why), key=lambda x: -x[1])]}
    return out
