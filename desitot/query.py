"""Query understanding for tip-of-the-tongue requests.

A ToT request mixes several kinds of statements (Arguello et al., CHIIR 2021; TREC ToT 2025):
plain memories, hedged memories ("I think the heroine was a doctor"), confident memories
("I clearly remember the train scene"), exclusions ("definitely not a comedy"), context
("I watched it on Doordarshan as a kid") and social niceties / earlier search attempts
("please help, Google was useless").

`parse_query` splits the request into clauses, tags each clause with these cues, and works out
the *scope* of every negation.  A negation is an **exclusion** only when it is about the film
itself (its genre, decade, language, cast, or "it wasn't ...").  Negations that describe the
story ("he can't see", "nobody believes her") stay positive evidence.  The scoring weights for
each clause type are applied later (`QueryAnalysis.term_weights`) so they can be learned.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .hinglish import HINGLISH_LEXICON, EntityMatcher, EntityMatch, SpellingCorrector
from .text import QUERY_FILLER, STOPWORDS, analyze, stem, tokenize


def fold_same_length(s: str) -> str:
    """Lower-case and strip accents character by character, so offsets still match `s`."""
    out = []
    for ch in s:
        if ch in "’‘`´":
            out.append("'")
            continue
        base = unicodedata.normalize("NFKD", ch)[:1] or ch
        low = base.lower()
        out.append(low[0] if low else ch)
    return "".join(out)


def _phrases(words) -> re.Pattern:
    words = {w.strip() for w in words if w and w.strip()}          # an empty alternative would match everywhere
    parts = sorted({re.escape(w).replace(r"\ ", r"\s+") for w in words}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9%])")


# --------------------------------------------------------------------------- cue lexicons
HEDGE_RX = _phrases("""
i think|i thought|i believe|i guess|i suppose|i feel like|feel like|i reckon|i want to say|maybe|perhaps|probably
possibly|presumably|apparently|supposedly|might|might have|might've|may have|could be|could have|could've
not sure|unsure|not certain|not 100% sure|not entirely sure|not completely sure|not totally sure|not too sure
i'm not sure|im not sure|not sure if|if i remember correctly|if i remember right|if i recall|if i recall correctly
iirc|if memory serves|or something|or so|something like|sort of|kind of|kinda|sorta|vaguely|vague|fuzzy|hazy
i could be wrong|could be wrong|i may be wrong|i might be wrong|don't quote me|dont quote me|i'm guessing
im guessing|likely|seems like|seemed like|i assume|or was it|or maybe|either|my memory|if i'm not wrong
if im not wrong|if i am not wrong|if i'm not mistaken|if im not mistaken|or else|something along those lines
""".replace("\n", "|").split("|"))
HEDGE_EXTRA_RX = re.compile(r"(?<![a-z])(?:around|roughly|approximately|circa|about)\s+(?:the\s+)?(?:early\s+|mid\s+|late\s+)?"
                            r"(?:\d|nineteen|twenty|fifties|sixties|seventies|eighties|nineties)|(?<=[0-9s])[- ]?ish(?![a-z])")
NEG_HEDGE_RX = _phrases("""
i don't think|i dont think|i do not think|don't think|dont think|i don't believe|i dont believe|i doubt|doubt it
probably not|maybe not|possibly not|not sure it was|i don't remember it being|i dont remember it being
pretty sure it wasn't|pretty sure it wasnt|pretty sure it was not
""".replace("\n", "|").split("|"))
CONFIDENT_RX = _phrases("""
definitely|definately|defo|for sure|i'm sure|im sure|i am sure|i'm certain|im certain|i am certain|certainly
clearly remember|i clearly remember|i remember clearly|distinctly|i distinctly remember|100%|without a doubt
absolutely|i know for a fact|for a fact|i'm positive|im positive|i am positive|surely|no doubt|i remember vividly
vividly|one thing i'm sure|one thing im sure|the one thing i remember
""".replace("\n", "|").split("|"))
# "no wait", "oh no" are discourse markers, not negations of a film property.
DISCOURSE_RX = re.compile(r"(?<![a-z])(?:no wait|no no|oh no|or not|whether or not|no matter|no one else but)(?![a-z])")
NEGATION_RX = re.compile(r"(?<![a-z])(?:not|no|never|neither|nor|without|none|nobody|nothing|cannot|wasnt|isnt|"
                         r"didnt|doesnt|dont|cant|couldnt|wont|wouldnt|arent|werent|hasnt|havent|hadnt|aint)"
                         r"(?![a-z])|n't(?![a-z])")
META_RX = re.compile(r"(?<![a-z])(?:please|pls|plz|thanks|thank you|thx|help|anyone|anybody|someone (?:know|knows|help)|"
                     r"(?:been|was|am|i'm|im) (?:looking|searching|trying)|searched|googl\w*|"
                     r"(?:can'?t|cannot|couldn'?t|could not|unable to) (?:find|remember|recall|figure)|"
                     r"driving me (?:crazy|nuts|mad)|tip of my tongue|(?:forgot|forget|forgotten) (?:the )?(?:name|title)|"
                     r"what(?:'s| is| was) (?:the|its|it's) (?:name|title)|name of (?:the|this|that) (?:film|movie)|"
                     r"bugging me|i(?:'ve| have) tried|no luck|which (?:film|movie) (?:is|was) (?:this|it)|any ideas?|"
                     r"identify|i would love to watch it again|want to watch it again|been years|no idea|no clue)(?![a-z])")
CONTEXT_RX = re.compile(r"(?<![a-z])(?:i (?:watched|saw|seen|remember watching)|(?:watched|saw|seen) (?:it|this|that)|"
                        r"on (?:tv|television|doordarshan|dd national|cable|star gold|set max|zee cinema|youtube|netflix|"
                        r"prime|hotstar|a bus|a train)|in (?:the )?(?:theatre|theater|cinema hall|cinema)s?|"
                        r"as a (?:kid|child|teen|teenager|boy|girl)|when i was (?:a kid|young|little|small|in school|\d+)|"
                        r"(?:my|our) (?:mom|mum|mother|dad|father|parents|family|grandma|grandmother|grandpa|grandfather|"
                        r"cousins?|friends?|uncle|aunt)|childhood|years ago|long (?:time )?ago)(?![a-z])")

FILM_SUBJECT_RX = re.compile(r"(?<![a-z])(?:it|this|that|this one|the (?:film|movie|picture|flick|one)|film|movie)\s+"
                             r"(?:\w+\s+){0,2}?(?:was\s+|is\s+|'s\s+|s\s+|had\s+|has\s+|did\s+|does\s+)?"
                             r"(?:definitely\s+|certainly\s+|probably\s+)?(?:not|never|n't|wasnt|isnt|didnt|doesnt)(?![a-z])")
# "Mr. India was not it", "there were no songs", "it had no interval"
EXCL_EXTRA_RX = re.compile(r"(?<![a-z])(?:(?:was|is|'s)\s+not\s+it|(?:wasn't|wasnt|isn't|isnt)\s+it|"
                           r"not\s+(?:it|the one|this one|that one)|there\s+(?:was|were|is|are)\s+(?:no|not any|never any)|"
                           r"(?:it|the film|the movie)\s+had\s+no|without any)(?![a-z])")
CAST_NEG_RX = re.compile(r"(?<![a-z])(?:(?:wasn't|wasnt|was not|isn't|isnt|is not|weren't|werent|were not)\s+"
                         r"(?:in it|in the (?:film|movie)|in this|starring|the (?:lead|hero|heroine|villain|star))|"
                         r"(?:didn't|didnt|did not|doesn't|doesnt|does not)\s+(?:star|have|feature|include|act)|"
                         r"not\s+(?:starring|featuring|with)|without)(?![a-z])")

GENRES = {
    "comedy": "comedy", "comedies": "comedy", "comic": "comedy", "funny": "comedy", "hilarious": "comedy",
    "slapstick": "comedy", "satire": "comedy", "spoof": "comedy", "parody": "comedy", "romcom": "comedy",
    "horror": "horror", "scary": "horror", "spooky": "horror", "thriller": "thriller", "suspense": "thriller",
    "romance": "romance", "romantic": "romance", "love story": "romance", "romcom ": "romance",
    "action": "action", "musical": "musical", "animated": "animation", "animation": "animation",
    "cartoon": "animation", "biopic": "biographical", "biographical": "biographical", "true story": "biographical",
    "sports": "sports", "sport": "sports", "war film": "war", "war movie": "war", "war drama": "war",
    "historical": "historical", "period drama": "historical", "period film": "historical", "costume drama": "historical",
    "superhero": "superhero", "sci-fi": "science fiction", "scifi": "science fiction", "science fiction": "science fiction",
    "mythological": "mythological", "devotional": "mythological", "crime": "crime", "gangster": "crime",
    "heist": "crime", "spy": "spy", "espionage": "spy", "mystery": "mystery", "whodunit": "mystery",
    "fantasy": "fantasy", "drama": "drama", "tearjerker": "drama", "family drama": "family",
    "family film": "family", "masala": "masala", "western": "western", "coming of age": "coming-of-age",
    "coming-of-age": "coming-of-age", "adventure": "adventure",
}
GENRE_RX = _phrases(GENRES)
LANGS = {"hindi": "hi", "bollywood": "hi", "hindustani": "hi", "telugu": "te", "tollywood": "te",
         "south indian": "te", "south movie": "te", "south film": "te", "english": "en", "hollywood": "en"}
LANG_RX = _phrases(LANGS)

DECADE_WORDS = {"fifties": 1950, "sixties": 1960, "seventies": 1970, "eighties": 1980, "nineties": 1990,
                "noughties": 2000}
DECADE_RX = re.compile(r"(?<![a-z0-9])(?:(early|mid|late)[\s-]*)?(?:the\s+)?"
                       r"(?:(19|20)?(\d)0\s*'?\s*s|(fifties|sixties|seventies|eighties|nineties|noughties))(?![a-z0-9])")
YEAR_RX = re.compile(r"(?<![0-9])((?:19[3-9]|20[0-2])\d)(?![0-9])")
OLD_RX = re.compile(r"(?<![a-z])(?:black and white|black-and-white|b&w|very old|old (?:hindi |telugu |english )?(?:film|movie|one)|"
                    r"classic|golden era|vintage|retro)(?![a-z])")
NEW_RX = re.compile(r"(?<![a-z])(?:recent|recently|new (?:film|movie|one)|latest|last year|this year|"
                    r"a few years (?:back|ago)|came out recently)(?![a-z])")
TOKEN_RX = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# "no" followed by these words is an idiom or a pronoun ("no one", "no means no"), not "no <noun>".
_NO_IDIOMS = {"means", "one", "longer", "more", "wonder", "way", "matter", "doubt", "idea", "clue", "sooner",
              "less", "body"}
# Words that may precede a clause-initial negation without turning it into a plot statement.
_NEG_HEAD_OK = {"definitely", "definately", "certainly", "probably", "maybe", "and", "but", "also", "so",
                "absolutely", "surely", "i'm", "im", "sure", "pretty", "it", "was", "is", "its", "it's", "that",
                "this", "one", "thing", "just", "and", "though", "however", "oh", "um", "well", "film", "movie",
                "the", "i", "am", "100%", "for"}
ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "no", "prof", "lt", "col", "capt", "sgt", "etc", "e.g", "i.e"}


# --------------------------------------------------------------------------- data classes
@dataclass
class Cue:
    kind: str          # hedge, confident, negation, meta, context, year, lang, genre, entity, correction, lexicon
    start: int
    end: int
    text: str
    info: dict = field(default_factory=dict)


@dataclass
class YearCue:
    lo: int
    hi: int
    mode: str          # "prefer" | "upper" (watched it in ...) | "exclude"
    text: str
    soft: float = 1.0  # < 1 for vague cues such as "old film"


@dataclass
class Token:
    tok: str
    start: int
    end: int
    in_scope: bool = False   # inside a negation scope
    cue: str = ""            # kind of cue phrase the token belongs to ("hedge", "year", "lang", ...)

    @property
    def term(self) -> bool:
        """Does this token count as a content term?  Cue words ("maybe", "early 90s") do not."""
        return self.cue not in ("hedge", "confident", "year", "meta") and _is_term(self.tok)


@dataclass
class Clause:
    text: str
    start: int
    end: int
    hedged: bool = False
    confident: bool = False
    meta: bool = False
    context: bool = False
    negated: bool = False
    exclusion: bool = False
    cues: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    extra: list = field(default_factory=list)        # (stem, factor, in_scope, reason)
    entities: list = field(default_factory=list)      # (EntityMatch, in_scope)
    years: list = field(default_factory=list)
    langs: list = field(default_factory=list)          # (lang, in_scope)
    genres: list = field(default_factory=list)         # (genre, in_scope)
    trust: float = 1.0      # set below 1 by the evidence check (C1) when the clue belongs to another film

    @property
    def kind(self) -> str:
        if self.meta:
            return "meta"
        if self.exclusion:
            return "exclusion"
        if self.hedged:
            return "hedged"
        if self.context:
            return "context"
        if self.confident:
            return "confident"
        return "plain"

    def base_weight(self, w: dict, use_clauses: bool = True) -> float:
        """c_j before the evidence check: multiplicative, so a hedged context clause gets c_ctx * c_h."""
        if not use_clauses:
            return 1.0
        if self.meta:
            return w.get("meta", 0.0)
        c = 1.0
        if self.context:
            c *= w.get("context", 0.5)
        if self.hedged:
            c *= w.get("hedged", 0.3)
        elif self.confident:
            c *= w.get("confident", 1.2)
        return c

    def weight(self, w: dict, use_clauses: bool = True) -> float:
        """c_j: the clause weight, times the trust the evidence check (C1) leaves it."""
        return self.base_weight(w, use_clauses) * self.trust

    def cue_weight(self, w: dict, use_clauses: bool = True, floor: float = 0.15) -> float:
        """Weight of the clause's structured cues (year, language, genre, names): never below `floor`, so a hedged
        "maybe the 90s" still counts a little, unless the evidence check withdrew the clause (trust 0)."""
        return max(self.base_weight(w, use_clauses), floor) * self.trust


@dataclass
class QueryAnalysis:
    text: str
    clauses: list
    dense_text: str | None = None    # text for the whole-query embedding when clauses were withdrawn (C1)

    # ---- term weights for the lexical scorer
    def term_weights(self, w: dict, use_clauses: bool = True, negation: str = "scope",
                     hinglish: bool = True, lex_factor: float = 0.6) -> tuple[dict, dict]:
        """Return (positive, negative) {stem: weight}.

        s(d) = sum_j c_j * BM25(clause_j, d) - mu * sum_j c_j * BM25(neg_j, d) is additive over terms,
        so it equals a weighted-term BM25 with q_t = sum of the weights of the clauses containing t.

        negation modes: "none" (negated words count as positive evidence, i.e. plain BM25 behaviour),
        "drop" (negated words ignored), "penalize_all" (every negated scope is penalised),
        "scope" (only exclusions about the film are penalised; plot negations stay positive).
        """
        pos: dict[str, float] = {}
        neg: dict[str, float] = {}
        mu = w.get("mu", 0.5)

        def add(d, s, v):
            if v > 0:
                d[s] = d.get(s, 0.0) + v

        for c in self.clauses:
            cw = c.weight(w, use_clauses)
            penal = c.exclusion if negation == "scope" else (c.negated if negation == "penalize_all" else False)
            items = [(stem(t.tok), 1.0, t.in_scope, t.cue) for t in c.tokens if t.term]
            if hinglish:
                items += [(s, f, sc, "") for s, f, sc, _ in c.extra]
            for s, f, in_scope, cue_kind in items:
                if in_scope and c.negated and negation != "none":
                    if negation == "drop":
                        continue
                    if penal:
                        if cue_kind == "lang" and negation == "scope":
                            continue          # "not the Hindi remake": handled by the language feature

                        # an exclusion is penalised by mu * (its confidence); meta clauses never penalise
                        conf = 1.0 if not use_clauses else (w.get("hedged", 0.3) if c.hedged else
                                                            w.get("confident", 1.2) if c.confident else 1.0)
                        add(neg, s, mu * conf * c.trust * f)
                        continue
                add(pos, s, cw * f)
        return pos, neg

    # ---- structured cues used as features
    def year_cues(self, w: dict, use_clauses: bool = True, negation: str = "scope"):
        out = []
        for c in self.clauses:
            for y in c.years:
                weight = c.base_weight(w, use_clauses) if not c.meta else 0.5 * c.base_weight(w, use_clauses)
                if c.context and y.mode == "prefer":
                    y = YearCue(y.lo, y.hi, "upper", y.text, y.soft)
                if y.mode == "exclude" and negation not in ("scope", "penalize_all"):
                    continue
                if c.trust > 0:
                    out.append((y, max(weight, 0.15) * c.trust * y.soft))
        return out

    def lang_prefs(self, w: dict, use_clauses: bool = True, negation: str = "scope"):
        pref, excl = {}, {}
        for c in self.clauses:
            cw = c.cue_weight(w, use_clauses)
            if cw <= 0:
                continue
            for lang, in_scope in c.langs:
                if in_scope and c.negated and negation in ("scope", "penalize_all"):
                    excl[lang] = max(excl.get(lang, 0), cw)
                elif not (in_scope and c.negated and negation == "drop"):
                    pref[lang] = max(pref.get(lang, 0), cw)
        return pref, excl

    def genre_prefs(self, w: dict, use_clauses: bool = True, negation: str = "scope"):
        pref, excl = {}, {}
        for c in self.clauses:
            cw = c.cue_weight(w, use_clauses)
            if cw <= 0:
                continue
            for g, in_scope in c.genres:
                if in_scope and c.negated and (negation == "penalize_all" or (negation == "scope" and c.exclusion)):
                    excl[g] = max(excl.get(g, 0), cw)
                elif not (in_scope and c.negated and negation == "drop"):
                    pref[g] = max(pref.get(g, 0), cw)
        return pref, excl

    def entity_prefs(self, w: dict, use_clauses: bool = True, negation: str = "scope", hinglish: bool = True):
        """{entity name: (weight, docs)} for mentioned and excluded people / characters / titles."""
        pos, neg = {}, {}
        for c in self.clauses:
            cw = c.cue_weight(w, use_clauses)
            if cw <= 0:
                continue
            for m, in_scope in c.entities:
                if not hinglish and m.how != "exact":
                    continue
                excluded = in_scope and c.negated and (negation == "penalize_all" or (negation == "scope" and c.exclusion))
                target = neg if excluded else pos
                if in_scope and c.negated and negation == "drop":
                    continue
                key = (m.entity.name, m.entity.kind)
                prev = target.get(key)
                val = cw * m.score
                if prev is None or prev[0] < val:
                    target[key] = (val, m.entity.docs, m.how, m.surface)
        return pos, neg

    def content_text(self, w: dict, use_clauses: bool = True) -> list[tuple[str, float]]:
        """Clause texts with their weights, for clause-level dense retrieval (exclusions and meta removed)."""
        out = []
        for c in self.clauses:
            if c.meta or c.exclusion:
                continue
            words = [t.tok for t in c.tokens if t.term]
            if len(words) >= 1:
                out.append((c.text, c.weight(w, use_clauses)))
        return out

    def to_json(self, w: dict) -> dict:
        return {"text": self.text, "clauses": [{
            "text": c.text, "start": c.start, "end": c.end, "kind": c.kind,
            "weight": round(c.weight(w), 3), "trust": c.trust, "negated": c.negated, "exclusion": c.exclusion,
            "hedged": c.hedged, "confident": c.confident, "context": c.context, "meta": c.meta,
            "cues": [{"kind": q.kind, "start": q.start, "end": q.end, "text": q.text, **q.info} for q in c.cues],
            "terms": sorted({stem(t.tok) for t in c.tokens if t.term and not (t.in_scope and c.exclusion)}),
            "excluded_terms": sorted({stem(t.tok) for t in c.tokens if t.term and t.in_scope and c.exclusion}),
        } for c in self.clauses]}


def _is_term(tok: str) -> bool:
    t = tok.replace("'", "")
    if t in STOPWORDS or t in QUERY_FILLER or len(t) < 2 and not t.isdigit():
        return False
    if re.fullmatch(r"\d0s|(19|20)\d\ds?|\d0", t) or t in DECADE_WORDS:
        return False          # dates are handled by the year feature, not by term matching
    return True


# --------------------------------------------------------------------------- clause splitting
_SPLIT_CONJ = re.compile(r"\s+(?:but|although|though|however|whereas|except that|and then|and also|plus|also|"
                         r"and i think|and maybe|and probably|and i'm sure|and im sure)\s+"
                         r"|\s+and\s+(?=(?:he|she|they|it|there|i|we|his|her|their)\b)"
                         r"|\s*,\s*|\s*\(\s*|\s*\)\s*|\s+-+\s+|\s*—\s*|\s*:\s+")


ACTOR_STATEMENT_RX = re.compile(r"(?<![a-z])(?:(?:was|were|is|are)\s+in\s+(?:it|the\s+(?:film|movie))|starred|starring|"
                                r"acted|played\s+the)(?![a-z])")


def split_clauses(text: str) -> list[tuple[int, int]]:
    """Character spans of clauses: split on sentence ends, commas and clause-joining conjunctions."""
    low = fold_same_length(text)
    # 1) sentence boundaries (skip abbreviations and initials such as "S. S. Rajamouli", "Mr. India")
    bounds, start = [], 0
    for m in re.finditer(r"[.!?;]+(?:\s+|$)|\.\.\.+|\n+", low):
        prev = re.search(r"([a-z0-9.]+)\.*$", low[start:m.start() + 1])
        word = prev.group(1).rstrip(".") if prev else ""
        abbrev = len(word) == 1 or word in ABBREV
        if word == "no" and not re.match(r"\d", low[m.end():m.end() + 1]):
            abbrev = False          # "No. 1" is an abbreviation; "no means no. I watched it in Hindi" ends a sentence
        if m.group(0).startswith(".") and not m.group(0).startswith("...") and abbrev:
            continue
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(text)))
    spans = []
    for sid, (s, e) in enumerate(bounds):
        seg = low[s:e]
        pos = 0
        for m in _SPLIT_CONJ.finditer(seg):
            spans.append((s + pos, s + m.start(), sid))
            pos = m.end()
        spans.append((s + pos, e, sid))
    spans = [(s, e, sid) for s, e, sid in spans if text[s:e].strip(" .,;:!?-")]
    # 2) merge fragments that carry no cue and almost no content (list items, "Also,") into a neighbour
    merged: list[list[int]] = []
    for s, e, sid in spans:
        frag = low[s:e]
        toks = [t for t in TOKEN_RX.findall(frag) if _is_term(t)]
        has_cue = bool(HEDGE_RX.search(frag) or CONFIDENT_RX.search(frag) or NEGATION_RX.search(frag)
                       or DECADE_RX.search(frag) or YEAR_RX.search(frag) or META_RX.search(frag)
                       or GENRE_RX.search(frag) or LANG_RX.search(frag))
        # an actor statement after a full stop ("Amitabh Bachchan was in it.") is a sentence of its own
        own_sentence = bool(merged) and merged[-1][2] != sid and bool(ACTOR_STATEMENT_RX.search(frag))
        if merged and not own_sentence and not has_cue and len(toks) <= 2 and (text[s:e].strip()[:1].isupper() or frag.strip().startswith("and ")):
            merged[-1][1] = e           # "Dharmendra, Amitabh and Hema" stays one clause
        elif merged and not toks and not has_cue:
            merged[-1][1] = e
        else:
            merged.append([s, e, sid])
    out = []
    for s, e, _ in merged:
        while s < e and text[s] in " \t,.;:-(":
            s += 1
        while e > s and text[e - 1] in " \t,.;:-)":
            e -= 1
        if e > s:
            out.append((s, e))
    return out


# --------------------------------------------------------------------------- the parser
def _decades(low: str, offset: int):
    cues = []
    for m in DECADE_RX.finditer(low):
        qual, century, digit, word = m.group(1), m.group(2), m.group(3), m.group(4)
        if word:
            d = DECADE_WORDS[word]
        else:
            dd = int(digit) * 10
            if century:
                d = int(century) * 100 + dd
            else:
                d = 2000 + dd if dd <= 20 else 1900 + dd
        lo, hi = d, d + 9
        if qual == "early":
            lo, hi = d, d + 4
        elif qual == "mid":
            lo, hi = d + 3, d + 7
        elif qual == "late":
            lo, hi = d + 5, d + 9
        cues.append((m.start() + offset, m.end() + offset, lo, hi, 1.0))
    taken = [(a, b) for a, b, *_ in cues]
    for m in YEAR_RX.finditer(low):
        a, b = m.start() + offset, m.end() + offset
        if any(x <= a < y for x, y in taken):
            continue
        y = int(m.group(1))
        around = bool(re.search(r"(around|about|roughly|circa|approximately|or so|ish)", low[max(0, m.start() - 12):m.end() + 6]))
        cues.append((a, b, y - (2 if around else 1), y + (2 if around else 1), 1.0))
    for m in OLD_RX.finditer(low):
        bw = "black" in m.group(0) or "b&w" in m.group(0)
        cues.append((m.start() + offset, m.end() + offset, 1930, 1964 if bw else 1985, 0.9 if bw else 0.5))
    for m in NEW_RX.finditer(low):
        cues.append((m.start() + offset, m.end() + offset, 2017, 2026, 0.5))
    return cues


def _mask(spans, n):
    mask = [False] * n
    for a, b in spans:
        for i in range(max(0, a), min(n, b)):
            mask[i] = True
    return mask


def parse_query(text: str, matcher: EntityMatcher | None = None, corrector: SpellingCorrector | None = None,
                vocab: set | None = None) -> QueryAnalysis:
    text = re.sub(r"\s+", " ", text).strip()
    low_all = fold_same_length(text)
    clauses = []
    for s, e in split_clauses(text):
        low = low_all[s:e]
        c = Clause(text=text[s:e], start=s, end=e)

        def cue(kind, a, b, **info):
            c.cues.append(Cue(kind, s + a, s + b, text[s + a:s + b], info))

        neg_hedges = [(m.start(), m.end()) for m in NEG_HEDGE_RX.finditer(low)]
        hedges = [(m.start(), m.end()) for m in HEDGE_RX.finditer(low)] + \
                 [(m.start(), m.end()) for m in HEDGE_EXTRA_RX.finditer(low)]
        confs = [(m.start(), m.end()) for m in CONFIDENT_RX.finditer(low)]
        for a, b in neg_hedges:
            cue("hedge", a, b, negated=True)
        for a, b in hedges:
            if not any(x <= a < y for x, y in neg_hedges):
                cue("hedge", a, b)
        for a, b in confs:
            cue("confident", a, b)
        c.hedged = bool(neg_hedges or hedges)
        c.confident = bool(confs) and not c.hedged
        meta = list(META_RX.finditer(low))
        ctx = list(CONTEXT_RX.finditer(low))
        for m in meta:
            cue("meta", m.start(), m.end())
        for m in ctx:
            cue("context", m.start(), m.end())
        c.context = bool(ctx)

        # negation cues, ignoring "not" inside hedge/confidence phrases ("not sure", "no doubt")
        # and discourse markers ("no wait")
        protected = _mask([(a, b) for a, b in hedges + confs if not any(x <= a < y for x, y in neg_hedges)]
                          + [(m.start(), m.end()) for m in DISCOURSE_RX.finditer(low)], len(low))
        negs = [(m.start(), m.end()) for m in NEGATION_RX.finditer(low) if not protected[m.start()]]
        negs += neg_hedges
        negs.sort()
        c.negated = bool(negs)
        for a, b in negs:
            if (a, b) not in neg_hedges:
                cue("negation", a, b)

        # structured cues (positions are relative to the clause) -----------------------
        year_spans = _decades(low, 0)
        for a, b, lo, hi, _ in year_spans:
            cue("year", a, b, lo=lo, hi=hi)
        genre_hits = [(GENRES[re.sub(r"\s+", " ", m.group(0))], m.start(), m.end()) for m in GENRE_RX.finditer(low)]
        lang_hits = [(LANGS[re.sub(r"\s+", " ", m.group(0))], m.start(), m.end()) for m in LANG_RX.finditer(low)]
        for g, a, b in genre_hits:
            cue("genre", a, b, genre=g)
        for l, a, b in lang_hits:
            cue("lang", a, b, lang=l)

        # tokens, labelled with the cue phrase they belong to (cue words are not content terms)
        tokens = [Token(m.group(0).replace("'", ""), s + m.start(), s + m.end()) for m in TOKEN_RX.finditer(low)]
        labelled = ([("hedge", a, b) for a, b in hedges + neg_hedges] + [("confident", a, b) for a, b in confs]
                    + [("year", a, b) for a, b, *_ in year_spans] + [("meta", m.start(), m.end()) for m in meta]
                    + [("lang", a, b) for _, a, b in lang_hits] + [("genre", a, b) for _, a, b in genre_hits])
        for t in tokens:
            for kind, a, b in labelled:
                if a <= t.start - s < b:
                    t.cue = kind
                    break
        content = [t for t in tokens if t.term]
        c.meta = bool(meta) and (len(content) <= 4 or len(meta) >= 2) and not (c.negated and not meta)

        # negation scope: the words after the first negation cue, or the subject before it
        # when nothing follows ("Shah Rukh wasn't in it") ----------------------------------
        in_scope = lambda pos: False
        if c.negated:
            first_a, first_b = negs[0]
            after = [t for t in tokens if t.start - s >= first_b]
            before = [t for t in tokens if t.end - s <= first_a]
            use_after = any(t.term or t.cue in ("lang", "genre", "year") for t in after)
            for t in (after if use_after else before):
                t.in_scope = True
            in_scope = (lambda pos: pos >= first_b) if use_after else (lambda pos: pos < first_a)

            # Is the negation about the film itself (an exclusion) or part of the story?
            # A clause-initial "not ..." or "no <noun>" ("no songs", "no villain") rules something out;
            # "nobody believes her", "no one knew", "no means no" are statements, not exclusions.
            head_words = re.findall(r"[a-z']+", low[:first_a])
            first_cue = low[first_a:first_b].strip()
            nxt = (re.match(r"\s*([a-z']+)", low[first_b:]) or [None, ""])[1]
            excl = all(w in _NEG_HEAD_OK for w in head_words) and (
                first_cue == "not" or (first_cue == "no" and nxt not in _NO_IDIOMS))
            excl |= bool(FILM_SUBJECT_RX.search(low) or CAST_NEG_RX.search(low) or EXCL_EXTRA_RX.search(low)
                         or neg_hedges)
            excl |= any(in_scope(a) for _, a, _ in genre_hits) and len(content) <= 8
            excl |= any(in_scope(a) for _, a, _ in lang_hits)
            excl |= any(in_scope(a) for a, *_ in year_spans)
            c.exclusion = excl and not c.meta
        exclude = lambda pos: c.exclusion and in_scope(pos)
        c.years = [YearCue(lo, hi, "exclude" if exclude(a) else "prefer", text[s + a:s + b], soft)
                   for a, b, lo, hi, soft in year_spans]
        c.genres = [(g, in_scope(a)) for g, a, _ in genre_hits]
        c.langs = [(l, in_scope(a)) for l, a, _ in lang_hits]
        c.tokens = tokens
        clauses.append(c)

    # explicit beats vague: in "old Hindi film, maybe the 90s" the film is from the 90s, not 1930-1985
    # ("old" is relative to today); a black-and-white cue is specific and stays
    generic = lambda y: bool((OLD_RX.fullmatch(y.text.lower()) and not re.search(r"black|b&w", y.text.lower()))
                             or NEW_RX.fullmatch(y.text.lower()))
    if any(y.mode == "prefer" and not generic(y) for c in clauses for y in c.years):
        for c in clauses:
            drop = {y.text for y in c.years if y.mode == "prefer" and generic(y)}
            if drop:
                c.years = [y for y in c.years if y.text not in drop]
                c.cues = [q for q in c.cues if not (q.kind == "year" and q.text in drop)]

    analysis = QueryAnalysis(text, clauses)
    _resolve_entities(analysis, matcher, corrector, vocab)
    return analysis


def _resolve_entities(qa: QueryAnalysis, matcher, corrector, vocab) -> None:
    """Entity matching, spelling correction and Hinglish lexicon expansion, clause by clause."""
    vocab = vocab or set()
    for c in qa.clauses:
        toks = c.tokens
        covered: set[int] = set()
        if matcher is not None:
            # words already explained by a language/genre/year/hedge cue cannot be (part of) a name
            words = ["the" if t.cue in ("lang", "genre", "year", "hedge", "confident", "meta") else t.tok
                     for t in toks]
            for m in matcher.match(words, vocab):
                in_scope = any(toks[i].in_scope for i in range(m.start, m.end))
                c.entities.append((m, in_scope))
                covered.update(range(m.start, m.end))
                a, b = toks[m.start].start, toks[m.end - 1].end
                if not any(q.kind == "entity" and q.start == a for q in c.cues):
                    c.cues.append(Cue("entity", a, b, qa.text[a:b],
                                      {"entity": m.entity.name, "type": m.entity.kind, "how": m.how,
                                       "score": round(m.score, 2)}))
                if m.corrected and m.entity.kind != "title":
                    # add only the corrected spelling's *new* words ("amitab bachan" -> amitabh, bachchan);
                    # words already typed, stop words and very common words ("singh", "inspector") are skipped
                    typed = set(m.surface.split())
                    common, proper = getattr(matcher, "common", set()), getattr(matcher, "proper", set())
                    for s_ in dict.fromkeys(stem(t) for t in tokenize(m.entity.name)
                                            if t not in typed and not (t in common and t not in proper)
                                            and t not in STOPWORDS and len(t) >= 3):
                        c.extra.append((s_, m.score, in_scope, f"{m.surface} -> {m.entity.name}"))
        for i, t in enumerate(toks):
            if i in covered or not t.term:
                continue
            gloss = HINGLISH_LEXICON.get(t.tok)
            if gloss:
                for g in gloss:
                    for s_ in analyze(g):
                        c.extra.append((s_, 0.6, t.in_scope, f"{t.tok} -> {g}"))
                c.cues.append(Cue("lexicon", t.start, t.end, qa.text[t.start:t.end], {"gloss": " ".join(gloss)}))
                continue
            if corrector is not None and t.tok not in vocab:
                fix = corrector.correct(t.tok)
                if fix:
                    word, conf = fix
                    for s_ in analyze(word):
                        c.extra.append((s_, conf, t.in_scope, f"{t.tok} -> {word}"))
                    c.cues.append(Cue("correction", t.start, t.end, qa.text[t.start:t.end], {"to": word}))
