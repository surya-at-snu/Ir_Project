"""Text preprocessing: folding, tokenising, stop-word removal and stemming (lectures 2-3).

The same analyser is used for documents and queries so that terms line up in the
inverted index.  Queries additionally drop "ToT filler" words ("movie", "remember",
"scene", "hero") that every tip-of-the-tongue request contains but no plot does.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")

# A standard English stop-word list (close to NLTK's) plus contractions with the apostrophe removed.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren arent as at be because been before
being below between both but by can cannot cant could couldn couldnt d did didn didnt do does doesn
doesnt doing don dont down during each few for from further had hadn hadnt has hasn hasnt have haven
havent having he hed hell hes her here heres hers herself him himself his how hows i id ill im ive if
in into is isn isnt it its itself just let lets ll m ma me mightn more most mustn my myself needn no
nor not now o of off on once only or other ought our ours ourselves out over own re s same shan she
shed shell shes should shouldn shouldnt so some such t than that thats the their theirs them
themselves then there theres these they theyd theyll theyre theyve this those through to too under
until up ve very was wasn wasnt we wed well were weren werent weve what whats when whens where wheres
which while who whom whos why whys will with won wont would wouldn wouldnt y you youd youll your
youre yours yourself yourselves ve also us one two get got go went gone yet still even ever though
""".split())

# Words that appear in nearly every ToT request but never describe the film's content.
QUERY_FILLER = frozenset("""
movie movies film films flick picture remember remembered recall recalled watch watched watching saw
seen see think thought guess believe maybe perhaps probably possibly might may must sure unsure
certain definitely remember scene scenes something someone somebody somewhere thing things stuff
guy guys hero heroine lead actor actress actors protagonist character characters story plot name
title called help please thanks thank anyone anybody know knows looking look find found trying try
kind sort like really pretty quite bit lot lots basically actually kinda sorta yeah lol idk
part around ish vaguely vague clearly distinctly exactly totally completely named called wait actually
sorry nobody nothing never none neither without anything everything somehow somewhat whole end ending
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def fold(text: str) -> str:
    """Lower-case and strip accents/diacritics ("Brahmāstra" -> "brahmastra")."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace("’", "'").replace("‘", "'").lower()


def tokenize(text: str) -> list[str]:
    """Folded word tokens; apostrophes are removed so "wasn't" -> "wasnt", "Khan's" -> "khans"."""
    text = fold(text).replace("'", "")
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 1 or t.isdigit()]


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    if token.isdigit():
        return token
    return _STEMMER.stemWord(token)


def analyze(text: str, query: bool = False) -> list[str]:
    """Tokenise, remove stop words (and ToT filler for queries), then stem."""
    out = []
    for tok in tokenize(text):
        if tok in STOPWORDS or (query and tok in QUERY_FILLER):
            continue
        out.append(stem(tok))
    return out


def is_content_token(tok: str, query: bool = True) -> bool:
    return tok not in STOPWORDS and not (query and tok in QUERY_FILLER)


def word_spans(text: str):
    """Yield (start, end, folded token) for every word in the *original* text (used for highlighting)."""
    for m in _WORD_RE.finditer(text):
        tok = fold(m.group(0)).replace("'", "")
        tok = re.sub(r"[^a-z0-9]", "", tok)
        if tok:
            yield m.start(), m.end(), tok
