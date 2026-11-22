"""Hinglish / transliteration handling.

Indian film names, character names and dialogue words are written in Roman script in
many ways ("Bachchan"/"Bachan", "Deewaar"/"Diwar", "Shahrukh"/"Shah Rukh"/"Sharukh").
Following the mixed-script IR line of work (Gupta et al., SIGIR 2014; FIRE MSIR tracks),
we normalise spelling variants with a phonetic key and use it in three places:

1. EntityMatcher   - finds (possibly misspelled) person names, character names and titles
                     in the query and rewrites them to the canonical corpus spelling.
2. SpellingCorrector - maps out-of-vocabulary query words to in-vocabulary words with the
                     same phonetic key ("gabar" -> "gabbar", "mogembo" -> "mogambo").
3. HINGLISH_LEXICON - translates common Hindi/Telugu words to English ("shaadi" -> wedding,
                     "andha" -> blind) because Wikipedia plots are written in English.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from .text import STOPWORDS, QUERY_FILLER, fold


# --------------------------------------------------------------------------- phonetic key
def phonetic_key(word: str) -> str:
    """Collapse common Roman-script spelling variants of Hindi/Telugu words to one key.

    Rules (applied in order): ph->f, q->k, x->ks, z->j, w->v; drop aspiration (bh->b, ch->c,
    dh->d, kh->k, sh->s, th->t, ...); drop 'h' after a vowel; long vowels -> short
    (aa->a, ee->i, oo->u); diphthongs (ai->e, au/ou->o); final -ay/-ey -> e; iya -> ia;
    and finally squeeze doubled letters.  Digits are kept, so sequels keep their own keys
    ("Dhoom 3" -> dum3, not the same key as "Dhoom").
    """
    w = re.sub(r"[^a-z0-9]", "", fold(word))
    if not w:
        return ""
    w = re.sub(r"h+", "h", w)
    w = w.replace("ph", "f").replace("q", "k").replace("x", "ks").replace("z", "j")
    w = w.replace("w", "v").replace("ck", "k")
    w = re.sub(r"([bcdgjkpstr])h", r"\1", w)          # aspirates and sh/ch
    w = re.sub(r"^h(?=[^aeiouy])", "", w)              # hrithik -> rithik
    w = re.sub(r"(?<=[aeiou])h", "", w)                # shahrukh -> sarukh, rahman -> raman
    w = re.sub(r"(ay|ey)$", "e", w)                    # sholay / sholey -> shole
    w = w.replace("iya", "ia")                         # dulhaniya -> dulhania
    w = w.replace("aa", "a").replace("ee", "i").replace("ii", "i")
    w = w.replace("oo", "u").replace("uu", "u")
    w = w.replace("ai", "e").replace("au", "o").replace("ou", "o")
    w = re.sub(r"(.)\1+", r"\1", w)                    # gabbar -> gabar, bachchan -> bacan
    return w


# --------------------------------------------------------------------------- lexicon
_LEX = """
maa amma mummy maiyya ammi=mother | baap pitaji babuji abba nanna=father | beta bete=son
beti bitiya=daughter | bhai bhaiya bhaiyya bhaiyaa anna=brother | behen bahen didi akka chelli=sister
dada=grandfather | dadi=grandmother | nani=grandmother | chacha chachaji mamaji=uncle | chachi mausi bua=aunt
saas=mother-in-law | sasur=father-in-law | bahu=daughter-in-law | damad jamai=son-in-law | sasural=in-laws
biwi patni pellam bharya=wife | pati shohar mogudu bhartha=husband | dulha dulhe=groom
dulhan dulhania dulhaniya=bride | dost yaar snehitudu nestham=friend | dosti yaari=friendship
premi aashiq ashiq=lover | premika mehbooba sanam=beloved | ladka chhokra abbai=boy | ladki chhori ammayi ammai=girl
bachcha bacha bachche bacche=child | aurat=woman | aadmi=man | budha buddhe=old
pyaar pyar prem prema ishq mohabbat muhabbat=love | dil manasu=heart | shaadi shadi vivah byah pelli=wedding marriage
rishta=proposal | sagai mangni=engagement | baraat baarat=wedding procession | dahej=dowry | talaq=divorce
judai=separation | bewafa=unfaithful | dhoka dhokha=betrayal
gaon gaanv gaun pind palle palli=village | ooru=town | sheher shahar=city | mandir=temple | masjid=mosque
ghar illu=house home | haveli kothi=mansion | mahal=palace | kotha=brothel | jail jel=prison
adalat kachehri=court | dhaba=roadside restaurant | bazaar bazar=market | mela=fair | khet=farm field
jangal=forest | pahad=mountain | nadi=river | samundar samandar=sea | videsh pardes=abroad
desh watan mulk=country nation | sarhad=border | chor=thief | chori=theft
daku daaku dakait=dacoit bandit | dakaiti=dacoity robbery | gunda goonda rowdy badmash=goon gangster
pulis polisu=police | havaldar=constable police | thanedar darogha=police inspector
sipahi fauji jawan=soldier | fauj sena=army | kisan kisaan=farmer | mazdoor=labourer worker
naukar=servant | naukrani=maid | malik=owner boss | seth=businessman | sahukar mahajan lala=moneylender
thakur zamindar jagirdar=landlord | sarpanch=village head | raja maharaja=king | rani maharani=queen
rajkumar shehzada shahzada=prince | rajkumari shehzadi=princess | badshah baadshah samrat shahenshah=emperor
nawab=nawab nobleman | tawaif=courtesan | mujra=courtesan dance | sadhu=ascetic | pujari=priest
tantrik=sorcerer occult | vakil wakeel=lawyer | daktar=doctor | masterji=teacher | pehlwan pahalwan=wrestler
neta=politician | mantri=minister | sarkar=government | angrez angrezi=british | firangi=foreigner
qaidi kaidi khaidi=prisoner | jasoos jasus=spy detective | deewana diwana=lover obsessed
pagal paagal=mad crazy | pagalpan=madness | sharabi sharaabi=drunkard alcoholic | juari jwari=gambler
bhikari bhikhari=beggar | anath yatim=orphan | vidhwa vidhva bewa=widow
bhoot bhut deyyam dayyam=ghost | chudail=witch ghost | pishach rakshas=demon | aatma atma=spirit soul
shaitan shaitaan=devil | bhagwan bhagavan devudu=god | devta=deity god | devi=goddess
jadoo jaadu=magic | chamatkar=miracle | punarjanam punarjanm=reincarnation | shraap shrap=curse
naag=snake serpent | nagin naagin=serpent woman snake
badla inteqam intaqam intekam paga pagaa=revenge | khoon raktham=blood murder | khooni=killer
qatil katil=murderer | qatl katl=murder | maut mout=death | laash=corpse | goli=bullet gunshot
bandook banduk=gun | talwar=sword | dushman=enemy | dushmani=feud enmity | jung jang yuddh yuddham=war battle
ladai=fight | apharan=kidnapping | taskar=smuggler | taskari=smuggling | nasha nashaa=drugs addiction
daaru daru sharab sharaab=alcohol liquor | jua=gambling | phansi faansi=hanging execution | saza=punishment
kanoon kanun=law | insaaf insaf nyay=justice | gawah gavah=witness | sach sacchai=truth | jhooth jhoot=lie
zindagi jindagi jeevan=life | sapna sapne khwab=dream | khushi=happiness | gham=sorrow grief | dard=pain
aansu=tears | darr=fear | himmat=courage | izzat=honour | imaan=integrity | imandar=honest
garib gareeb=poor | garibi=poverty | paisa paise=money | daulat=wealth | khazana=treasure
heera hira=diamond | zameen jameen=land | karz karza udhaar=debt loan | naukri=job | padhai=studies education
pariksha imtihaan imtihan=exam | kitaab=book | chitthi=letter | tasveer tasvir=photograph painting
aaina=mirror | gaadi=car | railgaadi=train | jahaz jahaaz=ship | kashti naav=boat
andha andhi andhe=blind | behra bahra=deaf | goonga gunga gungi=mute | langda=lame | apahij=disabled
aankh aankhen=eye eyes | haath=hand | bimari bimaari=illness disease | yaad yaadein=memory memories
yaaddasht yaadasht=memory amnesia
gaana gana gaane gane geet=song | sangeet sangeetham=music | naach nach naachna nritya=dance | nartaki=dancer
dhol=drum | qawwali=qawwali song | bhajan=devotional song | rakhi=rakhi sister brother
baarish barish barsaat=rain | chand chaand=moon | suraj=sun | aag=fire | paani pani=water | toofan tufaan=storm
kushti=wrestling | dangal akhada=wrestling | khel=game sport | daud=race running
chhota chota=small younger | bada=big elder | purana=old | naya=new | nakli=fake | asli=real
judwa judwaa=twins | jodi=couple | humshakal hamshakal=lookalike double | sautela sauteli=step
akela=alone lonely | anjaan ajnabi=stranger | padosi padosan=neighbour | katha kadha kahani=story
"""


def _parse_lexicon(spec: str) -> dict[str, list[str]]:
    lex: dict[str, list[str]] = {}
    for entry in spec.replace("\n", " | ").split("|"):
        if "=" not in entry:
            continue
        words, gloss = entry.split("=", 1)
        for w in words.split():
            lex[fold(w)] = gloss.split()
    return lex


HINGLISH_LEXICON = _parse_lexicon(_LEX)

# Nicknames and abbreviations people actually type, mapped to Wikipedia names.
ALIASES = {
    "srk": "Shah Rukh Khan", "king khan": "Shah Rukh Khan", "big b": "Amitabh Bachchan",
    "sallu": "Salman Khan", "mr perfectionist": "Aamir Khan", "akki": "Akshay Kumar",
    "khiladi kumar": "Akshay Kumar", "dharam paaji": "Dharmendra", "thalaiva": "Rajinikanth",
    "rajini": "Rajinikanth", "chiru": "Chiranjeevi", "megastar": "Chiranjeevi",
    "balayya": "Nandamuri Balakrishna", "nbk": "Nandamuri Balakrishna", "jr ntr": "N. T. Rama Rao Jr.",
    "tarak": "N. T. Rama Rao Jr.", "junior ntr": "N. T. Rama Rao Jr.", "bunny": "Allu Arjun",
    "icon star": "Allu Arjun", "power star": "Pawan Kalyan", "pspk": "Pawan Kalyan",
    "rebel star": "Prabhas", "mahesh": "Mahesh Babu", "nag": "Nagarjuna", "rgv": "Ram Gopal Varma",
    "rajamouli": "S. S. Rajamouli", "jakkanna": "S. S. Rajamouli", "mithun da": "Mithun Chakraborty",
    "bebo": "Kareena Kapoor", "arr": "A. R. Rahman", "rahman": "A. R. Rahman", "pancham": "R. D. Burman",
    "rd burman": "R. D. Burman", "kishore da": "Kishore Kumar", "lata": "Lata Mangeshkar",
    "dilip saab": "Dilip Kumar", "raj saab": "Raj Kapoor", "nolan": "Christopher Nolan",
    "spielberg": "Steven Spielberg", "srk's": "Shah Rukh Khan",
    # first names people use for superstars
    "salman": "Salman Khan", "salmaan": "Salman Khan", "aamir": "Aamir Khan", "amir khan": "Aamir Khan",
    "shahrukh": "Shah Rukh Khan", "sharukh": "Shah Rukh Khan", "amitabh": "Amitabh Bachchan",
    "amitab": "Amitabh Bachchan", "hrithik": "Hrithik Roshan", "akshay": "Akshay Kumar", "ranbir": "Ranbir Kapoor",
    "ranveer": "Ranveer Singh", "madhuri": "Madhuri Dixit", "aishwarya": "Aishwarya Rai", "deepika": "Deepika Padukone",
    "priyanka": "Priyanka Chopra", "kareena": "Kareena Kapoor", "ayushmann": "Ayushmann Khurrana",
    "rajkummar": "Rajkummar Rao", "nawazuddin": "Nawazuddin Siddiqui", "irrfan": "Irrfan Khan",
    "chiranjeevi": "Chiranjeevi", "prabhas": "Prabhas", "nagarjuna": "Nagarjuna", "venkatesh": "Venkatesh",
    "balakrishna": "Nandamuri Balakrishna", "allu arjun": "Allu Arjun", "ram charan": "Ram Charan",
    "vijay deverakonda": "Vijay Deverakonda", "nani": "Nani",
}


# --------------------------------------------------------------------------- entity matching
SEQUEL_MARKERS = {"2", "3", "4", "5", "6", "ii", "iii", "iv", "v"}
# NLTK lists "don" (from "don't") as a stop word, but queries tokenise "don't" as "dont", so a bare "don" is
# the film or the character: "not don 2" must be able to match the title "Don 2"
MATCH_STOP = set(STOPWORDS) - {"don"}
SEQUEL_TITLE_RX = re.compile(r"\s(?:\d{1,2}|ii|iii|iv)(?:\s|:|$)")     # "dhoom 3", "baahubali 2: the conclusion"


@dataclass
class Entity:
    name: str
    kind: str                      # "person" | "character" | "title"
    docs: set = field(default_factory=set)
    key: str = ""


@dataclass
class EntityMatch:
    entity: Entity
    surface: str                   # the words in the query that matched
    start: int                     # token offsets in the query
    end: int
    score: float                   # 1.0 exact key match, lower for fuzzy / partial
    how: str                       # "exact", "fuzzy", "partial", "alias"

    @property
    def corrected(self) -> bool:
        return fold(self.surface).replace(" ", "") != fold(self.entity.name).replace(" ", "")


class EntityMatcher:
    """Finds people, characters and titles in a query despite transliteration variants."""

    def __init__(self):
        self.entities: list[Entity] = []
        self._by_name: dict[tuple[str, str], int] = {}
        self.full_key: dict[str, list[int]] = defaultdict(list)
        self.token_key: dict[str, list[int]] = defaultdict(list)
        self._full_keys: list[str] = []

    def add(self, name: str, kind: str, doc: int) -> None:
        name = re.sub(r"\s+", " ", name).strip()
        if len(name) < 2:
            return
        k = (fold(name), kind)
        idx = self._by_name.get(k)
        if idx is None:
            idx = len(self.entities)
            self._by_name[k] = idx
            self.entities.append(Entity(name, kind, set(), phonetic_key(name)))
        self.entities[idx].docs.add(doc)

    def finalize(self, common_tokens: set[str], proper: set[str] | None = None,
                 english: set[str] | None = None) -> None:
        """Build key indexes.

        common_tokens  ordinary words (mostly lower-case in prose, or in 15+ plots): an entity made only of
                       them is generic ("Doctor", "Police Officer" in a cast list, or a film called "Murder")
        proper         words usually capitalised in running prose (names, places)
        english        whole English words from the sentence encoder's WordPiece vocabulary; an unknown
                       word that is English ("bubbly") is not a misspelled name and is never fuzzy-matched
        """
        self.generic: set[int] = set()
        self.common = set(common_tokens)
        self.proper = set(proper or ())
        self.english = set(english or ())
        ordinary = self.common - self.proper            # frequent *and* usually lower-case: "inspector", "doctor"
        for i, e in enumerate(self.entities):
            toks = [t for t in re.findall(r"[a-z0-9]+", fold(e.name)) if t not in STOPWORDS]
            if e.kind == "character":
                # cast lists say "X as Inspector Vijay Khanna" or "X as Pakistani Judge": keep only the name part
                toks = [t for t in toks if t not in ordinary]
                if not toks:
                    self.generic.add(i)
                    continue
                e.key = phonetic_key("".join(toks))
            generic = not toks or all(t in ordinary or t.isdigit() for t in toks)
            if generic:
                self.generic.add(i)
            if len(e.key) >= 3:
                self.full_key[e.key].append(i)
            if e.kind == "title" and re.search(r"\d", e.key) and not SEQUEL_TITLE_RX.search(fold(e.name)):
                # "3 Idiots", "1942: A Love Story": the number is part of the name, not a sequel number, so the
                # title is also found without it (a typed "1942" may already have been read as a year)
                alias = phonetic_key(re.sub(r"\d+", " ", e.name))
                if len(alias) >= 3 and i not in self.full_key[alias]:
                    self.full_key[alias].append(i)
            if len(toks) > 1 and e.kind != "title":          # partial (one-word) matches: people & characters only
                for t in toks:
                    k = phonetic_key(t)
                    if len(k) >= 4 and t not in common_tokens:
                        self.token_key[k].append(i)
        self._full_keys = [k for k in self.full_key if len(k) >= 5]
        self._by_initial: dict[str, list[str]] = defaultdict(list)
        for k in self._full_keys:
            self._by_initial[k[0]].append(k)

    def resolve_alias(self, alias: str) -> list[int]:
        target = ALIASES.get(alias)
        if not target:
            return []
        hit = self.full_key.get(phonetic_key(target))
        if hit:
            return hit
        persons = [(e.name, i) for i, e in enumerate(self.entities) if e.kind == "person"]
        m = process.extractOne(target, [p[0] for p in persons], scorer=fuzz.WRatio, score_cutoff=88)
        return [persons[m[2]][1]] if m else []

    def match(self, tokens: list[str], vocab: set[str]) -> list[EntityMatch]:
        """Greedy longest-first matching over query token windows (4 down to 1 token).

        Single tokens are matched fuzzily only when they are out of vocabulary (a word that
        never occurs in the corpus is likely a misspelled name), which keeps ordinary words
        such as "king" or "rain" from being read as names.
        """
        out: list[EntityMatch] = []
        used: set[int] = set()
        n = len(tokens)
        for size in (4, 3, 2, 1):
            for i in range(n - size + 1):
                span = range(i, i + size)
                if used.intersection(span):
                    continue
                win = tokens[i:i + size]
                if win[0] in MATCH_STOP or win[-1] in MATCH_STOP or win[0] in QUERY_FILLER:
                    continue
                surface = " ".join(win)
                hits, score, how = [], 1.0, "exact"
                alias = self.resolve_alias(surface)
                if alias:
                    hits, how = alias, "alias"
                if not hits:
                    key = phonetic_key("".join(win))
                    oov = size > 1 or win[0] not in vocab
                    if size == 1:
                        # A single word is read as a name only if it is (a) the full name of a mononymous
                        # person ("kajol", "tabu"), (b) a rare proper noun of the corpus, or (c) unknown to
                        # the corpus and not an English word (a likely misspelling: "gabar", "mogembo").
                        tok = win[0]
                        if len(key) < 4 or tok in QUERY_FILLER:
                            continue
                        mono = [h for h in self.full_key.get(key, []) if self.entities[h].kind == "person"]
                        name_like = tok in getattr(self, "proper", ()) and tok not in getattr(self, "common", ())
                        unknown_name = tok not in vocab and tok not in getattr(self, "english", ())
                        if mono:
                            hits = mono
                        elif not (name_like or unknown_name):
                            continue
                    if not hits:
                        hits = self.full_key.get(key, [])
                    if not hits and len(key) >= 6 and oov and (size > 1 or win[0] not in getattr(self, "english", ())):
                        m = process.extractOne(key, self._by_initial.get(key[0], []), scorer=fuzz.ratio,
                                               score_cutoff=88 if size > 1 else 85)
                        if m:
                            hits, score, how = self.full_key[m[0]], m[1] / 100.0, "fuzzy"
                    if not hits and size == 1 and len(key) >= 5:
                        partial = self.token_key.get(key, [])
                        if partial and (win[0] not in vocab or len(partial) <= 6):
                            hits, score, how = partial, 0.6, "partial"
                if hits and i + size < n and (tokens[i + size] in SEQUEL_MARKERS or (
                        tokens[i + size] == "part" and i + size + 1 < n and tokens[i + size + 1] in SEQUEL_MARKERS)):
                    # "not Race 2": a title followed by a sequel number names another film (longer titles that
                    # exist, such as "Dhoom 3", were already matched by the larger windows)
                    hits = [h for h in hits if self.entities[h].kind != "title"]
                if hits:
                    # prefer people over characters with the same key, and down-weight generic titles
                    hits = sorted(hits, key=lambda h: self.entities[h].kind != "person")
                    for h in hits[:12]:
                        s = score * (0.35 if h in getattr(self, "generic", ()) else 1.0)
                        out.append(EntityMatch(self.entities[h], surface, i, i + size, s, how))
                    used.update(span)
        return out


# --------------------------------------------------------------------------- spelling correction
class SpellingCorrector:
    """Out-of-vocabulary word -> most frequent corpus word with the same (or a very close) phonetic key.

    Only proper nouns (words usually capitalised in the corpus: names, places, Hindi words in plots)
    are allowed as targets, since transliteration variants are what this is meant to fix."""

    def __init__(self, token_counts: Counter, proper: set[str] | None = None, min_count: int = 2,
                 english: set[str] | None = None):
        self.counts = token_counts
        self.english = set(english or ())
        self.by_key: dict[str, list[str]] = defaultdict(list)
        for tok, c in token_counts.items():
            if c >= min_count and tok.isalpha() and len(tok) >= 4 and (proper is None or tok in proper):
                if len(phonetic_key(tok)) >= 4:
                    self.by_key[phonetic_key(tok)].append(tok)
        for k in self.by_key:
            self.by_key[k].sort(key=lambda t: -self.counts[t])
        self._keys = list(self.by_key)

    def correct(self, token: str) -> tuple[str, float] | None:
        if token in self.counts or len(token) < 4 or not token.isalpha() or token in getattr(self, "english", ()):
            return None
        key = phonetic_key(token)
        cands = self.by_key.get(key)
        if cands:
            return cands[0], 0.9
        if len(key) < 5:
            return None
        m = process.extractOne(key, self._keys, scorer=fuzz.ratio, score_cutoff=86)
        if m and abs(len(m[0]) - len(key)) <= 2 and m[0][0] == key[0]:
            return self.by_key[m[0]][0], 0.7
        return None
