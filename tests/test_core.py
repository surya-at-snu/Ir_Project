"""Unit tests for the core algorithms (no corpus or network needed).   Run:  python -m pytest tests -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desitot.graph import adjacency, hits, pagerank  # noqa: E402
from desitot.hinglish import HINGLISH_LEXICON, phonetic_key  # noqa: E402
from desitot.index import InvertedIndex  # noqa: E402
from desitot.metrics import binary_prf, per_query  # noqa: E402
from desitot.nb import MultinomialNB  # noqa: E402
from desitot.query import parse_query  # noqa: E402
from desitot.text import analyze  # noqa: E402

W = {"hedged": 0.3, "confident": 1.2, "context": 0.5, "meta": 0.0, "mu": 0.5}


# ----------------------------------------------------------------- text & Hinglish
def test_analyze_removes_stopwords_and_stems():
    assert analyze("The doctors were directing films") == ["doctor", "direct", "film"]
    assert "movi" not in analyze("an old movie i remember", query=True)       # ToT filler removed from queries


@pytest.mark.parametrize("a,b", [("Bachchan", "Bachan"), ("Deewaar", "Diwar"), ("Sholay", "Sholey"),
                                 ("Shahrukh", "Shah Rukh"), ("Dixit", "Dikshit"), ("Zindagi", "Jindagi"),
                                 ("Qayamat", "Kayamat"), ("Dulhania", "Dulhaniya")])
def test_phonetic_key_merges_transliteration_variants(a, b):
    assert phonetic_key(a) == phonetic_key(b.replace(" ", ""))


def test_lexicon_translates_hinglish():
    assert "wedding" in HINGLISH_LEXICON["shaadi"]
    assert "blind" in HINGLISH_LEXICON["andha"]


def test_phonetic_key_keeps_sequel_numbers():
    assert phonetic_key("Dhoom 3") != phonetic_key("Dhoom")              # a sequel is a different film
    assert phonetic_key("Dhoom 3") == phonetic_key("dhum3")


def test_entity_matcher_sequels():
    from desitot.hinglish import EntityMatcher
    m = EntityMatcher()
    for i, t in enumerate(["Race", "Dhoom", "Dhoom 3", "Don", "Don 2", "3 Idiots"]):
        m.add(t, "title", i)
    m.finalize(set())
    found = lambda q: {x.entity.name for x in m.match(q.split(), set(q.split()))}
    assert found("not dhoom 3") == {"Dhoom 3"}
    assert found("not race 2") == set()                                   # "Race 2" is not "Race"
    assert found("not don 2") == {"Don 2"}                                # "don" is an NLTK stop word
    assert found("3 idiots") == {"3 Idiots"}
    assert 5 in m.full_key[phonetic_key("Idiots")]                        # the number is part of the name: alias
    assert 2 not in m.full_key[phonetic_key("Dhoom")]                     # a sequel number is not: no alias


# ----------------------------------------------------------------- cast lists
def test_cast_parsing_roles_dual_roles_and_tables():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import mwparserfromhell as mw
    from build_corpus import cast_pairs, role_display, role_names, table_pairs
    text = "\n".join([
        "Amjad Khan as Gabbar Singh, a dacoit",
        "Sridevi in a dual role of identical twin sisters as",
        "Manju Prajapati – Suraj's girlfriend",
        "Anju Pandyekar – Jackie's girlfriend",
        "Mark Hamill as Luke Skywalker: A young adult raised by his aunt and uncle on Tatooine, who dreams of "
        "something more than his current life and joins the rebellion against the Galactic Empire",
        "Uncredited members of the cast include Tom Rosqui as Rocco Lampone, one of Michael's enforcers",
    ])
    pairs = cast_pairs(text)
    assert ("Amjad Khan", "Gabbar Singh, a dacoit") in pairs
    assert [p[0] for p in pairs[1:3]] == ["Sridevi", "Sridevi"]           # both twins belong to Sridevi
    assert any(a == "Mark Hamill" and r.startswith("Luke Skywalker") for a, r in pairs)   # long role kept
    assert ("Tom Rosqui", "Rocco Lampone") in pairs                       # "X as Y" inside prose
    assert role_display("Gabbar Singh, a dacoit") == ("Gabbar Singh", "a dacoit")
    assert role_names('Jaidev "Jai"') == ["Jai", "Jaidev"]
    assert role_names("Mark Donald aka Don") == ["Mark Donald", "Don"]
    assert role_names("twins Robert and Albert") == ["Robert", "Albert"]
    table = mw.parse('{| class="wikitable"\n! Actor !! Role !! Notes\n|-\n| Rishi Kapoor || Monty || a rock star\n|}')
    assert table_pairs(table) == [("Rishi Kapoor", "Monty, a rock star")]


# ----------------------------------------------------------------- clause analysis
def kinds(q):
    return [c.kind for c in parse_query(q).clauses]


def test_clause_types():
    qa = parse_query("Old Hindi film, maybe the 90s, the hero pretends to be blind, "
                     "I think the heroine was a doctor, definitely not a comedy.")
    assert [c.kind for c in qa.clauses] == ["plain", "hedged", "plain", "hedged", "exclusion"]
    pos, neg = qa.term_weights(W)
    assert pos["blind"] == 1.0 and pos["doctor"] == pytest.approx(0.3)       # hedged clause down-weighted
    assert neg == {"comedi": pytest.approx(0.5 * 1.2)}                         # confident exclusion: mu * c_conf


def test_negation_scope_exclusion_vs_plot():
    assert kinds("He can't see and nobody believes him") == ["plain"]          # plot negation stays positive
    assert kinds("I don't think Shah Rukh was in it") == ["exclusion"]          # hedged exclusion of a person
    assert kinds("not the hindi remake") == ["exclusion"]
    assert kinds("there were no songs in it") == ["exclusion"]


def test_negation_traps_are_not_exclusions():
    assert "exclusion" not in kinds("an old lawyer defends them, no means no")
    assert "exclusion" not in kinds("a therapist keeps saying it's not your fault")


def test_meta_and_context_clauses():
    qa = parse_query("please help!! been searching for years. watched it on doordarshan as a kid in the 80s")
    assert [c.kind for c in qa.clauses] == ["meta", "meta", "context"]
    (y, _), = qa.year_cues(W)
    assert (y.lo, y.hi, y.mode) == (1980, 1989, "upper")                        # "watched it in the 80s": made then or earlier


def test_decade_and_year_cues():
    qa = parse_query("early 90s i think, definitely not the 80s, it wasn't in black and white")
    modes = sorted((y.lo, y.hi, y.mode) for c in qa.clauses for y in c.years)
    assert (1990, 1994, "prefer") in modes and (1980, 1989, "exclude") in modes and (1930, 1964, "exclude") in modes


# ----------------------------------------------------------------- index & metrics
def test_bm25_prefers_rare_terms_and_cosine_is_bounded():
    docs = [{"title": "", "plot": "a dacoit attacks a village"},
            {"title": "", "plot": "a village wedding in a village"},
            {"title": "", "plot": "a doctor in the city"}]
    idx = InvertedIndex({"title": 2.0, "plot": 1.0}, {"title": 0.3, "plot": 0.75}).build(docs)
    assert idx.idf["dacoit"] > idx.idf["villag"]
    s = idx.bm25({"dacoit": 1.0, "villag": 1.0})
    assert int(np.argmax(s)) == 0
    c = idx.tfidf_cosine({"doctor": 1.0})
    assert int(np.argmax(c)) == 2 and c.max() <= 1.0 + 1e-6


def test_known_item_metrics():
    m = per_query(1)
    assert m["rr"] == 1 and m["s@1"] == 1 and m["ndcg10"] == 1
    m = per_query(3)
    assert m["rr"] == pytest.approx(1 / 3) and m["s@1"] == 0 and m["s@5"] == 1
    assert m["ndcg10"] == pytest.approx(1 / math.log2(4))
    assert m["p@10"] == pytest.approx(0.1) and m["f1@10"] == pytest.approx(2 * 0.1 * 1 / 1.1)
    assert per_query(math.inf)["rr"] == 0
    assert binary_prf([True, False, True], [True, True, False])["precision"] == pytest.approx(0.5)


# ----------------------------------------------------------------- link analysis & Naive Bayes
def test_pagerank_and_hits():
    # 0 and 1 both link to 2; 2 links back to 0; 3 is dangling and links nowhere
    A = adjacency(4, [(0, 2), (1, 2), (2, 0)])
    pr = pagerank(A)
    assert pr.sum() == pytest.approx(1.0) and int(np.argmax(pr)) in (0, 2)
    hub, auth = hits(A)
    assert int(np.argmax(auth)) == 2 and hub[0] > hub[3]


def test_naive_bayes():
    docs = [["ghost", "haunt", "night"], ["ghost", "spirit", "scream"], ["love", "wed", "song"], ["love", "romanc", "song"]] * 3
    labels = [["horror"], ["horror"], ["romance"], ["romance"]] * 3
    nb = MultinomialNB(alpha=0.5).fit(docs, labels)
    assert nb.predict([["ghost", "scream"]]) == ["horror"]
    p = nb.predict_proba([["love", "song"]])[0]
    assert p[nb.classes_.index("romance")] > 0.5 and p.sum() == pytest.approx(1.0)


# ----------------------------------------------------------------- learning to rank
def test_vectorised_mrr_matches_loop():
    from desitot.ltr import _mrr, _StackedGroups
    rng = np.random.default_rng(0)
    groups = []
    for g in range(40):
        n = 0 if g == 5 else 30
        X = rng.normal(size=(n, 6))
        X[:, 2] = np.round(X[:, 2])                                       # ties
        y = np.zeros(n)
        if n and g % 7:
            y[int(rng.integers(0, n))] = 1                               # some groups have no relevant film
        groups.append((X, y))
    stacked = _StackedGroups(groups)
    for w in [rng.normal(size=6) for _ in range(50)] + [np.zeros(6)]:
        assert stacked.mrr(w) == pytest.approx(_mrr(groups, [X @ w for X, _ in groups]), abs=1e-12)


def test_no_period_ends_a_sentence_unless_a_number_follows():
    qa = parse_query("three women are accused, an old lawyer defends them. no means no. I watched it in Hindi.")
    last = qa.clauses[-1]
    assert last.kind != "exclusion" and ("hi", False) in last.langs       # Hindi is heard, not ruled out
    assert len(parse_query("he wears jersey No. 7 and wins the final").clauses) == 1


# ----------------------------------------------------------------- evidence layer (C1-C4)
def test_withdrawn_clause_carries_no_weight():
    qa = parse_query("a dacoit attacks a village, I'm certain there was a ghost in the haveli")
    qa.clauses[1].trust = 0.0
    pos, _ = qa.term_weights(W)
    assert "dacoit" in pos and "ghost" not in pos
    assert qa.clauses[1].cue_weight(W) == 0.0 and qa.clauses[0].cue_weight(W) > 0


def test_actor_and_plot_clues():
    from desitot.conflicts import make_clues
    from desitot.hinglish import EntityMatcher
    m = EntityMatcher()
    m.add("Akshay Kumar", "person", 3)
    m.add("Venkatesh", "person", 4)
    m.finalize(set())
    qa = parse_query("a cable tv operator hides a crime using tricks from films. I'm certain Akshay Kumar was in it.",
                     m, None, {"akshay", "kumar", "cable", "operator"})
    clues = make_clues(qa)
    assert [k.kind for k in clues] == ["plot", "actor"]
    assert clues[1].docs == {3} and clues[1].certainty == "confident"


def test_story_clusters_and_dubs():
    from desitot.versions import VersionIndex, dub_languages
    films = [
        {"id": 0, "title": "Arjun Reddy", "year": 2017, "lang": "te", "lead": "The film was remade in Hindi as Kabir Singh (2019) and in Tamil as Adithya Varma (2019).", "other": ""},
        {"id": 1, "title": "Kabir Singh", "year": 2019, "lang": "hi", "lead": "A remake of Vanga's own Telugu film Arjun Reddy (2017), it stars Shahid Kapoor.", "other": ""},
        {"id": 2, "title": "Drishyam 2", "year": 2022, "lang": "hi", "lead": "It is a remake of the 2021 Malayalam film of the same name and sequel to the 2015 film Drishyam.", "other": ""},
        {"id": 3, "title": "Drishyam", "year": 2015, "lang": "hi", "lead": "A thriller.", "other": ""},
        {"id": 4, "title": "Aashiqui", "year": 1990, "lang": "hi", "lead": "A musical romance.", "other": ""},
        {"id": 5, "title": "Ishq", "year": 2012, "lang": "te", "lead": "It was remade in Bengali as Aashiqui in 2015. It was dubbed into Hindi and Tamil.", "other": ""},
    ]
    vi = VersionIndex.build(films)
    assert vi.same_story(0, 1)                       # remake pair
    assert not vi.same_story(2, 3)                   # a sequel is not a version
    assert not vi.same_story(4, 5)                   # "remade in Bengali as Aashiqui" names another film
    assert vi.dubs[5] == ["hi", "ta"]
    assert dub_languages("a Hindi-dubbed version was released on television") == ["hi"]


def test_event_pairs_follow_order_markers():
    from desitot.order import event_pairs
    qa = parse_query("a man loses his memory after an attack and then he tattoos clues all over his body")
    assert event_pairs(qa) == [(0, 1)]
    qa = parse_query("the hero wins the big race at the end, before that he trains with an old coach")
    assert event_pairs(qa) == [(1, 0)]


def test_information_gain():
    from desitot.questions import information_gain
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert information_gain(p, [[0], [1], [2], [3]]) == pytest.approx(2.0)
    assert information_gain(p, [[0, 1], [2, 3]]) == pytest.approx(1.0)
    assert information_gain(p, [[0, 1, 2, 3]]) == pytest.approx(0.0)


def test_coordinate_ascent_keeps_exclusion_weights_nonpositive():
    from desitot.ltr import CoordinateAscent
    rng = np.random.default_rng(1)
    groups = []
    for _ in range(60):                       # feature "excl" is *higher* for the right film: unconstrained CA likes it
        X = rng.normal(size=(20, 3))
        y = np.zeros(20)
        y[0] = 1
        X[0, 0] += 1.0
        X[0, 2] += 2.0
        groups.append((X, y))
    names = ["signal", "noise", "excl"]
    free = CoordinateAscent(names).fit(groups, init={"signal": 1.0})
    capped = CoordinateAscent(names).fit(groups, init={"signal": 1.0}, nonpositive=("excl",))
    assert free.w[2] > 0
    assert capped.w[2] <= 0 and capped.w[0] > 0
