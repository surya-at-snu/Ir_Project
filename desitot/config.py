"""Paths and default hyper-parameters shared by the scripts, the engine and the web app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CORPUS = DATA / "corpus" / "films.json"
POSTERS = DATA / "posters"
INDEX_DIR = DATA / "index"
QUERIES = DATA / "queries"
RESULTS = ROOT / "results"

# Sentence-embedding model used for dense retrieval (384-d, retrieval-tuned, runs on CPU).
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# BM25F: per-field weights and length-normalisation strengths (Robertson et al., 2004).
FIELD_WEIGHTS = {"title": 2.0, "lead": 1.0, "plot": 1.0, "cast": 0.8, "people": 0.8,
                 "genre": 0.6, "music": 0.4, "other": 0.3}
FIELD_B = {"title": 0.3, "lead": 0.6, "plot": 0.75, "cast": 0.5, "people": 0.3,
           "genre": 0.2, "music": 0.6, "other": 0.75}
BM25_K1 = 1.2

# Clause weights (defaults before learning; the evaluation script learns them by cross-validation).
DEFAULT_WEIGHTS = {
    "plain": 1.0,       # c_j for ordinary clauses
    "hedged": 0.3,      # c_j for clauses with "I think", "maybe", ...
    "confident": 1.2,   # c_j for clauses with "definitely", "I clearly remember", ...
    "meta": 0.0,        # social niceties / previous search attempts ("please help", "google failed")
    "mu": 0.5,          # penalty weight for exclusion clauses ("not a comedy")
}

CANDIDATES = 100        # candidate pool re-ranked by the learning-to-rank model
