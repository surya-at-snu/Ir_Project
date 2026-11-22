# Evaluation queries

| File | Written by | Count |
|---|---|---|
| `llm_queries_hi.jsonl`, `llm_queries_te.jsonl`, `llm_queries_en.jsonl`, `llm_queries_added.jsonl`, `llm_queries_hindi50.jsonl`, `llm_queries_te140.jsonl`, `llm_queries_en140.jsonl`, `llm_queries_add150.jsonl` | an LLM (Claude): one *detailed* description per curated film (names, songs, dialogue) | 200 + 50 + 50 + 50 + 50 + 140 + 140 + 150 = 830 |
| `llm_vague_hi.jsonl`, `llm_vague_te_en.jsonl`, `llm_vague_added.jsonl`, `llm_vague_hindi50.jsonl`, `llm_vague_te140.jsonl`, `llm_vague_en140.jsonl`, `llm_vague_add150.jsonl` | an LLM (Claude): one *vague* description per curated film, in the style of real ToT posts | 200 + 100 + 50 + 50 + 140 + 140 + 150 = 830 |
| `human_queries.jsonl` | **your team**: empty until you add queries | 0 |

The `*_added.jsonl` files cover the 50 famous 1990–2026 films added later (30 Hindi, 10 Telugu, 10 English); the
`*_hindi50.jsonl` files cover the 50 Hindi films from 1980–2026 added after that (10 per decade, *Shaan* to
*Border 2*), written from each film's Wikipedia plot and cast list. The `*_te140` / `*_en140` files cover the 140 Telugu
and 140 English popular, highly rated 1990–2026 films added after that (Telugu ones written from the Wikipedia plots and
cast lists), and the `*_add150` files the 50 Hindi + 50 Telugu + 50 English 2000–2026 films added last, each checked
against its Wikipedia plot and cast list. Many of the added descriptions test sequel and remake confusion on purpose ("not Race 2", "not the
first or the second one", "not the 1997 original", "the Telugu remake, not Dabangg", "not Endgame, the one before it", "not Karthikeya 2", "not The Godfather").
`scripts/evaluate.py` also builds a third, synthetic condition from the vague set by injecting one hedged false
memory into each query.

The LLM queries imitate real tip-of-the-tongue requests: hedges ("I think", "maybe"), exclusions
("not the remake", "definitely not a comedy"), deliberate false memories (Sholay "in the 80s",
Mr. India "early 90s"), Hinglish words and misspelled names ("amitab bachan", "salmaan"),
social niceties and viewing context. Each query carries `tags` naming the phenomena it contains,
so results can be broken down by phenomenon.

## Adding human-written queries (needed for the LLM-vs-human comparison)

The TREC ToT 2025 track compared system rankings on LLM-written and human-elicited queries
(Kendall's τ = 0.847 vs a real query set, and 0.703 for human-elicited queries). To repeat that
check on Indian films, each team member should describe films **from memory, without looking
them up**, then record the correct answer:

1. Run the web app (`python web/app.py`), type a description, and press **Add to test set** on
   the correct film (search for it by title if it isn't in the top 5).
2. Or append lines to `human_queries.jsonl` by hand:

```json
{"qid": "hum-001", "film": "Sholay", "year": 1975, "author": "your-name", "tags": ["hedge"], "query": "two crooks hired by a retired cop to catch a dacoit..."}
```

`python scripts/evaluate.py` picks the file up automatically and reports the human set
separately, plus Kendall's τ between the system rankings on the two sets.
