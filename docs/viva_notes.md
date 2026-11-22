# Viva cheat-sheet — formulas with worked examples

All numbers below come from this project's index (N = 1,249 films, k1 = 1.2, average plot length 295 terms).

## 1. Evaluation (known-item search: exactly one relevant film per query)

Let *r* be the rank of the right film (∞ if not retrieved).

| Measure | Formula | r = 1 | r = 3 | r = 10 | r = 11 |
|---|---|---|---|---|---|
| Reciprocal rank | 1 / r | 1 | 0.333 | 0.100 | 0.091 |
| Success@10 = Recall@10 | 1 if r ≤ 10 | 1 | 1 | 1 | 0 |
| Precision@10 | (1 if r ≤ 10) / 10 | 0.1 | 0.1 | 0.1 | 0 |
| F1@10 | 2PR/(P+R) | 0.182 | 0.182 | 0.182 | 0 |
| nDCG@10 | 1 / log₂(r+1) if r ≤ 10 | 1 | 0.5 | 0.289 | 0 |
| nDCG@1000 | 1 / log₂(r+1) if r ≤ 1000 | 1 | 0.5 | 0.289 | 0.279 |

- **nDCG by hand.** DCG@k = Σᵢ relᵢ / log₂(i+1). With one relevant film at rank r, DCG = 1/log₂(r+1), and
  the ideal ranking puts it first, so IDCG = 1/log₂2 = 1. Hence nDCG = 1/log₂(r+1): rank 3 → 1/log₂4 = 0.5.
  (With graded relevance, use the gain 2^rel − 1 instead of rel.)
- **MRR** is the mean of 1/r. Ranks 1, 2, 4 and "not found": (1 + 0.5 + 0.25 + 0) / 4 = 0.4375.
  With one relevant document, **MAP = MRR**.
- **Why is P@10 at most 0.1?** Only one film is relevant, so 9 of the 10 slots are always "wrong".
  Success@k (= R@k) and MRR are the meaningful measures here, as in the TREC ToT track.
- **Accuracy** in the report = Success@1 (right film ranked first).
- **Significance:** paired randomisation test — flip the sign of each query's RR difference at random 20,000 times
  and count how often the mean difference is as large as observed. 95% CIs come from 5,000 bootstrap resamples.
- **Kendall's τ** (LLM vs human query sets): τ = (concordant − discordant) / (n(n−1)/2) over all pairs of systems.

## 2. BM25 / BM25F

idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))

| term | df | idf |
|---|---|---|
| "pianist" | 13 | ln(1 + 1236.5/13.5) = **4.528** |
| "dacoit" | 20 | ln(1 + 1229.5/20.5) = **4.110** |
| "blind" | 76 | 2.794 |
| "village" | 297 | 1.435 |
| "film" | 1,249 (every film) | ln(1 + 0.5/1249.5) ≈ **0.0004** → useless, as it should be |

BM25F first combines the fields into one pseudo-frequency, normalising each field by its own length:

tf̃(t,d) = Σ_f w_f · tf_f(t,d) / ((1 − b_f) + b_f · len_f(d)/avglen_f)

score(q,d) = Σ_t q_t · idf(t) · tf̃ (k1 + 1) / (k1 + tf̃)

The saturating part tf̃(k1+1)/(k1+tf̃) can never exceed k1 + 1 = 2.2, so repeating a word 50 times
does not dominate. Example: "dacoit" in Sholay has tf̃ ≈ 6 → saturated part 1.839 →
contribution 4.110 × 1.839 = **7.56**. Field weights: title 2.0, lead 1.0, plot 1.0, cast 0.8, crew 0.8,
genre 0.6, music 0.4, other sections 0.3.

**TF-IDF cosine (VSM):** w(t,d) = (1 + ln tf) · ln(N/df); cos(q,d) = q·d / (|q||d|). Length normalisation
stops long articles from winning just by size.

## 3. Our clause-weighted score with exclusions

s(d) = Σ_j c_j · BM25F(clause_j, d) − μ · Σ_j c′_j · BM25F(neg_j, d)

Because BM25 adds up per-term contributions, this equals one weighted BM25 query in which each word's weight is
the sum of the weights of the clauses it appears in.

Example — *"a pianist pretends to be blind and witnesses a murder, I think around 2015, definitely not a horror film"*:

| clause | type | weight |
|---|---|---|
| a pianist pretends to be blind and witnesses a murder | plain | c = 1 → pianist, pretend, blind, witness, murder |
| I think around 2015 | hedged | year cue 2013–2017 with weight c_h |
| definitely not a horror film | confident **exclusion** | "horror" penalised with μ · c_conf = 0.5 × 1.2 = 0.6 |

Andhadhun: positive BM25F = 16.4, exclusion penalty 0 (not a horror film) → ranked first.

Negation scope: a negation is an **exclusion** only when it is about the film itself ("not a comedy",
"no songs", "it wasn't…", "X wasn't in it", "not the remake"); "he can't see", "nobody believes her" and
"no means no" describe the plot and stay positive evidence. Hedge/negation cue lexicons follow CoNLL-2010 and NegEx.

## 4. Hybrid fusion (RRF)

RRF(d) = Σ_systems 1 / (60 + rank_system(d)). A film ranked 1st by BM25 and 3rd by the dense model:
1/61 + 1/63 = 0.0164 + 0.0159 = **0.0323**. Only ranks matter, so BM25 and cosine scores need no calibration.

## 5. Dense retrieval (MaxP)

dense(q,d) = max over passages p of d of cos(E(q), E(p)), with E = bge-small-en-v1.5 (384-d). It matches
paraphrases ("pretends to be blind" ≈ "feigns visual impairment") but ignores negation (NevIR, 2024).

## 6. PageRank and HITS

PageRank: r = d · Mᵀ r + (1 − d)/n · **1**, d = 0.85, M row-normalised adjacency; dangling pages spread their
rank evenly. Toy graph A→B, B→C, C→A: by symmetry r = (1/3, 1/3, 1/3).

HITS: a = Aᵀ h, h = A a, normalise both every iteration. Here people's pages (filmographies) are **hubs** and
films are **authorities**: a film linked from many important actors' pages gets a high authority score.

Graph: 1,249 films + 2,032 people, 23.3k links; top PageRank: Star Wars, Titanic, Blade Runner, Avatar, The Godfather.

## 7. Naive Bayes

P(c | q) ∝ P(c) · Π_t P(t | c)^{n_t},  P(t | c) = (N_tc + α) / (N_c + α|V|),  α = 0.5 (Laplace/Lidstone smoothing).

Example: if "ghost" appears 40 times in horror plots (N_c = 20,000 words, |V| = 10,000), P(ghost|horror) =
40.5 / 25,000 = 0.00162. NB posteriors are over-confident (correlated words are multiplied as if
independent), so we temper them: P ∝ exp(log-likelihood / T) with T = √(#query words). The predictions are
**soft filters**: films whose genre/decade the query predicts get a small bonus; nothing is removed.

## 8. Learning to rank

- **Coordinate ascent** (Metzler & Croft 2007): score = w · x over 22 standardised features; optimise one
  weight at a time by line search, keeping a change only if training MRR improves. Optimises MRR directly.
- **RankNet-linear:** minimise Σ log(1 + exp(−w·(x_pos − x_neg))) over (right film, wrong film) pairs + L2.
- **LambdaMART:** gradient-boosted trees whose gradients are weighted by the change in nDCG from swapping two films.
- **No leakage:** 5-fold cross-validation by film: weights and models are learned on the queries of four fifths of the
  films (664 per query set) and tested on the other 166.

## 9. Likely questions

- *Why not just use the dense model?* It scored lowest here (MRR 0.62 on vague queries vs 0.87 for BM25):
  plots are full of names and rare words, and dense models ignore negation. It still helps inside the learned ranker.
- *Why learn the hedge weight instead of fixing 0.3?* The best value depends on how often hedged memories are
  wrong; on clean queries hedges carry real information, on queries with hedged false memories the learned weight
  drops to 0 (see `results/hedge_sweep_*.csv`).
- *Why are the exclusion weights of the deployed ranker kept ≤ 0?* Ruling something out should only ever lower a
  film's score. Trained freely, coordinate ascent gave the BM25 exclusion feature a small positive weight, because many
  test descriptions rule out a sequel or remake whose title words also appear in the right film's article ("not
  Karthikeya 2", "not the first Drushyam"); the app then ranked a comedy first for the brief's "…definitely not a
  comedy". A sign (monotonicity) constraint on the five exclusion features fixes it: Mohra is first again.
- *Why are your scores so much higher than TREC's?* 1,249 films versus 6.4 million Wikipedia articles of all kinds.
  Compare gains over each benchmark's own BM25 baseline instead.
- *How do you know the LLM queries are realistic?* TREC 2025 and He et al. (SIGIR 2025) found system rankings on
  LLM queries correlate with real ones (τ ≈ 0.85); we repeat the check with our team's human queries.
- *How do you use the cast?* Each Wikipedia cast list is parsed into (actor, character, role) triples, including
  `{{castlist}}` templates, cast tables and dual roles ("Sridevi in a dual role as Anju / Manju"). Actor and character
  names go into the entity matcher (phonetic keys, so "gabar" finds Gabbar Singh), role descriptions ("a retired police
  officer") into the BM25F cast field, and explanations say "Amitabh Bachchan (as Jai)". Fixing the parser added
  ~1,350 character names to the first 350 curated films (5,256 → 6,600); all 830 curated films now have 16,144.
- *What about sequels?* Title keys keep their numbers, so "Dhoom 3" ≠ "Dhoom"; "not Race 2" excludes nothing (Race 2 is not
  in the index) instead of excluding *Race*, and "the first one, not Don 2" excludes only *Don 2*.
- *What is new?* An Indian-cinema ToT collection; certainty-weighted clauses learned from data; scope-aware
  exclusions; Hinglish spelling handling inside ToT search; and a learned combination of lexical, dense, Naive
  Bayes, cue and link-analysis features with per-result explanations.
