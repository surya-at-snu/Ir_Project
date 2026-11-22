# Research papers for Desi ToT

This is the reading list behind the project, grouped by the part of the system each paper supports.
Section A lists the **core papers for the Literature Review** (the report asks for 5–7). Sections B–D
are the supporting references for the methodology, which you cite where you describe the method.

Papers in sections A and B were checked against their publisher, ACL Anthology, DBLP or arXiv
pages while building the project; the DOIs and arXiv IDs below come from those pages. Section C/D
entries are standard textbook references for well-known algorithms.

---

## A. Core papers for the Literature Review (tip-of-the-tongue retrieval)

### [1] Arguello et al., CHIIR 2021 — *the paper that defined the problem*
J. Arguello, A. Ferguson, E. Fine, B. Mitra, H. Zamani, F. Diaz. "Tip of the Tongue Known-Item
Retrieval: A Case Study in Movie Identification." *Proc. ACM CHIIR 2021*. DOI 10.1145/3406522.3446021 ·
arXiv:2101.07124.

- **What it does:** qualitative analysis of real "I remember this movie" requests: what people remember
  (plot, scenes, actors, context of watching), how they express **uncertainty (hedging)**, opinions,
  relative comparisons and previous failed searches; preliminary retrieval experiments.
- **Used in this project:** the clause types (plain / hedged / confident / context / small talk) and the
  idea that different statement types deserve different weights come straight from this analysis.

### [2] Bhargav, Sidiropoulos, Kanoulas, WSDM 2022 — *a large real ToT dataset*
S. Bhargav, G. Sidiropoulos, E. Kanoulas. "'It's on the tip of my tongue': A New Dataset for Known-Item
Retrieval." *Proc. ACM WSDM 2022*, pp. 48–56. DOI 10.1145/3488560.3498421.

- **What it does:** builds Reddit-TOMT (~15k query–item pairs for movies and books) from r/tipofmytongue
  with heuristics instead of paid annotation; shows dense retrieval beating lexical baselines on it.
- **Used in this project:** motivates building our own query sets cheaply, and the BM25-vs-dense comparison.

### [3] Lin, Lo, Gonzalez, Klein, Findings of EMNLP 2023 — *closest prior method*
K. Lin, K. Lo, J. E. Gonzalez, D. Klein. "Decomposing Complex Queries for Tip-of-the-tongue Retrieval."
*Findings of EMNLP 2023*, pp. 5521–5533. arXiv:2305.15053.

- **What it does:** an LLM splits a long ToT query into individual clues, each clue goes to a specialised
  retriever (text, cover image, dates), and results are combined; up to 7% relative gain in Recall@5 on a
  book-search collection.
- **Used in this project / novelty:** our clause splitter is a lightweight, rule-based, explainable version
  of this decomposition that runs without an LLM. Unlike Lin et al., we also **weight each clause by the
  searcher's certainty** and **penalise exclusions**, with the weights learned from data.

### [4] Arguello et al., TREC 2025 overview — *benchmark and comparison numbers*
J. Arguello, F. Diaz, M. Fröbe, T. E. Kim, B. Mitra. "Overview of the TREC 2025 Tip-of-the-Tongue track."
arXiv:2601.20671 (2026).

- **What it does:** 622 test queries (172 real MS-ToT movie queries, 150 human-elicited, 300 LLM-generated)
  over 6,407,814 Wikipedia articles; official metric nDCG@1000. Lists the ToT phenomena: uncertainty,
  exclusion criteria, relative comparisons, false memories, previous attempts, social niceties.
  Baselines (Table 1): BM25 PyTerrier nDCG@1000 = 0.1700, MRR = 0.1114; BM25 Anserini 0.1032 / 0.0724;
  dense Lightning IR 0.0514 / 0.0160; best run (scrb-tot-04) 0.6824 / 0.6258. System rankings on LLM
  queries correlate with real MS-ToT queries at Kendall's τ = 0.847, NIST human-elicited vs MS-ToT at τ = 0.703.
- **Used in this project:** the comparison table with an existing paper (results/trec_comparison.md),
  the metric choice (MRR, Success@k, nDCG), and the LLM-vs-human query experiment.

### [5] He, Kim, Diaz, Arguello, Mitra, SIGIR 2025 — *LLM-written ToT queries are valid test data*
Y. He, T. E. Kim, F. Diaz, J. Arguello, B. Mitra. "Tip of the Tongue Query Elicitation for Simulated
Evaluation." *Proc. ACM SIGIR 2025*. arXiv:2502.17776.

- **What it does:** an LLM-based ToT query simulator and a human elicitation interface; system rankings on
  simulated queries correlate with rankings on real community queries (Kendall's τ > 0.8).
- **Used in this project:** justifies evaluating with LLM-written queries, and the web app's
  "Add to test set" button implements a simple human elicitation interface for the team.

### [6] Gupta, Bali, Banchs, Choudhury, Rosso, SIGIR 2014 — *Hinglish / mixed-script search*
P. Gupta, K. Bali, R. E. Banchs, M. Choudhury, P. Rosso. "Query Expansion for Mixed-Script Information
Retrieval." *Proc. ACM SIGIR 2014*, pp. 677–686. DOI 10.1145/2600428.2609622.

- **What it does:** formalises mixed-script IR (Roman-script Hindi queries against documents in several
  scripts, e.g. Bollywood song lyrics) and expands queries with transliteration variants; reports ~12% MRR
  gains.
- **Used in this project:** the Hinglish layer: phonetic keys that merge spelling variants
  ("Bachchan"/"Bachan", "Deewaar"/"Diwar"), fuzzy name matching, and the Hinglish→English lexicon.

### [7] Weller, Lawrie, Van Durme, EACL 2024 — *neural retrievers ignore negation*
O. Weller, D. Lawrie, B. Van Durme. "NevIR: Negation in Neural Information Retrieval." *Proc. EACL 2024*
(Long Papers), pp. 2274–2287. arXiv:2305.07614.

- **What it does:** a benchmark of document pairs that differ only by negation; most retrievers, especially
  bi-encoders, perform near or below chance.
- **Used in this project:** the reason exclusions ("definitely not a comedy") are handled with an explicit
  lexical penalty and structured features rather than left to the dense model.

**Novelty of this project relative to [1]–[7].** None of these works targets Indian cinema or
code-mixed (Hinglish) ToT requests. We (i) build a Hindi/Telugu/English ToT test collection with three
query conditions, (ii) weight clauses by expressed certainty with weights *learned* by cross-validation,
(iii) separate film-level exclusions from plot-level negations ("scope-aware" negation), (iv) bring
mixed-script spelling handling ([6]) into ToT search, and (v) combine lexical, dense, Naive Bayes,
link-analysis and cue features in a learned ranker with per-result explanations.

---

## B. Supporting papers for specific components

| # | Reference | Where it is used |
|---|---|---|
| [8] | J. Arguello, S. Bhargav, F. Diaz, E. Kanoulas, B. Mitra. "Overview of the TREC 2023 Tip-of-the-Tongue Track." *TREC 2023*, NIST. | First ToT track (movie identification, nDCG@1000). Background. |
| [9] | J. Arguello, S. Bhargav, F. Diaz, T. E. Kim, Y. He, E. Kanoulas, B. Mitra. "Overview of the TREC 2024 Tip-of-the-Tongue Track." *TREC 2024*, NIST. | Extension to celebrities and landmarks; first use of LLM-elicited test queries. Background. |
| [10] | R. Farkas, V. Vincze, G. Móra, J. Csirik, G. Szarvas. "The CoNLL-2010 Shared Task: Learning to Detect Hedges and their Scope in Natural Language Text." *Proc. CoNLL 2010 Shared Task*, pp. 1–12 (ACL Anthology W10-3001). | Hedge cues and their scope: our hedge lexicon and clause-level hedge detection. |
| [11] | W. W. Chapman, W. Bridewell, P. Hanbury, G. F. Cooper, B. G. Buchanan. "A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries" (NegEx). *Journal of Biomedical Informatics* 34(5):301–310, 2001. | Trigger-word negation detection with a scope window: our negation cues and scope rule. |
| [12] | R. Sequiera, M. Choudhury, P. Gupta, P. Rosso, et al. "Overview of FIRE-2015 Shared Task on Mixed Script Information Retrieval." *FIRE 2015 Working Notes*, CEUR-WS Vol-1587, pp. 19–25. | Transliterated (Roman-script) search for Indian languages. |
| [13] | L. Azzopardi, M. de Rijke, K. Balog. "Building Simulated Queries for Known-Item Topics: An Analysis Using Six European Languages." *Proc. ACM SIGIR 2007*, pp. 455–462. DOI 10.1145/1277741.1277820. | Simulated known-item queries: our injected-false-memory ("vague + noise") condition. |

## C. Retrieval models, ranking and evaluation (methodology)

| # | Reference | Where it is used |
|---|---|---|
| [14] | S. Robertson, H. Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in IR* 3(4):333–389, 2009. | BM25 scoring and IDF. |
| [15] | S. Robertson, H. Zaragoza, M. Taylor. "Simple BM25 Extension to Multiple Weighted Fields." *Proc. ACM CIKM 2004*, pp. 42–49. | BM25F over title / lead / plot / cast / crew / genre / music / other fields. |
| [16] | L. Page, S. Brin, R. Motwani, T. Winograd. "The PageRank Citation Ranking: Bringing Order to the Web." Stanford InfoLab Technical Report, 1999. | PageRank on the Wikipedia link graph. |
| [17] | J. M. Kleinberg. "Authoritative Sources in a Hyperlinked Environment." *Journal of the ACM* 46(5):604–632, 1999. | HITS hubs (people's filmography pages) and authorities (films). |
| [18] | G. V. Cormack, C. L. A. Clarke, S. Büttcher. "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *Proc. ACM SIGIR 2009*, pp. 758–759. | Hybrid BM25 + dense fusion (RRF, k = 60). |
| [19] | D. Metzler, W. B. Croft. "Linear Feature-Based Models for Information Retrieval." *Information Retrieval* 10(3):257–274, 2007. | Coordinate-ascent learning to rank that directly maximises MRR (the deployed model). |
| [20] | C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, G. Hullender. "Learning to Rank Using Gradient Descent." *Proc. ICML 2005*, pp. 89–96. | RankNet: our pairwise logistic (RankNet-linear) ranker. |
| [21] | C. J. C. Burges. "From RankNet to LambdaRank to LambdaMART: An Overview." Microsoft Research Technical Report MSR-TR-2010-82, 2010. | LambdaMART (via LightGBM). |
| [22] | Z. Dai, J. Callan. "Deeper Text Understanding for IR with Contextual Neural Language Modeling." *Proc. ACM SIGIR 2019*, pp. 985–988. | MaxP: a film's dense score is its best-matching plot passage. |
| [23] | S. Xiao, Z. Liu, P. Zhang, N. Muennighoff. "C-Pack: Packaged Resources to Advance General Chinese Embedding." arXiv:2309.07597, 2023. | The BGE embedding models (we use BAAI/bge-small-en-v1.5). |
| [24] | A. McCallum, K. Nigam. "A Comparison of Event Models for Naive Bayes Text Classification." *AAAI-98 Workshop on Learning for Text Categorization*, 1998. | Multinomial Naive Bayes for genre and decade prediction. |
| [25] | K. Järvelin, J. Kekäläinen. "Cumulated Gain-Based Evaluation of IR Techniques." *ACM TOIS* 20(4):422–446, 2002. | nDCG. |
| [26] | M. D. Smucker, J. Allan, B. Carterette. "A Comparison of Statistical Significance Tests for Information Retrieval Evaluation." *Proc. ACM CIKM 2007*, pp. 623–632. | Paired randomisation test used for every p-value. |

## D. Textbook background (lecture topics)

- C. D. Manning, P. Raghavan, H. Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008 —
  tokenisation, stop words, stemming, inverted index (ch. 1–2), tf-idf and the vector space model (ch. 6),
  evaluation (ch. 8), Naive Bayes (ch. 13), link analysis (ch. 21).
- M. F. Porter. "An Algorithm for Suffix Stripping." *Program* 14(3):130–137, 1980 — the stemmer family
  (we use the Snowball/Porter2 English stemmer).

---

## Reference list in IEEE style (for the report's References section)

The screenshot of the report template was cut off before the required format; the list below uses IEEE style.
Switch the format if your instructor asked for APA or ACM.

[1] J. Arguello, A. Ferguson, E. Fine, B. Mitra, H. Zamani, and F. Diaz, "Tip of the tongue known-item retrieval: A case study in movie identification," in *Proc. ACM SIGIR Conf. Human Information Interaction and Retrieval (CHIIR)*, 2021, doi: 10.1145/3406522.3446021.

[2] S. Bhargav, G. Sidiropoulos, and E. Kanoulas, "'It's on the tip of my tongue': A new dataset for known-item retrieval," in *Proc. 15th ACM Int. Conf. Web Search and Data Mining (WSDM)*, 2022, pp. 48–56, doi: 10.1145/3488560.3498421.

[3] K. Lin, K. Lo, J. E. Gonzalez, and D. Klein, "Decomposing complex queries for tip-of-the-tongue retrieval," in *Findings of the Association for Computational Linguistics: EMNLP 2023*, 2023, pp. 5521–5533.

[4] J. Arguello, F. Diaz, M. Fröbe, T. E. Kim, and B. Mitra, "Overview of the TREC 2025 tip-of-the-tongue track," arXiv:2601.20671, 2026.

[5] Y. He, T. E. Kim, F. Diaz, J. Arguello, and B. Mitra, "Tip of the tongue query elicitation for simulated evaluation," in *Proc. 48th Int. ACM SIGIR Conf.*, 2025, arXiv:2502.17776.

[6] P. Gupta, K. Bali, R. E. Banchs, M. Choudhury, and P. Rosso, "Query expansion for mixed-script information retrieval," in *Proc. 37th Int. ACM SIGIR Conf.*, 2014, pp. 677–686, doi: 10.1145/2600428.2609622.

[7] O. Weller, D. Lawrie, and B. Van Durme, "NevIR: Negation in neural information retrieval," in *Proc. 18th Conf. European Chapter of the ACL (EACL)*, 2024, pp. 2274–2287.

[8] J. Arguello, S. Bhargav, F. Diaz, E. Kanoulas, and B. Mitra, "Overview of the TREC 2023 tip-of-the-tongue track," in *Proc. 32nd Text REtrieval Conf. (TREC)*, NIST, 2023.

[9] J. Arguello, S. Bhargav, F. Diaz, T. E. Kim, Y. He, E. Kanoulas, and B. Mitra, "Overview of the TREC 2024 tip-of-the-tongue track," in *Proc. 33rd Text REtrieval Conf. (TREC)*, NIST, 2024.

[10] R. Farkas, V. Vincze, G. Móra, J. Csirik, and G. Szarvas, "The CoNLL-2010 shared task: Learning to detect hedges and their scope in natural language text," in *Proc. 14th Conf. Computational Natural Language Learning — Shared Task*, 2010, pp. 1–12.

[11] W. W. Chapman, W. Bridewell, P. Hanbury, G. F. Cooper, and B. G. Buchanan, "A simple algorithm for identifying negated findings and diseases in discharge summaries," *J. Biomedical Informatics*, vol. 34, no. 5, pp. 301–310, 2001.

[12] R. Sequiera *et al.*, "Overview of FIRE-2015 shared task on mixed script information retrieval," in *Working Notes of FIRE 2015*, CEUR-WS vol. 1587, 2015, pp. 19–25.

[13] L. Azzopardi, M. de Rijke, and K. Balog, "Building simulated queries for known-item topics: An analysis using six European languages," in *Proc. 30th Int. ACM SIGIR Conf.*, 2007, pp. 455–462, doi: 10.1145/1277741.1277820.

[14] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.

[15] S. Robertson, H. Zaragoza, and M. Taylor, "Simple BM25 extension to multiple weighted fields," in *Proc. 13th ACM CIKM*, 2004, pp. 42–49.

[16] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank citation ranking: Bringing order to the web," Stanford InfoLab, Tech. Rep., 1999.

[17] J. M. Kleinberg, "Authoritative sources in a hyperlinked environment," *J. ACM*, vol. 46, no. 5, pp. 604–632, 1999.

[18] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in *Proc. 32nd Int. ACM SIGIR Conf.*, 2009, pp. 758–759.

[19] D. Metzler and W. B. Croft, "Linear feature-based models for information retrieval," *Information Retrieval*, vol. 10, no. 3, pp. 257–274, 2007.

[20] C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, and G. Hullender, "Learning to rank using gradient descent," in *Proc. 22nd Int. Conf. Machine Learning (ICML)*, 2005, pp. 89–96.

[21] C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An overview," Microsoft Research, Tech. Rep. MSR-TR-2010-82, 2010.

[22] Z. Dai and J. Callan, "Deeper text understanding for IR with contextual neural language modeling," in *Proc. 42nd Int. ACM SIGIR Conf.*, 2019, pp. 985–988.

[23] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, "C-Pack: Packaged resources to advance general Chinese embedding," arXiv:2309.07597, 2023.

[24] A. McCallum and K. Nigam, "A comparison of event models for naive Bayes text classification," in *Proc. AAAI-98 Workshop on Learning for Text Categorization*, 1998.

[25] K. Järvelin and J. Kekäläinen, "Cumulated gain-based evaluation of IR techniques," *ACM Trans. Information Systems*, vol. 20, no. 4, pp. 422–446, 2002.

[26] M. D. Smucker, J. Allan, and B. Carterette, "A comparison of statistical significance tests for information retrieval evaluation," in *Proc. 16th ACM CIKM*, 2007, pp. 623–632.

[27] C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge, U.K.: Cambridge Univ. Press, 2008.
