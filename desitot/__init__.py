"""Desi Tip-of-the-Tongue: known-item search for Indian (and some English) films.

Modules
  text      tokenising, case/diacritic folding, stop words, stemming
  hinglish  transliteration-aware phonetic keys, Hinglish lexicon, fuzzy entity matching
  index     inverted index with BM25F and TF-IDF cosine scoring
  query     clause splitting, hedge / negation / exclusion tagging, decade and language cues
  nb        multinomial Naive Bayes for genre and decade prediction
  graph     PageRank and HITS on the Wikipedia link graph
  dense     sentence-embedding retrieval over plot passages
  ltr       learning-to-rank (pairwise logistic RankNet-style model and LambdaMART)
  engine    the full search pipeline with explanations
  metrics   MRR, Success@k, P@k, R@k, F1@k, nDCG@k, significance tests, Kendall's tau
"""

__version__ = "1.0.0"
