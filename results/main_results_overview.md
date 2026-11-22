| System | MRR detailed | MRR vague | MRR vague+noise | S@1 detailed | S@1 vague | S@1 vague+noise |
|---|---|---|---|---|---|---|
| BM25F (baseline) | 0.992 | 0.865 | 0.460 | 0.986 | 0.818 | 0.160 |
| TF-IDF cosine (VSM) | 0.976 | 0.753 | 0.349 | 0.958 | 0.655 | 0.099 |
| Dense bi-encoder (bge-small, MaxP) | 0.835 | 0.624 | 0.365 | 0.780 | 0.527 | 0.217 |
| BM25 + dense (RRF) | 0.946 | 0.813 | 0.492 | 0.919 | 0.741 | 0.251 |
| Clause-weighted BM25 (learned c_h, c_conf, c_ctx) | 0.991 | 0.865 | 0.672 | 0.984 | 0.818 | 0.507 |
| + scope-aware negation (learned mu) | 0.989 | 0.866 | 0.672 | 0.982 | 0.819 | 0.506 |
| + Hinglish spelling handling | 0.988 | 0.865 | 0.645 | 0.981 | 0.818 | 0.476 |
| Full LTR: coordinate ascent (ours) | 0.988 | 0.917 | 0.819 | 0.980 | 0.880 | 0.748 |
| Full LTR: pairwise logistic / RankNet-linear (ours) | 0.991 | 0.917 | 0.765 | 0.983 | 0.876 | 0.628 |
| Full LTR: LambdaMART (ours) | 0.991 | 0.917 | 0.830 | 0.984 | 0.875 | 0.747 |
