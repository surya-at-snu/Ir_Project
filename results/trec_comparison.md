| System | Queries | Corpus | nDCG@10 | MRR | nDCG@1000 |
|---|---|---|---|---|---|
| TREC 2025: BM25 (PyTerrier) | 622 (2025 test) | 6,407,814 Wikipedia articles | 0.122 | 0.111 | 0.170 |
| TREC 2025: BM25 (Anserini) | 622 (2025 test) | 6,407,814 Wikipedia articles | 0.076 | 0.072 | 0.103 |
| TREC 2025: dense (Lightning IR) | 622 (2025 test) | 6,407,814 Wikipedia articles | 0.019 | 0.016 | 0.051 |
| TREC 2025: best run (scrb-tot-04) | 622 (2025 test) | 6,407,814 Wikipedia articles | 0.658 | 0.626 | 0.682 |
| Desi ToT [detailed]: BM25F (baseline) | 830 | 1249 film articles | 0.993 | 0.992 | 0.994 |
| Desi ToT [detailed]: Dense bi-encoder (bge-small, MaxP) | 830 | 1249 film articles | 0.856 | 0.835 | 0.871 |
| Desi ToT [detailed]: Full LTR: coordinate ascent (ours) | 830 | 1249 film articles | 0.991 | 0.988 | 0.991 |
| Desi ToT [vague]: BM25F (baseline) | 830 | 1249 film articles | 0.882 | 0.865 | 0.894 |
| Desi ToT [vague]: Dense bi-encoder (bge-small, MaxP) | 830 | 1249 film articles | 0.663 | 0.624 | 0.701 |
| Desi ToT [vague]: Full LTR: coordinate ascent (ours) | 830 | 1249 film articles | 0.932 | 0.917 | 0.936 |
| Desi ToT [vague+noise]: BM25F (baseline) | 830 | 1249 film articles | 0.560 | 0.460 | 0.585 |
| Desi ToT [vague+noise]: Dense bi-encoder (bge-small, MaxP) | 830 | 1249 film articles | 0.418 | 0.365 | 0.488 |
| Desi ToT [vague+noise]: Full LTR: coordinate ascent (ours) | 830 | 1249 film articles | 0.846 | 0.819 | 0.858 |
