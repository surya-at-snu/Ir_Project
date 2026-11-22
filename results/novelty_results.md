# Evaluation of the evidence capabilities (C1-C4)

Base = the deployed ranker; every row compares the same queries with and without the capability.

## C1: borrowed memories

| Query set | n | MRR base | MRR +C1 | S@1 base | S@1 +C1 | p | Flagged queries | Precision | Recall | F1 | Source named correctly |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean_vague | 810 | 0.921 | 0.919 | 0.884 | 0.883 | 0.346 | 0.038 | — | — | — | — |
| clean_detailed | 812 | 0.986 | 0.984 | 0.974 | 0.972 | 0.378 | 0.106 | — | — | — | — |
| hedged_false | 830 | 0.774 | 0.790 | 0.649 | 0.682 | 0.038 | 0.871 | 0.889 | 0.651 | 0.751 | 0.825 |
| plain_false | 830 | 0.680 | 0.663 | 0.512 | 0.508 | 0.031 | 0.840 | 0.608 | 0.371 | 0.461 | 0.732 |
| certain_false | 830 | 0.656 | 0.649 | 0.481 | 0.484 | 0.362 | 0.828 | 0.608 | 0.363 | 0.455 | 0.740 |
| maybe_true_actor | 830 | 0.941 | 0.945 | 0.905 | 0.907 | 0.058 | 0.073 | — | — | — | — |
| certain_true_actor | 830 | 0.966 | 0.967 | 0.941 | 0.943 | 0.676 | 0.073 | — | — | — | — |

## C2: remakes and dubs

| Variant | System | n | Story@1 | Exact@1 | Both in top 2 | Credit | Says ambiguous | MRR |
|---|---|---|---|---|---|---|---|---|
| original | base | 33 | 0.879 | 0.848 | 0.273 | 0.848 | 0.000 | 0.878 |
| original | c2 | 33 | 0.879 | 0.848 | 0.303 | 0.848 | 0.182 | 0.878 |
| original | full | 33 | 0.879 | 0.848 | 0.303 | 0.848 | 0.182 | 0.878 |
| plot_only | base | 33 | 0.818 | 0.667 | 0.394 | 0.758 | 0.000 | 0.758 |
| plot_only | c2 | 33 | 0.818 | 0.667 | 0.485 | 0.788 | 0.455 | 0.763 |
| plot_only | full | 33 | 0.818 | 0.667 | 0.485 | 0.788 | 0.455 | 0.763 |
| plot+language_heard | base | 33 | 0.848 | 0.727 | 0.424 | 0.727 | 0.000 | 0.801 |
| plot+language_heard | c2 | 33 | 0.848 | 0.818 | 0.424 | 0.818 | 0.000 | 0.847 |
| plot+language_heard | full | 33 | 0.848 | 0.818 | 0.455 | 0.818 | 0.030 | 0.854 |
| plot+lead_actor | base | 33 | 0.909 | 0.758 | 0.333 | 0.758 | 0.000 | 0.856 |
| plot+lead_actor | c2 | 33 | 0.909 | 0.909 | 0.394 | 0.909 | 0.030 | 0.947 |
| plot+lead_actor | full | 33 | 0.909 | 0.909 | 0.394 | 0.909 | 0.000 | 0.949 |
| plot+other_language_heard | base | 31 | 0.806 | 0.677 | 0.484 | 0.774 | 0.000 | 0.771 |
| plot+other_language_heard | c2 | 31 | 0.806 | 0.000 | 0.774 | 0.774 | 0.032 | 0.432 |
| plot+other_language_heard | full | 31 | 0.806 | 0.000 | 0.774 | 0.774 | 0.032 | 0.432 |
| telugu_dubbed_into_hindi+heard_hindi | base | 61 | nan | 0.918 | nan | nan | nan | 0.929 |
| telugu_dubbed_into_hindi+heard_hindi | c2 | 61 | nan | 0.918 | nan | nan | nan | 0.929 |

## C3: event order

| Measure | Base | +C3 |
|---|---|---|
| Pairwise order accuracy (rank) | 0.501 | 0.510 |
| Pairwise order accuracy (score) | 0.525 | 0.972 |
| MRR, events in plot order | 0.987 | 0.987 |
| MRR, events reversed | 0.987 | 0.980 |
| MRR, ordinary vague queries | 0.921 | 0.921 |

Order evidence prefers the plot order for the target in 0.995 of 772 pairs.

## C4: one question (simulated viewers)

| Query set | Policy | Viewer knowledge | Ask rate | Answer rate | S@1 | MRR |
|---|---|---|---|---|---|---|
| vague | no question | — | 0 | — | 0.882 | 0.919 |
| vague | fixed_decade | priors | 0.141 | 0.644 | 0.897 | 0.929 |
| vague | targeted | priors | 0.158 | 0.687 | 0.905 | 0.934 |
| vague | fixed_decade | all 0.6 | 0.141 | 0.600 | 0.896 | 0.928 |
| vague | targeted | all 0.6 | 0.158 | 0.600 | 0.904 | 0.933 |
| certain_false | no question | — | 0 | — | 0.484 | 0.649 |
| certain_false | fixed_decade | priors | 0.605 | 0.671 | 0.606 | 0.722 |
| certain_false | targeted | priors | 0.653 | 0.784 | 0.711 | 0.777 |
| certain_false | fixed_decade | all 0.6 | 0.605 | 0.600 | 0.597 | 0.716 |
| certain_false | targeted | all 0.6 | 0.653 | 0.600 | 0.657 | 0.746 |
