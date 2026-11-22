"""Link analysis on the Wikipedia hyperlink graph: PageRank (Page et al., 1999) and HITS (Kleinberg, 1999).

Nodes are film articles and the articles of people credited on at least two films.
Edges: film -> film hyperlinks, film -> person credits (Wikidata cast/crew), and
person -> film hyperlinks (an actor's article linking to the films they appeared in).
Person pages behave like *hubs* (filmographies) and films like *authorities*, which is
exactly the structure HITS was designed for.

PageRank:   r = d * M^T r + (1 - d) / n * 1      (M row-stochastic; dangling rows spread uniformly)
HITS:       a = A^T h,   h = A a,   both L2-normalised every iteration
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def adjacency(n: int, edges: list[tuple[int, int]]) -> sparse.csr_matrix:
    if not edges:
        return sparse.csr_matrix((n, n))
    src, dst = zip(*edges)
    A = sparse.csr_matrix((np.ones(len(edges)), (src, dst)), shape=(n, n))
    A.data[:] = 1.0                       # collapse duplicate edges
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def pagerank(A: sparse.csr_matrix, d: float = 0.85, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    n = A.shape[0]
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, out_deg, out=np.zeros(n), where=out_deg > 0)
    M = sparse.diags(inv) @ A                       # row-stochastic for non-dangling nodes
    dangling = out_deg == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = d * (M.T @ r + r[dangling].sum() / n) + (1 - d) / n
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r / r.sum()


def hits(A: sparse.csr_matrix, tol: float = 1e-10, max_iter: int = 500) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    h = np.ones(n) / np.sqrt(n)
    a = h.copy()
    for _ in range(max_iter):
        a_new = A.T @ h
        a_new /= np.linalg.norm(a_new) or 1.0
        h_new = A @ a_new
        h_new /= np.linalg.norm(h_new) or 1.0
        if np.abs(a_new - a).sum() + np.abs(h_new - h).sum() < tol:
            a, h = a_new, h_new
            break
        a, h = a_new, h_new
    return h, a
