"""Independent, dict-based interpolated Witten-Bell trigram (test plan 2026-09-24, T1.12).

Written from the definition, with plain Python counters and floats (no numpy, no package code):

    p1(w)          = (c(w) + T0 / V) / (N0 + T0)
    p2(w | h1)     = (c(h1, w) + T(h1) p1(w)) / (N(h1) + T(h1))          or p1(w)      if N(h1) = 0
    p3(w | h2, h1) = (c(h2, h1, w) + T(h2, h1) p2(w | h1)) / (N + T)      or p2(w | h1) if N = 0

with ``N`` the token count after a context, ``T`` the number of DISTINCT continuation types after it,
V the 40 predicted symbols, and every line padded with two BOS (so no context crosses a line). BOS is
a context only, never predicted, and there is no end-of-sentence symbol.
"""

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


class WittenBellOracle:
    def __init__(self, lines: Iterable[Sequence[int]], n_types: int, bos: int):
        self.n_types, self.bos = int(n_types), int(bos)
        self.c1: Counter = Counter()
        self.c2: Dict[int, Counter] = defaultdict(Counter)
        self.c3: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        for line in lines:
            h2, h1 = self.bos, self.bos
            for w in line:
                w = int(w)
                assert 0 <= w < self.n_types
                self.c1[w] += 1
                self.c2[h1][w] += 1
                self.c3[(h2, h1)][w] += 1
                h2, h1 = h1, w
        self.n0 = sum(self.c1.values())
        self.t0 = sum(1 for v in self.c1.values() if v > 0)
        self._p1 = [self._unigram(w) for w in range(self.n_types)]

    def _unigram(self, w: int) -> float:
        if self.n0 == 0:
            return 1.0 / self.n_types
        return (self.c1[w] + self.t0 / self.n_types) / (self.n0 + self.t0)

    def p1(self, w: int) -> float:
        return self._p1[w]

    def p2(self, w: int, h1: int) -> float:
        cnt = self.c2.get(h1)
        n = sum(cnt.values()) if cnt else 0
        if n == 0:
            return self.p1(w)
        t = sum(1 for v in cnt.values() if v > 0)
        return (cnt[w] + t * self.p1(w)) / (n + t)

    def p3(self, w: int, h2: int, h1: int) -> float:
        cnt = self.c3.get((h2, h1))
        n = sum(cnt.values()) if cnt else 0
        if n == 0:
            return self.p2(w, h1)
        t = sum(1 for v in cnt.values() if v > 0)
        return (cnt[w] + t * self.p2(w, h1)) / (n + t)

    def log_prob(self, seq: Sequence[int], order: int) -> float:
        """Sum of per-token log-probabilities with two-BOS padding and no end term."""
        h2, h1, total = self.bos, self.bos, 0.0
        for w in seq:
            if order == 1:
                p = self.p1(w)
            elif order == 2:
                p = self.p2(w, h1)
            else:
                p = self.p3(w, h2, h1)
            total += math.log(p)
            h2, h1 = h1, w
        return total

    def perplexity(self, lines: Sequence[Sequence[int]], order: int) -> float:
        total, n = 0.0, 0
        for line in lines:
            total += self.log_prob(line, order)
            n += len(line)
        return math.exp(-total / max(n, 1))

    def tables(self) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        """``(log p1 [V], log p2 [V+1][V], log p3 [(V+1)^2][V])`` in the npz row layout
        (bigram row h1, trigram row h2 * (V + 1) + h1)."""
        v, n_ctx = self.n_types, self.n_types + 1
        uni = [math.log(self.p1(w)) for w in range(v)]
        bi = [[math.log(self.p2(w, h1)) for w in range(v)] for h1 in range(n_ctx)]
        tri = [[math.log(self.p3(w, h2, h1)) for w in range(v)] for h2 in range(n_ctx) for h1 in range(n_ctx)]
        return uni, bi, tri
