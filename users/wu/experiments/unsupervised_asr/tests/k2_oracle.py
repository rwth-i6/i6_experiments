"""Brute-force oracle for the k2 lexicon term (test plan 2026-09-24, T1.18-T1.23).

The fixture: phones A, B, C, SIL (``n_phones`` 4, ``sil_id`` 3); words a=[A], ab=[A,B], bc=[B,C],
x=[C], y=[C] (the homophones x / y and the prefix a < ab force disambiguation symbols); a tiny
NORMALISED back-off word LM of order 3 (and an order-4 extension) over
{<s>, </s>, <unk>, a, ab, bc, x, y}; escape phone probabilities [0.3, 0.3, 0.2, 0.2].

INDEPENDENCE.  Nothing here calls the package's graph or scoring code.  The package is used only in
:func:`build_resources` / :func:`build_hlg`, which build the fixture's input files through the
PRODUCTION build path (``lexlat.parse_arpa_word_lm``, ``lexlat.build_trie``,
``lexlat.derange_pronunciations``, ``lexlat.save_resources``, ``lexlat_k2.main(["build-hlg",...])``)
-- that path is what is under test.  The oracle reads the resulting npz with ``numpy.load`` and
states the graph's intended semantics itself:

* :func:`enumerate_paths` -- every path of L o G for a token string ``y``, one tuple per path,
  written from the Design's description (START -> LOOP / SILST, SILST -SIL-> LOOP, words read
  through explicit G arcs, ``#0`` back-off moves, the escape phone loop), never from ``L``'s arcs.
* :class:`ArpaScorer` -- an ARPA text read into dicts and the textbook back-off recursion.
* :func:`proper_paths` -- the same parse space priced by the exact (single-route) back-off LM.
* :func:`brute_force` -- ``Z = sum_{a in 4^T} exp(sum_t e_t(a_t) / tau) W_tau(B(a))`` in float64
  torch, so autograd yields the exact gradient.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import os
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------------------------------------
# the fixture's constants
# ------------------------------------------------------------------------------------------------
PHONES = ("A", "B", "C", "SIL")
PHONE2ID = {p: i for i, p in enumerate(PHONES)}
N_PHONES = 4
SIL_ID = 3
SIL = "SIL"
#: the lexicon: homophones x / y and the proper prefix a < ab need disambiguation symbols
LEXICON: Dict[str, List[str]] = {"a": ["A"], "ab": ["A", "B"], "bc": ["B", "C"], "x": ["C"],
                                 "y": ["C"]}
#: the LM vocabulary, in the ARPA's unigram order (= the LM word ids)
VOCAB = ("<s>", "</s>", "<unk>", "a", "ab", "bc", "x", "y")
#: the raw escape phone log probabilities (order-1 phone prior of the escape convention)
ESCAPE_PHONE_PROB = (0.3, 0.3, 0.2, 0.2)
ESCAPE_PHONE_LOG_PROB = tuple(math.log(p) for p in ESCAPE_PHONE_PROB)
#: the escape's per-phone length term, log 0.5 (the Design's convention, stated here independently)
ESCAPE_LENGTH_LOG_PROB = math.log(0.5)
#: the optional silence probability passed to the build (``--sil-prob``)
SIL_PROB = 0.5
#: the bed's d_min / stride passed to the build
D_MIN = 2
RECOGNIZER_STRIDE = 3

# ------------------------------------------------------------------------------------------------
# the word LMs, as explicit probabilities; back-off weights are DERIVED so every context sums to 1
# ------------------------------------------------------------------------------------------------
#: unigram probabilities (sum 1; <s> is written as -99, i.e. probability 1e-99)
UNIGRAM = {"<s>": 0.0, "</s>": 0.1, "<unk>": 0.05, "a": 0.25, "ab": 0.15, "bc": 0.15,
           "x": 0.299, "y": 0.001}
#: explicit bigrams / trigrams: context -> {word: probability}.  Every trigram context is a listed
#: bigram.  (bc x) sits close to its back-off estimate (pruned at theta >= 0.5) while the trigram
#: state (a bc), whose back-off path runs through it, keeps (a bc x) / (a bc y): the S3 fixture.
LM3_EXPLICIT: Dict[Tuple[str, ...], Dict[str, float]] = {
    ("<s>",): {"a": 0.4, "ab": 0.2, "bc": 0.1},
    ("a",): {"bc": 0.5, "x": 0.1, "</s>": 0.2},
    ("ab",): {"x": 0.3, "y": 0.3},
    ("bc",): {"a": 0.45, "x": 0.17},
    ("x",): {"<unk>": 0.2, "a": 0.3},
    ("<unk>",): {"a": 0.5},
    ("<s>", "a"): {"bc": 0.7},
    ("a", "bc"): {"x": 0.6, "</s>": 0.1, "y": 0.2},
    ("ab", "x"): {"a": 0.5},
    ("x", "<unk>"): {"a": 0.6},
}
#: the order-4 extension: every 4-gram context is a listed trigram
LM4_EXPLICIT: Dict[Tuple[str, ...], Dict[str, float]] = {
    **LM3_EXPLICIT,
    ("<s>", "a", "bc"): {"x": 0.7, "y": 0.1},
    ("a", "bc", "x"): {"a": 0.4},
}


def _lm_order(explicit: Mapping[Tuple[str, ...], Mapping[str, float]]) -> int:
    return 1 + max(len(h) for h in explicit)


def derive_backoffs(explicit: Mapping[Tuple[str, ...], Mapping[str, float]],
                    unigram: Mapping[str, float] = UNIGRAM) -> Dict[Tuple[str, ...], float]:
    """``alpha(h)`` (a probability ratio) for every context with explicit continuations.

    ``alpha(h) = (1 - sum_{w in E(h)} p(w|h)) / (1 - sum_{w in E(h)} p(w|h[1:]))``, bottom-up, so
    every context of the resulting back-off LM sums to one over the vocabulary.
    """
    alpha: Dict[Tuple[str, ...], float] = {}

    def prob(h: Tuple[str, ...], w: str) -> float:
        if not h:
            return float(unigram[w])
        if h in explicit and w in explicit[h]:
            return float(explicit[h][w])
        return alpha.get(h, 1.0) * prob(h[1:], w)

    for h in sorted(explicit, key=len):
        e = explicit[h]
        num = 1.0 - sum(e.values())
        den = 1.0 - sum(prob(h[1:], w) for w in e)
        assert num > 0 and den > 0, (h, num, den)
        alpha[h] = num / den
    return alpha


def _log10(p: float) -> float:
    return -99.0 if p <= 0.0 else math.log10(p)


def write_arpa(path: str, explicit: Mapping[Tuple[str, ...], Mapping[str, float]],
               unigram: Mapping[str, float] = UNIGRAM) -> str:
    """Write the back-off LM as an ARPA file (log10, tab separated, 12 decimals)."""
    order = _lm_order(explicit)
    alpha = derive_backoffs(explicit, unigram)
    grams: Dict[int, List[Tuple[Tuple[str, ...], float]]] = {o: [] for o in range(1, order + 1)}
    for w in VOCAB:
        grams[1].append(((w,), float(unigram[w])))
    for h in sorted(explicit, key=lambda t: (len(t), [VOCAB.index(x) for x in t])):
        for w in sorted(explicit[h], key=VOCAB.index):
            grams[len(h) + 1].append((h + (w,), float(explicit[h][w])))
    lines = ["\\data\\"] + [f"ngram {o}={len(grams[o])}" for o in range(1, order + 1)] + [""]
    for o in range(1, order + 1):
        lines.append(f"\\{o}-grams:")
        for ngram, p in grams[o]:
            row = f"{_log10(p):.12f}\t{' '.join(ngram)}"
            if o < order:
                row += f"\t{_log10(alpha.get(ngram, 1.0)) if ngram in alpha else 0.0:.12f}"
            lines.append(row)
        lines.append("")
    lines.append("\\end\\")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


class ArpaScorer:
    """An ARPA text in dicts and the textbook back-off recursion, in nats (independent of PKG)."""

    def __init__(self, path: str):
        self.lp: Dict[Tuple[str, ...], float] = {}
        self.bo: Dict[Tuple[str, ...], float] = {}
        self.grams: Dict[int, List[Tuple[str, ...]]] = {}
        cur = 0
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("ngram ") or s == "\\data\\":
                    continue
                if s == "\\end\\":
                    break
                if s.startswith("\\") and s.endswith("-grams:"):
                    cur = int(s[1:].split("-")[0])
                    self.grams[cur] = []
                    continue
                f = s.split()
                ngram = tuple(f[1:1 + cur])
                self.lp[ngram] = float(f[0]) * math.log(10.0)
                if len(f) == cur + 2:
                    self.bo[ngram] = float(f[cur + 1]) * math.log(10.0)
                self.grams[cur].append(ngram)
        self.order = max(self.grams)
        self.vocab = [g[0] for g in self.grams[1]]

    def logp(self, hist: Sequence[str], w: str) -> float:
        """``log p(w | hist)`` in nats, the history truncated to ``order - 1`` words."""
        h = tuple(hist)[max(0, len(hist) - (self.order - 1)):] if self.order > 1 else ()
        acc = 0.0
        while True:
            if h + (w,) in self.lp:
                return acc + self.lp[h + (w,)]
            assert h, f"{w!r} has no unigram"
            acc += self.bo.get(h, 0.0)
            h = h[1:]

    def state_contexts(self) -> List[Tuple[str, ...]]:
        """The CSR state numbering of an ARPA with no blank context: (), (w,) per word, then the
        listed k-grams for k = 2 .. order - 1 in file order."""
        out: List[Tuple[str, ...]] = [()] + [(w,) for w in self.vocab]
        for k in range(2, self.order):
            out += list(self.grams[k])
        return out


# ------------------------------------------------------------------------------------------------
# the production build path (the thing under test), used only to MAKE the fixture's files
# ------------------------------------------------------------------------------------------------
def build_resources(tmpdir: str, *, explicit=LM3_EXPLICIT) -> Dict[str, object]:
    """ARPA -> ``lexlat_resources.npz`` exactly as ``word_lm.LexiconTrieBuildJob.run`` does."""
    from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat

    os.makedirs(tmpdir, exist_ok=True)
    arpa = write_arpa(os.path.join(tmpdir, "lm3.arpa"), explicit)
    lm = lexlat.parse_arpa_word_lm(arpa)
    word2id = {w: i for i, w in enumerate(lm["words"])}
    trie = lexlat.build_trie(LEXICON, PHONE2ID, word2id, n_phones=N_PHONES, sil=SIL)
    shuffled, _stats = lexlat.derange_pronunciations(LEXICON, seed=0)
    trie_null = lexlat.build_trie(shuffled, PHONE2ID, word2id, n_phones=N_PHONES, sil=SIL)
    npz = os.path.join(tmpdir, "lexlat_resources.npz")
    lexlat.save_resources(npz, trie=trie, lm=lm, escape_phone_log_prob=list(ESCAPE_PHONE_LOG_PROB),
                          n_phones=N_PHONES, sil_id=SIL_ID, shuffled_word_id=trie_null["word_id"])
    return {"arpa": arpa, "npz": npz, "shuffled_lexicon": {w: list(p) for w, p in shuffled.items()}}


def build_hlg(npz: str, out_dir: str, *, backoff_loops: str = "word_boundary",
              strict: bool = False, shuffled: bool = False, theta: float = 0.0) -> Dict[str, str]:
    """``lexlat_k2.main(["build-hlg", ...])`` -- the production CLI -- into ``out_dir``."""
    from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat_k2

    os.makedirs(out_dir, exist_ok=True)
    hlg, js = os.path.join(out_dir, "HLG.pt"), os.path.join(out_dir, "build.json")
    argv = ["build-hlg", "--resources", npz, "--out-hlg", hlg, "--out-json", js,
            "--prune-theta", repr(float(theta)), "--sil-prob", repr(SIL_PROB),
            "--d-min", str(D_MIN), "--recognizer-stride", str(RECOGNIZER_STRIDE),
            "--backoff-loops", backoff_loops]
    if strict:
        argv.append("--strict-lexicon")
    if shuffled:
        argv.append("--shuffled")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rc = lexlat_k2.main(argv)
    assert rc == 0, log.getvalue()[-2000:]
    return {"hlg": hlg, "stats": js, "log": log.getvalue()}


# ------------------------------------------------------------------------------------------------
# G as the oracle reads it: the npz arrays, np.load only
# ------------------------------------------------------------------------------------------------
class GTables:
    """The CSR back-off automaton of the resource npz, as dicts (float32 values up-cast)."""

    def __init__(self, npz: str):
        with np.load(npz, allow_pickle=False) as z:
            v = int(z["n_words"])
            key = np.asarray(z["arc_key"], dtype=np.int64)
            lp = np.asarray(z["arc_logp"], dtype=np.float32).astype(np.float64)
            nx = np.asarray(z["arc_next"], dtype=np.int64)
            self.backoff = np.asarray(z["backoff"], dtype=np.float32).astype(np.float64)
            self.backoff_state = np.asarray(z["backoff_state"], dtype=np.int64)
            self.words = [str(w) for w in z["words"]]
            self.begin_state = int(z["begin_state"])
            self.unk = int(z["unk_word"])
            self.n_words = v
        self.arc: Dict[Tuple[int, int], Tuple[float, int]] = {
            (int(k // v), int(k % v)): (float(l), int(n)) for k, l, n in zip(key, lp, nx)}
        self.wid = {w: i for i, w in enumerate(self.words)}

    def backoff_move(self, s: int) -> Optional[Tuple[float, int]]:
        """G's ``#0`` arc out of ``s``: every state but the null context has one."""
        if s == 0:
            return None
        return float(self.backoff[s]), int(self.backoff_state[s])


def csr_walk(tables: Mapping[str, object], s: int, w: int) -> Tuple[float, int]:
    """``(log p(w | s), successor)`` of a CSR table by the plain back-off walk to the null state
    (any depth): the explicit arc if ``(s, w)`` has one, else ``backoff[s]`` plus the walk from
    ``backoff_state[s]``."""
    key = np.asarray(tables["arc_key"], dtype=np.int64)
    lp = np.asarray(tables["arc_logp"], dtype=np.float32).astype(np.float64)
    nx = np.asarray(tables["arc_next"], dtype=np.int64)
    bo = np.asarray(tables["backoff"], dtype=np.float32).astype(np.float64)
    bs = np.asarray(tables["backoff_state"], dtype=np.int64)
    v = int(tables["n_words"])
    acc = 0.0
    s = int(s)
    while True:
        k = np.int64(s) * v + int(w)
        pos = int(np.searchsorted(key, k))
        if pos < key.size and key[pos] == k:
            return acc + float(lp[pos]), int(nx[pos])
        assert s != 0, f"word {w} has no unigram arc"
        acc += float(bo[s])
        s = int(bs[s])


def csr_state(tables: Mapping[str, object], hist: Sequence[int]) -> int:
    """The state a word history leads to, walked from the null context 0 by :func:`csr_walk`."""
    s = 0
    for w in hist:
        s = csr_walk(tables, s, int(w))[1]
    return s


# ------------------------------------------------------------------------------------------------
# the explicit L o G path enumerator (T1.19)
# ------------------------------------------------------------------------------------------------
class Path(NamedTuple):
    score: float  # nats, at temperature 1
    n_words: int  # word-emitting transitions, <unk> included (the monitor's convention)
    n_unk: int  # escape spans
    adjacent_escape: bool  # an escape span opened right after another one closed (no word / SIL)
    n_final_backoff: int  # back-off moves taken at LOOP after the last token, before accepting


def needs_disambig(lexicon: Mapping[str, Sequence[str]]) -> Dict[str, bool]:
    """A pronunciation needs a disambiguation symbol when it is a proper prefix of another one or
    is shared by more than one word (the icefall rule, restated)."""
    prons = [tuple(p) for p in lexicon.values()]
    out = {}
    for w, p in lexicon.items():
        p = tuple(p)
        shared = prons.count(p) > 1
        prefix = any(len(q) > len(p) and q[:len(p)] == p for q in prons)
        out[w] = shared or prefix
    return out


def escape_price(k: int) -> float:
    """The per-phone escape price: the raw phone log probability + log 0.5 (non-SIL only)."""
    assert k != SIL_ID
    return ESCAPE_PHONE_LOG_PROB[k] + ESCAPE_LENGTH_LOG_PROB


def enumerate_paths(y: Sequence[int], g: GTables, *, lexicon: Mapping[str, Sequence[str]] = LEXICON,
                    placement: str = "word_boundary", escape: bool = True,
                    sil_prob: float = SIL_PROB) -> List[Path]:
    """Every L o G path consuming the token string ``y`` (phone ids), one :class:`Path` each.

    START -> LOOP by eps at log(1-p), or -> SILST at log p.  SILST -> LOOP on a SIL token.  At LOOP:
    accept at the end of ``y``; back off ``s -> backoff_state[s]`` at ``backoff[s]`` (repeatable,
    never from the null context); read a word whose pronunciation matches the next tokens via an
    EXPLICIT G arc, then -> LOOP at log(1-p) or -> SILST at log p; or (escape) open an escape on a
    non-SIL token via the explicit G arc (s, <unk>) plus the phone's escape price -> ESC.  ESC extends
    on a non-SIL token at its price or exits by eps to LOOP (log(1-p)) / SILST (log p).  Every G
    state is final at 0 and there is no </s>.  ``placement = "all"``: the back-off move is also
    available at SILST and at every word-interior position (after each token of the word's token
    sequence but the last, the disambiguation symbol counting as a token).
    """
    y = [int(t) for t in y]
    n = len(y)
    lsil, lnosil = math.log(sil_prob), math.log(1.0 - sil_prob)
    dis = needs_disambig(lexicon)
    words = []
    for w, pron in lexicon.items():
        ids = [PHONE2ID[p] for p in pron]
        n_tokens = len(ids) + (1 if dis[w] else 0)
        words.append((g.wid[w], ids, n_tokens - 1))  # (LM id, phones, interior positions)
    everywhere = placement == "all"
    assert placement in ("all", "word_boundary"), placement
    out: List[Path] = []

    def backoffs(s: int, score: float):
        """(state, score, moves) after 0, 1, 2, ... back-off moves from ``s``."""
        k = 0
        yield s, score, k
        while True:
            mv = g.backoff_move(s)
            if mv is None:
                return
            score += mv[0]
            s = mv[1]
            k += 1
            yield s, score, k

    def after_word(pos, s, score, nw, nu, adj):
        # the word's last token -> LOOP (log(1-p)) or SILST (log p); an escape run is broken
        loop(pos, s, score + lnosil, nw, nu, adj, just_escaped=False)
        silst(pos, s, score + lsil, nw, nu, adj)

    def interior(k_left, pos, s, score, nw, nu, adj):
        # ``k_left`` word-interior positions remain; each may take back-off moves under "all"
        if k_left == 0:
            after_word(pos, s, score, nw, nu, adj)
            return
        chain = backoffs(s, score) if everywhere else [(s, score, 0)]
        for s2, sc2, _k in chain:
            interior(k_left - 1, pos, s2, sc2, nw, nu, adj)

    def silst(pos, s, score, nw, nu, adj):
        chain = backoffs(s, score) if everywhere else [(s, score, 0)]
        for s2, sc2, _k in chain:
            if pos < n and y[pos] == SIL_ID:
                loop(pos + 1, s2, sc2, nw, nu, adj, just_escaped=False)

    def loop(pos, s, score, nw, nu, adj, just_escaped):
        for s2, sc2, k in backoffs(s, score):
            if pos == n:
                # accept: every G state is final at 0, so a path that backs off k times and then
                # accepts is its own path (k = 0, 1, ...), exactly as the graph has it
                out.append(Path(sc2, nw, nu, adj, k))
            for wid, ids, n_int in words:
                if tuple(y[pos:pos + len(ids)]) != tuple(ids):
                    continue
                arc = g.arc.get((s2, wid))
                if arc is None:
                    continue
                interior(n_int, pos + len(ids), arc[1], sc2 + arc[0], nw + 1, nu, adj)
            if escape and pos < n and y[pos] != SIL_ID:
                arc = g.arc.get((s2, g.unk))
                if arc is not None:
                    esc(pos + 1, arc[1], sc2 + arc[0] + escape_price(y[pos]), nw + 1, nu + 1,
                        adj or just_escaped)

    def esc(pos, s, score, nw, nu, adj):
        if pos < n and y[pos] != SIL_ID:
            esc(pos + 1, s, score + escape_price(y[pos]), nw, nu, adj)
        loop(pos, s, score + lnosil, nw, nu, adj, just_escaped=True)
        silst(pos, s, score + lsil, nw, nu, adj)

    s0 = g.begin_state
    loop(0, s0, lnosil, 0, 0, False, just_escaped=False)
    silst(0, s0, lsil, 0, 0, False)
    return out


def proper_paths(y: Sequence[int], scorer: ArpaScorer, *,
                 lexicon: Mapping[str, Sequence[str]] = LEXICON, escape: bool = True,
                 sil_prob: float = SIL_PROB) -> List[float]:
    """The same parse space as :func:`enumerate_paths` (words, escape spans, optional SIL), each
    word transition priced ONCE by the exact back-off LM on the word history -- the single-route
    proper-LM reading the over-count is measured against.  No back-off moves: they are inside
    ``scorer.logp``.  Returns the path scores."""
    y = [int(t) for t in y]
    n = len(y)
    lsil, lnosil = math.log(sil_prob), math.log(1.0 - sil_prob)
    out: List[float] = []
    entries = [(w, [PHONE2ID[p] for p in pron]) for w, pron in lexicon.items()]

    def after_word(pos, hist, score):
        loop(pos, hist, score + lnosil)
        silst(pos, hist, score + lsil)

    def silst(pos, hist, score):
        if pos < n and y[pos] == SIL_ID:
            loop(pos + 1, hist, score)

    def loop(pos, hist, score):
        if pos == n:
            out.append(score)
        for w, ids in entries:
            if tuple(y[pos:pos + len(ids)]) == tuple(ids):
                after_word(pos + len(ids), hist + (w,), score + scorer.logp(hist, w))
        if escape and pos < n and y[pos] != SIL_ID:
            esc(pos + 1, hist + ("<unk>",),
                score + scorer.logp(hist, "<unk>") + escape_price(y[pos]))

    def esc(pos, hist, score):
        if pos < n and y[pos] != SIL_ID:
            esc(pos + 1, hist, score + escape_price(y[pos]))
        after_word(pos, hist, score)

    loop(0, ("<s>",), lnosil)
    silst(0, ("<s>",), lsil)
    return out


# ------------------------------------------------------------------------------------------------
# strings, collapse, brute force over frame strings
# ------------------------------------------------------------------------------------------------
def collapse(a: Sequence[int]) -> Tuple[int, ...]:
    """The adjacent-run collapse B(a) (every symbol, SIL included)."""
    out: List[int] = []
    for k in a:
        if not out or out[-1] != int(k):
            out.append(int(k))
    return tuple(out)


def token_strings(max_len: int) -> List[Tuple[int, ...]]:
    """Every B(a) for a in {0..3}^T, T <= ``max_len``: strings without an adjacent repeat."""
    out = set()
    for t in range(1, max_len + 1):
        for a in itertools.product(range(N_PHONES), repeat=t):
            out.add(collapse(a))
    return sorted(out, key=lambda s: (len(s), s))


def log_w(paths: Sequence[Path], tau: float, *, weight=None) -> float:
    """``log sum_paths exp(score / tau) [* weight(path)]``; ``-inf`` for no path / zero weight."""
    vals = []
    for p in paths:
        c = 1.0 if weight is None else float(weight(p))
        if c > 0:
            vals.append(p.score / tau + math.log(c))
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


class PathCache:
    """``y -> [Path]`` for one graph variant, computed on demand."""

    def __init__(self, g: GTables, **kw):
        self.g, self.kw, self.cache = g, kw, {}

    def __call__(self, y: Tuple[int, ...]) -> List[Path]:
        if y not in self.cache:
            self.cache[y] = enumerate_paths(y, self.g, **self.kw)
        return self.cache[y]


def brute_force(e, tau: float, paths: PathCache, *, weight=None):
    """``log sum_{a in 4^T} exp(sum_t e_t(a_t)/tau) W_tau(B(a))`` for one utterance ``e [T, K]``.

    ``e`` is a float64 torch tensor; the result is a 0-dim tensor differentiable in ``e``.  With
    ``weight`` the path weight is multiplied by ``weight(path)`` (posterior expectations).
    """
    import torch

    t_len, k = int(e.shape[0]), int(e.shape[1])
    frames = list(itertools.product(range(k), repeat=t_len))
    lw = {}
    for a in frames:
        y = collapse(a)
        if y not in lw:
            lw[y] = log_w(paths(y), tau, weight=weight)
    idx = torch.tensor(frames, dtype=torch.long)  # [N, T]
    em = e[torch.arange(t_len).unsqueeze(0), idx].sum(dim=1) / tau
    logw = torch.tensor([lw[collapse(a)] for a in frames], dtype=torch.float64)
    ok = torch.isfinite(logw)
    if not bool(ok.any()):
        return torch.tensor(-math.inf, dtype=torch.float64)
    return torch.logsumexp(em[ok] + logw[ok], dim=0)
