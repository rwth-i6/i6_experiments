"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lexlat.py.

Port note: the RESOURCE half only -- the trie / word-LM builders, ``LexResources``, ``lm_score``
and the npz persistence, which the lexicon / HLG build jobs and the k2 runtime read.  The source's
fixed-string scorers ``string_best_segmentation`` / ``string_log_sum`` (the reference of its
equivalence tests; no ported caller) are cut, and the docstrings that name them refer to the source.  The trie DP
(``LexlatParams`` .. ``lexlat_viterbi``, source lines 785-1984) is CUT: no in-scope config
states ``lexlat_resources``, the only training path that ran it.  The text below that
describes the DP is the source's and refers to the cut code.

The LEXICON INSIDE the marginalised blank-free lattice -- the DP core of `SAE_4A_lexlat.md`.

``emc.lattice`` marginalises over phone strings under a phone trigram prior.  This module adds the
lexicalised prior of ``emc.prior_gap`` (a pronunciation lexicon + a KenLM WORD trigram, with the
pre-registered ESCAPE convention) to the SAME lattice as an additive arc term, so that the exact
marginal ``log Z`` the bed optimises is taken under ``beta * log P3 + lam_lex * (word-LM increment
+ escape increment)`` instead of ``beta * log P3`` alone.  Round 1 deliverable: the core, its
resources, its tests and the two probe jobs of ``emc.lexlat_jobs``; NOTHING here is wired into a
training step yet.

A NEW MODULE, never an addition to ``lattice.py`` / ``prior_gap.py``: a file-sha ``__sis_version__``
moves every job in the file it stamps and both of those files carry banked, still-running jobs
(memory: sis-version-file-hash-moves-on-any-edit).  Every numeric primitive of the bed --
``NEG_INF``, ``_logmm``'s float64 accumulation, ``_exact_fp32_matmul``, ``_band_matrix``,
``scaled_seg_pad``, ``_arc_weights``, ``_legality_mask`` -- is IMPORTED from ``lattice`` and not
re-implemented, so the two DPs share one floating point.

THE STATE SPACE (`SAE_4A_lexlat.md` Design 1).  The bed's context ``h`` (the phone trigram history)
is augmented with the lexical state

    (trie node n, word-LM state sigma, escape-open flag e)

so a DP context is ``(n, sigma, e, h)`` and the band offset ``o = s - stride*t + W`` and the CTC
repeat flag ``f`` stay exactly what they are in ``lattice.py``.  The arcs, all of which also carry
the bed's own ``w_sym[k] + beta/tau * log P3(k | h) + G[k, s, d]/tau``:

=========== ================================================== ==================================
family      source -> destination                              lexical increment (x ``lam_lex``)
=========== ================================================== ==================================
CONT        ``(n, sigma, 0, h)`` -> ``(child[n,k], sigma, 0,``  0
            ``h')``, ``n != ROOT``
START       a BOUNDARY entry -> ``(child[ROOT,k], sigma_e,``    the entry's own ``w_e``
            ``0, h')``
SIL         a BOUNDARY entry -> ``(ROOT, sigma_e, 0, h')``,     the entry's own ``w_e``
            ``k = SIL``
ESC-open    a BOUNDARY entry -> ``(ROOT, unk_next[sigma_e],``   ``w_e + log p(<unk>|sigma_e)``
            ``1, h')``                                          ``+ esc[k]``
ESC-ext     ``(ROOT, sigma, 1, h)`` -> ``(ROOT, sigma, 1, h')`` ``esc[k]``
=========== ================================================== ==================================

A BOUNDARY entry is a context that may start a new word at this frame: a context at the ROOT node
(a closed boundary, or an OPEN ESCAPE -- closing an escape is free, exactly as in
``prior_gap.best_segmentation``) contributes one entry of weight 0, and a context at a word-end node
contributes one entry PER HOMOPHONE of that node, of weight ``log p(word | sigma)`` and successor
state.  FINALITY (Design 2 amendment 3): an utterance may end only at a closed boundary, so the
final weight of a context is the logsumexp of its own boundary entries' weights (0 at the ROOT, the
word-close weight at a word end, NEG_INF mid-word) -- a mid-word context is NOT final.

WHY THE ARC LIST IS BUILT AS SLOTS AND DEDUPED.  Two different sources can reach the same
destination (the design review's finding: the injective ``index_copy_`` of ``lattice.py:882`` does
not hold here), and the prior term ``log P3(k|h)`` resolves the FULL history ``h`` while a
destination only carries ``last(h)``.  Every arc is therefore materialised as a SLOT
``(source, k)`` carrying its own weight, the slots are sorted by destination key and
segment-reduced, and the per-frame top-C is taken over the deduped destinations.  Because the
band/duration reduction is the expensive one, the ranking is done PRE-BAND on
``T[o,k] = logsumexp_d band(o, d, k)``: ``rank(dst) = logsumexp_slots (logmm(mass, T)[src,k] +
prior + w_sym + lexical)`` is EXACTLY the completed frame's mass of that destination (the band
reduction and the ``o`` contraction commute), so the pruning keeps the true top-C and the expensive
reduction runs on C destinations, not on all of them.

PRUNING (Design 2, amendment 3).  ``max_contexts`` = C destinations survive per frame, of which
``escape_budget`` = C_esc slots are RESERVED for escape-open contexts (without the reserve the
escape branch, which is always below a matching lexical branch, is pruned away at frame 1 and the
convention becomes STRICT by accident).  The discarded forward mass is reported per frame and per
utterance as median / p95 / max, together with the count of utterances whose ``log Z`` came out at
NEG_INF -- the three numbers the funding rule of Design 6 reads.

NORMALISATION (Design 4).  Nothing here is normalised: the lexical increment is a log-probability
increment of a proper word LM plus the escape convention's own geometric length term, added to the
arc, and ``log Z`` is the exact marginal over the state space above.  Every term carries the ``1/tau``
of the objective, the lexical one included (``lattice.scaled_seg_pad``'s rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from .lattice import NEG_INF

__all__ = [
    "ROOT",
    "LexResources",
    "build_trie",
    "derange_pronunciations",
    "parse_arpa_word_lm",
    "lm_score",
    "DEFAULT_LM_ORDER",
    "RESOURCE_ARRAYS",
    "RESOURCE_SCALARS",
    "save_resources",
    "load_resources",
]

#: the trie's root node -- the state "no word is open"
ROOT = 0
#: natural log of 10; KenLM / ARPA scores are log10, everything in the DP is nats
LN10 = math.log(10.0)
#: the escape's per-phone length term, ``log(0.5)`` (``prior_gap.ESCAPE_LENGTH_LOG_PROB``)
ESCAPE_LENGTH_LOG_PROB = math.log(0.5)
#: the escape word, priced by the word LM as an OOV (``prior_gap.ESCAPE_WORD``)
ESCAPE_WORD = "<unk>"
#: the order of the word LM assumed where nothing says otherwise: ``parse_arpa_word_lm`` reads an
#: order-3 ARPA, and every resource npz written before the official-LM round is one, so this is
#: what :func:`lm_score` walks and what a resource without a stored order loads as.  It is the
#: DEPTH OF THE BACK-OFF WALK, and it must equal the LM's own order: from a context of length
#: ``order - 1`` a word that has only a unigram arc is ``order`` lookups away.
DEFAULT_LM_ORDER = 3


# ---------------------------------------------------------------------------------------------------
# resources: the flat trie
# ---------------------------------------------------------------------------------------------------


def build_trie(
    word_to_phones: Mapping[str, Sequence[str]],
    phone2id: Mapping[str, int],
    word2id: Mapping[str, int],
    *,
    n_phones: int,
    sil: str = "SIL",
) -> Dict[str, np.ndarray]:
    """The pronunciation lexicon as FLAT INT32 ARRAYS: ``child[node, phone]``, word ends, word ids.

    The same lexicon ``prior_gap.Lexicon`` builds (same entries, same homophone grouping, same
    "a pronunciation outside the inventory is not an entry" rule), laid out for the DP: a walk is
    ``node = child[node, k]`` and ``child[node, k] == -1`` is "no word continues this way".

    Homophones are a RANGE of the word-id array, not a single id: ``word_id[word_start[n] :
    word_start[n+1]]`` are the words that end at node ``n``, each of which closes with its own
    ``log p(word | sigma)``.  Word order inside a node is by word id, and nodes are created over the
    SORTED SET OF PRONUNCIATIONS (never in word order), so the node numbering is a function of the
    pronunciation multiset alone and the shuffled-pronunciation null moves ``word_id`` and nothing
    else (test (e)).

    A pronunciation containing SIL is REJECTED (SIL is a word boundary in this DP, never a phone
    inside a word) -- asserted, not silently dropped, because the bliss lexicon has none.
    """
    sil_id = int(phone2id[sil])
    children: List[Dict[int, int]] = [{}]
    n_rejected = 0
    n_oov_phone = 0
    entries: List[Tuple[Tuple[int, ...], int]] = []
    for word in sorted(word_to_phones):
        if word not in word2id:
            continue
        ids = [phone2id.get(p) for p in word_to_phones[word]]
        if not ids or any(i is None for i in ids):
            n_oov_phone += 1
            continue
        assert sil_id not in ids, f"{word}: a pronunciation containing {sil} is not a lexicon entry"
        entries.append((tuple(int(i) for i in ids), int(word2id[word])))
    # PASS 1 creates the nodes in a canonical order over the SET of pronunciations, never in word
    # order: the node numbering is then a function of the pronunciation multiset alone, which is
    # exactly what makes :func:`derange_pronunciations` leave ``child`` / ``word_start`` /
    # ``is_word_end`` bit-identical (a bijection of words onto pronunciations moves no string).
    for ids in sorted({e[0] for e in entries}):
        node = ROOT
        for k in ids:
            nxt = children[node].get(k)
            if nxt is None:
                nxt = len(children)
                children[node][k] = nxt
                children.append({})
            node = nxt
    n_nodes = len(children)
    ends: List[List[int]] = [[] for _ in range(n_nodes)]
    for ids, wid in entries:  # PASS 2 attaches the word ids; every node already exists
        node = ROOT
        for k in ids:
            node = children[node][k]
        ends[node].append(wid)
    child = np.full((n_nodes, n_phones), -1, dtype=np.int32)
    for node, kids in enumerate(children):
        for k, v in kids.items():
            child[node, k] = v
    starts = np.zeros(n_nodes + 1, dtype=np.int64)
    flat: List[int] = []
    for node in range(n_nodes):
        starts[node] = len(flat)
        flat.extend(sorted(ends[node]))
    starts[n_nodes] = len(flat)
    assert not ends[ROOT], "the empty pronunciation is not a word"
    return {
        "child": child,
        "word_start": starts.astype(np.int64),
        "word_id": np.asarray(flat, dtype=np.int64),
        "is_word_end": (np.diff(starts) > 0),
        "n_nodes": np.int64(n_nodes),
        "n_entries": np.int64(len(flat)),
        "n_oov_phone": np.int64(n_oov_phone),
        "n_rejected": np.int64(n_rejected),
    }


def derange_pronunciations(
    word_to_phones: Mapping[str, Sequence[str]], *, seed: int = 0
) -> Tuple[Dict[str, List[str]], Dict[str, object]]:
    """The SHUFFLED-PRONUNCIATION NULL: a fixed-point-free permutation of word -> pronunciation.

    Sattolo's algorithm under ``numpy.random.default_rng(seed)`` produces a single cycle over the
    sorted word list, so NO word keeps its own pronunciation -- a true derangement, asserted here.
    A bijection of words onto pronunciations leaves the MULTISET of pronunciations untouched, so the
    trie arrays (``child``, ``word_start``, ``is_word_end``) are bit-identical to the real lexicon's
    and only ``word_id`` moves: the null changes WHICH word the LM prices at each word end and
    nothing else about the lattice's shape.  ``kept_by_homophony`` counts the words whose shuffled
    pronunciation is nonetheless a HOMOPHONE of their own (the derangement is over words, and two
    words can share a pronunciation), which is the honest residual of the null.
    """
    words = sorted(word_to_phones)
    n = len(words)
    perm = list(range(n))
    rng = np.random.default_rng(seed)
    for i in range(n - 1, 0, -1):  # Sattolo: j < i, one cycle, no fixed point
        j = int(rng.integers(0, i))
        perm[i], perm[j] = perm[j], perm[i]
    assert n < 2 or all(perm[i] != i for i in range(n)), "not a derangement"
    out = {w: list(word_to_phones[words[perm[i]]]) for i, w in enumerate(words)}
    kept = sum(1 for w in words if list(out[w]) == list(word_to_phones[w]))
    return out, {
        "n_words": n,
        "fixed_points": 0,
        "kept_by_homophony": int(kept),
        "seed": int(seed),
        "algorithm": "sattolo_single_cycle",
    }


# ---------------------------------------------------------------------------------------------------
# resources: the word LM as a CSR back-off automaton with integer state ids
# ---------------------------------------------------------------------------------------------------


def parse_arpa_word_lm(path: str, *, unk: str = ESCAPE_WORD, bos: str = "<s>") -> Dict[str, np.ndarray]:
    """An ARPA word trigram -> the CSR back-off automaton the DP looks arcs up in.

    STATE IDS.  ``0`` is the null (unigram) context, ``1 + w`` is the context of the single word
    ``w``, and ``1 + V + j`` is the ``j``-th 2-gram of the ARPA in file order.  ``successor(state,
    word)`` is the LONGEST context the tables know -- KenLM's own rule.  Where a state is not in the
    tables the successor falls back to the shorter one, which is SCORE-EQUIVALENT: an n-gram the
    ARPA lists without a back-off column has back-off 0, so backing off through it costs nothing.
    The equivalence is not argued, it is MEASURED: ``lexlat_jobs.LexlatEquivalenceProbeJob`` and
    ``test_lexlat`` score real strings through this table and through ``kenlm`` itself.

    ARCS.  One sorted int64 key ``state * V + word`` per (state, word) the ARPA scores, with its log
    probability (converted to NATS) and its successor state; a lookup is a ``searchsorted``.  Every
    state also carries its back-off weight, its back-off state and a PRECOMPUTED ``<unk>`` arc (the
    escape's word cost), so the escape never pays a search.
    """
    counts: Dict[int, int] = {}
    order = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    i = 0
    while i < len(lines) and not lines[i].startswith("\\data\\"):
        i += 1
    i += 1
    while i < len(lines) and lines[i].strip():
        if lines[i].startswith("ngram "):
            o, c = lines[i][6:].split("=")
            counts[int(o)] = int(c)
            order = max(order, int(o))
        i += 1
    assert order == 3, f"{path}: this DP wants an order-3 word LM, the ARPA is order {order}"

    grams: Dict[int, List[Tuple[float, List[str], float]]] = {1: [], 2: [], 3: []}
    cur = 0
    for line in lines[i:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("\\") and s.endswith("-grams:"):
            cur = int(s[1:].split("-")[0])
            continue
        if s == "\\end\\":
            break
        if cur == 0:
            continue
        parts = s.split("\t")
        lp = float(parts[0])
        toks = parts[1].split(" ")
        bo = float(parts[2]) if len(parts) > 2 else 0.0
        grams[cur].append((lp, toks, bo))
    for o, c in counts.items():
        assert len(grams[o]) == c, f"{path}: {len(grams[o])} {o}-grams, the header says {c}"

    words = [g[1][0] for g in grams[1]]
    word2id = {w: i for i, w in enumerate(words)}
    n_words = len(words)
    assert unk in word2id, f"{path}: the ARPA has no {unk} unigram (the escape has no price)"

    n_states = 1 + n_words + len(grams[2])
    backoff = np.zeros(n_states, dtype=np.float32)
    backoff_state = np.zeros(n_states, dtype=np.int64)
    bi_state: Dict[Tuple[int, int], int] = {}
    for j, (_lp, toks, _bo) in enumerate(grams[2]):
        bi_state[(word2id[toks[0]], word2id[toks[1]])] = 1 + n_words + j

    def state_of(ws: Sequence[int]) -> int:
        """The longest context in the tables for the word sequence ``ws`` (most recent last)."""
        if len(ws) >= 2 and (ws[-2], ws[-1]) in bi_state:
            return bi_state[(ws[-2], ws[-1])]
        if len(ws) >= 1:
            return 1 + ws[-1]
        return 0

    keys: List[int] = []
    logp: List[float] = []
    nxt: List[int] = []
    for lp, toks, bo in grams[1]:
        w = word2id[toks[0]]
        keys.append(0 * n_words + w)
        logp.append(lp * LN10)
        nxt.append(state_of([w]))
        backoff[1 + w] = bo
        backoff_state[1 + w] = 0
    for lp, toks, bo in grams[2]:
        w1, w2 = word2id[toks[0]], word2id[toks[1]]
        keys.append((1 + w1) * n_words + w2)
        logp.append(lp * LN10)
        nxt.append(state_of([w1, w2]))
        s = bi_state[(w1, w2)]
        backoff[s] = bo
        backoff_state[s] = 1 + w2
    for lp, toks, _bo in grams[3]:
        w1, w2, w3 = (word2id[t] for t in toks)
        st = bi_state.get((w1, w2))
        if st is None:  # a trigram whose context is not a listed 2-gram: unreachable, skip
            continue
        keys.append(st * n_words + w3)
        logp.append(lp * LN10)
        nxt.append(state_of([w2, w3]))
    key = np.asarray(keys, dtype=np.int64)
    perm = np.argsort(key, kind="stable")
    key = key[perm]
    assert np.all(np.diff(key) > 0), f"{path}: duplicate (state, word) arcs"
    out = {
        "arc_key": key,
        "arc_logp": np.asarray(logp, dtype=np.float32)[perm],
        "arc_next": np.asarray(nxt, dtype=np.int64)[perm],
        "backoff": backoff * np.float32(LN10),
        "backoff_state": backoff_state,
        "n_states": np.int64(n_states),
        "n_words": np.int64(n_words),
        "begin_state": np.int64(state_of([word2id[bos]]) if bos in word2id else 0),
        "unk_word": np.int64(word2id[unk]),
        "words": np.asarray(words, dtype=object),
        "n_1gram": np.int64(len(grams[1])),
        "n_2gram": np.int64(len(grams[2])),
        "n_3gram": np.int64(len(grams[3])),
    }
    return out


# ---------------------------------------------------------------------------------------------------
# resources: the object the DP reads
# ---------------------------------------------------------------------------------------------------


@dataclass
class LexResources:
    """Trie + word-LM automaton + escape prices, as device tensors.  A CONSTANT of a DP call."""

    child: torch.Tensor  # [n_nodes, K] long, -1 where no word continues
    word_start: torch.Tensor  # [n_nodes + 1] long
    word_id: torch.Tensor  # [E] long, LM word ids of the words ending at each node
    word_cnt: torch.Tensor  # [n_nodes] long
    arc_key: torch.Tensor  # [A] long, sorted state * V + word
    arc_logp: torch.Tensor  # [A] float (nats)
    arc_next: torch.Tensor  # [A] long
    backoff: torch.Tensor  # [S] float (nats)
    backoff_state: torch.Tensor  # [S] long
    unk_logp: torch.Tensor  # [S] float: log p(<unk> | state), the escape's word cost
    unk_next: torch.Tensor  # [S] long: the state an opened escape moves to
    escape_price: torch.Tensor  # [K] float: order-1 phone log prob + log 0.5 (NEG_INF at SIL)
    n_nodes: int
    n_states: int
    n_words: int
    begin_state: int
    unk_word: int
    n_phones: int
    sil_id: int
    words: Optional[Sequence[str]] = None  # host-side, for the decode readout only
    max_homophones: int = 1
    #: the ORDER of the word LM these tables came from -- the depth of :func:`lm_score`'s back-off
    #: walk when a caller does not name one.  ``3`` is the default because every npz written before
    #: the official-LM round carries no order and IS an order-3 LM (``parse_arpa_word_lm`` reads
    #: 1-, 2- and 3-grams only); ``load_resources`` fills it from the npz where the writer stored
    #: it.  A wrong value here is silent: too shallow a walk returns ``log p = 0`` (probability
    #: one) for a word that only has a lower-order arc.
    order: int = DEFAULT_LM_ORDER

    @property
    def device(self):
        return self.child.device

    @property
    def dtype(self):
        return self.arc_logp.dtype

    def to(self, device, dtype: Optional[torch.dtype] = None) -> "LexResources":
        dt = dtype or self.arc_logp.dtype
        mv = lambda t: t.to(device=device, dtype=dt) if t.is_floating_point() else t.to(device)
        return LexResources(
            **{
                **{f: mv(getattr(self, f)) for f in
                   ("child", "word_start", "word_id", "word_cnt", "arc_key", "arc_logp", "arc_next",
                    "backoff", "backoff_state", "unk_logp", "unk_next", "escape_price")},
                **{f: getattr(self, f) for f in
                   ("n_nodes", "n_states", "n_words", "begin_state", "unk_word", "n_phones",
                    "sil_id", "words", "max_homophones", "order")},
            }
        )

    @property
    def key_stride(self) -> Tuple[int, int, int]:
        """``(n_states, n_nodes, 2)`` -- the radices of the packed destination key."""
        return (int(self.n_states), int(self.n_nodes), 2)

    @classmethod
    def build(
        cls,
        trie: Mapping[str, np.ndarray],
        lm: Mapping[str, np.ndarray],
        escape_phone_log_prob: Sequence[float],
        *,
        n_phones: int,
        sil_id: int,
        device=None,
        dtype: torch.dtype = torch.float32,
        words: Optional[Sequence[str]] = None,
        order: Optional[int] = None,
    ) -> "LexResources":
        """Assemble from :func:`build_trie` / :func:`parse_arpa_word_lm` and the order-1 phone prior.

        ``escape_phone_log_prob[k]`` is the ORDER-1 phone log probability of ``prior_gap``'s escape
        convention (``witten_bell_utt_log_prob(prior, [p], 1)``); ``log 0.5`` is added here, once,
        and SIL is floored to NEG_INF because an escape span is non-SIL by construction.

        ``order`` is the word LM's order, i.e. the depth of the back-off walk these tables need.
        It defaults to ``lm["order"]`` where the tables carry it (``lexlat_k2_official.parse_arpa_lm``
        writes it at any order) and to :data:`DEFAULT_LM_ORDER` where they do not, which is every
        table :func:`parse_arpa_word_lm` produces and every npz banked before the official-LM
        round.  ``unk_logp`` -- the escape's word cost, read straight out of this table by the DP
        -- is computed at that depth, so an order-4 LM's ``<unk>`` cost is a real number and not
        the ``log p = 0`` a three-level walk returns.
        """
        dev = device or torch.device("cpu")
        t = lambda a, d=torch.int64: torch.as_tensor(np.asarray(a), dtype=d, device=dev)
        esc = torch.as_tensor(np.asarray(escape_phone_log_prob), dtype=dtype, device=dev).clone()
        assert esc.numel() == n_phones, (esc.numel(), n_phones)
        esc += ESCAPE_LENGTH_LOG_PROB
        esc[sil_id] = NEG_INF
        start = t(trie["word_start"])
        res = cls(
            child=t(trie["child"]),
            word_start=start,
            word_id=t(trie["word_id"]),
            word_cnt=(start[1:] - start[:-1]),
            arc_key=t(lm["arc_key"]),
            arc_logp=t(lm["arc_logp"], dtype),
            arc_next=t(lm["arc_next"]),
            backoff=t(lm["backoff"], dtype),
            backoff_state=t(lm["backoff_state"]),
            unk_logp=torch.zeros(int(lm["n_states"]), dtype=dtype, device=dev),
            unk_next=torch.zeros(int(lm["n_states"]), dtype=torch.int64, device=dev),
            escape_price=esc,
            n_nodes=int(trie["n_nodes"]),
            n_states=int(lm["n_states"]),
            n_words=int(lm["n_words"]),
            begin_state=int(lm["begin_state"]),
            unk_word=int(lm["unk_word"]),
            n_phones=int(n_phones),
            sil_id=int(sil_id),
            words=words if words is not None else list(lm.get("words", [])),
            max_homophones=int((start[1:] - start[:-1]).max()) if int(trie["n_nodes"]) > 1 else 1,
            order=int(order if order is not None else lm.get("order", DEFAULT_LM_ORDER)),
        )
        assert res.order >= 1, res.order
        states = torch.arange(res.n_states, dtype=torch.int64, device=dev)
        lp, nx = lm_score(res, states, torch.full_like(states, res.unk_word), levels=res.order)
        res.unk_logp, res.unk_next = lp, nx
        assert int(res.child.shape[1]) == n_phones, res.child.shape
        assert torch.all(res.child[ROOT] >= -1)
        return res


def lm_score(res: LexResources, state: torch.Tensor, word: torch.Tensor,
             *, levels: int = DEFAULT_LM_ORDER):
    """``(log p(word | state) in nats, successor state)`` -- the back-off walk, vectorised.

    ``levels`` is the DEPTH of the walk and MUST be the LM's own order.  Three levels answer an
    order-3 automaton: a 2-gram state backs off to a 1-gram state, which backs off to the null
    state, where every word has a unigram arc.  Exactly ``KenLMWordLM.score`` (``prior_gap``) on
    the same ARPA, to the tolerance the probe and the tests measure.

    TOO SHALLOW A WALK FAILS SILENTLY: from an order-4 context a word with only a unigram arc is
    four lookups away, and a three-level walk leaves ``found`` false and returns the initialised
    ``log p = 0`` -- probability ONE -- with successor state 0.  The default is kept at
    :data:`DEFAULT_LM_ORDER` rather than at ``res.order`` so that every caller written for the
    banked order-3 resources scores exactly the number it scored before this argument existed;
    the two fixed-string scorers below take their default from ``res.order`` instead, which is the
    LM's own order wherever the npz stored it.
    """
    assert int(levels) >= 1, levels
    shape = state.shape
    st, wd = state.reshape(-1), word.reshape(-1)
    v = res.n_words
    total = torch.zeros(st.shape, dtype=res.arc_logp.dtype, device=st.device)
    out_lp = torch.zeros_like(total)
    out_nx = torch.zeros_like(st)
    found = torch.zeros_like(st, dtype=torch.bool)
    cur = st
    last = res.arc_key.numel() - 1
    for _ in range(int(levels)):
        key = cur * v + wd
        pos = torch.searchsorted(res.arc_key, key).clamp(max=last)
        hit = (~found) & (res.arc_key[pos] == key)
        out_lp = torch.where(hit, total + res.arc_logp[pos], out_lp)
        out_nx = torch.where(hit, res.arc_next[pos], out_nx)
        found = found | hit
        total = torch.where(found, total, total + res.backoff[cur])
        cur = torch.where(found, cur, res.backoff_state[cur])
    return out_lp.reshape(shape), out_nx.reshape(shape)


# ---------------------------------------------------------------------------------------------------
# persistence: one npz carries the whole resource (the build job writes it, every consumer reads it)
# ---------------------------------------------------------------------------------------------------

#: the arrays :func:`save_resources` writes; ``shuffled_word_id`` is the null's only difference
RESOURCE_ARRAYS = (
    "child", "word_start", "word_id", "shuffled_word_id",
    "arc_key", "arc_logp", "arc_next", "backoff", "backoff_state",
    "escape_phone_log_prob", "words",
)
#: the scalars it writes beside them
RESOURCE_SCALARS = (
    "n_nodes", "n_entries", "n_states", "n_words", "begin_state", "unk_word", "n_phones", "sil_id",
    "n_1gram", "n_2gram", "n_3gram",
)


def save_resources(
    path: str,
    *,
    trie: Mapping[str, np.ndarray],
    lm: Mapping[str, np.ndarray],
    escape_phone_log_prob: Sequence[float],
    n_phones: int,
    sil_id: int,
    shuffled_word_id: Optional[np.ndarray] = None,
) -> None:
    """Write trie + CSR automaton + escape prices as ONE npz -- the phase's frozen resource.

    ``escape_phone_log_prob[k]`` is stored RAW (the order-1 Witten-Bell phone log probability,
    ``prior_gap``'s convention); ``log 0.5`` is added by :meth:`LexResources.build`, once, so the
    stored vector is exactly the number ``PriorGapAnalysisJob`` banks under
    ``escape_phone_log_probs``.  ``shuffled_word_id`` is the derangement's word-id column: the
    arm and its null read the same file and differ in one array (Design 6).

    ``order`` is written as an OPTIONAL extra key when ``lm`` carries one, and is deliberately NOT
    in :data:`RESOURCE_SCALARS`: the banked order-3 npz files do not have it and must keep
    loading, and they load as :data:`DEFAULT_LM_ORDER`, which is what they are.  Where it IS
    stored, :func:`load_resources` makes it the default depth of the two fixed-string scorers'
    back-off walk, so an order-4 resource is priced at four levels without its readers being told.
    """
    words = np.asarray(list(lm.get("words", [])), dtype=np.str_)
    out = {
        "child": np.asarray(trie["child"], dtype=np.int32),
        "word_start": np.asarray(trie["word_start"], dtype=np.int64),
        "word_id": np.asarray(trie["word_id"], dtype=np.int64),
        "arc_key": np.asarray(lm["arc_key"], dtype=np.int64),
        "arc_logp": np.asarray(lm["arc_logp"], dtype=np.float32),
        "arc_next": np.asarray(lm["arc_next"], dtype=np.int64),
        "backoff": np.asarray(lm["backoff"], dtype=np.float32),
        "backoff_state": np.asarray(lm["backoff_state"], dtype=np.int64),
        "escape_phone_log_prob": np.asarray(escape_phone_log_prob, dtype=np.float64),
        "words": words,
        "n_nodes": np.int64(trie["n_nodes"]),
        "n_entries": np.int64(trie["n_entries"]),
        "n_states": np.int64(lm["n_states"]),
        "n_words": np.int64(lm["n_words"]),
        "begin_state": np.int64(lm["begin_state"]),
        "unk_word": np.int64(lm["unk_word"]),
        "n_phones": np.int64(n_phones),
        "sil_id": np.int64(sil_id),
        "n_1gram": np.int64(lm.get("n_1gram", 0)),
        "n_2gram": np.int64(lm.get("n_2gram", 0)),
        "n_3gram": np.int64(lm.get("n_3gram", 0)),
    }
    if "order" in lm:
        out["order"] = np.int64(lm["order"])
    if shuffled_word_id is not None:
        out["shuffled_word_id"] = np.asarray(shuffled_word_id, dtype=np.int64)
        assert out["shuffled_word_id"].shape == out["word_id"].shape
    np.savez(path, **out)


def load_resources(
    path: str, *, device=None, dtype: torch.dtype = torch.float32, shuffled: bool = False
) -> LexResources:
    """Read :func:`save_resources`' npz back into a :class:`LexResources`.

    ``shuffled=True`` swaps in the derangement's word-id column and NOTHING else -- the trie
    topology, the automaton and the escape prices are the arm's own (the null of Design 6).

    An npz that stores ``order`` (the optional key :func:`save_resources` writes when the tables
    carry one) sets :attr:`LexResources.order`, and with it the depth at which
    :func:`string_best_segmentation` and :func:`string_log_sum` walk the back-off chain; one that
    does not loads as :data:`DEFAULT_LM_ORDER` and scores exactly as it always did.
    """
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in RESOURCE_ARRAYS + RESOURCE_SCALARS
                   if k not in z and (k != "shuffled_word_id" or shuffled)]
        assert not missing, f"{path} is not a lexlat resource: {missing} are missing"
        trie = {"child": z["child"], "word_start": z["word_start"],
                "word_id": z["shuffled_word_id"] if shuffled else z["word_id"],
                "n_nodes": int(z["n_nodes"]), "n_entries": int(z["n_entries"])}
        lm = {k: z[k] for k in ("arc_key", "arc_logp", "arc_next", "backoff", "backoff_state")}
        lm.update({k: int(z[k]) for k in ("n_states", "n_words", "begin_state", "unk_word",
                                          "n_1gram", "n_2gram", "n_3gram")})
        return LexResources.build(
            trie, lm, z["escape_phone_log_prob"], n_phones=int(z["n_phones"]),
            sil_id=int(z["sil_id"]), device=device, dtype=dtype,
            words=[str(w) for w in z["words"]],
            order=int(z["order"]) if "order" in z else DEFAULT_LM_ORDER,
        )
