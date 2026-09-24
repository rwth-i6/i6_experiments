"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lexlat_k2_official.py (verbatim but imports).

emc.lexlat_k2_official -- the OFFICIAL LibriSpeech lexicon and LMs as ``L`` and ``G``.

WHAT THIS IS.  ``lexlat_k2`` compiles ``H . L . G`` from the phase's OWN resources: the 151,731-word
trie of ``LexiconTrieBuildJob`` and the CSR back-off form of our 1,000,000-line word trigram.  This
module compiles the SAME ``H . L . G``, through the SAME functions, from the OFFICIAL LibriSpeech
resources instead -- the openslr ``librispeech-lexicon.txt`` and an ARPA of ANY ORDER -- so the k2
cost probe can be read beside the standard recognition graphs of the field
(icefall's ``3-gram.pruned.1e-7``; Galvez et al. 2023's ``3-gram.pruned.3e-7``; the full 4-gram).

WHAT IS SHARED AND WHAT IS NEW.  Everything downstream of the tables is ``lexlat_k2``'s, imported
and never re-implemented: :func:`~.lexlat_k2.h_topology`, :func:`~.lexlat_k2.add_lex_disambig`,
:func:`~.lexlat_k2.lexicon_to_fst` (with its ESCAPE phone loop), :func:`~.lexlat_k2.escape_prices`,
:func:`~.lexlat_k2.g_fsa`, :func:`~.lexlat_k2.prune_lm_tables` and
:func:`~.lexlat_k2.compile_hlg`, at :data:`~.lexlat_k2.SIL_PROB` and with ``determinize = False``
always.  What is new here is only what produces their INPUTS:

* :func:`parse_arpa_lm` -- an ARPA of any order into the CSR back-off layout
  ``lexlat.parse_arpa_word_lm`` writes for order 3, PLUS an ``order`` field and the per-state
  ``state_order`` array that ``lexlat_k2.state_orders`` needs above order 3.  Back-off stays an
  epsilon (``#0``) arc per state, exactly as now.
* :func:`read_official_lexicon` / :func:`official_prons` -- the openslr lexicon, stress digits
  stripped, mapped onto the bed's ``prior.PHONES`` (39 ARPAbet + SIL) and restricted to ``G``'s
  vocabulary.
* :func:`build_trie_multi` -- ``lexlat.build_trie``'s flat int32 trie for a lexicon with SEVERAL
  pronunciations per word (the official lexicon has 200,130 pronunciations for 200,000 words; the
  banked trie keeps one per word and its builder takes a word -> pronunciation MAPPING).
* :func:`derange_official_prons` -- the NULL's assignment, ``lexlat.derange_pronunciations``
  (Sattolo, seed 0) reused verbatim on the official lexicon: a word's whole pronunciation block
  moves, so the pronunciation multiset of ``L`` is untouched and only the word identity -- which
  word-LM arc a pronunciation ends at -- changes.

TWO BUILD OPTIONS, both recorded in the build record and both hash-excluded at their default by
:class:`~.hlg.LexlatOfficialHLGBuildJob` so the three official graphs of
2026-09-21 keep their hashes: ``backoff_loops`` (``lexlat_k2.BACKOFF_LOOP_PLACEMENTS``; the
``word_boundary`` placement is what removes the positional multiplicity that failed amendment
9.5's over-count read in-house) and ``shuffled`` (the derangement above).

THE ESCAPE CONVENTION IS NOT RE-DECLARED.  The per-phone escape price vector is READ FROM THE
PHASE'S OWN ``lexlat_resources.npz`` (:func:`read_escape_prices`) -- the same order-1 Witten-Bell
phone log probabilities plus ``lexlat.ESCAPE_LENGTH_LOG_PROB`` the banked graph carries, on the
same 40-phone inventory -- so the escape leg of an official graph and of the banked graph are
priced identically and the comparison is a comparison of ``L`` and ``G`` alone.  The word cost of
a span is the LM's own unknown-word arc, whatever that LM spells it (``<unk>`` in ours, ``<UNK>``
in the official ARPAs); :func:`unk_word_id` finds it and ``build.json`` records the spelling.

NOT DETERMINIZED, as in ``lexlat_k2``: k2's ``determinize`` is tropical-only and would under-count
the log-semiring total the probe measures, and the escape loop is not determinizable at all.

THE PRUNING LADDER DROPS ARCS, NOT STATES.  ``lexlat_k2.prune_lm_tables`` removes n-gram arcs and
re-derives the back-off weights; the STATE set is untouched, so ``G``'s state count is the ARPA's
at every rung and only its n-gram arcs shrink.  The states no surviving arc reaches cost ``G``'s
own memory and nothing else: ``k2.compose`` explores the product forward from the two start states
and ``connect`` drops the rest.  This is stated in the build job's ``summary.txt`` rather than
worked around, because a rung is chosen on whether the graph COMPILES, which is what is measured.

Run as a CHILD MODULE under the k2 environment (``hlg.LexlatOfficialHLGBuildJob``: ``python -m
<package>.lm.lexlat_k2_official --help`` lists the sub-commands), with the directory holding the
``i6_experiments`` package as the child's whole ``PYTHONPATH``; nothing it imports needs sisyphus.  ``arpa-to-npz`` is a separate command so
that an ARPA is parsed ONCE per build job and every rung of the pruning ladder re-reads the npz
instead of the 4.4 GB text.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# RELATIVE imports: this file runs as ``python -m`` inside its package (``hlg.py``), under the k2
# environment, with the directory holding ``i6_experiments`` as the child's whole ``PYTHONPATH``.
from ..model.lexlat import ESCAPE_LENGTH_LOG_PROB, derange_pronunciations
from ..model.lexlat_k2 import (
    BACKOFF_LOOPS_ALL,
    BACKOFF_LOOP_PLACEMENTS,
    NEG_INF,
    SIL_PROB,
    add_lex_disambig,
    compile_hlg,
    emission_min_frames,
    escape_prices,
    g_fsa,
    h_topology,
    lexicon_to_fst,
    prune_lm_tables,
    read_lm_tables,
    state_orders,
    _lookup,
    _max_rss_gib,
    _sizes,
)
from ..phones import PHONE2ID, PHONES, SIL


__all__ = [
    "LN10",
    "UNK_SPELLINGS",
    "BOS_WORD",
    "EOS_WORD",
    "NON_LEXICAL_WORDS",
    "parse_arpa_lm",
    "save_lm_npz",
    "unk_word_id",
    "read_escape_prices",
    "read_official_lexicon",
    "official_prons",
    "derange_official_prons",
    "DERANGEMENT_SEED",
    "read_raw_escape_prices",
    "build_trie_multi",
    "build_official_hlg",
    "compact_lm_tables",
    "predict_hlg_cost",
]


#: log(10): the ARPA's log10 probabilities become nats here, as in ``lexlat.parse_arpa_word_lm``
LN10 = math.log(10.0)

#: THE REFERENCE BUILD, measured, not assumed: ``LexlatHLGBuildJob.rtX44PBJFNy1/output/build.json``
#: -- the banked word trigram compiled with this very code path (escape on, determinize off) under
#: the same 200 GB allocation.  ``predict_hlg_cost`` extrapolates from it linearly in the G FSA's
#: ARC COUNT, which is the only pre-compose number available.
REF_G_ARCS = 20578578
REF_HLG_ARCS = 143874554
REF_PEAK_GIB = 26.0406494140625
#: 6.99 HLG arcs per G arc, and 194 bytes of peak RSS per HLG arc
HLG_ARCS_PER_G_ARC = REF_HLG_ARCS / REF_G_ARCS
GIB_PER_HLG_ARC = REF_PEAK_GIB / REF_HLG_ARCS

#: ``build-hlg``'s exit code for "the size guard fired, nothing was composed" -- distinct from 0
#: (built) and from a crash, so the parent can record the prediction and move to the next rung
EXIT_SIZE_GUARD = 3

#: the spellings an ARPA may give the unknown word.  Ours writes ``<unk>`` (``lmplz``); the
#: official LibriSpeech ARPAs write ``<UNK>``.  The escape's word cost is THAT word's arc.
UNK_SPELLINGS: Tuple[str, ...] = ("<unk>", "<UNK>")
BOS_WORD = "<s>"
EOS_WORD = "</s>"
#: the words of the LM that the lexicon must never carry as an ORTHOGRAPHY: the two sentence
#: markers have no pronunciation and the bed adds no end-of-sequence term, and the unknown word is
#: reached only through the ESCAPE phone loop, never through a pronunciation of its own.
NON_LEXICAL_WORDS: Tuple[str, ...] = (BOS_WORD, EOS_WORD) + UNK_SPELLINGS

#: the seed of the shuffled-pronunciation null (`SAE_4A_lexlat.md` Design 6: "a fixed-seed
#: derangement").  IT IS ``word_lm.DERANGEMENT_SEED``, the in-house null's own seed, restated
#: here only because this file runs as a child module under the k2 environment, where ``sisyphus``
#: (and hence ``word_lm``) is not importable; ``tests/test_lm_word_lm.py`` asserts the two are
#: equal, so the two nulls can never drift apart.
DERANGEMENT_SEED = 0


# ===================================================================================================
# G -- an ARPA of any order into the CSR back-off layout
# ===================================================================================================
def _open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if str(path).endswith(".gz") \
        else open(path, "r", encoding="utf-8", errors="replace")


def _encode(words: np.ndarray, v: int) -> np.ndarray:
    """``[N, L]`` word ids (most recent LAST) -> one int64 key per row, base ``v``."""
    words = np.asarray(words)
    n_cols = int(words.shape[1])
    assert n_cols >= 1
    # v ** n_cols must fit an int64 key; above order 5 at a 200 k vocabulary it does not
    assert n_cols * math.log(max(v, 2)) < 63.0 * math.log(2.0), (
        f"a context of {n_cols} words over a vocabulary of {v} does not fit an int64 key")
    out = words[:, 0].astype(np.int64)
    for c in range(1, n_cols):
        out = out * np.int64(v) + words[:, c].astype(np.int64)
    return out


class _Block:
    """The listed k-grams of one ARPA order, as a sorted key table for ``searchsorted``."""

    def __init__(self, words: np.ndarray, v: int, base_state: int):
        self.base_state = int(base_state)
        key = _encode(words, v)
        self.order_perm = np.argsort(key, kind="stable")
        self.key = key[self.order_perm]
        assert self.key.size == 0 or np.all(np.diff(self.key) > 0), (
            "the ARPA lists the same n-gram twice")

    def find(self, key: np.ndarray) -> np.ndarray:
        """File-order index of each key, or -1 where the ARPA does not list it."""
        if self.key.size == 0:
            return np.full(key.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.key, key)
        pos = np.minimum(pos, self.key.size - 1)
        hit = self.key[pos] == key
        return np.where(hit, self.order_perm[pos], np.int64(-1))


def _context_state(grams: Dict[int, np.ndarray], blocks: Dict[int, _Block], cols: np.ndarray,
                   *, v: int, order: int) -> np.ndarray:
    """State id of each context row ``cols [N, L]`` (most recent LAST), backing off to a shorter one.

    ``lexlat.parse_arpa_word_lm``'s ``state_of``, vectorised and for any order: the LONGEST listed
    context, at most ``order - 1`` words; a single word is always a state (``1 + w``) and the empty
    context is the null state 0.
    """
    n = int(cols.shape[0])
    out = np.zeros(n, dtype=np.int64)
    found = np.zeros(n, dtype=bool)
    for length in range(min(int(cols.shape[1]), order - 1), 0, -1):
        sub = cols[:, cols.shape[1] - length:]
        if length == 1:
            out = np.where(found, out, 1 + sub[:, 0].astype(np.int64))
            found = np.ones(n, dtype=bool)
            break
        idx = blocks[length].find(_encode(sub, v))
        hit = (~found) & (idx >= 0)
        out = np.where(hit, blocks[length].base_state + np.maximum(idx, 0), out)
        found = found | hit
    return out


def parse_arpa_lm(path: str, *, log=print) -> Dict[str, object]:
    """An ARPA word LM of ANY order -> the CSR back-off automaton ``lexlat_k2`` reads.

    THE LAYOUT IS ``lexlat.parse_arpa_word_lm``'s, extended.  ``0`` is the null (unigram) context,
    ``1 + w`` is the context of the single word ``w``, and the contexts of the listed k-grams for
    ``k = 2 ... order - 1`` follow in file order, one block per k.  ``arc_key = state * V + word``
    is a sorted int64 table with the n-gram's log probability in NATS and its successor state --
    the longest listed context, KenLM's own rule, which is what ``lexlat_k2._lookup`` walks.  Every
    state carries its back-off weight and its back-off state (the context with its oldest word
    dropped, backed off to a listed one), so back-off stays an epsilon arc in ``g_fsa`` exactly as
    for the order-3 tables.

    TWO NEW FIELDS, and the only reason ``read_lm_tables`` had to learn anything: ``order``, and
    ``state_order[s]`` -- the order of the n-grams leaving state ``s``.  Above order 3 the block
    boundaries are the ARPA's own counts and cannot be derived from the vocabulary size, which is
    all the order-3 layout ever needed.

    A LISTED N-GRAM WHOSE CONTEXT THE ARPA DOES NOT LIST gets that context CREATED, which is what
    KenLM does and therefore what "this G scores like KenLM" requires.  A pruned ARPA may list
    ``A B C`` without listing ``A B``; the created context carries back-off weight 0 (KenLM's rule
    for an absent context), backs off to its own longest listed suffix, and is entered by a new
    arc whose log probability is the BACKED-OFF value ``p(B | A)`` the lower orders already give.
    Without it the listed ``A B C`` would be unreachable and the walk would answer with ``p(C |
    B)``: 0.46 nats adrift of KenLM on the order-4 fixture.  ``n_{o}gram_blank`` counts them, and
    on an ARPA with no hole -- every ``lmplz`` output, the banked trigram included -- none is
    created and the tables are unchanged.
    """
    started = time.monotonic()
    counts: Dict[int, int] = {}
    order = 0
    with _open_text(path) as fh:
        line = fh.readline()
        while line and not line.startswith("\\data\\"):
            line = fh.readline()
        assert line, f"{path}: no \\data\\ header"
        while True:
            line = fh.readline()
            if not line or not line.strip():
                break
            if line.startswith("ngram "):
                o, c = line[6:].split("=")
                counts[int(o)] = int(c)
                order = max(order, int(o))
        assert order >= 1 and sorted(counts) == list(range(1, order + 1)), counts
        log(f"ARPA {path}: order {order}, counts {counts}", flush=True)

        words: List[str] = []
        word2id: Dict[str, int] = {}
        gram_words: Dict[int, np.ndarray] = {}
        gram_lp: Dict[int, np.ndarray] = {}
        gram_bo: Dict[int, np.ndarray] = {}
        for o in range(1, order + 1):
            gram_words[o] = np.zeros((counts[o], o), dtype=np.int32)
            # the log10 probability is kept in FLOAT64 until it is multiplied by ``LN10`` and cast
            # once, which is what ``lexlat.parse_arpa_word_lm`` does; rounding to float32 first
            # moves the stored nats by up to 2.4e-7 and the order-3 tables would stop being that
            # function's bit for bit
            gram_lp[o] = np.zeros(counts[o], dtype=np.float64)
            gram_bo[o] = np.zeros(counts[o], dtype=np.float32)

        cur, filled = 0, 0
        n_oov_token = 0
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("\\") and s.endswith("-grams:"):
                assert cur == 0 or filled == counts[cur], (cur, filled, counts.get(cur))
                cur, filled = int(s[1:].split("-")[0]), 0
                continue
            if s == "\\end\\":
                break
            if cur == 0:
                continue
            fields = s.split()
            assert len(fields) in (cur + 1, cur + 2), f"{path}: {s!r} is not a {cur}-gram line"
            if cur == 1:
                w = fields[1]
                if w not in word2id:
                    word2id[w] = len(words)
                    words.append(w)
                gram_words[1][filled, 0] = word2id[w]
            else:
                row = gram_words[cur][filled]
                bad = False
                for j in range(cur):
                    i = word2id.get(fields[1 + j])
                    if i is None:  # a word with no unigram: the ARPA cannot score it at all
                        bad = True
                        break
                    row[j] = i
                if bad:
                    n_oov_token += 1
                    gram_lp[cur][filled] = np.nan
                    filled += 1
                    continue
            gram_lp[cur][filled] = float(fields[0])
            if len(fields) == cur + 2:
                gram_bo[cur][filled] = float(fields[cur + 1])
            filled += 1
        assert cur == order and filled == counts[order], (cur, filled, counts[order])
    v = len(words)
    assert v == counts[1], (v, counts[1])
    log(f"  parsed in {time.monotonic() - started:.0f} s: {v} words, "
        f"{sum(counts.values())} n-grams, {n_oov_token} with an unlisted word", flush=True)

    # --- KenLM's BLANK CONTEXTS: create the contexts the ARPA lists an n-gram over but not ------
    # itself, top down, so that a context created at level k is itself given one at level k - 1.
    # A context of ONE word is always a state (``1 + w``) and every word has a unigram line
    # (``words`` is built from them), so the pass stops at k = 3.
    n_rows = {o: int(gram_words[o].shape[0]) for o in range(1, order + 1)}
    blank = {o: np.zeros(n_rows[o], dtype=bool) for o in range(1, order + 1)}
    for k in range(order, 2, -1):
        pref = gram_words[k][~np.isnan(gram_lp[k])][:, :k - 1]
        if pref.size == 0:
            continue
        uniq, first = np.unique(_encode(pref, v), return_index=True)
        listed = gram_words[k - 1][~np.isnan(gram_lp[k - 1])]
        missing = ~np.isin(uniq, _encode(listed, v), assume_unique=False)
        n_new = int(missing.sum())
        if n_new == 0:
            continue
        rows = pref[first[missing]].astype(np.int32)
        gram_words[k - 1] = np.concatenate([gram_words[k - 1], rows])
        gram_lp[k - 1] = np.concatenate([gram_lp[k - 1], np.zeros(n_new, dtype=np.float64)])
        gram_bo[k - 1] = np.concatenate([gram_bo[k - 1], np.zeros(n_new, dtype=np.float32)])
        blank[k - 1] = np.concatenate([blank[k - 1], np.ones(n_new, dtype=bool)])
        n_rows[k - 1] += n_new
        log(f"  {n_new} {k - 1}-gram contexts created: the ARPA lists {k}-grams over them but "
            f"not the {k - 1}-gram itself (KenLM's blank, back-off 0)", flush=True)

    # --- the state blocks: 0, then 1 .. V, then the k-grams for k = 2 .. order - 1 ---------------
    blocks: Dict[int, _Block] = {}
    base = 1 + v
    state_order_blocks: List[Tuple[int, int, int]] = [(0, 1, 1), (1, 1 + v, 2)]
    for k in range(2, order):
        blocks[k] = _Block(gram_words[k], v, base)
        state_order_blocks.append((base, base + n_rows[k], k + 1))
        base += n_rows[k]
    n_states = base
    state_order = np.zeros(n_states, dtype=np.int8)
    for lo, hi, o in state_order_blocks:
        state_order[lo:hi] = o
    backoff = np.zeros(n_states, dtype=np.float32)
    backoff_state = np.zeros(n_states, dtype=np.int64)

    keys: List[np.ndarray] = []
    logp: List[np.ndarray] = []
    nxt: List[np.ndarray] = []
    n_unreachable = 0
    per_order: Dict[int, int] = {}
    for o in range(1, order + 1):
        w = gram_words[o]
        ok = ~np.isnan(gram_lp[o])
        # EVERY listed n-gram below the top order OWNS a state, whether or not its OWN arc
        # survives the reachability filter below -- ``_Block`` numbers one per listed n-gram, and
        # a surviving (o+1)-gram can land on it through ``arc_next``.  ``ok_state`` is therefore
        # the mask BEFORE that filter, and the back-off block is assigned from it: assigning from
        # the filtered mask left such a state at ``backoff = 0, backoff_state = 0``, so
        # ``_lookup`` skipped straight to the null context and paid no back-off weight (measured
        # at up to 2.07 nats per sentence on the order-4 fixture with exactly that hole).  Below
        # order 4 the filter never runs, so ``ok_state is ok`` and the order-3 tables do not move.
        ok_state = ok
        if o == 1:
            src = np.zeros(int(ok.sum()), dtype=np.int64)
        else:
            src = _context_state(gram_words, blocks, w[ok][:, :o - 1], v=v, order=order)
            # the context must be a state of its OWN order, not a backed-off shorter one.  The
            # blank pass above created every one that the ARPA left out, so this filter now only
            # catches an n-gram whose context carries a word the ARPA never listed.
            if o >= 3:
                want = blocks[o - 1].find(_encode(w[ok][:, :o - 1], v))
                live = want >= 0
                n_unreachable += int((~live).sum())
                src = src[live]
                sel = np.zeros(ok.size, dtype=bool)
                sel[np.nonzero(ok)[0][live]] = True
                ok = sel
        last = w[ok][:, o - 1].astype(np.int64)
        lp_o = gram_lp[o][ok].astype(np.float64) * LN10
        is_blank = blank[o][ok]
        if bool(is_blank.any()):
            # a created context's own arc carries the BACKED-OFF probability the levels below it
            # already give -- KenLM's blank -- so it is read off the automaton built so far.  The
            # walk always terminates: level 1 has an arc for every word (``v == counts[1]``).
            # ``levels = o``, not ``o - 1``: the walk starts at a context of order
            # ``o - 1``, whose own arcs are the level-``o`` ones being built and are
            # therefore always a miss, so the search really begins one level down.
            part = {"arc_key": np.concatenate(keys), "arc_logp": np.concatenate(logp),
                    "arc_next": np.concatenate(nxt), "backoff": backoff * np.float32(LN10),
                    "backoff_state": backoff_state, "n_words": v}
            perm_p = np.argsort(part["arc_key"], kind="stable")
            for f in ("arc_key", "arc_logp", "arc_next"):
                part[f] = np.ascontiguousarray(part[f][perm_p])
            part["arc_logp"] = part["arc_logp"].astype(np.float32)
            lp_o[is_blank] = _lookup(part, src[is_blank], last[is_blank], levels=o)[0]
            assert bool(np.all(lp_o[is_blank] < 0.0)), (
                "a created context's backed-off probability must come out of the lower orders")
        keys.append(src * np.int64(v) + last)
        logp.append(lp_o)
        nxt.append(_context_state(gram_words, blocks, w[ok], v=v, order=order))
        per_order[o] = int(ok.sum())
        if o < order:  # these n-grams ARE states: give them their back-off weight and target
            states = (blocks[o].base_state + np.nonzero(ok_state)[0] if o >= 2
                      else 1 + w[ok_state][:, 0].astype(np.int64))
            backoff[states] = gram_bo[o][ok_state]
            # the back-off target is the longest LISTED suffix.  Skipping an ABSENT intermediate
            # context is exactly KenLM's rule -- an unlisted context has back-off 0 -- and the
            # order-4 fixture exercises it (``A B D`` whose suffix ``B D`` is unlisted).
            backoff_state[states] = (
                np.zeros(states.size, dtype=np.int64) if o == 1 else
                _context_state(gram_words, blocks, w[ok_state][:, 1:], v=v, order=order))
    key = np.concatenate(keys)
    perm = np.argsort(key, kind="stable")
    key = key[perm]
    assert np.all(np.diff(key) > 0), f"{path}: duplicate (state, word) arcs"
    unk = unk_word_id(words)
    out: Dict[str, object] = {
        "arc_key": key,
        "arc_logp": np.concatenate(logp)[perm].astype(np.float32),
        "arc_next": np.concatenate(nxt)[perm],
        "backoff": backoff * np.float32(LN10),
        "backoff_state": backoff_state,
        "state_order": state_order,
        "order": int(order),
        "n_states": int(n_states),
        "n_words": int(v),
        "begin_state": int(1 + word2id[BOS_WORD]) if BOS_WORD in word2id else 0,
        "unk_word": int(unk),
        "words": words,
        "arpa": os.path.realpath(str(path)),
        "n_unreachable": int(n_unreachable),
        "n_unlisted_token": int(n_oov_token),
        "seconds": time.monotonic() - started,
    }
    for o in range(1, order + 1):
        out[f"n_{o}gram"] = int(per_order[o])
        out[f"n_{o}gram_listed"] = int(counts[o])
        out[f"n_{o}gram_blank"] = int(blank[o].sum())
    out["n_blank"] = int(sum(int(blank[o].sum()) for o in range(1, order + 1)))
    log(f"  automaton: {n_states} states, {key.size} arcs, order {order}, "
        f"{out['n_blank']} created contexts, "
        f"{n_unreachable} unreachable n-grams dropped, peak RSS {_max_rss_gib():.1f} GiB",
        flush=True)
    return out


def compact_lm_tables(lm: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, int]]:
    """Drop the states no path from the begin state reaches, and renumber.  Scores cannot move.

    WHY IT IS NEEDED.  ``lexlat_k2.prune_lm_tables`` drops n-gram ARCS and leaves the STATE set
    in place, and ``lexlat_k2.g_fsa`` gives EVERY state a back-off arc and a ``-1`` final arc.  So
    a pruned G still carries ``2 * n_states`` arcs whatever the threshold: on the official 4-gram
    that is 168.7 M arcs before a single n-gram arc is counted, and a ladder rung would move the
    compiled G by a factor of about 1.8 rather than by the factor its arc ratio suggests.  Pruning
    strands most of the higher-order context states -- nothing reaches them once their incoming
    n-gram arcs are gone -- and this is what removes them.

    IT CANNOT CHANGE A SCORE.  Reachability is computed forward from ``begin_state`` over the
    n-gram arcs (``arc_next``) AND the back-off arcs (``backoff_state``), i.e. over every arc
    ``g_fsa`` emits, so a dropped state has no path from the start of the FSA; no word sequence
    reaches it and ``k2.connect`` after the composition would have removed its image anyway.  What
    changes is the size of the object the compiler has to hold.

    The null context stays state 0 -- it is reachable (every state's back-off chain ends there)
    and it is the lowest id, and the renumbering is monotone -- which is what ``g_fsa``'s
    begin-state swap and its "the null context has no back-off arc" line need.
    """
    started = time.monotonic()
    v = int(lm["n_words"])
    n_states = int(lm["n_states"])
    order = int(lm.get("order", 3))
    arc_key = np.asarray(lm["arc_key"])
    arc_next = np.asarray(lm["arc_next"])
    backoff_state = np.asarray(lm["backoff_state"])
    src = (arc_key // np.int64(v)).astype(np.int64)

    reach = np.zeros(n_states, dtype=bool)
    reach[int(lm["begin_state"])] = True
    rounds, seen = 0, 0
    while True:
        rounds += 1
        reach[backoff_state[reach]] = True
        reach[arc_next[reach[src]]] = True
        now = int(reach.sum())
        if now == seen:
            break
        seen = now
    assert bool(reach[0]), "the null context must be reachable: every back-off chain ends there"

    new_id = np.cumsum(reach, dtype=np.int64) - 1
    new_id[~reach] = -1
    keep = reach[src]
    new_key = new_id[src[keep]] * np.int64(v) + (arc_key[keep] % np.int64(v))
    assert new_key.size == 0 or np.all(np.diff(new_key) > 0), (
        "the renumbering is monotone, so the key table must stay sorted")
    n_after = int(reach.sum())
    out = dict(lm)
    out.update({
        "arc_key": new_key,
        "arc_logp": np.asarray(lm["arc_logp"])[keep],
        "arc_next": new_id[arc_next[keep]],
        "backoff": np.asarray(lm["backoff"])[reach],
        "backoff_state": new_id[backoff_state[reach]],
        "state_order": state_orders(lm)[reach].astype(np.int8),
        "n_states": n_after,
        "begin_state": int(new_id[int(lm["begin_state"])]),
    })
    assert int(out["arc_next"].min(initial=0)) >= 0 and int(out["backoff_state"].min()) >= 0, (
        "a surviving arc points at a dropped state")
    orders = np.asarray(out["state_order"])[(new_key // np.int64(v))]
    for o in range(1, order + 1):
        out[f"n_{o}gram"] = int((orders == o).sum())
    stats = {
        "states_before": n_states, "states_after": n_after,
        "ngram_arcs_before": int(arc_key.size), "ngram_arcs_after": int(new_key.size),
        # what ``g_fsa`` will emit: one arc per n-gram, one back-off arc and one final arc per
        # state, minus the null context's back-off arc.  An UPPER BOUND: ``g_fsa`` also drops the
        # arcs of the words L can never emit (``<s>``, ``</s>``), a handful.  The size guard
        # itself prices the COMPILED G's own arc count, not this one.
        "g_arcs_before": int(arc_key.size) + 2 * n_states - 1,
        "g_arcs_after": int(new_key.size) + 2 * n_after - 1,
        "rounds": rounds, "seconds": time.monotonic() - started,
    }
    return out, stats


def predict_hlg_cost(g_arcs: int) -> Dict[str, float]:
    """Predict ``H . L . G``'s arc count and peak RSS from the G FSA's arc count.

    Linear extrapolation from THE REFERENCE BUILD (:data:`REF_G_ARCS`, :data:`REF_HLG_ARCS`,
    :data:`REF_PEAK_GIB` -- the banked trigram graph, measured, same code path and same
    allocation): 6.99 HLG arcs per G arc and 194 bytes of peak RSS per HLG arc.  It is a
    ONE-POINT fit and it is used only to SKIP a rung that cannot fit, never to claim one will.
    """
    predicted_arcs = float(g_arcs) * HLG_ARCS_PER_G_ARC
    return {
        "g_arcs": int(g_arcs),
        "predicted_hlg_arcs": predicted_arcs,
        "predicted_peak_gib": predicted_arcs * GIB_PER_HLG_ARC,
        "reference": {"g_arcs": REF_G_ARCS, "hlg_arcs": REF_HLG_ARCS,
                      "peak_gib": REF_PEAK_GIB,
                      "job": "LexlatHLGBuildJob.rtX44PBJFNy1"},
    }


def save_lm_npz(lm: Dict[str, object], path: str) -> None:
    """Write the tables of :func:`parse_arpa_lm` where ``lexlat_k2.read_lm_tables`` reads them."""
    order = int(lm["order"])
    payload = {
        k: np.asarray(lm[k]) for k in
        ("arc_key", "arc_logp", "arc_next", "backoff", "backoff_state", "state_order")
    }
    payload["words"] = np.asarray([str(w) for w in lm["words"]])
    for k in ("n_states", "n_words", "begin_state", "unk_word", "order"):
        payload[k] = np.int64(lm[k])
    for o in range(1, order + 1):
        payload[f"n_{o}gram"] = np.int64(lm[f"n_{o}gram"])
    # ``read_lm_tables`` asks every npz for the bed's inventory; an LM npz carries no trie, so
    # these two are the only lexicon-side fields it needs and they are the bed's own constants
    payload["n_phones"] = np.int64(len(PHONES))
    payload["sil_id"] = np.int64(PHONE2ID[SIL])
    np.savez(path, **payload)


def unk_word_id(words: Sequence[str]) -> int:
    """The id of the LM's unknown word -- the escape's word cost -- whatever it spells it."""
    index = {str(w): i for i, w in enumerate(words)}
    for spelling in UNK_SPELLINGS:
        if spelling in index:
            return int(index[spelling])
    raise AssertionError(
        f"the LM lists none of {UNK_SPELLINGS}, so the ESCAPE span has no word cost")


def read_raw_escape_prices(resources_npz: str, *, n_phones: int, sil_id: int) -> np.ndarray:
    """The RAW ``escape_phone_log_prob`` column of ``lexlat_resources.npz``, inventory asserted.

    RAW means exactly what the npz stores and what ``PriorGapAnalysisJob`` banks: the order-1
    Witten-Bell phone log probabilities, WITHOUT ``lexlat.ESCAPE_LENGTH_LOG_PROB`` and without the
    SIL floor.  :func:`read_escape_prices` adds both (through ``lexlat_k2.escape_prices``, which is
    what the graph is built with) and ``lexlat.LexResources.build`` adds both itself, so an
    official resources npz must be written from THIS vector and never from the priced one.
    """
    with np.load(resources_npz, allow_pickle=False) as z:
        raw = np.asarray(z["escape_phone_log_prob"], dtype=np.float64)
        assert int(z["n_phones"]) == int(n_phones), (int(z["n_phones"]), n_phones)
        assert int(z["sil_id"]) == int(sil_id), (int(z["sil_id"]), sil_id)
    return raw


def read_escape_prices(resources_npz: str, *, n_phones: int, sil_id: int) -> np.ndarray:
    """The PHASE'S OWN per-phone escape price vector, read off ``lexlat_resources.npz``.

    The official graphs change ``L`` and ``G``; they do NOT change the escape convention, so this
    is ``lexlat_k2.escape_prices`` applied to the very ``escape_phone_log_prob`` the banked graph
    carries -- the order-1 Witten-Bell phone log probabilities of ``PriorGapAnalysisJob`` plus
    ``lexlat.ESCAPE_LENGTH_LOG_PROB``, SIL floored to ``NEG_INF``.  The inventory is asserted, so
    a resource built for another bed cannot price an official graph.
    """
    raw = read_raw_escape_prices(resources_npz, n_phones=n_phones, sil_id=sil_id)
    return escape_prices(raw, n_phones=int(n_phones), sil_id=int(sil_id))


# ===================================================================================================
# L -- the official pronunciation lexicon
# ===================================================================================================
_STRESS = re.compile(r"\d+$")


def read_official_lexicon(path: str) -> Tuple[List[Tuple[str, Tuple[str, ...]]], Dict[str, int]]:
    """openslr ``librispeech-lexicon.txt`` -> ``[(word, (phone, ...))]`` on the BED's inventory.

    The rule is ``w2vu2.word_decode.convert_lexicon_lines``', restated here because this file runs
    as a child module under the k2 environment, where only this package is importable:
    stress digits are stripped (``AH0 -> AH``) so the entry lands in the 39 ARPAbet inventory of
    ``prior.PHONES``, an entry with any remaining phone outside it is DROPPED and counted, and the
    duplicate ``(word, pronunciation)`` pairs stress-stripping creates are removed.  SIL is never
    a phone inside a word (it is the word boundary of this bed), so an entry carrying it is
    dropped with its own count rather than silently kept.

    Every phone of every kept entry is asserted to be in ``prior.PHONES``; the counters go into
    the build job's ``build.json``.
    """
    inventory = set(PHONES)
    entries: List[Tuple[str, Tuple[str, ...]]] = []
    seen = set()
    stats = {"lines": 0, "no_pron": 0, "oov_phone": 0, "has_sil": 0, "duplicate": 0, "kept": 0}
    with _open_text(path) as fh:
        for line in fh:
            fields = line.split()
            if not fields:
                continue
            stats["lines"] += 1
            word, phones = fields[0], fields[1:]
            if not phones:
                stats["no_pron"] += 1
                continue
            pron = tuple(_STRESS.sub("", p) for p in phones)
            if any(p not in inventory for p in pron):
                stats["oov_phone"] += 1
                continue
            if SIL in pron:
                stats["has_sil"] += 1
                continue
            if (word, pron) in seen:
                stats["duplicate"] += 1
                continue
            seen.add((word, pron))
            entries.append((word, pron))
    stats["kept"] = len(entries)
    assert entries, f"{path}: no entry survived the mapping onto {len(PHONES)} phones"
    for _w, pron in entries:
        assert all(p in inventory for p in pron), _w
    return entries, stats


def official_prons(
    entries: Sequence[Tuple[str, Tuple[str, ...]]],
    words: Sequence[str],
    *,
    drop_words: Sequence[str] = NON_LEXICAL_WORDS,
) -> Tuple[List[Tuple[int, List[int]]], Dict[str, int]]:
    """``[(word, pron)]`` + the LM's vocabulary -> ``[(lm_word_id, [phone ids])]``, G-restricted.

    ``L`` may only emit words ``G`` scores, so an entry whose orthography is not in the LM's
    vocabulary is dropped and counted.  ``<s>`` / ``</s>`` / ``<unk>`` / ``<UNK>`` are dropped AS
    WORDS: the two markers have no pronunciation and the bed adds no end-of-sequence term, and the
    unknown word is reached only through the ESCAPE phone loop, which takes its transition from
    ``G`` directly.  The order is ``(word id, pronunciation)``, the order
    ``lexlat_k2.prons_from_trie`` returns for the banked graph, so ``add_lex_disambig`` hands out
    the same indices it would there.
    """
    word2id = {str(w): i for i, w in enumerate(words)}
    drop = set(str(w) for w in drop_words)
    out: List[Tuple[int, List[int]]] = []
    stats = {"entries": len(entries), "oov_word": 0, "dropped_marker": 0,
             "words_with_pron": 0, "kept": 0}
    kept_words = set()
    for word, pron in entries:
        if word in drop:
            stats["dropped_marker"] += 1
            continue
        wid = word2id.get(word)
        if wid is None:
            stats["oov_word"] += 1
            continue
        out.append((int(wid), [PHONE2ID[p] for p in pron]))
        kept_words.add(word)
    out.sort(key=lambda t: (t[0], t[1]))
    stats["kept"] = len(out)
    stats["words_with_pron"] = len(kept_words)
    assert out, "no pronunciation survived the restriction to the LM's vocabulary"
    return out, stats


def derange_official_prons(
    prons: Sequence[Tuple[int, Sequence[int]]],
    words: Sequence[str],
    *,
    seed: int = None,  # type: ignore[assignment]
) -> Tuple[List[Tuple[int, List[int]]], Dict[str, object]]:
    """THE NULL's word -> pronunciation assignment: ``lexlat.derange_pronunciations``, seed 0.

    THE SAME CONSTRUCTION AS THE IN-HOUSE NULL, the function itself reused: Sattolo's algorithm
    under ``numpy.random.default_rng(seed)`` over the SORTED word list, one cycle, no fixed point
    (``lexlat.derange_pronunciations``, which ``LexiconTrieBuildJob`` calls for
    ``shuffled_word_id`` and hence for ``LexlatHLGBuildJob(shuffled=True)``).  The in-house lexicon
    gives each word ONE pronunciation and the official one gives it several, so what the
    permutation moves here is a word's WHOLE pronunciation block: the multiset of pronunciations in
    ``L`` is untouched (asserted), the number of entries is untouched, and only WHICH WORD -- i.e.
    which word-LM arc -- each pronunciation ends at moves.  That is Design 6's prior-weight control
    on the official lexicon: the null differs from the treatment in the word identity and in
    nothing else (the same trie shape, the same escape prices, the same ``G``).

    ``kept_by_homophony`` is ``lexlat.derange_pronunciations``' own honest residual -- the words
    whose shuffled block happens to equal their own, which a derangement over words cannot exclude
    when two words share pronunciations -- and it is reported, never repaired.

    Returns ``(prons, stats)`` in :func:`official_prons`' own order and format, so the deranged
    list is a drop-in for the real one in :func:`build_official_hlg` and :func:`build_trie_multi`.
    """
    seed = DERANGEMENT_SEED if seed is None else int(seed)
    by_word: Dict[str, List[Tuple[int, ...]]] = {}
    for wid, pron in prons:
        by_word.setdefault(str(words[int(wid)]), []).append(tuple(int(p) for p in pron))
    blocks = {w: sorted(v) for w, v in by_word.items()}
    deranged, stats = derange_pronunciations(blocks, seed=int(seed))
    index = {str(words[int(wid)]): int(wid) for wid, _pron in prons}
    out: List[Tuple[int, List[int]]] = [
        (index[w], list(pron)) for w, block in deranged.items() for pron in block]
    out.sort(key=lambda t: (t[0], t[1]))
    before = sorted(tuple(int(p) for p in pron) for _w, pron in prons)
    after = sorted(tuple(int(p) for p in pron) for _w, pron in out)
    assert after == before, (
        "the derangement moved the MULTISET of pronunciations: it is not the prior-weight control")
    return out, {**stats, "n_pronunciations": len(out), "n_words_with_pronunciation": len(blocks)}


def build_trie_multi(prons: Sequence[Tuple[int, Sequence[int]]], *, n_phones: int
                     ) -> Dict[str, np.ndarray]:
    """``lexlat.build_trie``'s flat int32 trie, for SEVERAL pronunciations per word.

    ``lexlat.build_trie`` takes a word -> pronunciation MAPPING (the banked trie keeps the first
    pronunciation of each word); the official lexicon gives a word several, so this takes the
    ``[(word id, pronunciation)]`` list :func:`official_prons` returns.  Everything else is that
    function's, verbatim in effect: nodes are created over the SORTED SET of pronunciations so the
    numbering is a function of the pronunciation multiset alone, homophones are a RANGE of the
    word-id array, and the empty pronunciation is not a word.  The trie is what
    ``lexlat.LexResources`` needs, i.e. what the max-plus equivalence check reads; the ``L``
    transducer itself is built from ``prons`` directly.
    """
    children: List[Dict[int, int]] = [{}]
    entries = [(tuple(int(p) for p in pron), int(word)) for word, pron in prons]
    assert all(pron for pron, _w in entries), "the empty pronunciation is not a word"
    for pron in sorted({e[0] for e in entries}):
        node = 0
        for k in pron:
            nxt = children[node].get(k)
            if nxt is None:
                nxt = len(children)
                children[node][k] = nxt
                children.append({})
            node = nxt
    n_nodes = len(children)
    ends: List[List[int]] = [[] for _ in range(n_nodes)]
    for pron, wid in entries:
        node = 0
        for k in pron:
            node = children[node][k]
        ends[node].append(wid)
    child = np.full((n_nodes, int(n_phones)), -1, dtype=np.int32)
    for node, kids in enumerate(children):
        for k, nxt in kids.items():
            child[node, k] = nxt
    starts = np.zeros(n_nodes + 1, dtype=np.int64)
    flat: List[int] = []
    for node in range(n_nodes):
        starts[node] = len(flat)
        flat.extend(sorted(ends[node]))
    starts[n_nodes] = len(flat)
    assert not ends[0], "the empty pronunciation is not a word"
    return {
        "child": child,
        "word_start": starts,
        "word_id": np.asarray(flat, dtype=np.int64),
        "is_word_end": (np.diff(starts) > 0),
        "n_nodes": np.int64(n_nodes),
        "n_entries": np.int64(len(flat)),
    }


# ===================================================================================================
# the official H . L . G
# ===================================================================================================
def build_official_hlg(
    *,
    lm: Dict[str, object],
    prons: Sequence[Tuple[int, Sequence[int]]],
    escape_price: Optional[np.ndarray],
    n_phones: int,
    sil_id: int,
    d_min: int,
    recognizer_stride: int,
    sil_prob: float = SIL_PROB,
    backoff_loops: str = BACKOFF_LOOPS_ALL,
    mem_guard_gib: Optional[float] = None,
    log=print,
) -> Tuple[Optional[object], Dict[str, object]]:
    """``H . L . G`` from the official tables, through ``lexlat_k2``'s own factors.

    ``escape_price = None`` builds the STRICT lexicon (no escape phone loop and no unknown-word
    arc in ``G``), a DISCLOSED DEVIATION from `SAE_4A_lexlat.md` Design 1, which fixes ESCAPE
    inside the marginal.  Never determinized, for the reason ``lexlat_k2.compile_hlg`` states.

    ``backoff_loops`` is ``lexlat_k2.lexicon_to_fst``'s own parameter, passed through unchanged:
    ``"all"`` is icefall's placement and the DEFAULT every official graph built before 2026-09-21
    carries, ``"word_boundary"`` puts the ``#0`` self-loops only where a word transition can start.
    The two differ in the LOG-semiring total alone -- under ``"all"`` one back-off route of a word
    transition is consumable at every position inside the word and an undeterminized graph sums
    each position as a separate path (``lexlat_k2._add_self_loops``,
    ``reports/audit_k2_overcount_2026-09-21.md`` section 3) -- so the placement is recorded in the
    build record and a graph must never be read for the other.

    THE SIZE GUARD.  ``H``, ``L`` and ``G`` are cheap (the reference build spends 0.5 s on all
    three and 142 s on the composition, at 2.7 GiB against a 26.0 GiB peak), so the graph that
    decides the cost is measured BEFORE the expensive step: with ``mem_guard_gib`` set, the peak
    RSS of the composition is predicted from ``G``'s arc count by :func:`predict_hlg_cost` and a
    rung over budget returns ``(None, record)`` with ``skipped_by_size_guard: True`` instead of
    composing.  That matters because the alternative is Slurm killing the JOB, which ends the
    whole ladder -- the subprocess boundary only catches an attempt that dies by itself.  The
    prediction is a one-point linear fit and is used ONLY to skip, never to promise a fit.
    """
    import k2

    assert backoff_loops in BACKOFF_LOOP_PLACEMENTS, backoff_loops
    n_words = int(lm["n_words"])
    unk = int(lm["unk_word"])
    escape = escape_price is not None
    entries, max_disambig = add_lex_disambig(prons)
    log(f"lexicon: {len(prons)} pronunciations, max disambig #{max_disambig}", flush=True)
    min_frames = emission_min_frames(d_min, recognizer_stride)
    H = h_topology(int(n_phones), min_frames=min_frames)
    log(f"backoff loops: {backoff_loops}"
        + ("  (icefall's placement, every L state with a non-eps token)"
           if backoff_loops == BACKOFF_LOOPS_ALL else
           "  (only where a word transition can start, so each back-off route of a word "
           "transition is consumed exactly once under the log semiring)"), flush=True)
    L = lexicon_to_fst(entries, n_phones=int(n_phones), sil_id=int(sil_id), n_words=n_words,
                       max_disambig=max_disambig, sil_prob=float(sil_prob),
                       escape_word=unk if escape else None,
                       escape_price=escape_price if escape else None,
                       backoff_loops=str(backoff_loops))
    drop = (BOS_WORD, EOS_WORD) if escape else (BOS_WORD, EOS_WORD, str(lm["words"][unk]))
    G = g_fsa(lm, drop_words=drop)
    log("escape: ON (the Design's ESCAPE convention)" if escape else
        "escape: OFF -- the STRICT lexicon, a DISCLOSED DEVIATION from the Design", flush=True)
    g_size = _sizes(G)
    guard = predict_hlg_cost(int(g_size["arcs"]))
    guard["mem_guard_gib"] = None if mem_guard_gib is None else float(mem_guard_gib)
    log(f"size guard: G has {g_size['arcs']} arcs over {g_size['states']} states -> "
        f"{guard['predicted_hlg_arcs'] / 1e6:.1f} M HLG arcs, "
        f"{guard['predicted_peak_gib']:.1f} GiB predicted peak "
        f"(budget {guard['mem_guard_gib']})", flush=True)
    record: Dict[str, object] = {
        "n_phones": int(n_phones), "sil_id": int(sil_id), "n_words": n_words,
        "sil_prob": float(sil_prob), "d_min": int(d_min),
        "recognizer_stride": int(recognizer_stride), "min_frames": int(min_frames),
        "n_pronunciations": len(prons), "max_disambig": int(max_disambig),
        "escape": bool(escape), "backoff_loops": str(backoff_loops), "determinized": False,
        "escape_word": str(lm["words"][unk]),
        "escape_length_log_prob": float(ESCAPE_LENGTH_LOG_PROB),
        "escape_disambig": (int(n_phones) + 1 + int(max_disambig) + 1) if escape else None,
        "escape_price_min": None if not escape else float(
            min(p for p in escape_price if p > NEG_INF / 2)),
        "escape_price_max": None if not escape else float(
            max(p for p in escape_price if p > NEG_INF / 2)),
        "dropped_words": list(drop),
        "h": _sizes(H), "l": _sizes(L), "g": g_size,
        "size_guard": guard, "k2": k2.__dev_version__,
    }
    if mem_guard_gib is not None and guard["predicted_peak_gib"] > float(mem_guard_gib):
        record["skipped_by_size_guard"] = True
        log(f"SKIPPED by the size guard: {guard['predicted_peak_gib']:.1f} GiB predicted peak "
            f"is over the {float(mem_guard_gib):.1f} GiB budget; not composing", flush=True)
        return None, record
    record["skipped_by_size_guard"] = False
    HLG, stages = compile_hlg(H, L, G, n_phones=int(n_phones), n_words=n_words, determinize=False)
    record["hlg"] = _sizes(HLG)
    record["stages"] = stages
    return HLG, record


# ===================================================================================================
# the sub-commands (this file is run as a child module under the k2 env python)
# ===================================================================================================
def _cmd_arpa_to_npz(args) -> int:
    started = time.monotonic()
    lm = parse_arpa_lm(args.arpa)
    save_lm_npz(lm, args.out_npz)
    order = int(lm["order"])
    record = {
        "arpa": lm["arpa"], "order": order, "n_states": int(lm["n_states"]),
        "n_words": int(lm["n_words"]), "n_arcs": int(np.asarray(lm["arc_key"]).size),
        "unk_word": str(lm["words"][int(lm["unk_word"])]),
        "begin_state": int(lm["begin_state"]),
        "n_unreachable": int(lm["n_unreachable"]),
        "n_unlisted_token": int(lm["n_unlisted_token"]),
        "ngram_counts": {str(o): int(lm[f"n_{o}gram"]) for o in range(1, order + 1)},
        "ngram_counts_listed": {str(o): int(lm[f"n_{o}gram_listed"])
                                for o in range(1, order + 1)},
        "seconds": time.monotonic() - started, "max_rss_gib": _max_rss_gib(),
    }
    with open(args.out_json, "w") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps(record), flush=True)
    return 0


def _cmd_build_hlg(args) -> int:
    import torch

    started = time.monotonic()
    lm = read_lm_tables(args.lm_npz)
    prune_stats = {"theta_nats": float(args.prune_theta)}
    if float(args.prune_theta) > 0.0:
        lm, prune_stats = prune_lm_tables(lm, float(args.prune_theta))
        print(f"pruned at theta = {args.prune_theta}: {prune_stats}", flush=True)
    # ALWAYS compact, at every rung including theta = 0: pruning drops arcs and not states, and
    # even the unpruned tables carry the states of the n-grams whose context the ARPA does not
    # list.  It cannot move a score (see compact_lm_tables) and it is what makes a rung shrink
    # the graph the compiler sees rather than only its n-gram arcs.
    lm, compact_stats = compact_lm_tables(lm)
    print(f"compacted: {compact_stats}", flush=True)
    n_phones, sil_id = len(PHONES), PHONE2ID[SIL]
    assert int(lm["n_phones"]) == n_phones and int(lm["sil_id"]) == sil_id, (
        "the LM npz was written for another inventory")
    entries, lex_stats = read_official_lexicon(args.lexicon)
    prons, restrict_stats = official_prons(entries, lm["words"])
    print(f"lexicon {args.lexicon}: {lex_stats}; restricted: {restrict_stats}", flush=True)
    null_stats: Optional[Dict[str, object]] = None
    if bool(args.shuffled):
        prons, null_stats = derange_official_prons(prons, lm["words"])
        print(f"THE NULL: seed-{DERANGEMENT_SEED} derangement of the word -> pronunciation "
              f"assignment ({null_stats})", flush=True)
    esc = (None if bool(args.strict_lexicon) else
           read_escape_prices(args.escape_resources, n_phones=n_phones, sil_id=sil_id))
    print("determinized: False -- k2's determinize is tropical-only, so a determinized graph "
          "would under-count the log-semiring total the probe measures (see compile_hlg)",
          flush=True)
    HLG, record = build_official_hlg(
        lm=lm, prons=prons, escape_price=esc, n_phones=n_phones, sil_id=sil_id,
        d_min=int(args.d_min), recognizer_stride=int(args.recognizer_stride),
        sil_prob=float(args.sil_prob), backoff_loops=str(args.backoff_loops),
        mem_guard_gib=(None if args.mem_guard_gib is None else float(args.mem_guard_gib)))
    if HLG is not None:
        torch.save(HLG.as_dict(), args.out_hlg)
    order = int(lm["order"])
    record.update({
        "lm_npz": os.path.realpath(args.lm_npz),
        "lexicon": os.path.realpath(args.lexicon),
        "shuffled": bool(args.shuffled),
        "derangement": null_stats,
        "escape_resources": (None if esc is None
                             else os.path.realpath(args.escape_resources)),
        "order": order,
        "lexicon_stats": lex_stats,
        "restrict_stats": restrict_stats,
        "vocab_size": int(lm["n_words"]),
        "n_words_with_pronunciation": int(restrict_stats["words_with_pron"]),
        "n_lexicon_entries_dropped": int(
            lex_stats["oov_phone"] + lex_stats["has_sil"] + lex_stats["no_pron"]
            + restrict_stats["oov_word"] + restrict_stats["dropped_marker"]),
        "prune": prune_stats,
        "compact": compact_stats,
        "ngram_counts": {str(o): int(lm[f"n_{o}gram"]) for o in range(1, order + 1)},
        "seconds": time.monotonic() - started, "max_rss_gib": _max_rss_gib(),
    })
    for o in range(1, order + 1):
        record[f"n_{o}gram"] = int(lm[f"n_{o}gram"])
    with open(args.out_json, "w") as fh:
        json.dump(record, fh, indent=2)
    if HLG is None:  # the size guard fired: a DISTINCT exit code, so the parent tries the next rung
        print(json.dumps({k: record[k] for k in
                          ("size_guard", "seconds", "max_rss_gib", "prune", "compact")}),
              flush=True)
        return EXIT_SIZE_GUARD
    print(json.dumps({k: record[k] for k in
                      ("hlg", "seconds", "max_rss_gib", "prune", "compact")}), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("arpa-to-npz", help="parse an ARPA of any order into the CSR tables")
    p.add_argument("--arpa", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    p.set_defaults(fn=_cmd_arpa_to_npz)

    p = sub.add_parser("build-hlg", help="compile H . L . G from the official lexicon and LM")
    p.add_argument("--lm-npz", required=True, help="the output of arpa-to-npz")
    p.add_argument("--lexicon", required=True, help="openslr librispeech-lexicon.txt")
    p.add_argument("--escape-resources", required=True,
                   help="LexiconTrieBuildJob's lexlat_resources.npz -- the escape price vector")
    p.add_argument("--out-hlg", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--prune-theta", type=float, default=0.0)
    p.add_argument("--sil-prob", type=float, default=SIL_PROB)
    p.add_argument("--d-min", type=int, required=True)
    p.add_argument("--recognizer-stride", type=int, required=True)
    p.add_argument("--mem-guard-gib", type=float, default=None,
                   help="skip this rung WITHOUT composing when the peak RSS predicted from G's "
                        "arc count exceeds this many GiB; the parent then tries the next rung "
                        f"(exit code {EXIT_SIZE_GUARD})")
    p.add_argument("--strict-lexicon", action="store_true",
                   help="build WITHOUT the escape phone loop (a disclosed deviation from the "
                        "Design, which fixes ESCAPE inside the marginal)")
    p.add_argument("--backoff-loops", choices=list(BACKOFF_LOOP_PLACEMENTS),
                   default=BACKOFF_LOOPS_ALL,
                   help="where L's #0 back-off self-loops sit: 'all' is icefall's placement (the "
                        "default, and every official graph built before 2026-09-21), "
                        "'word_boundary' only where a word transition can start")
    p.add_argument("--shuffled", action="store_true",
                   help=f"build the NULL: the seed-{DERANGEMENT_SEED} derangement of the word -> "
                        "pronunciation assignment (the same pronunciation multiset; only the word "
                        "identity moves)")
    p.set_defaults(fn=_cmd_build_hlg)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
