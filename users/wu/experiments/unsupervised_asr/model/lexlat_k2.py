"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lexlat_k2.py.

Port note: the changes are the relative import of the escape constants from ``.lexlat`` and
the scratch mount point of ``gpu-check``, now the caller's ``--scratch-prefix`` (see
``FORBID_SCRATCH_MAPS_ENV``).
The source ran this file AS A SCRIPT (``<k2 python> lexlat_k2.py build-hlg ...``); the port
must be run as a module, ``<k2 python> -m
i6_experiments.users.wu.experiments.unsupervised_asr.model.lexlat_k2 build-hlg ...``.  The k2
chunking code is the snapshot's; it already contains the L2-0 chunk fix (speech-llm
6fd3d02e, an ancestor of c49559ce).

emc.lexlat_k2 -- H . L . G for the k2 settling probe of `SAE_4A_lexlat.md` (Results "Cost analysis").

WHAT THIS IS.  `SAE_4A_lexlat.md` State: E1 measured the augmented ``lexlat.py`` DP at 802-912 s per
complete update against a 1202 s bar for a whole sub-epoch, and the literature read attributes the
gap to the ARC COUNT, not to the kernels.  Route (A) of the cost analysis is a PRUNED marginal with
gradients over a standard ``H . L . G`` through ``k2.intersect_dense_pruned`` at
``max_active`` 1,000 / 3,000 / 10,000.  This module builds that graph and runs that measurement.
It decides nothing: the probe is a READ against the E1 bar as written (<= 1202 s per sub-epoch,
<= 80 GiB), and a PASS licenses a design amendment, not an arm.

TWO ENVIRONMENTS, ONE MODULE.  ``k2`` is built with CUDA only in the scratch clone
``/e/scratch/spell/wu24/envs/sae_k2`` (``reports/impl_k2_build_2026-09-21.md``); sisyphus jobs run
under the shared training env, where ``k2`` is absent.  Every function here is therefore importable
WITHOUT k2 (the import is local to the functions that need it), and the jobs in
``lexlat_k2_jobs`` call this file as a SCRIPT under the scratch env's python, handing data over on
disk and results back as json.  ``python lexlat_k2.py --help`` lists the sub-commands.

------------------------------------------------------------------------------------------------
H -- the bed's blank-free topology, as an FSA over the RECOGNIZER frames
------------------------------------------------------------------------------------------------
The bed's DP (``lattice.py`` at ``topology="blankfree"``, ``recognizer_stride=3``, the configuration
``sae_blankfree.SaeBlankfreeModelV1`` pins) consumes ``log_q [B, T_rec, 40]`` -- 39 ARPAbet phones
plus SIL, no blank column -- where ``T_rec = (T_feat + 2) // 3``.  Its three arcs out of a state
``(t, s, h, f)`` reduce, ON THE RECOGNIZER AXIS ALONE, to this state machine:

* the **blank** arc is dead: ``lattice._arc_weights`` fills ``w_blank`` with ``NEG_INF`` outside
  ``topology == "ctc"`` (``lattice.py:511-516``);
* the **repeat** arc re-emits ``last(h)`` at frame ``t``, adds no prior term and opens no new token;
* the **emit** arc emits ``k`` and opens a new token, and ``lattice._prior_history`` forbids
  ``k == last(h)`` whenever ``f = 1`` (``same_nonsil`` at ``lattice.py:347-350``; the ``k != SIL``
  exception there is CTC-only).  ``f`` can only be reset to 0 by a blank arc, which is dead, so from
  the first emission on ``f = 1`` and ``k == last(h)`` is forbidden for every symbol including SIL.

So the frame label sequence is any string over the 40 symbols and its token string is the
ADJACENT-RUN COLLAPSE of it -- the very map ``blankfree.expected_run_counts`` is written for --
and every frame string has exactly one reading.  :func:`h_topology` is that map as an FSA: one
state per "the run now open carries phone p", a self-loop that continues the run and emits no
token, and a transition to every other phone that emits one.

MINIMUM DURATION.  ``d_min = 2`` is standing (``LatticeConfig.d_min``; the memory rule
``min-duration-topology-standing``) and it is a duration on the REVERSE/unit clock at 50 Hz, i.e.
on ``s``, not on ``t``: the emit arc moves ``s`` by ``d in [d_min, D_k]`` while the recognizer
advances ONE frame of its own ``stride = 3`` clock.  Expressed on the recognizer axis the bed's
minimum duration is therefore ``ceil(d_min / recognizer_stride) = ceil(2 / 3) = 1`` frame
(:func:`emission_min_frames`), which is what :func:`h_topology` is built with at this bed.  The
function takes a general ``min_frames >= 1`` and unrolls that many states per phone, so a bed whose
stride did not already absorb ``d_min`` would get a real min-duration chain; here it does, and H is
the plain run-collapse topology.  This mapping is the ONE reading choice of this module and it is
stated in ``reports/impl_lexlat_k2_r1_2026-09-21.md``.

COLUMN 0.  ``k2.DenseFsaVec`` reads the arc label as a COLUMN INDEX of the ``[N, T, C]`` log-prob
tensor and reserves column 0 for blank/epsilon.  The bed has no blank, so phone ``p`` is carried in
column ``p + 1`` and column 0 is filled with :data:`NEG_INF` (the bed's own finite -inf,
``psi_align.NEG_INF``); no arc of ``HLG`` ever carries label 0 after epsilon removal, so the column
is never read and never receives gradient.

------------------------------------------------------------------------------------------------
L -- the banked pronunciation lexicon, with SIL optional between words
------------------------------------------------------------------------------------------------
:func:`lexicon_to_fst` and :func:`add_lex_disambig` are icefall's ``prepare_lang.py`` functions,
VENDORED (icefall is not installed here) and reduced to the arguments this bed uses.  The
pronunciations are the banked ones: they are recovered from the trie of
``LexiconTrieBuildJob``'s ``lexlat_resources.npz`` by :func:`prons_from_trie`, so the word set, the
phone ids and the word-LM ids are the ones the E-1 equivalence probe and the E0 census already ran
on, and the seed-0 derangement (the phase's null) is available from the same file.

SIL is OPTIONAL between words with probability :data:`SIL_PROB` = 0.5 -- the dispatch's constant for
the bed's text convention, and the convention ``prior_gap.py`` already implements (SIL closes a word
and is never consumed inside one).  In L that is icefall's silence construction: a ``start -> loop``
arc at ``log(1 - p)``, a ``start -> sil`` arc at ``log p``, and after every word's last phone one
arc to ``loop`` at ``log(1 - p)`` and one to ``sil`` at ``log p``.  The silence model is therefore
part of the graph score and is NOT part of the word LM; the source's ``sil_model_log_prob`` (not
ported) prices it, which is what makes the equivalence check of the source's ``test_lexlat_k2`` exact.

DISAMBIGUATION.  ``#0`` is reserved for the word LM's back-off arcs (Kaldi's convention, which
icefall keeps); ``#1 ...`` disambiguate pronunciations that repeat or that are a prefix of another
pronunciation, so the LEXICAL ``L_disambig . G`` is determinizable (the escape loop is not; see
below).

WHERE THE ``#0`` LOOPS SIT is a BUILD OPTION, ``backoff_loops`` (:data:`BACKOFF_LOOP_PLACEMENTS`).
``"all"``, the DEFAULT and what every banked graph was compiled with, is icefall's own placement:
a loop at every L state with a non-epsilon outgoing token, word interiors included.  Under the LOG
semiring that placement lets ONE back-off route of a word transition be consumed at any position
inside the word, and since nothing here is determinized or epsilon-removed in the log semiring the
copies are summed -- the dominant term of the over-count measured on the banked graph
(``reports/audit_k2_overcount_2026-09-21.md`` section 3).  ``"word_boundary"`` puts the loops only
on the states from which a word-labelled arc leaves, so each route is consumed exactly once; the
MAX-PLUS total is unchanged (:func:`_add_self_loops`).

ESCAPE, the Design's convention and not an option of this module (`SAE_4A_lexlat.md` Design 1:
"ESCAPE, verbatim from ``prior_gap.py``: one ``<unk>`` word-LM transition per contiguous non-SIL
span plus, per phone, the live Witten-Bell order-1 phone log probability plus ``log 0.5``";
ESCAPE and not STRICT, because a ``-inf`` on non-segmentable prefixes is the zero-probability trap
inside a marginalised sum).  In ``L`` that is one extra state -- the ESCAPE PHONE LOOP: from the
word-boundary state, one arc per non-SIL phone into the loop state carrying the word ``<unk>`` and
the phone's escape price, a self-loop per non-SIL phone at the same price that extends the span,
and an exit arc (through the escape's own disambiguation symbol, which keeps the escape reading
apart from every lexical reading of the same phone string) back to the word boundary with the same
optional silence as a word's last phone.  The word cost of the span is
``G``'s own ``<unk>`` arc -- the very ``log p(<unk> | state)`` that ``lexlat.LexResources``
precomputes as ``unk_logp`` -- and the per-phone price is
:func:`escape_prices`, which IS ``lexlat.LexResources.build``'s ``escape_price``
(``lexlat.ESCAPE_LENGTH_LOG_PROB`` imported, never re-declared; SIL floored to :data:`NEG_INF`
because an escape span is non-SIL by construction).  Closing a span is free, as in the source's
``lexlat.string_best_segmentation`` (not ported); the span's word-LM state does not move while it is open, so
the escape state carries no ``#0`` back-off self-loop.  A STRICT-lexicon graph (no escape, the
``<unk>`` arcs dropped from ``G``) is still buildable -- ``escape = False`` -- but it is a
DISCLOSED DEVIATION from the Design, and the build job's ``summary.txt`` says which of the two was
compiled.

NOT DETERMINIZED -- NO GRAPH THE PROBE PRICES, ESCAPE OR STRICT.  k2's ``determinize`` is
TROPICAL-ONLY: its own docstring (``k2/fsa_algo.py``) promises a result "equivalent to the input
`fsa` under tropical semiring", i.e. it preserves the BEST path and merges the readings of a
label string by MAX.  The probe measures a LOG-SEMIRING total -- a sum over readings -- and a
tropical determinization is not weight-preserving for it: a determinized graph would under-count
that marginal.  :func:`compile_hlg` therefore never determinizes; epsilon removal, ``connect`` and
``arc_sort`` are kept, and every score the probe reports is read off the UNDETERMINIZED graph.
The escape graph could not be determinized in any case: the escape loop reads phone ``p`` at the
escape price while the lexical branch reads the same ``p`` on its way into a word, and the two
weights differ by an amount that grows with the span, so the weighted transducer violates the
twins property and ``determinize`` does not terminate on it -- measured on the tiny fixture of
``test_lexlat_k2``: >240 s with no result, against 0.01 s for the WHOLE strict build.  The graph
priced is consequently the larger, ambiguous one; ``build.json`` carries ``determinized`` (always
``False``) and the job's ``summary.txt`` prints it for every rung of the pruning ladder.

------------------------------------------------------------------------------------------------
G -- the banked word trigram
------------------------------------------------------------------------------------------------
The ARPA of the banked build (``LexiconTrieBuildJob.rlMsnTBSZXsB``) was written to that job's WORK
directory and the work directory has been cleaned; the job's surviving outputs are the KenLM binary
and the CSR back-off automaton in ``lexlat_resources.npz``, which ``lexlat.parse_arpa_word_lm``
parsed from that very ARPA and which carries every n-gram of it (151,734 + 3,393,577 + 10,419,405 =
13,964,716 arcs, no n-gram dropped).  :func:`g_fsa` therefore converts THOSE TABLES -- the same
object the phase's own scorer reads -- into the back-off FSA, instead of round-tripping an ARPA
through ``kaldilm``.  ``kaldilm`` 1.15.4 does install into the scratch env (aarch64 wheel) and is
used in ``test_lexlat_k2`` as an INDEPENDENT CROSS-CHECK of :func:`g_fsa` on a tiny ARPA, beside
KenLM itself.

No end-of-sentence arc.  ``prior.PhoneNgramPrior`` pre-registers "no end-of-sequence term" and
the source's ``lexlat.string_best_segmentation`` (not ported) scores no ``</s>``; G is built with EVERY state final at score
0 and with the ``<s>`` / ``</s>`` arcs dropped (L can never emit them).  The ``<unk>`` arcs are
KEPT: under ESCAPE they are the escape's word cost, reached from any state through the same
back-off arcs ``lexlat.lm_score`` walks.  They are dropped only in the strict build
(:data:`NON_EMITTABLE_WORDS_STRICT`), where nothing emits ``<unk>``.

PRUNING, only if the full graph does not compile.  :func:`prune_lm_tables` drops an n-gram arc whose
explicit log probability is already reproduced by its back-off estimate to within ``theta`` NATS and
re-normalises the back-off weights; ``theta = 0`` drops nothing.  The criterion is the
count-free form of Seymore and Rosenfeld's; the LADDER of thresholds is declared by the config, and
the rung actually used, with the resulting n-gram counts, is written into the build job's output.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# the ESCAPE convention's own constants, imported so this module can never re-declare them
# (``log 0.5`` per escaped phone, and ``<unk>`` as the escape word); ``lexlat`` needs torch but no
# k2, and the k2 env this file is run under has both.  RELATIVE (port): a build child runs this
# module as ``python -m <package>.model.lexlat_k2 ...`` with the recipe root on PYTHONPATH.
from .lexlat import ESCAPE_LENGTH_LOG_PROB, ESCAPE_WORD


__all__ = [
    "NEG_INF",
    "SIL_PROB",
    "SEARCH_BEAM",
    "OUTPUT_BEAM",
    "MIN_ACTIVE_STATES",
    "MAX_ACTIVE_LADDER",
    "NON_EMITTABLE_WORDS",
    "NON_EMITTABLE_WORDS_STRICT",
    "DEFAULT_LM_ORDER",
    "bed_min_duration",
    "emission_min_frames",
    "h_topology",
    "prons_from_trie",
    "add_lex_disambig",
    "escape_prices",
    "lexicon_to_fst",
    "g_fsa",
    "state_orders",
    "prune_lm_tables",
    "compile_hlg",
    "CHUNK_SEQS",
    "CHUNK_SEQS_ENV",
    "K2_PRUNE_WINDOW",
    "resolve_chunk_seqs",
    "chunk_bounds",
    "max_out_degree",
    "window_arc_bound",
    "chunk_guard",
    "chunked_tot_scores",
    "lattice_sizes",
    "run_probe",
    "FORBID_SCRATCH_MAPS_ENV",
    "mapped_objects_under",
]


#: the order of a CSR back-off automaton that does not say what its order is.  Every npz written
#: before the official-LM round came from ``lexlat.parse_arpa_word_lm``, which ASSERTS order 3.
DEFAULT_LM_ORDER = 3


#: the bed's own finite -inf (``psi_align.NEG_INF``), used for the unused blank column of the dense
#: FSA.  A real ``-inf`` would be as correct (no arc reads column 0) but this value is the bed's.
NEG_INF = -1.0e30

#: SIL between words, as an OPTIONAL boundary: p = 0.5 (dispatch 2026-09-21, the bed's text
#: convention; ``prior_gap.py:461-467`` treats SIL as the word boundary anchor).
SIL_PROB = 0.5

#: icefall's MMI search settings, disclosed in `SAE_4A_lexlat.md` State as the probe's operating
#: point.  They are NOT swept here.
SEARCH_BEAM = 20.0
OUTPUT_BEAM = 8.0
MIN_ACTIVE_STATES = 30
#: the three operating points the probe is registered for
MAX_ACTIVE_LADDER: Tuple[int, ...] = (1000, 3000, 10000)

#: the words of the word LM that L can never emit under the Design's ESCAPE convention: ``<s>``
#: and ``</s>`` have no pronunciation and the bed adds no end-of-sequence term.  ``<unk>`` is NOT
#: here -- it IS emittable, as the escape word of the phone loop (module docstring, ESCAPE).
NON_EMITTABLE_WORDS = ("<s>", "</s>")
#: the same list for a STRICT-lexicon build (``escape = False``): with no escape phone loop nothing
#: emits ``<unk>`` either.  A strict graph is a DISCLOSED DEVIATION from the Design.
NON_EMITTABLE_WORDS_STRICT = ("<s>", "</s>", ESCAPE_WORD)

#: where ``L``'s ``#0`` back-off self-loops sit (:func:`_add_self_loops`).  ``"all"`` is icefall's
#: own placement and the DEFAULT, so every banked build keeps its hash; ``"word_boundary"`` is the
#: placement that makes each back-off route of a word transition consumable exactly once.
BACKOFF_LOOPS_ALL = "all"
BACKOFF_LOOPS_WORD_BOUNDARY = "word_boundary"
BACKOFF_LOOP_PLACEMENTS: Tuple[str, ...] = (BACKOFF_LOOPS_ALL, BACKOFF_LOOPS_WORD_BOUNDARY)

#: THE 90-DAY LEASE, as an assertion rather than as advice (design review Q1.3 / Q2.1).  The k2
#: build was made in a scratch clone and its shared objects carry ``RPATH`` entries under the
#: scratch mount; the copy the arms launch under resolves its own ``libtorch`` first, so the
#: ``RPATH`` should never be consulted -- but only ``/proc/<pid>/maps`` PROVES it, and a launch
#: that passes today and dies at the next resume (after the lease is purged) is what this check
#: exists to prevent.  Port note: the source hardcoded the mount point as a module constant
#: (``/e/scratch``); it is cluster-specific, so it is now the ``prefix`` argument of
#: :func:`mapped_objects_under` and the ``--scratch-prefix`` option of ``gpu-check`` (no default).
#:
#: Set to ``1`` to make :func:`_cmd_gpu_check` RAISE when any shared object under the stated
#: ``--scratch-prefix`` is mapped (and refuse to run without one).  Default OFF, so the banked probe
#: jobs' check is the check they ran; the pre-launch step probe sets it.
FORBID_SCRATCH_MAPS_ENV = "LEXLAT_K2_FORBID_SCRATCH_MAPS"


# ===================================================================================================
# H -- the bed's blank-free topology
# ===================================================================================================
#: ``sae_blankfree.py``'s own ``replace(self.lattice_cfg, topology="blankfree",
#: recognizer_stride=3)`` -- the stride every blank-free arm of this bed runs at.  It is a MODEL
#: fact, so it is never trusted on its own: ``LexlatK2ProbeJob`` asserts the graph's value against
#: the live ``model.lattice_cfg`` before a single frame is dumped.
BLANKFREE_RECOGNIZER_STRIDE = 3


def bed_min_duration() -> Tuple[int, int]:
    """``(d_min, recognizer_stride)`` of the blank-free bed, for a config that must not re-type them.

    ``d_min`` comes from ``reverse.ReverseConfig``, whose ``d_min = 2`` is the phase's standing
    constraint and which no arm of this bed overrides; the stride is
    :data:`BLANKFREE_RECOGNIZER_STRIDE`.  Both are asserted against the arm's own
    ``model.lattice_cfg`` in :class:`~.lexlat_k2_jobs.LexlatK2ProbeJob`.
    """
    from .reverse import ReverseConfig

    return int(ReverseConfig().d_min), int(BLANKFREE_RECOGNIZER_STRIDE)


def emission_min_frames(d_min: int, recognizer_stride: int) -> int:
    """The bed's minimum duration, expressed in RECOGNIZER frames (module docstring, H).

    ``d_min`` is a duration on the 50 Hz unit clock; the recognizer advances one frame per
    ``recognizer_stride`` unit frames, so a symbol occupies at least ``ceil(d_min / stride)``
    recognizer frames.  At the bed (``d_min = 2``, ``stride = 3``) this is 1.
    """
    d_min, recognizer_stride = int(d_min), int(recognizer_stride)
    assert d_min >= 1 and recognizer_stride >= 1, (d_min, recognizer_stride)
    return max(1, -(-d_min // recognizer_stride))


def h_topology(n_phones: int, *, min_frames: int = 1, device: str = "cpu"):
    """The blank-free adjacent-run-collapse topology of the bed, as a k2 FSA.

    States ``(p, i)`` for ``p`` in ``0 .. n_phones - 1`` and ``i`` in ``0 .. min_frames - 1``:
    "the run now open carries phone ``p`` and has already taken ``i + 1`` frames" (capped).  Plus a
    start state and a final state.  Arcs, with the emission label of phone ``p`` being ``p + 1``
    (column 0 of the dense FSA is the unused blank/epsilon column):

    * ``start -(p+1 : p+1)-> (p, 0)``            -- the first token opens;
    * ``(p, i) -(p+1 : 0)-> (p, min(i+1, m-1))`` -- the run continues, no token emitted;
    * ``(p, m-1) -(q+1 : q+1)-> (q, 0)`` for ``q != p`` -- a new token opens;
    * ``(p, m-1) -(-1 : 0)-> final`` and ``start -(-1 : 0)-> final``.

    ``labels`` are the emission columns; ``aux_labels`` are the TOKENS L consumes (the same ids,
    ``p + 1``, so tokens and emissions share one alphabet and ``0`` is epsilon in both).
    """
    import k2
    import torch

    n_phones, m = int(n_phones), int(min_frames)
    assert n_phones >= 1 and m >= 1, (n_phones, m)
    start = 0
    state = lambda p, i: 1 + p * m + i
    final = 1 + n_phones * m
    arcs: List[Tuple[int, int, int, int, float]] = []
    for p in range(n_phones):
        arcs.append((start, state(p, 0), p + 1, p + 1, 0.0))
        for i in range(m):
            arcs.append((state(p, i), state(p, min(i + 1, m - 1)), p + 1, 0, 0.0))
        for q in range(n_phones):
            if q != p:
                arcs.append((state(p, m - 1), state(q, 0), q + 1, q + 1, 0.0))
        arcs.append((state(p, m - 1), final, -1, 0, 0.0))
    arcs.append((start, final, -1, 0, 0.0))  # the empty utterance; no batch ever has T = 0
    return _fsa_from_arcs(arcs, k2=k2, torch=torch, device=device)


def _fsa_from_arcs(arcs: Sequence[Tuple[int, int, int, int, float]], *, k2, torch, device="cpu"):
    """``(src, dst, label, aux_label, score)`` rows -> a k2 ``Fsa`` (arcs sorted by src state)."""
    rows = sorted(arcs, key=lambda a: (a[0], a[1], a[2]))
    src = torch.tensor([a[0] for a in rows], dtype=torch.int32)
    dst = torch.tensor([a[1] for a in rows], dtype=torch.int32)
    lab = torch.tensor([a[2] for a in rows], dtype=torch.int32)
    aux = torch.tensor([a[3] for a in rows], dtype=torch.int32)
    score = torch.tensor([a[4] for a in rows], dtype=torch.float32)
    tensor = torch.stack([src, dst, lab, score.view(torch.int32)], dim=1)
    fsa = k2.Fsa(tensor, aux_labels=aux)
    return fsa.to(device) if device != "cpu" else fsa


# ===================================================================================================
# L -- the lexicon
# ===================================================================================================
def prons_from_trie(child: np.ndarray, word_start: np.ndarray, word_id: np.ndarray):
    """Walk the banked flat trie back into ``[(lm_word_id, [phone ids]), ...]``.

    ``lexlat.build_trie``'s arrays: ``child[node, phone]`` is the child node or -1, and the words
    ending at ``node`` are ``word_id[word_start[node] : word_start[node + 1]]`` (homophones share a
    node).  Reading the pronunciations back off the trie rather than off ``lexicon.json.gz`` keeps
    the phone ids and the word-LM ids EXACTLY the ones the DP runs on, and makes the shuffled null
    (``shuffled_word_id``) a drop-in.
    """
    child = np.asarray(child)
    n_nodes, n_phones = child.shape
    out: List[Tuple[int, List[int]]] = []
    stack: List[Tuple[int, List[int]]] = [(0, [])]
    while stack:
        node, pron = stack.pop()
        for e in range(int(word_start[node]), int(word_start[node + 1])):
            out.append((int(word_id[e]), list(pron)))
        row = child[node]
        for p in range(n_phones - 1, -1, -1):
            nxt = int(row[p])
            if nxt >= 0:
                stack.append((nxt, pron + [p]))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def add_lex_disambig(prons: Sequence[Tuple[int, Sequence[int]]]):
    """icefall ``prepare_lang.add_lex_disambig``, vendored, on integer pronunciations.

    Returns ``(entries, max_disambig)`` where ``entries`` is ``[(word, pron, disambig or None)]``:
    a pronunciation gets a disambiguation index when it is a PROPER PREFIX of another pronunciation
    or when it is shared by more than one entry, and repeated pronunciations get DISTINCT indices.
    ``#0`` is reserved for the word LM's back-off arcs, so the first index handed out is 1.
    """
    first_allowed = 1
    counts: Dict[Tuple[int, ...], int] = {}
    prefixes = set()
    for _w, pron in prons:
        key = tuple(int(p) for p in pron)
        counts[key] = counts.get(key, 0) + 1
        for i in range(len(key)):
            prefixes.add(key[:i])  # every PROPER prefix, the empty one included
    last_used: Dict[Tuple[int, ...], int] = {}
    entries: List[Tuple[int, List[int], Optional[int]]] = []
    max_disambig = first_allowed - 1
    for word, pron in prons:
        key = tuple(int(p) for p in pron)
        if key not in prefixes and counts[key] == 1:
            entries.append((int(word), list(key), None))
            continue
        cur = last_used.get(key, first_allowed - 1) + 1
        last_used[key] = cur
        max_disambig = max(max_disambig, cur)
        entries.append((int(word), list(key), cur))
    return entries, int(max_disambig)


def _add_self_loops(arcs, *, disambig_token: int, disambig_word: int,
                    placement: str = BACKOFF_LOOPS_ALL):
    """The ``#0:#0`` loops through which ``L`` passes the word LM's back-off arcs.

    ``placement = "all"`` (the DEFAULT, and what every banked graph was built with) is icefall
    ``prepare_lang.add_self_loops`` VERBATIM: a loop at every state with a non-epsilon outgoing
    INPUT label, i.e. at the word boundary AND at every state inside a word.

    ``placement = "word_boundary"`` puts a loop only on the states from which an arc carrying a
    non-epsilon WORD (auxiliary) label leaves -- the trie root, which is also where the SIL and
    the ESCAPE closures return to and where the escape word's own opening arcs start.  WHY: the
    word LM's state moves only on a word-labelled arc, so under ``"all"`` the same back-off route
    of one word transition can be consumed at any of the positions inside that word, and because
    :func:`compile_hlg` never determinizes and ``k2.remove_epsilon`` is tropical-only, each
    position survives as a separate path whose weight the LOG-semiring total ADDS.  That
    positional multiplicity is the dominant part of the measured over-count of the banked graph
    (``reports/audit_k2_overcount_2026-09-21.md`` section 3: a median +0.24 gold / +0.11 private
    nats per token, against +0.033 / +0.017 for the plain back-off routes themselves).  With the
    loops at the boundary alone each route is consumed exactly once.  The MAX-PLUS total is
    unchanged: a loop inside a word only ever adds a SECOND route of the same weight, since the
    loop costs 0 and the G arcs traversed are the same ones.
    """
    assert placement in BACKOFF_LOOP_PLACEMENTS, placement
    col = 2 if placement == BACKOFF_LOOPS_ALL else 3  # 2 = input token, 3 = output word
    need = {int(a[0]) for a in arcs if int(a[col]) != 0}
    return list(arcs) + [(s, s, disambig_token, disambig_word, 0.0) for s in sorted(need)]


def escape_prices(escape_phone_log_prob: Sequence[float], *, n_phones: int, sil_id: int):
    """``lexlat.LexResources.build``'s ``escape_price`` vector, in numpy: ``[K]`` nats per phone.

    ``escape_phone_log_prob[k]`` is what the banked npz stores RAW -- ``prior_gap``'s order-1
    Witten-Bell phone log probability, the number ``PriorGapAnalysisJob`` banks under
    ``escape_phone_log_probs``.  The escape's own per-phone length term
    ``lexlat.ESCAPE_LENGTH_LOG_PROB`` (``log 0.5``) is added here ONCE, and SIL is floored to
    :data:`NEG_INF` because an escape span is non-SIL by construction: the three lines of
    ``LexResources.build``, with ``lexlat``'s constant imported and never re-declared.
    ``test_lexlat_k2`` asserts this vector IS ``LexResources.build(...).escape_price``.
    """
    esc = np.asarray(escape_phone_log_prob, dtype=np.float64).copy()
    assert esc.size == int(n_phones), (esc.size, n_phones)
    esc += ESCAPE_LENGTH_LOG_PROB
    esc[int(sil_id)] = NEG_INF
    return esc


def lexicon_to_fst(
    entries: Sequence[Tuple[int, Sequence[int], Optional[int]]],
    *,
    n_phones: int,
    sil_id: int,
    n_words: int,
    max_disambig: int,
    sil_prob: float = SIL_PROB,
    escape_word: Optional[int] = None,
    escape_price: Optional[Sequence[float]] = None,
    need_self_loops: bool = True,
    backoff_loops: str = BACKOFF_LOOPS_ALL,
    device: str = "cpu",
):
    """icefall ``prepare_lang.lexicon_to_fst`` (``sil_prob`` variant), vendored, plus ESCAPE.

    Label spaces, one place: token ``0`` is epsilon, token ``p + 1`` is phone ``p``
    (``0 <= p < n_phones``), token ``n_phones + 1 + j`` is ``#j``; word ``0`` is epsilon, word
    ``w + 1`` is word-LM id ``w``, word ``n_words + 1`` is ``#0``.  ``labels`` are tokens,
    ``aux_labels`` are words -- the orientation ``k2.compose(L, G)`` wants.

    ESCAPE (``escape_word`` = the word LM's ``<unk>`` id, ``escape_price`` =
    :func:`escape_prices`) adds ONE state, the escape phone loop of the module docstring: one
    opening arc per non-SIL phone from the word boundary carrying the word ``<unk>`` (so ``G``
    prices the span's word cost, exactly ``lexlat``'s ``unk_logp``), one self-loop per non-SIL
    phone that extends the span at the same per-phone price, and an exit pair carrying the
    escape's OWN disambiguation symbol ``#(max_disambig + 1)`` -- without it, determinization of
    ``L_disambig . G`` would have to merge the escape reading of a phone string with its lexical
    readings.  The escape state gets NO ``#0`` self-loop: an open span emits no word, so its
    word-LM state does not move, which is the source's ``lexlat.string_best_segmentation``'s convention.
    ``escape_word = None`` builds the STRICT lexicon, a disclosed deviation from the Design.

    ``backoff_loops`` is where the ``#0`` self-loops sit -- :func:`_add_self_loops`'s own
    parameter, the DEFAULT ``"all"`` being icefall's placement and the one every banked graph was
    built with.  ``"word_boundary"`` puts them only where a word transition can start, which is
    what makes each back-off route consumable exactly once under the LOG semiring.
    """
    import k2
    import torch

    assert backoff_loops in BACKOFF_LOOP_PLACEMENTS, backoff_loops
    assert 0.0 < float(sil_prob) < 1.0, sil_prob
    sil_score = math.log(float(sil_prob))
    no_sil_score = math.log(1.0 - float(sil_prob))
    sil_token = int(sil_id) + 1
    start_state, loop_state, sil_state = 0, 1, 2
    next_state = 3
    eps = 0
    arcs: List[Tuple[int, int, int, int, float]] = [
        (start_state, loop_state, eps, eps, no_sil_score),
        (start_state, sil_state, eps, eps, sil_score),
        (sil_state, loop_state, sil_token, eps, 0.0),
    ]
    for word, pron, disambig in entries:
        tokens = [int(p) + 1 for p in pron]
        if disambig is not None:
            tokens = tokens + [int(n_phones) + 1 + int(disambig)]
        assert tokens, f"word {word} has an empty pronunciation"
        cur = loop_state
        w = int(word) + 1
        for i in range(len(tokens) - 1):
            arcs.append((cur, next_state, tokens[i], w if i == 0 else eps, 0.0))
            cur = next_state
            next_state += 1
        i = len(tokens) - 1
        out = w if i == 0 else eps
        arcs.append((cur, loop_state, tokens[i], out, no_sil_score))
        arcs.append((cur, sil_state, tokens[i], out, sil_score))
    if need_self_loops:
        # BEFORE the escape arcs, so the escape state is not given a ``#0`` back-off self-loop:
        # an open span emits no word and its word-LM state must not move (docstring).  Under
        # ``word_boundary`` the escape state would not qualify in any case (its arcs carry no
        # word label), and the escape's OPENING arcs start at ``loop_state``, which does.
        arcs = _add_self_loops(
            arcs, disambig_token=int(n_phones) + 1, disambig_word=int(n_words) + 1,
            placement=backoff_loops)
        loop_state_has_backoff = any(
            int(a[0]) == loop_state and int(a[1]) == loop_state
            and int(a[2]) == int(n_phones) + 1 for a in arcs)
        assert loop_state_has_backoff, (
            "the word-boundary state carries no #0 self-loop: G's back-off arcs could not be "
            f"consumed at all ({backoff_loops})")
    if escape_word is not None:
        assert escape_price is not None, "ESCAPE needs its per-phone prices (escape_prices)"
        esc = np.asarray(escape_price, dtype=np.float64)
        assert esc.size == int(n_phones), (esc.size, n_phones)
        esc_state = next_state
        next_state += 1
        esc_w = int(escape_word) + 1
        esc_disambig = int(n_phones) + 1 + int(max_disambig) + 1
        for p in range(int(n_phones)):
            price = float(esc[p])
            if price <= NEG_INF / 2:
                continue  # SIL: an escape span is non-SIL by construction (lexlat's floor)
            arcs.append((loop_state, esc_state, p + 1, esc_w, price))  # open: one <unk>
            arcs.append((esc_state, esc_state, p + 1, eps, price))  # extend the span
        arcs.append((esc_state, loop_state, esc_disambig, eps, no_sil_score))
        arcs.append((esc_state, sil_state, esc_disambig, eps, sil_score))
    final_state = next_state
    arcs.append((loop_state, final_state, -1, 0, 0.0))
    return _fsa_from_arcs(arcs, k2=k2, torch=torch, device=device)


def first_token_disambig_id(n_phones: int) -> int:
    """``#0``'s token id: phones occupy ``1 .. n_phones``."""
    return int(n_phones) + 1


def first_word_disambig_id(n_words: int) -> int:
    """``#0``'s word id: the word LM's words occupy ``1 .. n_words``."""
    return int(n_words) + 1


# ===================================================================================================
# G -- the word trigram, from the banked CSR back-off automaton
# ===================================================================================================
def read_lm_tables(npz_path: str) -> Dict[str, object]:
    """The CSR back-off word LM and the trie of ``LexiconTrieBuildJob``'s npz, as numpy arrays.

    ORDER.  The banked npz of ``LexiconTrieBuildJob`` carries no ``order`` field -- it was written
    by ``lexlat.parse_arpa_word_lm``, which asserts order 3 -- so a missing field reads as
    :data:`DEFAULT_LM_ORDER` = 3 and every number this function returns for that file is bit for
    bit what it returned before.  ``lexlat_k2_official.parse_arpa_lm`` writes ``order`` (and the
    per-state ``state_order``) for an ARPA of any order, and only then do the counts
    ``n_1gram ... n_{order}gram`` run past three.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        out: Dict[str, object] = {
            k: np.asarray(z[k]) for k in
            ("arc_key", "arc_logp", "arc_next", "backoff", "backoff_state")
        }
        for k in ("child", "word_start", "word_id", "escape_phone_log_prob", "shuffled_word_id",
                  "state_order"):
            if k in z:
                out[k] = np.asarray(z[k])
        order = int(z["order"]) if "order" in z else DEFAULT_LM_ORDER
        out["order"] = order
        out.update({k: int(z[k]) for k in
                    ("n_states", "n_words", "begin_state", "unk_word", "n_phones", "sil_id")})
        for k in ("n_nodes", "n_entries"):
            if k in z:
                out[k] = int(z[k])
        for o in range(1, order + 1):
            out[f"n_{o}gram"] = int(z[f"n_{o}gram"])
        out["words"] = [str(w) for w in z["words"]]
    return out


def state_orders(lm: Dict[str, object]) -> np.ndarray:
    """``[n_states]`` int64: the ORDER of the n-grams leaving each state (1 = the null context).

    The CSR layout numbers the null context 0, the ``n_words`` unigram contexts ``1 .. V`` and
    every higher-order context after them, so for the banked order-3 tables this reproduces
    ``lexlat.parse_arpa_word_lm``'s own numbering exactly -- and the order-3 expression
    ``where(s == 0, 1, where(s <= V, 2, 3))`` :func:`prune_lm_tables` used to inline.  An LM
    written by ``lexlat_k2_official.parse_arpa_lm`` carries the array explicitly, because above
    order 3 the block boundaries are the ARPA's own n-gram counts and cannot be derived from ``V``.
    """
    if "state_order" in lm:
        return np.asarray(lm["state_order"], dtype=np.int64)
    order = int(lm.get("order", DEFAULT_LM_ORDER))
    assert order == 3, (
        f"an order-{order} LM must carry its own state_order array; only the banked order-3 "
        "layout is derivable from n_words alone")
    v, n = int(lm["n_words"]), int(lm["n_states"])
    out = np.full(n, 3, dtype=np.int64)
    out[0] = 1
    out[1:v + 1] = 2
    return out


def _lookup(lm: Dict[str, object], state: np.ndarray, word: np.ndarray,
            *, levels: Optional[int] = None):
    """``lexlat.lm_score`` in numpy: ``(log p(word | state) in nats, successor state)``.

    ``levels`` is the depth of the back-off walk and defaults to the LM's own order (3 where the
    tables carry none, which is every npz written before the official-LM round), so an order-4
    automaton is walked four levels and the order-3 answers do not move.
    """
    if levels is None:
        levels = int(lm.get("order", DEFAULT_LM_ORDER))
    arc_key = lm["arc_key"]
    arc_logp = lm["arc_logp"].astype(np.float64)
    arc_next = lm["arc_next"]
    backoff = lm["backoff"].astype(np.float64)
    backoff_state = lm["backoff_state"]
    v = int(lm["n_words"])
    st = np.asarray(state, dtype=np.int64).reshape(-1)
    wd = np.asarray(word, dtype=np.int64).reshape(-1)
    total = np.zeros(st.shape, dtype=np.float64)
    out_lp = np.zeros_like(total)
    out_nx = np.zeros_like(st)
    found = np.zeros(st.shape, dtype=bool)
    cur = st.copy()
    last = arc_key.size - 1
    for _ in range(int(levels)):
        key = cur * v + wd
        pos = np.minimum(np.searchsorted(arc_key, key), last)
        hit = (~found) & (arc_key[pos] == key)
        out_lp = np.where(hit, total + arc_logp[pos], out_lp)
        out_nx = np.where(hit, arc_next[pos], out_nx)
        found = found | hit
        total = np.where(found, total, total + backoff[cur])
        cur = np.where(found, cur, backoff_state[cur])
    return out_lp, out_nx


def prune_lm_tables(lm: Dict[str, object], theta: float) -> Tuple[Dict[str, object], Dict[str, int]]:
    """Back-off-gain pruning of the CSR tables; ``theta`` in NATS, ``theta = 0`` prunes nothing.

    An arc ``(s, w)`` with ``s != 0`` (order >= 2) is DROPPED when its explicit log probability is
    already reproduced by the back-off estimate to within ``theta``::

        | arc_logp[s, w] - (backoff[s] + log p(w | backoff_state[s])) |  <  theta

    -- the count-free form of Seymore and Rosenfeld's criterion (we hold no n-gram counts: the ARPA
    of the banked build no longer exists on disk, only the automaton it was parsed into).  The
    lower-order probability is read from the ORIGINAL tables, as in the published one-at-a-time
    approximation.  Every state's back-off weight is then RE-NORMALISED so the state still sums to
    one over the vocabulary::

        alpha(s) = log( (1 - sum_{w explicit} p(w | s)) / (1 - sum_{w explicit} p(w | bo(s))) )

    with ``alpha(s) = 0`` where the explicit set became empty.  Returns the new tables and the
    n-gram counts, which the build job records.

    THIS CRITERION IS AN IMPLEMENTATION CHOICE of ``lexlat_k2``; the LADDER of thresholds and the
    rule that picks a rung ("the smallest that compiles inside the build job's budget") are the
    config's and the dispatch's.

    ANY ORDER.  "order >= 2" is read off :func:`state_orders` rather than off the order-3 block
    boundary ``s <= n_words``, and the reported counts run ``n_1gram ... n_{order}gram``.  On an
    order-3 table :func:`state_orders` returns exactly the expression this function used to
    inline, so the three counts and every kept arc are unchanged.
    """
    theta = float(theta)
    order = int(lm.get("order", DEFAULT_LM_ORDER))
    st_order = state_orders(lm)
    n_states = int(lm["n_states"])
    arc_key = np.asarray(lm["arc_key"])
    arc_logp = np.asarray(lm["arc_logp"]).astype(np.float64)
    arc_next = np.asarray(lm["arc_next"])
    backoff = np.asarray(lm["backoff"]).astype(np.float64)
    backoff_state = np.asarray(lm["backoff_state"])
    v = int(lm["n_words"])
    src = arc_key // v
    wrd = arc_key % v
    stats = {"arcs_before": int(arc_key.size), "order": order,
             "theta_nats": theta, "arcs_after": int(arc_key.size)}
    for o in range(1, order + 1):
        stats[f"n_{o}gram"] = int(lm[f"n_{o}gram"])
        stats[f"n_{o}gram_after"] = int(lm[f"n_{o}gram"])
    if theta <= 0.0:  # the identity, bit for bit: no arc dropped and no back-off re-derived
        return dict(lm), stats
    lower_lp, _ = _lookup(lm, backoff_state[src], wrd)
    keep = (st_order[src] == 1) | (np.abs(arc_logp - (backoff[src] + lower_lp)) >= theta)
    new_key, new_logp, new_next = arc_key[keep], arc_logp[keep], arc_next[keep]
    # re-normalise every state's back-off weight against the surviving explicit set
    ksrc = new_key // v
    explicit_mass = np.bincount(ksrc, weights=np.exp(new_logp), minlength=n_states)
    lower_kept, _ = _lookup(lm, backoff_state[ksrc], new_key % v)
    lower_mass = np.bincount(ksrc, weights=np.exp(lower_kept), minlength=n_states)
    num = np.clip(1.0 - explicit_mass, 1e-30, None)
    den = np.clip(1.0 - lower_mass, 1e-30, None)
    new_backoff = np.log(num) - np.log(den)
    new_backoff[0] = 0.0  # the null state never backs off
    out = dict(lm)
    out.update({
        "arc_key": new_key, "arc_logp": new_logp.astype(np.float32), "arc_next": new_next,
        "backoff": new_backoff.astype(np.float32), "backoff_state": backoff_state,
    })
    orders = st_order[ksrc]
    stats.update({"theta_nats": theta, "arcs_after": int(new_key.size)})
    for o in range(1, order + 1):
        out[f"n_{o}gram"] = int((orders == o).sum())
        stats[f"n_{o}gram_after"] = out[f"n_{o}gram"]
    return out, stats


def g_fsa(lm: Dict[str, object], *, device: str = "cpu", drop_words: Sequence[str] = NON_EMITTABLE_WORDS):
    """The CSR back-off automaton as a k2 FSA over WORD labels.

    One arc per explicit n-gram (label ``w + 1``, score the n-gram's log probability in nats), one
    back-off arc per state (label ``#0`` = ``n_words + 1``, score the state's back-off weight), and
    one ``-1`` arc from EVERY state to the single final state at score 0 -- the bed's "no
    end-of-sequence term" convention.  State ``begin_state`` (the ``<s>`` context) is swapped with
    state 0, because a k2 FSA starts at state 0.

    ``drop_words`` removes the arcs L can never emit; leaving them in would only cost memory,
    since ``connect`` after the composition removes them anyway.  Under ESCAPE that list is
    :data:`NON_EMITTABLE_WORDS` and the ``<unk>`` arcs STAY -- they are the escape's word cost;
    the strict build passes :data:`NON_EMITTABLE_WORDS_STRICT`.
    """
    import k2
    import torch

    n_states = int(lm["n_states"])
    v = int(lm["n_words"])
    arc_key = np.asarray(lm["arc_key"])
    arc_logp = np.asarray(lm["arc_logp"]).astype(np.float32)
    arc_next = np.asarray(lm["arc_next"])
    backoff = np.asarray(lm["backoff"]).astype(np.float32)
    backoff_state = np.asarray(lm["backoff_state"])
    words = list(lm["words"])
    src = (arc_key // v).astype(np.int64)
    wrd = (arc_key % v).astype(np.int64)
    if drop_words:
        bad = {i for i, w in enumerate(words) if w in set(drop_words)}
        if bad:
            mask = ~np.isin(wrd, np.fromiter(sorted(bad), dtype=np.int64))
            src, wrd, arc_logp, arc_next = src[mask], wrd[mask], arc_logp[mask], arc_next[mask]

    begin = int(lm["begin_state"])
    perm = np.arange(n_states, dtype=np.int64)
    if begin != 0:  # a k2 FSA starts at state 0: swap the <s> context with the null context
        perm[0], perm[begin] = begin, 0
    final = n_states
    states = np.arange(n_states, dtype=np.int64)
    n_arcs = src.size + n_states + n_states
    a_src = np.empty(n_arcs, dtype=np.int64)
    a_dst = np.empty(n_arcs, dtype=np.int64)
    a_lab = np.empty(n_arcs, dtype=np.int64)
    a_sco = np.empty(n_arcs, dtype=np.float32)
    n = src.size
    a_src[:n], a_dst[:n], a_lab[:n], a_sco[:n] = perm[src], perm[arc_next], wrd + 1, arc_logp
    a_src[n:n + n_states] = perm[states]
    a_dst[n:n + n_states] = perm[backoff_state]
    a_lab[n:n + n_states] = first_word_disambig_id(v)
    a_sco[n:n + n_states] = backoff
    # the null context has no back-off arc: drop it by pointing it at itself with label -1 below
    a_src[n + n_states:] = perm[states]
    a_dst[n + n_states:] = final
    a_lab[n + n_states:] = -1
    a_sco[n + n_states:] = 0.0
    null_new = perm[0]
    keep = np.ones(n_arcs, dtype=bool)
    keep[n + int(np.nonzero(perm[states] == null_new)[0][0])] = False
    a_src, a_dst, a_lab, a_sco = a_src[keep], a_dst[keep], a_lab[keep], a_sco[keep]
    order = np.lexsort((a_lab, a_dst, a_src))
    tensor = torch.from_numpy(
        np.stack([a_src[order].astype(np.int32), a_dst[order].astype(np.int32),
                  a_lab[order].astype(np.int32),
                  a_sco[order].view(np.int32)], axis=1).copy())
    fsa = k2.Fsa(tensor)
    return fsa.to(device) if device != "cpu" else fsa


# ===================================================================================================
# HLG
# ===================================================================================================
def _sizes(fsa) -> Dict[str, int]:
    """``{n_fsas, states, arcs}`` of an ``Fsa`` OR an ``FsaVec``, off the ragged arc structure.

    ``Fsa.shape`` is ``(num_states, None)`` for an Fsa and ``(num_fsas, None, None)`` for an
    FsaVec (k2 ``fsa.py:1218-1229``), so ``int(fsa.shape[1])`` is ``int(None)`` and raises, and
    ``fsa.shape[0]`` is the FSA COUNT on a vector, not a state count.  ``fsa.arcs`` is the ragged
    tensor ``[state][arc]`` / ``[fsa][state][arc]``: the states are its total size one axis above
    the arcs, and ``num_arcs`` is the element count.  Verified against a real Fsa and a real
    FsaVec in the sae_k2 env (``test_lexlat_k2.test_sizes_read_an_fsa_and_an_fsa_vec``).
    """
    axes = int(fsa.arcs.num_axes())
    assert axes in (2, 3), axes
    return {"n_fsas": 1 if axes == 2 else int(fsa.arcs.dim0()),
            "states": int(fsa.arcs.tot_size(axes - 2)),
            "arcs": int(fsa.num_arcs)}


def compile_hlg(H, L, G, *, n_phones: int, n_words: int, determinize: bool = False,
                log=print) -> Tuple[object, List[dict]]:
    """icefall's ``compile_hlg.py`` recipe WITHOUT determinization, with a stage log.

    ``L_disambig . G``, connect, drop the disambiguation tokens, remove epsilons, compose with
    ``H``, arc-sort.  Nothing here is a knob.

    NOT DETERMINIZED, for every graph the probe prices -- the ESCAPE graph and the STRICT one
    alike.  k2's ``determinize`` is TROPICAL-ONLY (``k2/fsa_algo.py``: the result is "equivalent
    to the input `fsa` under tropical semiring"), so it merges the readings of a label string by
    MAX and does not preserve the LOG-semiring total the probe measures: a determinized graph
    would under-count that marginal.  The escape graph is not determinizable at all -- its phone
    loop and the lexical branch read the same phone at different weights, the transducer violates
    the twins property, and ``determinize`` does not return in >240 s on a 21-state G / 16-state L
    fixture whose strict build compiles in 0.01 s.  The graph priced is therefore the larger,
    ambiguous one, and ``build.json`` / the job's ``summary.txt`` record ``determinized: False``.

    ``determinize`` survives only as a test hook for icefall's own stage; it defaults to ``False``
    and :class:`~.lexlat_k2_jobs.LexlatHLGBuildJob` never passes ``True``.
    """
    import k2

    stages: List[dict] = []

    def stage(name, fsa):
        rec = {"stage": name, **_sizes(fsa), "seconds": time.monotonic() - stage.t0,
               "max_rss_gib": _max_rss_gib()}
        stage.t0 = time.monotonic()
        stages.append(rec)
        log(f"  [hlg] {name}: {rec['states']} states, {rec['arcs']} arcs, "
            f"{rec['seconds']:.1f} s, peak RSS {rec['max_rss_gib']:.1f} GiB", flush=True)
        return fsa

    stage.t0 = time.monotonic()
    L = stage("L_disambig", k2.arc_sort(L))
    G = stage("G", k2.arc_sort(G))
    LG = stage("compose(L, G)", k2.compose(L, G))
    LG = stage("connect", k2.connect(LG))
    if determinize:
        LG = stage("determinize", k2.determinize(LG))
        LG = stage("connect", k2.connect(LG))
    else:
        log("  [hlg] determinize: SKIPPED (k2's determinize is tropical-only; a determinized "
            "graph would under-count the log-semiring total the probe measures)", flush=True)
    LG.labels[LG.labels >= first_token_disambig_id(n_phones)] = 0
    LG.__dict__["_properties"] = None
    LG = stage("remove_epsilon", k2.remove_epsilon(LG))
    LG = stage("connect", k2.connect(LG))
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)
    LG = stage("arc_sort", k2.arc_sort(LG))
    HLG = stage("compose(H, LG)", k2.compose(H, LG, inner_labels="tokens"))
    HLG = stage("connect", k2.connect(HLG))
    HLG = stage("arc_sort", k2.arc_sort(HLG))
    return HLG, stages


def _max_rss_gib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


# ===================================================================================================
# CHUNKING -- one ``intersect_dense_pruned`` call per at most ``chunk_seqs`` sequences
# ===================================================================================================
# ``k2.intersect_dense_pruned`` EXPANDS every out-arc of every active state BEFORE it prunes
# (``intersect_dense_pruned.cu`` ``GetArcs`` / ``ai_lambda``) and its backward prune sums the arc
# counts of a 30-frame window into an ``int32_t`` (``PruneTimeRange``, ``intersect_dense_pruned.cu``
# 1383-1395).  The undeterminized escape ``H . L . G`` has word-boundary states with up to 399,785
# out-arcs -- one per successor pronunciation -- so that sum overflows 2^31 once enough sequences
# share one call, and the negative size reaches k2's device kernels as a CUDA fault
# (``reports/debug_k2_probe_timeout_2026-09-21.md``: 2 and 16 utterances pass on CPU, 57 and 114
# fail; on the GPU the child faulted and then hung).  The fix is to CALL IT PER CHUNK: the pruning,
# the beams and ``max_active_states`` are all PER SEQUENCE in k2, so chunking changes no
# per-sequence result -- only how many of them share one kernel launch.
#: the sequences per ``intersect_dense_pruned`` call.  16 is the largest size VERIFIED to pass in
#: the CPU reproduction of that report; it is a RUNTIME knob (see :func:`resolve_chunk_seqs`), not
#: an experimental constant -- it moves no number, only the launch granularity.
CHUNK_SEQS = 16
#: the environment variable that overrides it, for the job's child and for the training arm alike.
#: It is NOT a job parameter: no sisyphus hash carries it, so re-chunking never re-runs anything.
CHUNK_SEQS_ENV = "LEXLAT_K2_CHUNK_SEQS"
#: k2's own backward-prune window, in frames (``intersect_dense_pruned.cu`` ``PruneTimeRange``),
#: used ONLY by the pre-flight bound of :func:`window_arc_bound`.
K2_PRUNE_WINDOW = 30
#: the type the overflowing sum is taken in
INT32_MAX = 2 ** 31 - 1


def resolve_chunk_seqs(value: Optional[int] = None) -> int:
    """``value``, else ``$LEXLAT_K2_CHUNK_SEQS``, else :data:`CHUNK_SEQS`."""
    if value is None:
        value = os.environ.get(CHUNK_SEQS_ENV) or None
    if value is None:
        return int(CHUNK_SEQS)
    n = int(value)
    assert n >= 1, f"chunk_seqs = {n}: at least one sequence per intersect call"
    return n


def chunk_bounds(n_seqs: int, chunk_seqs: int) -> List[Tuple[int, int]]:
    """``[(start, stop), ...]`` covering ``range(n_seqs)`` in blocks of at most ``chunk_seqs``."""
    n = int(chunk_seqs)
    assert n >= 1, n
    return [(s, min(s + n, int(n_seqs))) for s in range(0, int(n_seqs), n)]


def max_out_degree(fsa) -> int:
    """The largest number of arcs leaving one state of ``fsa`` (an ``Fsa`` or an ``FsaVec``).

    ``fsa.arcs`` is the ragged ``[state][arc]`` / ``[fsa][state][arc]`` structure, so the
    difference of consecutive row splits of its LAST axis is the out-degree of each state.  This
    is the quantity that overflows k2's pruned intersection (see above), not the arc total.
    """
    axes = int(fsa.arcs.num_axes())
    assert axes in (2, 3), axes
    splits = fsa.arcs.shape().row_splits(axes - 1)
    if int(splits.numel()) < 2:
        return 0
    return int((splits[1:] - splits[:-1]).max())


def window_arc_bound(*, chunk_seqs: int, out_degree: int, max_active_states: int,
                     window: int = K2_PRUNE_WINDOW) -> int:
    """The worst case of the sum k2 takes in an ``int32``: sequences x fan-out x window x active.

    A WORST CASE and nothing finer: every one of the ``max_active_states`` states k2 keeps per
    sequence is assumed to be the graph's largest-fan-out state, on every frame of the 30-frame
    window ``PruneTimeRange`` sums over, for every sequence of the chunk.  The real count is far
    smaller (the beams cut it), so a bound below 2^31 is a guarantee and a bound above it is a
    WARNING, not a prediction.
    """
    return int(chunk_seqs) * int(out_degree) * int(window) * int(max_active_states)


def chunk_guard(hlg, *, chunk_seqs: int, max_active_ladder: Sequence[int], log=print) -> dict:
    """Log the graph's fan-out, the chunk size and the worst-case window sum, before the first call.

    Returns the record it logs (it also goes into the probe's json), so the number that was
    checked is on disk next to the measurement rather than only in a log line.
    """
    degree = max_out_degree(hlg)
    bounds = {str(int(ma)): window_arc_bound(chunk_seqs=chunk_seqs, out_degree=degree,
                                             max_active_states=int(ma))
              for ma in max_active_ladder}
    record = {"chunk_seqs": int(chunk_seqs), "max_out_degree": int(degree),
              "prune_window_frames": int(K2_PRUNE_WINDOW), "int32_max": INT32_MAX,
              "worst_case_window_arcs": bounds}
    log(f"chunking: {chunk_seqs} sequences per intersect_dense_pruned call; the graph's largest "
        f"out-degree is {degree} arcs from one state", flush=True)
    for key, bound in bounds.items():
        share = bound / float(INT32_MAX)
        note = ("OK" if share < 0.5 else
                "WARNING: approaching k2's int32 window sum" if share < 1.0 else
                "WARNING: ABOVE k2's int32 window sum -- reduce chunk_seqs or max_active")
        log(f"  max_active {key:>6s}: worst-case 30-frame window sum "
            f"{bound} arcs = {share:.2f} x 2^31-1  [{note}]", flush=True)
    return record


def chunked_tot_scores(graph, dense_in, lens, *, search_beam: float, output_beam: float,
                       min_active_states: int, max_active_states: int, chunk_seqs: int,
                       backward: bool = False, sync: bool = False):
    """``intersect_dense_pruned`` + log-semiring ``get_tot_scores`` per chunk of the batch.

    Returns ``(tot_scores, chunks, times)``:

    * ``tot_scores`` -- the per-sequence totals of the WHOLE batch, in the batch's own order,
      concatenated from the chunks.  Each sequence's value is the one an unchunked call returns:
      k2's beams, ``min_active_states`` and ``max_active_states`` are per sequence, so a chunk
      changes nothing but how many sequences share a kernel launch
      (``test_lexlat_k2.test_chunking_is_bit_identical_to_one_call``).
    * ``chunks`` -- ``[(start, lattice), ...]``, the pruned lattice of each chunk together with the
      batch index its first sequence has, for the callers that read lattice sizes or arc posteriors.
    * ``times`` -- ``{"intersect", "tot_scores", "backward"}``, the SUM over the chunks of each
      stage's seconds.  The sum is the true cost of the leg: the chunks are run one after the other.

    ``backward = True`` runs ``tot.sum().backward()`` per chunk, so the gradient of every chunk
    lands in ``dense_in.grad`` (each chunk is a slice of that one tensor) and no chunk's autograd
    graph outlives it; the returned scores are then detached.  ``backward = False`` keeps the
    autograd graph, which is what a training step needs.  ``sync = True`` CUDA-synchronises around
    each stage, which is what the probe's timings are read with.
    """
    import torch

    import k2

    b = int(dense_in.shape[0])
    lens_i32 = lens.detach().cpu().to(torch.int32)
    assert int(lens_i32.numel()) == b, (lens_i32.shape, dense_in.shape)
    parts: List[object] = []
    chunks: List[Tuple[int, object]] = []
    times = {"intersect": 0.0, "tot_scores": 0.0, "backward": 0.0}
    do_sync = bool(sync) and bool(dense_in.is_cuda)

    def _sync():
        if do_sync:
            torch.cuda.synchronize()

    for start, stop in chunk_bounds(b, chunk_seqs):
        n = stop - start
        seg = torch.stack([torch.arange(n, dtype=torch.int32),
                           torch.zeros(n, dtype=torch.int32),
                           lens_i32[start:stop]], dim=1)
        _sync()
        t0 = time.perf_counter()
        dense = k2.DenseFsaVec(dense_in[start:stop], seg)
        lattice = k2.intersect_dense_pruned(
            graph, dense, search_beam=float(search_beam), output_beam=float(output_beam),
            min_active_states=int(min_active_states),
            max_active_states=int(max_active_states))
        _sync()
        times["intersect"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        tot = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
        _sync()
        times["tot_scores"] += time.perf_counter() - t0
        if backward:
            t0 = time.perf_counter()
            tot.sum().backward()
            _sync()
            times["backward"] += time.perf_counter() - t0
            tot = tot.detach()
        parts.append(tot)
        chunks.append((start, lattice))
    return torch.cat(parts), chunks, times


def lattice_sizes(chunks) -> Dict[str, int]:
    """``_sizes`` summed over the chunks' lattices -- the batch's own lattice, as it was before."""
    total = {"n_fsas": 0, "states": 0, "arcs": 0}
    for _start, lattice in chunks:
        for key, value in _sizes(lattice).items():
            total[key] += int(value)
    return total


# ===================================================================================================
# the probe body
# ===================================================================================================
def run_probe(
    *,
    hlg_path: str,
    batch_dir: str,
    out_path: str,
    max_active_ladder: Sequence[int] = MAX_ACTIVE_LADDER,
    search_beam: float = SEARCH_BEAM,
    output_beam: float = OUTPUT_BEAM,
    min_active_states: int = MIN_ACTIVE_STATES,
    temperature: float = 1.0,
    partial_path: Optional[str] = None,
    step_abort_sec: Optional[float] = None,
    chunk_seqs: Optional[int] = None,
    log=print,
) -> dict:
    """``intersect_dense_pruned`` -> ``get_tot_scores`` -> ``backward`` on the dumped E1 batches.

    ``batch_dir`` holds one ``batch_<index>.pt`` per timed batch, written by
    ``lexlat_k2_jobs.LexlatK2ProbeJob`` from the SHARED env: ``log_q [B, T_rec, 40]`` (the tensor
    ``lattice.py`` is handed, float32), ``feat_lens [B]`` (recognizer frames) and ``retained [B]``
    (the 50 Hz unit-frame count ``l_tau`` divides by).

    The bed's DP divides every arc weight by ``tau`` (``lattice._arc_weights``), so BOTH the dense
    emissions and the HLG scores are scaled by ``1 / tau`` here: ``log Z`` then comes out in the
    bed's own tempered nats and the stability read is on the same scale as E0's 0.05 nats per
    retained frame.  Column 0 of the dense tensor is the unused blank column
    (:data:`NEG_INF`); no HLG arc carries label 0.

    THE BATCH IS INTERSECTED IN CHUNKS of at most ``chunk_seqs`` sequences
    (:func:`chunked_tot_scores`; ``None`` takes ``$LEXLAT_K2_CHUNK_SEQS`` and then
    :data:`CHUNK_SEQS`), because one call over a whole 114-sequence batch overflows k2's int32
    window sum on this graph and faults the device (see the chunking section above).  The chunks
    run one after the other, so ``seconds`` is the SUM of their stage timings -- the true cost of
    the leg -- while the peak memory read below is torch's running maximum over the whole loop,
    i.e. the MAX over the chunks and never their sum.  Per-sequence results do not depend on the
    chunk size.

    THE TIMED REGION, per (batch, ``max_active``), is the intersection, the LOG-SEMIRING total
    scores and ``sum().backward()`` to the emissions -- and NOTHING else.  Each of the three is
    CUDA-synchronised, and ``seconds`` (the number read against the bar) is their sum; the peak
    memory of the bar is read at the end of that region.  The max-plus total score -- the sanity
    check ``log-sum >= max``, which the objective does not contain -- is taken AFTER it, behind a
    synchronisation, with its own timer and its own peak reset, and is reported separately
    (``seconds_max_plus``): k2 launches it asynchronously, so inside the region its kernels would
    drain into ``seconds_tot_scores`` and its buffers into the peak (review finding 2).

    ``partial_path`` is rewritten after EVERY cell, with ``partial = true`` and the records so
    far, so a job killed by its Slurm clock still leaves every cell it measured (E1's ruling of
    2026-09-21, ``lexlat_train_jobs.LexlatEfficiencyProbeJob``).  ``step_abort_sec`` is E1's
    ``STEP_ABORT_SEC``, passed in by the job and never re-typed here: a cell above it is recorded
    with ``over_step_abort``, its ``max_active`` rung is ABORTED (no further batch is timed at that
    rung, and the job reads the rung as a measured FAIL) and the remaining rungs continue -- a
    measured FAIL is the deliverable, a Slurm timeout is not.
    """
    import torch

    import k2

    dev = torch.device("cuda", 0)
    hlg = k2.Fsa.from_dict(torch.load(hlg_path, map_location="cpu", weights_only=False))
    hlg.scores = hlg.scores / float(temperature)
    hlg = hlg.to(dev)
    if not hasattr(hlg, "shape") or len(hlg.shape) == 2:
        hlg = k2.create_fsa_vec([hlg])
    graph = _sizes(hlg)
    log(f"HLG on {torch.cuda.get_device_name(0)}: {graph}", flush=True)

    names = sorted(f for f in os.listdir(batch_dir) if f.startswith("batch_") and f.endswith(".pt"))
    assert names, f"{batch_dir} holds no batch_*.pt"
    records: List[dict] = []
    ladder = [int(x) for x in max_active_ladder]
    chunk = resolve_chunk_seqs(chunk_seqs)
    # BEFORE the first intersect call: the graph's fan-out against k2's int32 window sum
    guard = chunk_guard(hlg, chunk_seqs=chunk, max_active_ladder=ladder, log=log)
    #: ``max_active`` -> the batch index whose cell went over ``step_abort_sec``; that rung is
    #: timed on no further batch and the job reads it as a measured FAIL (E1's rule)
    aborted: Dict[str, int] = {}

    def _write_partial(current: Optional[dict] = None) -> None:
        """The cells measured so far, after EVERY cell: a killed job still leaves its data."""
        if partial_path is None:
            return
        done = records + ([current] if current is not None else [])
        with open(partial_path, "w") as fh:
            json.dump({"partial": True, "graph": graph, "temperature": float(temperature),
                       "search_beam": float(search_beam), "output_beam": float(output_beam),
                       "min_active_states": int(min_active_states),
                       "max_active_ladder": ladder,
                       "chunk_seqs": int(chunk), "chunk_guard": guard,
                       "step_abort_seconds": (None if step_abort_sec is None
                                              else float(step_abort_sec)),
                       "aborted_max_active": dict(aborted),
                       "n_batches_planned": len(names), "batches": done,
                       "note": "written after every timed cell; probe.json is the completed run",
                       }, fh)

    for name in names:
        payload = torch.load(os.path.join(batch_dir, name), map_location="cpu", weights_only=True)
        index = int(payload["index"])
        lens = payload["feat_lens"].to(torch.int32)
        retained = payload["retained"].to(torch.float64)
        b, t_max, k_n = payload["log_q"].shape
        base = payload["log_q"].to(dev, torch.float32) / float(temperature)
        row: dict = {"index": index, "n_seqs": b, "t_rec": t_max, "n_symbols": k_n,
                     "t_feat": int(payload["t_feat"]), "is_longest": bool(payload["is_longest"]),
                     "retained_total": float(retained.sum()), "by_max_active": {}}
        for max_active in ladder:
            key = str(int(max_active))
            if key in aborted:  # this rung already spent more than one whole sub-epoch's bar
                log(f"  batch {index} max_active {max_active}: SKIPPED, the rung was aborted at "
                    f"batch {aborted[key]}", flush=True)
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            dense_in = torch.full((b, t_max, k_n + 1), NEG_INF, dtype=torch.float32, device=dev)
            dense_in[:, :, 1:] = base
            dense_in.requires_grad_(True)
            torch.cuda.synchronize()
            # --- THE TIMED REGION: forward + log-semiring total score + backward, and nothing else
            # (per CHUNK of at most ``chunk`` sequences; the seconds are the SUM over the chunks
            # and the peak memory below is torch's running maximum, i.e. the MAX over them)
            tot_log, chunks, stage_sec = chunked_tot_scores(
                hlg, dense_in, lens, search_beam=float(search_beam),
                output_beam=float(output_beam), min_active_states=int(min_active_states),
                max_active_states=int(max_active), chunk_seqs=chunk, backward=True, sync=True)
            t_intersect = stage_sec["intersect"]
            t_scores = stage_sec["tot_scores"]
            t_backward = stage_sec["backward"]
            peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 30
            peak_allocated = torch.cuda.max_memory_allocated() / 2 ** 30
            # --- OUTSIDE it: the max-plus sanity check, its own timer and its own peak
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                tot_max = torch.cat([lat.get_tot_scores(log_semiring=False,
                                                        use_double_scores=True)
                                     for _start, lat in chunks])
            torch.cuda.synchronize()
            t_max_plus = time.perf_counter() - t0
            peak_reserved_max_plus = torch.cuda.max_memory_reserved() / 2 ** 30
            grad = dense_in.grad
            tot_cpu = tot_log.detach().cpu().to(torch.float64)
            max_cpu = tot_max.detach().cpu().to(torch.float64)
            finite = torch.isfinite(tot_cpu)
            seconds = float(t_intersect + t_scores + t_backward)
            over = step_abort_sec is not None and seconds > float(step_abort_sec)
            sizes = lattice_sizes(chunks)
            cell = {
                "max_active": int(max_active),
                # the launch granularity, not an experimental constant: the per-sequence results
                # are the unchunked call's and the seconds are the sum over the chunks
                "chunk_seqs": int(chunk),
                "n_chunks": len(chunks),
                "seconds": seconds,
                "seconds_intersect": float(t_intersect),
                "seconds_tot_scores": float(t_scores),
                "seconds_backward": float(t_backward),
                # the max-plus pass is the ``log-sum >= max`` check, NOT part of the objective and
                # NOT part of ``seconds``; it is taken after the timed region (review finding 2)
                "seconds_max_plus": float(t_max_plus),
                "peak_reserved_gib": peak_reserved,
                "peak_allocated_gib": peak_allocated,
                # read after the timed region's own reservations, so it is not comparable with the
                # bar; it says what the extra pass costs on top of what was already held
                "peak_reserved_gib_max_plus": peak_reserved_max_plus,
                # ``lattice_states`` / ``lattice_arcs`` off the ragged structure: the lattice is an
                # FsaVec, whose ``shape[1]`` is ``None`` (review finding 1); SUMMED over the
                # chunks, which is the batch's own lattice
                **{f"lattice_{k}": v for k, v in sizes.items()},
                "tot_scores": [float(x) for x in tot_cpu],
                "tot_scores_max_plus": [float(x) for x in max_cpu],
                # ONE count per utterance: a genuine -inf is BOTH non-finite and below the finite
                # floor, and counting it twice doubled the reported number (review finding 5)
                "n_neg_inf": int(((~finite) | (tot_cpu <= NEG_INF / 2)).sum()),
                "log_sum_ge_max": bool(torch.all(tot_cpu[finite] >= max_cpu[finite] - 1e-6)),
                "grad_finite": bool(torch.isfinite(grad).all()),
                "grad_nonzero": bool(torch.any(grad != 0)),
                "grad_l2": float(grad.detach().norm()),
                "retained": [float(x) for x in retained],
                "over_step_abort": bool(over),
            }
            row["by_max_active"][key] = cell
            log(f"  batch {index} max_active {max_active}: "
                f"{seconds:.3f} s "
                f"(int {t_intersect:.3f} / tot {t_scores:.3f} / bwd {t_backward:.3f}; "
                f"max-plus {t_max_plus:.3f} s, outside the timed region), "
                f"reserved {peak_reserved:.2f} GiB, "
                f"arcs {sizes['arcs']} over {len(chunks)} chunk(s) of <= {chunk} seqs",
                flush=True)
            if over:
                aborted[key] = index
                log(f"  ABORT: batch {index} at max_active {max_active} took {seconds:.1f} s, "
                    f"above STEP_ABORT_SEC = {float(step_abort_sec):.0f} s (the Design's bar for "
                    "a WHOLE sub-epoch); no further batch is timed at this rung and the rung's "
                    "read is FAIL", flush=True)
            del dense_in, chunks, tot_log, tot_max, grad
            _write_partial(row)
        records.append(row)
        _write_partial()
        if len(aborted) == len(ladder):
            log("every rung of the ladder is aborted; no cell is left to time", flush=True)
            break
    out = {"graph": graph, "temperature": float(temperature),
           "search_beam": float(search_beam), "output_beam": float(output_beam),
           "min_active_states": int(min_active_states),
           "max_active_ladder": ladder,
           "chunk_seqs": int(chunk), "chunk_guard": guard,
           "step_abort_seconds": None if step_abort_sec is None else float(step_abort_sec),
           "aborted_max_active": dict(aborted),
           "n_batches_planned": len(names),
           "batches": records}
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    _write_partial()
    return out


# ===================================================================================================
# the sub-commands (this file is run as a SCRIPT under the scratch env's python)
# ===================================================================================================
def mapped_objects_under(prefix: str, maps_path: str = "/proc/self/maps") -> List[str]:
    """The DISTINCT shared objects mapped into this process from under ``prefix``, sorted.

    A file is counted when its path starts with ``prefix`` and its name carries ``.so`` (``.so``
    itself or a versioned ``.so.12.6.85``).  One file occupies several lines of ``maps`` (its
    text, rodata and data segments); the caller wants FILES, so they are de-duplicated here.

    This is the only evidence that an ``import torch, k2`` resolved every library where the
    launcher lives: ``ldd`` and ``RPATH`` say what COULD be loaded, the maps file says what WAS
    (design review Q1.3b).
    """
    seen: List[str] = []
    with open(maps_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(maxsplit=5)
            if len(parts) < 6:
                continue
            path = parts[5].strip()
            if not path.startswith(prefix):
                continue
            name = os.path.basename(path)
            if not (name.endswith(".so") or ".so." in name):
                continue
            if path not in seen:
                seen.append(path)
    return sorted(seen)


def _cmd_gpu_check(args) -> int:
    # Port: the scratch mount point is the caller's --scratch-prefix (the source hardcoded it).
    # Without one nothing is counted; a caller that made the check fatal without stating where the
    # lease lives is refused here, before any GPU work.
    prefix = getattr(args, "scratch_prefix", None)
    forbid = os.environ.get(FORBID_SCRATCH_MAPS_ENV, "") == "1"
    if forbid and not prefix:
        raise ValueError(
            f"{FORBID_SCRATCH_MAPS_ENV}=1 makes scratch-mapped shared objects fatal, but no "
            "--scratch-prefix states the scratch mount point to check")

    import torch

    import k2

    dev = torch.device("cuda", 0)
    s = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0], [1, 2, 2, 0], [1, 1, 1, 0], [2, 3, -1, 0]],
                     dtype=torch.int32)
    fsa = k2.Fsa(s).to(dev)
    vec = k2.create_fsa_vec([k2.arc_sort(fsa)])
    log_probs = torch.log_softmax(
        torch.randn(1, 6, 3, device=dev, generator=torch.Generator(device=dev).manual_seed(0)),
        dim=-1).requires_grad_(True)
    seg = torch.tensor([[0, 0, 6]], dtype=torch.int32)
    lat = k2.intersect_dense_pruned(vec, k2.DenseFsaVec(log_probs, seg), search_beam=20.0,
                                   output_beam=8.0, min_active_states=30, max_active_states=10000)
    tot = lat.get_tot_scores(log_semiring=True, use_double_scores=True)
    tot.sum().backward()
    ok = bool(torch.isfinite(tot).all()) and bool(torch.any(log_probs.grad != 0))
    # AFTER the intersection, so every library the CUDA path needs has been dlopen'ed: which of
    # them came from the 90-day scratch lease (design review Q1.3b).  Reported always, fatal only
    # under $LEXLAT_K2_FORBID_SCRATCH_MAPS=1, so the banked jobs' check is unchanged.
    scratch = mapped_objects_under(prefix) if prefix else []
    print(json.dumps({"ok": ok, "device": torch.cuda.get_device_name(0),
                      "k2": k2.__dev_version__, "with_cuda": bool(k2.with_cuda),
                      "tot_scores": [float(x) for x in tot.detach().cpu()],
                      "grad_norm": float(log_probs.grad.norm()),
                      "python": sys.executable,
                      "torch": str(torch.__file__),
                      "scratch_prefix": prefix,
                      "forbid_scratch_maps": forbid,
                      "n_scratch_mapped": len(scratch),
                      "scratch_mapped": scratch}))
    for path in scratch:
        print(f"scratch-mapped shared object: {path}", flush=True)
    if scratch and forbid:
        raise RuntimeError(
            f"{len(scratch)} shared object(s) of this process are mapped from {prefix}, which is "
            f"a 90-day lease: {scratch}.  Under {FORBID_SCRATCH_MAPS_ENV}=1 that is fatal -- a run "
            "launched like this passes today and dies at the next resume.  Patchelf the RPATH or "
            "rebuild in place before any pack is submitted (design review Q1.3b)")
    return 0 if ok else 1


def _cmd_build_hlg(args) -> int:
    import torch

    import k2

    started = time.monotonic()
    lm = read_lm_tables(args.resources)
    if args.shuffled:
        assert "shuffled_word_id" in lm, "the resource npz carries no derangement"
        lm["word_id"] = lm["shuffled_word_id"]
    prune_stats = {"theta_nats": float(args.prune_theta)}
    if float(args.prune_theta) > 0.0:
        lm, prune_stats = prune_lm_tables(lm, float(args.prune_theta))
        print(f"pruned at theta = {args.prune_theta}: {prune_stats}", flush=True)
    n_phones, sil_id, n_words = int(lm["n_phones"]), int(lm["sil_id"]), int(lm["n_words"])
    prons = prons_from_trie(lm["child"], lm["word_start"], lm["word_id"])
    entries, max_disambig = add_lex_disambig(prons)
    print(f"lexicon: {len(prons)} pronunciations, max disambig #{max_disambig}", flush=True)
    min_frames = emission_min_frames(args.d_min, args.recognizer_stride)
    H = h_topology(n_phones, min_frames=min_frames)
    escape = not bool(args.strict_lexicon)
    unk_word = int(lm["unk_word"])
    esc_price = (escape_prices(lm["escape_phone_log_prob"], n_phones=n_phones, sil_id=sil_id)
                 if escape else None)
    assert not escape or str(lm["words"][unk_word]) == ESCAPE_WORD, (
        f"the escape word of the banked LM is {lm['words'][unk_word]!r}, not {ESCAPE_WORD!r}")
    print("escape: ON (the Design's ESCAPE convention)" if escape else
          "escape: OFF -- the STRICT lexicon, a DISCLOSED DEVIATION from the Design", flush=True)
    print("determinized: False -- k2's determinize is tropical-only, so a determinized graph "
          "would under-count the log-semiring total the probe measures (see compile_hlg)",
          flush=True)
    print(f"backoff loops: {args.backoff_loops}"
          + ("  (icefall's placement, every L state with a non-eps token)"
             if args.backoff_loops == BACKOFF_LOOPS_ALL else
             "  (only where a word transition can start, so each back-off route of a word "
             "transition is consumed exactly once under the log semiring)"), flush=True)
    L = lexicon_to_fst(entries, n_phones=n_phones, sil_id=sil_id, n_words=n_words,
                       max_disambig=max_disambig, sil_prob=float(args.sil_prob),
                       escape_word=unk_word if escape else None, escape_price=esc_price,
                       backoff_loops=str(args.backoff_loops))
    G = g_fsa(lm, drop_words=NON_EMITTABLE_WORDS if escape else NON_EMITTABLE_WORDS_STRICT)
    # never determinized, escape or strict: compile_hlg's default, stated here so the one graph
    # the probe prices cannot be built any other way
    HLG, stages = compile_hlg(H, L, G, n_phones=n_phones, n_words=n_words, determinize=False)
    torch.save(HLG.as_dict(), args.out_hlg)
    record = {
        "resources": os.path.realpath(args.resources), "shuffled": bool(args.shuffled),
        "n_phones": n_phones, "sil_id": sil_id, "n_words": n_words,
        "sil_prob": float(args.sil_prob), "d_min": int(args.d_min),
        "recognizer_stride": int(args.recognizer_stride), "min_frames": int(min_frames),
        "n_pronunciations": len(prons), "max_disambig": int(max_disambig),
        "escape": bool(escape),
        "backoff_loops": str(args.backoff_loops),
        "determinized": False,
        "escape_word": str(lm["words"][unk_word]),
        "escape_length_log_prob": float(ESCAPE_LENGTH_LOG_PROB),
        "escape_disambig": (int(n_phones) + 1 + int(max_disambig) + 1) if escape else None,
        "escape_price_min": None if esc_price is None else float(
            min(p for p in esc_price if p > NEG_INF / 2)),
        "escape_price_max": None if esc_price is None else float(
            max(p for p in esc_price if p > NEG_INF / 2)),
        "dropped_words": list(NON_EMITTABLE_WORDS if escape else NON_EMITTABLE_WORDS_STRICT),
        "prune": prune_stats,
        "n_1gram": int(lm["n_1gram"]), "n_2gram": int(lm["n_2gram"]), "n_3gram": int(lm["n_3gram"]),
        "h": _sizes(H), "l": _sizes(L), "g": _sizes(G), "hlg": _sizes(HLG),
        "stages": stages, "seconds": time.monotonic() - started,
        "max_rss_gib": _max_rss_gib(), "k2": k2.__dev_version__,
    }
    with open(args.out_json, "w") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps({k: record[k] for k in ("hlg", "seconds", "max_rss_gib", "prune")}), flush=True)
    return 0


def _cmd_probe(args) -> int:
    run_probe(hlg_path=args.hlg, batch_dir=args.batch_dir, out_path=args.out_json,
              max_active_ladder=[int(x) for x in args.max_active],
              search_beam=args.search_beam, output_beam=args.output_beam,
              min_active_states=args.min_active_states, temperature=args.temperature,
              partial_path=args.partial_json, step_abort_sec=args.step_abort_sec,
              chunk_seqs=args.chunk_seqs)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gpu-check", help="the k2 CUDA smoke test; exits non-zero on a dead build")
    p.add_argument("--scratch-prefix", default=None,
                   help="the scratch mount point whose mapped shared objects are counted and "
                        "printed after the CUDA intersection (the source used /e/scratch); unset "
                        f"= not counted; fatal only under {FORBID_SCRATCH_MAPS_ENV}=1, which "
                        "requires it (design review Q1.3b)")
    p.set_defaults(fn=_cmd_gpu_check)

    p = sub.add_parser("build-hlg", help="compile H . L . G and save it")
    p.add_argument("--resources", required=True, help="LexiconTrieBuildJob's lexlat_resources.npz")
    p.add_argument("--out-hlg", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--prune-theta", type=float, default=0.0)
    p.add_argument("--sil-prob", type=float, default=SIL_PROB)
    p.add_argument("--d-min", type=int, required=True)
    p.add_argument("--recognizer-stride", type=int, required=True)
    p.add_argument("--shuffled", action="store_true")
    p.add_argument("--strict-lexicon", action="store_true",
                   help="build WITHOUT the escape phone loop (a disclosed deviation from the "
                        "Design, which fixes ESCAPE inside the marginal)")
    p.add_argument("--backoff-loops", choices=list(BACKOFF_LOOP_PLACEMENTS),
                   default=BACKOFF_LOOPS_ALL,
                   help="where L's #0 back-off self-loops sit: 'all' is icefall's placement (the "
                        "default, and what every banked graph was built with); 'word_boundary' "
                        "puts them only where a word transition can start, so the log-semiring "
                        "total does not add one copy per insertion position")
    p.set_defaults(fn=_cmd_build_hlg)

    p = sub.add_parser("probe", help="intersect_dense_pruned + backward on the dumped batches")
    p.add_argument("--hlg", required=True)
    p.add_argument("--batch-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--max-active", nargs="+", type=int, default=list(MAX_ACTIVE_LADDER))
    p.add_argument("--search-beam", type=float, default=SEARCH_BEAM)
    p.add_argument("--output-beam", type=float, default=OUTPUT_BEAM)
    p.add_argument("--min-active-states", type=int, default=MIN_ACTIVE_STATES)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--partial-json", default=None,
                   help="rewritten after EVERY timed cell, so a killed job still leaves its data")
    p.add_argument("--step-abort-sec", type=float, default=None,
                   help="E1's STEP_ABORT_SEC, passed in by the job: a cell above it aborts its "
                        "max_active rung with a MEASURED FAIL instead of a Slurm timeout")
    p.add_argument("--chunk-seqs", type=int, default=None,
                   help="sequences per intersect_dense_pruned call (default: "
                        f"${CHUNK_SEQS_ENV}, else {CHUNK_SEQS}).  A RUNTIME knob: one call over a "
                        "whole batch overflows k2's int32 window sum on this graph, and the "
                        "per-sequence results do not depend on it")
    p.set_defaults(fn=_cmd_probe)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
