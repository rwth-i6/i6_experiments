"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lexlat_k2_train.py.

Port note: the training runtime only (``LexlatK2Spec``, ``LexlatK2Runtime`` and the constants),
imports made relative; the source's test oracle ``h_log_z_closed_form`` (named below) is cut.  The offline tools the
source also ran as a script (``overcount_strings``, ``collapse_phones``,
``lattice_best_path_phones``, ``run_step_probe``, ``main``; source lines 868-1383) are cut:
their callers are the over-count / step-probe jobs, none of them in the ported closure.

emc.lexlat_k2_train -- the k2 lexicon leg INSIDE the training step (`SAE_4A_lexlat.md` a9).

`SAE_4A_lexlat.md` "Design amendment 9, DRAFT" (2026-09-21) replaces the trie DP of round 2 by a
SECOND GRAPH beside the bed's lattice.  Per utterance, with ``e`` the recognizer's dense per-frame
phone log-probabilities and ``temperature`` the arm's own anneal at this sub-epoch (point 1):

    L_lex = log Z_HLG(e / temperature) - log Z_H(e / temperature)

-- the log-semiring total of the PRUNED intersection of ``e`` with the priced ``H . L . G``, minus
the UNPRUNED total under ``H`` alone.  ``L_lex <= 0`` is the log posterior mass of word-decomposable
phone strings under the recognizer's tempered frame-wise posterior, ESCAPE included; the ``Z_H``
denominator makes it a probability and removes the emissions' free scale.  The bed's own lattice
term is untouched and the two graphs are ADDED:

    loss += lam_lex * (-L_lex) / n_unit_frames

reduced exactly as the lattice loss is (the mean over the batch's KEPT utterances).  ``lam_lex`` is
Design 3's curriculum verbatim -- :func:`~.lexlat_train.lexlat_lambda`, imported, never re-declared
(point 2).

WHAT THIS MODULE IS NOT.  It is not an ``nn.Module``: :class:`LexlatK2Runtime` is a plain object
hung on the model exactly as :class:`~.lexlat_train.LexlatRuntime` is, so nothing of it enters
``state_dict`` and a banked checkpoint still loads strictly.  It owns no experimental constant that
the arm does not state: ``max_active`` and the HLG come from the config's constants block, filled
from the settling probe's ``summary.txt`` before launch (points 3 and 4).

GRADIENT.  To the recognizer's emissions alone, through k2's autograd (the probe's backward).  The
graph's own scores are frozen tensors that require no gradient, so no lexicon, word-LM or escape
price is trained; ``test_lexlat_k2_train`` asserts that no other parameter of the model receives a
gradient from this term.

TWO ENVIRONMENTS, ONE MODULE (as ``lexlat_k2``).  ``k2`` exists only in the scratch clone
``/e/scratch/spell/wu24/envs/sae_k2``; the shared training env has none.  Every ``k2`` import here
is therefore LOCAL to the function that needs it, so this module imports under both interpreters,
and the arm's train job runs under the k2 python (``returnn_python_exe``, point 10).  The same
rule lets :class:`~.lexlat_k2_arm_jobs.LexlatK2OvercountJob` -- a SISYPHUS job, hence the shared env
-- call :func:`overcount_strings` here as a SCRIPT under the scratch python:
``python lexlat_k2_train.py overcount --help``.  That is why the pre-launch over-count measurement
of point 5 lives in this file and not in the jobs module.

COST NOTE.  ``Z_H`` is taken UNPRUNED, as the amendment writes it, through ``k2.intersect_dense``
on ``H`` replicated once per utterance: about ``n_phones^2`` arcs per recognizer frame (1,600 at
the bed's 40 symbols), i.e. tens of millions of arcs for a full batch.  It is the larger of the two
legs in arcs and it is not tunable here -- the amendment fixes it.  The source's ``h_log_z_closed_form``
(not ported) states what that total IS at the bed's ``min_frames = 1`` (the frame-wise ``logsumexp`` sum, since
``H`` then accepts every frame string exactly once) and the source's test suite uses it as an independent
check of the whole dense / temperature plumbing; the objective itself always goes through k2.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import lexlat_k2 as K
from .lattice import LatticeConfig
from .lexlat_train import (
    LEXLAT_FULL_LAM, LEXLAT_ONSET, LEXLAT_RAMP, assert_ramp, lexlat_lambda,
)


__all__ = [
    "ABORT_MARKER",
    "EMPTY_RAISE_FRAC",
    "EMPTY_UNINFORMATIVE_FRAC",
    "EXPECTED_BUILD_FIELDS",
    "H_OUTPUT_BEAM",
    "OVERCOUNT_TOLERANCE",
    "STABILITY_REFERENCE_MAX_ACTIVE",
    "STABILITY_SEQS",
    "STABILITY_CHUNK_SEQS",
    "LexlatK2Spec",
    "LexlatK2Runtime",
    # re-exported from ``lexlat_train``, NEVER re-declared (amendment 9.2: the curriculum is
    # Design 3's, verbatim): the model definition's guard imports this module alone.
    "assert_ramp",
    "lexlat_lambda",
]


#: ``k2.intersect_dense`` takes an ``output_beam`` and has no "no pruning" switch; the ``H`` leg is
#: the UNPRUNED total of amendment 9.1, so the beam is set beyond any reachable score difference.
#: Every ``H`` arc is a real emission (the dense's unused blank column is never read), so nothing
#: is discarded at this beam -- unlike the full HLG, where the ``NEG_INF`` column would survive it.
H_OUTPUT_BEAM = 1.0e9
#: amendment 9.6: more than this fraction of a batch with an EMPTY lattice raises (the analogue of
#: Design 2 amendment 3's NEG_INF rule).  Per BATCH, and per arm -- the arms of a pack are separate
#: ``rnn.py`` processes, so the raise ends this arm alone.
EMPTY_RAISE_FRAC = 0.10
#: amendment 9.6's READING rule, never raised on: a SUB-EPOCH mean above this reads the arm
#: uninformative at that ``max_active`` rung.  The per-step ``lexlat_k2_empty_frac`` monitor is
#: marked ``as_error``, so RETURNN's own sub-epoch mean of it is the number the rule is read on.
EMPTY_UNINFORMATIVE_FRAC = 0.02
#: amendment 9.5's pre-registered acceptance: MEDIAN over-count <= 0.05 nats per token, per string
#: set.  Above it the term is read as an approximate word constraint and the arm is not funded
#: without a further amendment.  :class:`~.lexlat_k2_arm_jobs.LexlatK2OvercountJob` measures it.
OVERCOUNT_TOLERANCE = 0.05
#: amendment 9.5 (Design, k2 route item 3) reads the pruning stability against the LADDER'S LARGEST
#: rung: ``|log Z(max_active) - log Z(10000)|`` per retained frame.  The same reference the settling
#: probe's ladder ends on (``config_sae_4a_lexlat_k2_v1.MAX_ACTIVE``), so the arm's per-sub-epoch
#: monitor and the probe's summary read the SAME quantity.  Not an arm constant: no config states
#: it and the runtime's default is this number.
STABILITY_REFERENCE_MAX_ACTIVE = 10_000
#: the sequences the per-sub-epoch stability read runs on (amendment 9.5 / design review Q1.7:
#: "the first chunk (16 sequences) of the first batch").  A DIAGNOSTIC's size, not an experimental
#: constant: it bounds what the read costs, and the read is a median over the utterances it covers.
STABILITY_SEQS = 16
#: the sequences per ``intersect_dense_pruned`` call INSIDE the stability read -- the read's launch
#: granularity, decoupled from its size (:data:`STABILITY_SEQS`) and from the training leg's
#: ``chunk_seqs``.  At the reference rung on an exactly uniform posterior (a flat theta at the k2
#: on-set) one sequence expands about 2e8 arcs in k2's 30-frame window and 16 in one call overflow
#: its int32 sum (``reports/debug_l20_preflight_overflow_2026-09-23.md``: 2.41e9 at 16).  Like
#: ``chunk_seqs`` it moves no number: k2 prunes per sequence.
STABILITY_CHUNK_SEQS = 1
#: amendment 9.6 / design review Q2.4: the marker an arm leaves in its own work dir when the empty
#: -lattice rule fires.  The arm then EXITS 0, so the pack marks it finished (``pack_jobs.run_arms``
#: requires only that the arm wrote its ``learning_rates``), the sibling arms' reads run, and this
#: arm reads UNINFORMATIVE from this file.
ABORT_MARKER = "lexlat_k2_ABORT.json"
#: what ``pack_jobs.run_arms`` hard-links out of a finished arm's cwd (``ArmRun.learning_rates_src``
#: is RETURNN's own ``learning_rate_file``): the exit-0 path is taken only when it is already on
#: disk, since an arm that exits 0 without it makes the pack raise in the assert instead.
LEARNING_RATES_FILE = "learning_rates"
#: the ``build.json`` fields a config may pre-register through ``lexlat_k2_expected_build``
#: (amendment 9.1 / design review Q1.2), and where each one is read from the record.  ``shuffled``
#: is DELIBERATELY absent: it is the one field the treatment and the null differ in.
EXPECTED_BUILD_FIELDS = {
    "backoff_loops": ("backoff_loops",),
    "escape": ("escape",),
    "sil_prob": ("sil_prob",),
    # the compile records the word-LM pruning threshold inside its ``prune`` block
    "theta": ("prune", "theta_nats"),
}


# ===================================================================================================
# the runtime
# ===================================================================================================
@dataclass(frozen=True)
class LexlatK2Spec:
    """Everything an arm states about the k2 lexicon leg; every field is an EXPERIMENTAL CONSTANT."""

    hlg: str
    stats: str
    max_active: int
    resources: Optional[str] = None
    onset: int = LEXLAT_ONSET
    ramp: int = LEXLAT_RAMP
    full_lam: float = LEXLAT_FULL_LAM
    search_beam: float = K.SEARCH_BEAM
    output_beam: float = K.OUTPUT_BEAM
    min_active_states: int = K.MIN_ACTIVE_STATES
    h_output_beam: float = H_OUTPUT_BEAM
    empty_raise_frac: float = EMPTY_RAISE_FRAC
    #: what the arm's config STATES the graph is (amendment 9.1): the ``build.json`` fields of
    #: :data:`EXPECTED_BUILD_FIELDS`, asserted by :meth:`LexlatK2Runtime.check_bed`.  ``None`` --
    #: every arm registered before this existed -- checks nothing.
    expected_build: Optional[Dict[str, object]] = None
    #: the stability read's reference rung and the sequences it covers (amendment 9.5); both are
    #: the runtime's own defaults and no config states them
    stability_reference_max_active: int = STABILITY_REFERENCE_MAX_ACTIVE
    stability_seqs: int = STABILITY_SEQS
    #: where the amendment 9.6 abort marker is written; ``None`` = the process's cwd, which IS the
    #: arm's work dir under ``pack_jobs`` (only the tests state a directory)
    abort_dir: Optional[str] = None
    #: the ONE field that is NOT an experimental constant: the sequences per
    #: ``intersect_dense_pruned`` call (:data:`lexlat_k2.CHUNK_SEQS`).  It moves no number -- the
    #: per-sequence totals and their gradients are the unchunked call's -- only the launch
    #: granularity, which is what keeps k2's int32 window sum under 2^31 on this graph.
    chunk_seqs: int = K.CHUNK_SEQS


class LexlatK2Runtime:
    """The k2 lexicon leg inside the training step.  A PLAIN object: nothing is in ``state_dict``.

    :param hlg: ``LexlatHLGBuildJob``'s ``HLG.pt`` -- the ESCAPE graph of the treatment, or the
        SHUFFLED one of the null (amendment 9.8; the null differs in this file and nothing else).
    :param stats: that job's ``build.json``.  It is not decoration: :meth:`check_bed` asserts its
        ``n_phones`` / ``sil_id`` / ``d_min`` / ``recognizer_stride`` / ``min_frames`` against the
        LIVE ``model.lattice_cfg`` before the first term is added, exactly as
        :class:`~.lexlat_k2_jobs.LexlatK2ProbeJob` does before it dumps a frame, so a graph built
        for another bed can never enter this objective.
    :param max_active: amendment 9.3 -- the smallest probe rung that passes the E1 bar AND E0's
        stability rule.  Fixed before launch from the probe's ``summary.txt``, in the arm's config.
    :param resources: the ``LexiconTrieBuildJob`` npz the graph was compiled from, for the
        ``<unk>`` word id the escape monitor counts.  ``None`` takes the path ``build.json``
        recorded (``resources``), which is the graph's own provenance and cannot disagree with it.
    :param onset / ramp / full_lam: Design 3's curriculum, verbatim (amendment 9.2).
    :param search_beam / output_beam / min_active_states: the probe's operating point, icefall's
        MMI setting, imported from ``lexlat_k2`` and NOT swept (amendment 9.3).
    :param h_output_beam: see :data:`H_OUTPUT_BEAM`.
    :param empty_raise_frac: see :data:`EMPTY_RAISE_FRAC`.
    :param expected_build: what the CONFIG says this graph is -- a subset of
        :data:`EXPECTED_BUILD_FIELDS` (``backoff_loops``, ``escape``, ``sil_prob``, ``theta``),
        asserted against ``build.json`` in :meth:`check_bed` (amendment 9.1, design review Q1.2).
        The treatment and the null state the SAME dict, which is what makes "the same placement,
        the same escape, the same sil_prob, the same theta; ``shuffled`` is the only differing
        field" a property of the code instead of an executor's pre-submit read.  ``None`` (the
        default, and every arm registered before this existed) checks nothing.
    :param stability_reference_max_active / stability_seqs: the per-sub-epoch stability read's
        reference rung and its size; see :data:`STABILITY_REFERENCE_MAX_ACTIVE`.
    :param abort_dir: where the amendment 9.6 marker goes; ``None`` = cwd = the arm's work dir.
    :param chunk_seqs: sequences per ``intersect_dense_pruned`` call.  ``None`` takes
        ``$LEXLAT_K2_CHUNK_SEQS`` and then ``lexlat_k2.CHUNK_SEQS`` (16, the size verified in
        ``reports/debug_k2_probe_timeout_2026-09-21.md``).  It is a RUNTIME knob and NOT an arm
        constant: k2's beams and ``max_active`` are per sequence, so the term, its monitors and
        its gradients are the same at every chunk size; what changes is that a whole batch's
        expanded arcs no longer overflow the ``int32`` sum k2 takes over a 30-frame window
        (``lexlat_k2``'s chunking section).  No config states it, so no arm's hash carries it.
    """

    def __init__(
        self,
        *,
        hlg: str,
        stats: str,
        max_active: int,
        resources: Optional[str] = None,
        onset: int = LEXLAT_ONSET,
        ramp: int = LEXLAT_RAMP,
        full_lam: float = LEXLAT_FULL_LAM,
        search_beam: float = K.SEARCH_BEAM,
        output_beam: float = K.OUTPUT_BEAM,
        min_active_states: int = K.MIN_ACTIVE_STATES,
        h_output_beam: float = H_OUTPUT_BEAM,
        empty_raise_frac: float = EMPTY_RAISE_FRAC,
        chunk_seqs: Optional[int] = None,
        expected_build: Optional[Dict[str, object]] = None,
        stability_reference_max_active: int = STABILITY_REFERENCE_MAX_ACTIVE,
        stability_seqs: int = STABILITY_SEQS,
        abort_dir: Optional[str] = None,
    ):
        assert int(onset) >= 1, onset
        assert int(ramp) >= 1, ramp
        assert float(full_lam) > 0.0, (
            f"full_lam = {full_lam} turns the lexicon off at every sub-epoch; state no graph "
            "instead of a zero weight, so the arm runs the bed's banked path (Design 5)"
        )
        assert int(max_active) >= 1, (
            f"max_active = {max_active}: amendment 9.3 takes it from the settling probe's "
            "summary.txt (the smallest rung that passes the E1 bar and E0's stability rule)"
        )
        assert float(search_beam) > 0.0 and float(output_beam) > 0.0, (search_beam, output_beam)
        assert int(min_active_states) >= 0, min_active_states
        assert 0.0 < float(empty_raise_frac) <= 1.0, empty_raise_frac
        assert int(stability_reference_max_active) >= 1, stability_reference_max_active
        assert int(stability_seqs) >= 1, stability_seqs
        expected = None if expected_build is None else dict(expected_build)
        if expected is not None:
            unknown = sorted(set(expected) - set(EXPECTED_BUILD_FIELDS))
            assert not unknown, (
                f"lexlat_k2_expected_build states {unknown}, which this runtime does not read out "
                f"of build.json; the fields it checks are {sorted(EXPECTED_BUILD_FIELDS)}"
            )
            assert "shuffled" not in expected_build, (
                "``shuffled`` is the ONE build.json field the treatment and the null differ in "
                "(amendment 9.8); stating it here would force one value on both arms"
            )
        self.spec = LexlatK2Spec(
            hlg=str(hlg), stats=str(stats), max_active=int(max_active),
            resources=None if resources is None else str(resources),
            onset=int(onset), ramp=int(ramp), full_lam=float(full_lam),
            search_beam=float(search_beam), output_beam=float(output_beam),
            min_active_states=int(min_active_states), h_output_beam=float(h_output_beam),
            empty_raise_frac=float(empty_raise_frac),
            expected_build=expected,
            stability_reference_max_active=int(stability_reference_max_active),
            stability_seqs=int(stability_seqs),
            abort_dir=None if abort_dir is None else str(abort_dir),
            chunk_seqs=K.resolve_chunk_seqs(chunk_seqs),
        )
        #: device -> {"vec": the HLG as an FsaVec on that device, "raw": its scores at
        #: temperature 1, "temperature": the temperature its scores currently carry}
        self._graph: Dict[str, Dict[str, object]] = {}
        #: (device, batch size) -> ``H`` replicated once per utterance (``intersect_dense`` does
        #: NOT broadcast a single FSA over the segments, unlike ``intersect_dense_pruned``)
        self._h: Dict[Tuple[str, int], object] = {}
        self._facts: Optional[Dict[str, object]] = None
        self._checked = False
        #: the sub-epoch the stability read was last taken in, and its value -- the monitor is
        #: EMITTED every step (so RETURNN's sub-epoch mean of it IS this number) and COMPUTED once
        self._stability_epoch: Optional[int] = None
        self._stability_value: float = float("nan")

    # -- the curriculum ---------------------------------------------------------------------------
    def lam(self, epoch: int) -> float:
        """``lam_lex`` of this sub-epoch -- ``lexlat_train``'s own schedule (amendment 9.2)."""
        return lexlat_lambda(epoch, onset=self.spec.onset, ramp=self.spec.ramp,
                             full=self.spec.full_lam)

    def active(self, epoch: int) -> bool:
        """Is the lexicon leg in the objective at this sub-epoch at all?"""
        return self.lam(epoch) > 0.0

    # -- the frozen graph -------------------------------------------------------------------------
    def facts(self) -> Dict[str, object]:
        """``build.json`` plus the two word ids the escape monitor needs, read ONCE.

        ``unk_word`` is not in ``build.json`` (only the escape word's NAME is), so it comes from the
        resource npz the graph itself records, and ``n_words`` is cross-checked between the two: a
        mismatch would silently mislabel every aux label the monitors count.
        """
        if self._facts is None:
            with open(self.spec.stats) as fh:
                build = json.load(fh)
            res = self.spec.resources or str(build["resources"])
            assert os.path.exists(res), (
                f"the resource npz of this graph ({res}) is not on disk; it carries the <unk> word "
                "id the escape monitor counts.  State lexlat_k2_resources explicitly"
            )
            with np.load(res, allow_pickle=False) as z:
                unk_word, n_words = int(z["unk_word"]), int(z["n_words"])
            assert n_words == int(build["n_words"]), (
                f"{res} has {n_words} words and the graph was built over {build['n_words']}; the "
                "monitors would count another lexicon's word ids"
            )
            self._facts = {
                "build": build, "resources": res, "unk_word": unk_word, "n_words": n_words,
                "n_phones": int(build["n_phones"]), "sil_id": int(build["sil_id"]),
                "d_min": int(build["d_min"]), "recognizer_stride": int(build["recognizer_stride"]),
                "min_frames": int(build["min_frames"]), "escape": bool(build["escape"]),
                "shuffled": bool(build.get("shuffled", False)),
                "sil_prob": float(build["sil_prob"]),
                "determinized": bool(build.get("determinized", False)),
            }
        return self._facts

    def check_bed(self, cfg: LatticeConfig) -> None:
        """The graph's own facts against the LIVE bed, once per process (the probe job's defence).

        A graph compiled for another ``d_min``, stride or symbol set would price a different
        marginal from the one the recognizer emits into, and nothing downstream would say so.

        AND, when the arm states ``expected_build`` (amendment 9.1, design review Q1.2), the four
        fields that make the treatment and the null the same objective: ``backoff_loops``,
        ``escape``, ``sil_prob`` and the word-LM pruning ``theta``.  The null differs from the
        treatment in ``shuffled`` ALONE, and that is what this assert makes true of the code rather
        than of a pre-submit read: the null was compiled by a helper that took the loop placement
        from a DEFAULT, so a null carrying icefall's all-states placement (and with it the
        positional-multiplicity mass amendment 9.5 removed from the treatment) would otherwise
        train beside the treatment and inflate the null's own ``lexlat_k2_term_mean``.
        """
        if self._checked:
            return
        facts = self.facts()
        stride = int(cfg.recognizer_stride)
        assert cfg.topology == "blankfree", (
            f"the k2 lexicon leg is written for the blank-free bed (H is its run-collapse "
            f"topology); this model runs topology = {cfg.topology!r}"
        )
        for key, seen in (("n_phones", int(cfg.n_phones)), ("sil_id", int(cfg.sil_id)),
                          ("d_min", int(cfg.d_min)), ("recognizer_stride", stride)):
            assert int(facts[key]) == seen, (
                f"the graph was built with {key} = {facts[key]} and this arm runs {key} = {seen} "
                f"({self.spec.stats})"
            )
        expect = K.emission_min_frames(int(cfg.d_min), stride)
        assert int(facts["min_frames"]) == expect, (facts["min_frames"], expect)
        assert not facts["determinized"], (
            "this graph is DETERMINIZED; k2's determinize is tropical-only, so its log-semiring "
            "total under-counts the marginal this term is (lexlat_k2.compile_hlg)"
        )
        self.check_expected_build()
        self._checked = True

    def check_expected_build(self) -> Dict[str, object]:
        """``expected_build`` against ``build.json``; returns what was read (``{}`` when unstated).

        Split out of :meth:`check_bed` so a job can take the same read without a
        ``LatticeConfig``.  A field the record does not carry is a MISMATCH, not a pass: a graph
        compiled before the field existed is a graph whose placement nobody wrote down.
        """
        expected = self.spec.expected_build
        if not expected:
            return {}
        build = self.facts()["build"]
        seen: Dict[str, object] = {}
        for key, path in EXPECTED_BUILD_FIELDS.items():
            if key not in expected:
                continue
            node: object = build
            for part in path:
                assert isinstance(node, dict) and part in node, (
                    f"{self.spec.stats} carries no {'.'.join(path)}, so this graph's {key} was "
                    f"never recorded; the arm states {key} = {expected[key]!r} (amendment 9.1)"
                )
                node = node[part]
            seen[key] = node
            want = expected[key]
            if isinstance(want, bool) or isinstance(node, bool):
                ok = bool(node) == bool(want)
            elif isinstance(want, (int, float)) and not isinstance(want, bool):
                ok = abs(float(node) - float(want)) <= 1e-12
            else:
                ok = str(node) == str(want)
            assert ok, (
                f"the graph {self.spec.hlg} was compiled with {key} = {node!r} and this arm "
                f"states {key} = {want!r} ({self.spec.stats}).  The treatment and the null must "
                "agree on placement, escape, sil_prob and theta and differ in `shuffled` alone "
                "(amendment 9.8)"
            )
        print(f"lexlat_k2: build.json matches the stated {seen} "
              f"(shuffled = {self.facts()['shuffled']})", flush=True)
        return seen

    def graph(self, device, temperature: float):
        """The HLG on ``device``, its scores divided by ``temperature`` (``run_probe``'s rule).

        The bed's DP divides every arc weight by the temperature (``lattice._arc_weights``), so
        BOTH the dense emissions and the graph's own scores are scaled here and ``log Z`` comes out
        in the bed's own tempered nats.  Loaded ONCE per device; the rescale is done from the saved
        raw copy whenever the anneal moves (once per sub-epoch), never by reloading the file.
        """
        import k2

        key = str(device)
        entry = self._graph.get(key)
        if entry is None:
            fsa = k2.Fsa.from_dict(torch.load(self.spec.hlg, map_location="cpu",
                                              weights_only=False))
            if not hasattr(fsa, "shape") or len(fsa.shape) == 2:
                fsa = k2.create_fsa_vec([fsa])
            fsa = fsa.to(device)
            entry = {"vec": fsa, "raw": fsa.scores.detach().clone(), "temperature": None}
            self._graph[key] = entry
            print(f"lexlat_k2: HLG on {device}: {K._sizes(fsa)} from {self.spec.hlg}", flush=True)
        if entry["temperature"] != float(temperature):
            entry["vec"].scores = entry["raw"] / float(temperature)
            entry["temperature"] = float(temperature)
        return entry["vec"]

    def h_graph(self, device, batch: int):
        """``H`` replicated ``batch`` times: ``k2.intersect_dense`` needs one FSA per segment.

        ``H``'s arcs all score 0, so the temperature does not touch it.  Cached per (device, batch)
        -- the pack's batches repeat their shapes, and the object is a few thousand arcs.
        """
        import k2

        facts = self.facts()
        key = (str(device), int(batch))
        if key not in self._h:
            one = K.h_topology(int(facts["n_phones"]), min_frames=int(facts["min_frames"]),
                               device="cpu")
            self._h[key] = k2.create_fsa_vec([one] * int(batch)).to(device)
        return self._h[key]

    # -- the term ---------------------------------------------------------------------------------
    def dense(self, log_q: torch.Tensor, temperature: float) -> torch.Tensor:
        """``run_probe``'s dense tensor: column 0 the unused blank, the rest ``e / temperature``.

        ``lexlat_k2.run_probe`` builds it as ``torch.full(NEG_INF)`` with ``[:, :, 1:] = base``;
        this is the same tensor written as a ``cat`` so the gradient reaches ``log_q`` (the probe
        only needed a gradient w.r.t. the dense tensor itself).  k2's dense scores are float32.
        """
        b, t, _ = log_q.shape
        blank = torch.full((b, t, 1), K.NEG_INF, dtype=torch.float32, device=log_q.device)
        return torch.cat([blank, log_q.to(torch.float32) / float(temperature)], dim=2)

    def log_z_hlg(self, dense_in: torch.Tensor, feat_lens: torch.Tensor, temperature: float):
        """``(log Z_HLG, chunks)``: the PRUNED intersection, log semiring, double scores.

        The batch is intersected in CHUNKS of at most ``spec.chunk_seqs`` sequences
        (``lexlat_k2.chunked_tot_scores``, the probe's own path): one call over a whole batch
        overflows the ``int32`` sum k2 takes over a 30-frame window of expanded arcs on this
        graph.  ``log Z_HLG`` is the per-sequence total in the BATCH's order either way, with its
        autograd graph intact; ``chunks`` is ``[(first batch index, lattice), ...]`` for the
        monitors, which are the only readers of the lattice itself.
        """
        return K.chunked_tot_scores(
            self.graph(dense_in.device, temperature), dense_in, feat_lens,
            search_beam=float(self.spec.search_beam), output_beam=float(self.spec.output_beam),
            min_active_states=int(self.spec.min_active_states),
            max_active_states=int(self.spec.max_active),
            chunk_seqs=int(self.spec.chunk_seqs))[:2]

    def log_z_h(self, dense_in: torch.Tensor, feat_lens: torch.Tensor):
        """``log Z_H``: the UNPRUNED total under the bed's topology alone (amendment 9.1).

        ``k2.intersect_dense`` requires the supervision segments in DECREASING duration and does
        not broadcast one FSA over the batch, so the segments are sorted here and the scores are
        scattered back into the batch's own order.
        """
        import k2

        b = int(dense_in.shape[0])
        lens = feat_lens.detach().cpu().to(torch.int32)
        order = torch.argsort(lens.to(torch.int64), descending=True, stable=True)
        seg = torch.stack([order.to(torch.int32), torch.zeros(b, dtype=torch.int32),
                           lens[order]], dim=1)
        dense = k2.DenseFsaVec(dense_in, seg)
        lattice = k2.intersect_dense(self.h_graph(dense_in.device, b), dense,
                                     output_beam=float(self.spec.h_output_beam))
        sorted_scores = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
        inverse = torch.argsort(order).to(sorted_scores.device)
        return sorted_scores[inverse]

    def step(self, log_q: torch.Tensor, *, feat_lens: torch.Tensor, retained: torch.Tensor,
             keep: torch.Tensor, epoch: int, temperature: float, cfg: LatticeConfig,
             global_step: Optional[int] = None):
        """The term this step adds, and its monitors: ``(term, {name: 0-dim tensor})``.

        ``term`` is ``mean_kept[(-L_lex) / n_unit_frames]`` -- the lattice loss's own reduction
        (``(x * keep).sum() / keep.sum()``), so the two graphs are on one scale; the arm's config
        hands it to ``mark_as_loss`` at ``scale = lam_lex``.  ``retained`` is the utterance's 50 Hz
        UNIT-frame count, the divisor ``l_tau`` itself uses -- never a per-token mean (Design 4,
        the ``lm_prior_norm = "units"`` sign guarantee).

        ``keep`` is the bed's own non-``z_zero`` mask; an utterance whose PRUNED lattice came out
        EMPTY is dropped from the numerator on top of it and counted (amendment 9.6).  The
        denominator stays the lattice loss's ``keep.sum()``, so a step that loses utterances to
        pruning contributes a smaller term rather than a rescaled one.

        ``global_step`` is recorded in the abort marker alone; it is optional so a caller that has
        no step counter (the tests, a probe) is unchanged.

        THE TIMED REGION (``lexlat_k2_sec``) and the MEMORY READ
        (``lexlat_k2_peak_reserved_gib``) cover the WHOLE leg: the pruned intersection, both total
        scores and the arc-posterior monitor pass, which is what the arm actually pays per step and
        what the pre-launch step probe's clause (b) is read against (design review Q1.4 -- the
        settling probe left ``Z_H`` and the monitors out of its number).  The once-per-sub-epoch
        stability read is taken BEFORE that region and is in neither: it is a diagnostic, and the
        seconds the cost is read on must not carry it.
        """
        assert self.active(epoch), (
            f"step() at sub-epoch {epoch}, where lam_lex = 0; the train step must not call the "
            "lexicon leg before the on-set (Design 3's curriculum)"
        )
        self.check_bed(cfg)
        facts = self.facts()
        assert int(log_q.shape[2]) == int(facts["n_phones"]), (log_q.shape, facts["n_phones"])
        b = int(log_q.shape[0])
        cuda = log_q.is_cuda
        dense_in = self.dense(log_q, temperature)
        stability = self._stability(dense_in, feat_lens=feat_lens, retained=retained, epoch=epoch,
                                    temperature=temperature)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        z_hlg, chunks = self.log_z_hlg(dense_in, feat_lens, temperature)
        z_h = self.log_z_h(dense_in, feat_lens)

        empty = ~torch.isfinite(z_hlg)
        n_empty = int(empty.sum())
        assert bool(torch.isfinite(z_h).all()), (
            "log Z_H is not finite on every utterance: H accepts EVERY frame string, so this is a "
            "dense-tensor or supervision-segment defect, not a pruning loss"
        )
        if n_empty > self.spec.empty_raise_frac * b:
            self._abort_on_empty(n_empty=n_empty, b=b, epoch=epoch, global_step=global_step)
        scored = (keep > 0) & (~empty)
        w = scored.to(z_hlg.dtype)
        l_lex = torch.where(scored, z_hlg - z_h, torch.zeros_like(z_hlg))
        n_keep = keep.sum().clamp(min=1)
        term = ((-l_lex / retained.to(l_lex.dtype)) * w).sum() / n_keep.to(l_lex.dtype)
        monitors = self._monitors(chunks, l_lex=l_lex, scored=w, retained=retained,
                                  feat_lens=feat_lens, epoch=epoch, n_empty=n_empty, b=b)
        if cuda:
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        peak_gib = (torch.cuda.max_memory_reserved() / 2 ** 30) if cuda else 0.0
        dtype, device = l_lex.dtype, l_lex.device
        monitors["lexlat_k2_sec"] = torch.as_tensor(float(sec), dtype=dtype, device=device)
        monitors["lexlat_k2_peak_reserved_gib"] = torch.as_tensor(
            float(peak_gib), dtype=dtype, device=device)
        monitors["lexlat_k2_stability"] = torch.as_tensor(
            float(stability), dtype=dtype, device=device)
        return term.to(log_q.dtype), monitors

    # -- the two reads beside the term ------------------------------------------------------------
    def _stability(self, dense_in: torch.Tensor, *, feat_lens, retained, epoch: int,
                   temperature: float) -> float:
        """``|log Z(max_active) - log Z(reference)|`` per retained frame, median over utterances.

        THE GATE'S PRUNING CLAUSE FOR THE k2 FORM (amendment 9.5 / 9.6, design review Q1.7): the
        settling probe measured this at the ``ctrl_50`` ep10 posterior, but after the on-set the
        posterior is what the term reshapes, so the arm re-measures it as it trains.  Once per
        sub-epoch, on the FIRST :data:`STABILITY_SEQS` sequences of the first batch the leg runs in,
        under ``no_grad``, intersected :data:`STABILITY_CHUNK_SEQS` sequences per k2 call (the
        sample is NOT the training leg's chunk); the value is CACHED and the monitor is emitted on every
        step, so RETURNN's sub-epoch mean of ``lexlat_k2_stability`` is exactly this number and the
        Gate's "median > 0.05 for 3 consecutive sub-epochs from 11" is read off that column.

        The convention is the probe's own (``lexlat_k2_jobs.probe_reads_and_summary``): the
        absolute difference of the two log totals divided by the utterance's RETAINED (50 Hz unit)
        frame count, over the utterances whose two totals are both finite, and the MEDIAN of that.

        IT IS A DIAGNOSTIC AND MAY NOT KILL THE ARM: the reference rung allocates more than the
        arm's own, so an out-of-memory (or any k2 failure that raises in Python) here is caught,
        reported and recorded as ``nan`` -- a missing read, which the Gate reads as a missing read.
        A ``K2_CHECK`` failing in a k2 pool thread calls ``std::terminate`` and aborts the process;
        no Python guard catches that, which is why the per-call size is bounded above.
        """
        if self._stability_epoch == int(epoch):
            return self._stability_value
        self._stability_epoch = int(epoch)
        self._stability_value = float("nan")
        # the SAMPLE: the first ``stability_seqs`` utterances of the batch, whatever the training
        # leg's chunk is; each k2 call below takes STABILITY_CHUNK_SEQS of them
        n = min(int(self.spec.stability_seqs), int(dense_in.shape[0]))
        ref = int(self.spec.stability_reference_max_active)
        if ref <= int(self.spec.max_active):
            # the read compares this rung with a LARGER reference rung; at or below this rung the
            # comparison is trivially zero, which would read as a settled rung it has not shown.
            # ``nan`` is the Gate's missing read, and is what it gets
            print(f"lexlat_k2: the stability reference rung ({ref}) is not above this arm's rung "
                  f"({self.spec.max_active}); lexlat_k2_stability is nan (no read), not 0",
                  flush=True)
            return self._stability_value
        try:
            with torch.no_grad():
                d = dense_in.detach()[:n].contiguous()
                lens = feat_lens[:n]
                graph = self.graph(d.device, temperature)
                totals = []
                for max_active in (int(self.spec.max_active), ref):
                    tot, chunks, _sec = K.chunked_tot_scores(
                        graph, d, lens, search_beam=float(self.spec.search_beam),
                        output_beam=float(self.spec.output_beam),
                        min_active_states=int(self.spec.min_active_states),
                        max_active_states=int(max_active), chunk_seqs=STABILITY_CHUNK_SEQS)
                    totals.append(tot.detach().cpu().to(torch.float64))
                    del chunks
                keep = retained[:n].detach().cpu().to(torch.float64)
                gaps = [float(abs(a - c) / float(k))
                        for a, c, k in zip(totals[0], totals[1], keep)
                        if math.isfinite(float(a)) and math.isfinite(float(c)) and float(k) > 0]
            if gaps:
                self._stability_value = float(np.median(np.asarray(gaps, dtype=np.float64)))
            print(f"lexlat_k2: stability at sub-epoch {epoch}: median "
                  f"{self._stability_value:.4f} nats per retained frame over {len(gaps)} of {n} "
                  f"utterances (|log Z({self.spec.max_active}) - log Z({ref})|)", flush=True)
        except Exception as exc:  # a diagnostic never takes the arm down
            print(f"lexlat_k2: the stability read at sub-epoch {epoch} FAILED ({type(exc).__name__}"
                  f": {exc}); lexlat_k2_stability is nan for this sub-epoch", flush=True)
        finally:
            if dense_in.is_cuda:
                torch.cuda.empty_cache()
        return self._stability_value

    def _abort_on_empty(self, *, n_empty: int, b: int, epoch: int,
                        global_step: Optional[int]) -> None:
        """Amendment 9.6's raise, as design review Q2.4 registers it: a MARKER and exit 0.

        A raise here ends this arm's ``rnn.py`` with a non-zero code, ``pack_jobs.run_arms`` raises
        ``ArmFailure`` at the end of the pack, the pack job errors, and EVERY downstream read of
        EVERY arm of that pack waits on an errored job -- one lost arm blocks the cohort.  So the
        arm writes :data:`ABORT_MARKER` into its own work dir and exits 0 instead: the pack then
        marks it finished (it requires only that the arm left its ``learning_rates``, which the
        sub-epochs before the abort already wrote), the sibling arms' reads run, and this arm reads
        UNINFORMATIVE from the marker.

        ``os._exit`` and not ``sys.exit``: the call sits inside RETURNN's train step, several
        ``try``/``finally`` frames deep, and the one thing this path must guarantee is the exit
        CODE.  The marker and the streams are flushed first; the checkpoint of the sub-epoch in
        flight is deliberately not written, since that sub-epoch was never completed.

        WITHOUT a ``learning_rates`` on disk the exit-0 path would trip ``run_arms``' own assert,
        so the marker is still written and the original RuntimeError is raised: that can only
        happen before the first sub-epoch ends, i.e. never after an on-set of 5 or 8.
        """
        reason = (
            f"lexlat_k2: {n_empty} of {b} utterances have an EMPTY pruned lattice at sub-epoch "
            f"{epoch} (max_active = {self.spec.max_active}), above the "
            f"{self.spec.empty_raise_frac:.0%} of amendment 9.6.  The arm stops here; the next "
            "arm is a larger max_active, not a different conclusion"
        )
        work = os.path.abspath(self.spec.abort_dir or os.getcwd())
        marker = os.path.join(work, ABORT_MARKER)
        payload = {
            "reason": reason,
            "rule": "SAE_4A_lexlat.md amendment 9.6 (design review Q2.4: marker + exit 0)",
            "read": "UNINFORMATIVE at this max_active rung",
            "step": None if global_step is None else int(global_step),
            "subepoch": int(epoch),
            "n_empty": int(n_empty),
            "batch_size": int(b),
            "empty_frac": float(n_empty) / float(max(b, 1)),
            "empty_raise_frac": float(self.spec.empty_raise_frac),
            "max_active": int(self.spec.max_active),
            "hlg": self.spec.hlg,
            "work_dir": work,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(marker, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"{reason}\nlexlat_k2: wrote {marker}", flush=True)
        if not os.path.exists(os.path.join(work, LEARNING_RATES_FILE)):
            raise RuntimeError(
                f"{reason}  No {LEARNING_RATES_FILE} in {work}, so exiting 0 would only move the "
                f"failure into the pack's own assert; {ABORT_MARKER} is written and this raises"
            )
        print(f"lexlat_k2: exiting 0 so the pack marks this arm finished and its siblings' reads "
              f"run ({ABORT_MARKER} carries the cause)", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def _monitors(self, chunks, *, l_lex, scored, retained, feat_lens, epoch, n_empty, b):
        """The per-step monitors of amendment 9.9, as 0-dim tensors.

        ``lexlat_k2_lam``
            this sub-epoch's ``lam_lex``, so the ramp is visible in the log.
        ``lexlat_k2_term_mean``
            ``L_lex`` per UNIT frame, averaged over the SCORED utterances -- RAW, i.e. not
            multiplied by ``lam_lex``, so the arm's and the null's magnitudes stay comparable
            through the ramp (the Gate reads the magnitude match over sub-epochs 8-11).  Negative:
            it is a log probability.  The DP arm's ``lexlat_term_mean`` is the same convention on
            the same divisor and a DIFFERENT quantity (a sum of increments, not a log mass) --
            amendment 9.2's disclosed scale note.
        ``lexlat_k2_n_empty`` / ``lexlat_k2_empty_frac``
            utterances whose pruned lattice is empty (amendment 9.6): the count and the fraction
            of the batch.  RETURNN's sub-epoch mean of the fraction is what the 2 % reading rule
            is read on; the 10 % raise happens in :meth:`step`.
        ``lexlat_k2_lattice_arcs_per_frame`` / ``lexlat_k2_lattice_states_per_frame``
            the pruned lattice's size over the batch's RECOGNIZER frames -- the settling probe's
            own size read, per frame so it is comparable across batches.  SUMMED over the chunks
            of :meth:`log_z_hlg`, which together are the batch's own lattice.
        ``lexlat_k2_expected_words`` / ``lexlat_k2_expected_escape_words``
            posterior-expected word-emitting and ``<unk>`` transitions per SCORED utterance, from
            the lattice's arc posteriors (``get_arc_post``, log semiring): the aux labels are the
            word ids of ``G``, so ``1 .. n_words`` are words and ``unk_word + 1`` is the escape
            word, while ``n_words + 1`` is the back-off ``#0`` and is counted as neither.

            THE ESCAPE WORD IS A WORD OF ``G``, so ``lexlat_k2_expected_words`` INCLUDES the
            ``<unk>`` transitions that ``lexlat_k2_expected_escape_words`` counts -- the two are
            NOT disjoint and must never be added.  THE GATE READ IS THE RATIO
            ``lexlat_k2_expected_escape_words / lexlat_k2_expected_words``, the share of word
            transitions that escaped; at 1.0 the lexicon is inactive (everything escapes) and the
            arm is priced as if it had no lexicon at all, which is visible here without a second
            job.
        :meth:`step` adds three more after this pass, because they are read on the WHOLE leg:

        ``lexlat_k2_sec``
            the leg's wall clock, CUDA-synchronised: the pruned intersection, both total scores and
            the arc-posterior pass of this method -- the true leg, i.e. everything the arm pays for
            the lexicon on this step.  The settling probe's number covers the first of the three
            only, which is why it under-states the arm (design review Q1.4).
        ``lexlat_k2_peak_reserved_gib``
            ``torch.cuda.max_memory_reserved()`` over that same region, with the peak stats reset
            at its start: the arm holds every chunk's lattice with its autograd graph until
            RETURNN's outer backward, which the probe (backward per chunk) did not, so this is the
            only number that says what the card carries.  Read against the 80 GiB clause.
        ``lexlat_k2_stability``
            :meth:`_stability`'s once-per-sub-epoch read, emitted on every step.
        """
        device = l_lex.device
        dtype = l_lex.dtype
        n_scored = scored.sum().clamp(min=1)
        sizes = K.lattice_sizes(chunks)
        frames = float(feat_lens.sum())
        with torch.no_grad():
            words, escapes = self._expected_words(chunks, b=b)
        as_t = lambda v: torch.as_tensor(float(v), dtype=dtype, device=device)
        return {
            "lexlat_k2_lam": as_t(self.lam(epoch)),
            "lexlat_k2_term_mean": ((l_lex / retained.to(dtype)) * scored).sum() / n_scored,
            "lexlat_k2_n_empty": as_t(n_empty),
            "lexlat_k2_empty_frac": as_t(n_empty / max(b, 1)),
            "lexlat_k2_lattice_arcs_per_frame": as_t(sizes["arcs"] / max(frames, 1.0)),
            "lexlat_k2_lattice_states_per_frame": as_t(sizes["states"] / max(frames, 1.0)),
            "lexlat_k2_expected_words": (words.to(dtype) * scored).sum() / n_scored,
            "lexlat_k2_expected_escape_words": (escapes.to(dtype) * scored).sum() / n_scored,
        }

    def _expected_words(self, chunks, *, b: int):
        """``(expected word transitions, expected <unk> transitions)`` per utterance of the batch.

        The arc posteriors of the LOG semiring (``get_arc_post``) summed over the arcs whose AUX
        label is a word of ``G``.  ``aux_labels`` is ragged after ``compile_hlg``'s
        ``remove_values_eq(0)``, so a value's arc comes from its own row id and the arc's utterance
        from the lattice's arc -> state -> fsa row ids.

        ``chunks`` is :meth:`log_z_hlg`'s ``[(first batch index, lattice), ...]``: each chunk's
        lattice numbers its sequences from 0, so its counts are scattered back at that offset and
        the two vectors come out in the BATCH's own order.
        """
        facts = self.facts()
        n_words, unk = int(facts["n_words"]), int(facts["unk_word"])
        parts = [self._expected_words_one(lattice, n_words=n_words, unk=unk)
                 for _start, lattice in chunks]
        words = torch.cat([p[0] for p in parts])
        escapes = torch.cat([p[1] for p in parts])
        assert int(words.numel()) == int(b), (
            f"the chunks cover {int(words.numel())} sequences and the batch has {b}; "
            "intersect_dense_pruned returned one FSA per supervision segment until now"
        )
        return words, escapes

    def _expected_words_one(self, lattice, *, n_words: int, unk: int):
        """:meth:`_expected_words` on ONE chunk's lattice, numbered from 0."""
        b = int(lattice.arcs.dim0())
        post = lattice.get_arc_post(use_double_scores=True, log_semiring=True).exp()
        shape = lattice.arcs.shape()
        fsa_of_arc = shape.row_ids(1)[shape.row_ids(2).long()].long()
        aux = lattice.aux_labels
        if isinstance(aux, torch.Tensor):
            values, arc_of_value = aux, torch.arange(aux.numel(), device=aux.device)
        else:
            values, arc_of_value = aux.values, aux.shape.row_ids(1).long()
        out = []
        for mask in ((values >= 1) & (values <= n_words), values == unk + 1):
            rows = arc_of_value[mask]
            acc = torch.zeros(b, dtype=post.dtype, device=post.device)
            acc.index_add_(0, fsa_of_arc[rows], post[rows])
            out.append(acc)
        return out[0], out[1]

    def describe(self) -> Dict[str, object]:
        """The spec as a json-able dict, for a job's own record."""
        return {
            "hlg": self.spec.hlg, "stats": self.spec.stats, "resources": self.spec.resources,
            "max_active": self.spec.max_active, "onset": self.spec.onset, "ramp": self.spec.ramp,
            "full_lam": self.spec.full_lam, "search_beam": self.spec.search_beam,
            "output_beam": self.spec.output_beam, "min_active_states": self.spec.min_active_states,
            "h_output_beam": self.spec.h_output_beam,
            "empty_raise_frac": self.spec.empty_raise_frac,
            # what the config states the graph is (amendment 9.1) and the stability read's shape;
            # in the log of every arm, so the null's placement is readable off its own log
            "expected_build": self.spec.expected_build,
            "stability_reference_max_active": self.spec.stability_reference_max_active,
            "stability_seqs": self.spec.stability_seqs,
            "stability_chunk_seqs": STABILITY_CHUNK_SEQS,
            # the launch granularity of the intersection, not an arm constant (see the spec)
            "chunk_seqs": self.spec.chunk_seqs,
        }


