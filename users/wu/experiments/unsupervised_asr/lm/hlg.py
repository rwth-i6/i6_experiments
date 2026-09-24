"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lexlat_k2_jobs.py (``LexlatHLGBuildJob``,
``_child_env``), src/speech_llm/sae/emc/lexlat_k2_official_jobs.py (``LexlatOfficialHLGBuildJob``,
``MEM_GUARD_FRACTION``) and src/speech_llm/sae/emc/lexlat_k2_official_resources_jobs.py
(``LexlatOfficialResourcesJob``, ``EXACT_REFERENCE_MAX_ORDER``), with the graph wiring of the
configs config_sae_4a_lexlat_k2_v1.py / _pack_v1.py / _official_v1.py / _pack3_v1.py as
:func:`get_hlg`.

The three ``H . L . G`` graphs of phase 4a and the lexlat resources npz each is read with:

* ``"inhouse_3gram"`` -- :class:`LexlatHLGBuildJob` over the in-house trie + word trigram
  (``word_lm.LexiconTrieBuildJob``), ``#0`` loops at the word boundary; its null is the same build
  over the seed-0 derangement (``shuffled=True``).
* ``"official_4gram"`` -- :class:`LexlatOfficialHLGBuildJob` over the openslr lexicon and 4-gram,
  expected to settle at the theta = 5 nats pruning rung; null as above.
* ``"official_3gram_1e7"`` -- the same over the openslr 3-gram pruned 1e-7 (no null, as in the
  source).

The official graphs' resources npz is :class:`LexlatOfficialResourcesJob`'s (the null takes the
TREATMENT's, as in the source); both kinds price the escape with the in-house trie's own vector.

THE k2 LEG RUNS IN A CHILD PROCESS under the k2 python (``default_tools.K2_PYTHON_EXE``), which has
no sisyphus: ``python -m <package>.model.lexlat_k2 build-hlg`` for the in-house graph and ``python -m
<package>.lm.lexlat_k2_official arpa-to-npz | build-hlg`` for the official ones, with
:func:`_src_root` -- the directory holding the top-level ``i6_experiments`` package -- as the
child's whole ``PYTHONPATH``.  The source ran the same files as scripts by path.

Cut from the source modules: ``LexlatK2ProbeJob`` and ``probe_reads_and_summary`` (the finished
settling probe), ``WordLmPerplexityJob`` (a finished text-only read), ``STABILITY_BAR_NATS``,
``SLURM_TIME_CLAMP_HOURS``, ``GPU_CHECK_TIMEOUT_SEC`` and ``_run_with_timeout`` (probe-only), and the
``escape=False`` strict-lexicon path of both build jobs (raises ``ValueError``).
"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence

from sisyphus import Job, Task, tk

__all__ = [
    "MEM_GUARD_FRACTION",
    "EXACT_REFERENCE_MAX_ORDER",
    "HLG_KINDS",
    "LexlatHLGBuildJob",
    "LexlatOfficialHLGBuildJob",
    "LexlatOfficialResourcesJob",
    "get_hlg",
]

#: ``i6_experiments.users.wu.experiments.unsupervised_asr`` -- this package's parent
_PKG_ROOT = __package__.rsplit(".", 1)[0]
#: the two k2 children, run as ``python -m``
_INHOUSE_CHILD = f"{_PKG_ROOT}.model.lexlat_k2"
_OFFICIAL_CHILD = f"{__package__}.lexlat_k2_official"


def _src_root() -> str:
    """The directory holding the top-level ``i6_experiments`` package -- the child's whole ``PYTHONPATH``.

    Walks up one directory per component of this module's package name, from this file's
    directory WITHOUT resolving symlinks, so the child imports the same tree the parent did.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, *([os.pardir] * len(__package__.split(".")))))


def _child_env() -> Dict[str, str]:
    """A minimal environment for the k2-env python.

    ``PYTHONPATH`` is :func:`_src_root` only (the shared env's own entries would drag its torch
    in); ``TMPDIR`` is passed AT THE CALL SITE rather than inherited (the memory rule
    ``bash-dead-tmp-full``).  The k2 build resolves every shared library from its own RPATH, so no
    ``LD_LIBRARY_PATH`` and no ``module load`` are needed (build report section 5).

    ``K2_ABORT=1`` is SET, always.  Without it a k2 fatal error prints its trace and throws a
    ``std::runtime_error`` (``k2/csrc/log.h``), and on a CUDA context that has already taken a
    sticky fault the child then stalls in C++ unwinding instead of exiting -- which is how the
    probe job of 2026-09-21 spent two hours doing nothing
    (``reports/debug_k2_probe_timeout_2026-09-21.md``).  With it, a k2 fatal aborts the child at
    once and the parent sees a returncode instead of a silence.  ``$LEXLAT_K2_CHUNK_SEQS`` is
    forwarded when it is set, so the chunk size can be overridden from the environment of the
    sisyphus worker without touching any job hash.  ``PYTHONDONTWRITEBYTECODE=1`` keeps the child
    from writing ``__pycache__`` into the package tree it imports.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": _src_root(),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "K2_ABORT": "1",
    }
    for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_STEP_ID",
                "LEXLAT_K2_CHUNK_SEQS"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


# ===================================================================================================
# the graph
# ===================================================================================================
class LexlatHLGBuildJob(Job):
    """Compile ``H . L . G`` for the k2 probe and save it as ``HLG.pt`` (CPU).

    THE THREE FACTORS, each traced in ``lexlat_k2``'s module docstring:

    * ``H`` -- the bed's blank-free adjacent-run-collapse topology over the RECOGNIZER frames, with
      the bed's minimum duration expressed on that clock (``ceil(d_min / recognizer_stride) = 1``
      at ``d_min = 2`` / stride 3).  40 symbols including SIL, no blank; column 0 of the dense FSA
      is the unused blank column.
    * ``L`` -- the pronunciations of the banked ``LexiconTrieBuildJob`` (151,731 words, read back
      off its trie so the phone and word ids are the DP's own), SIL optional between words at
      ``sil_prob``, icefall ``prepare_lang`` disambiguation, PLUS the Design's ESCAPE phone loop
      (``escape = True``): one ``<unk>`` word-LM transition per contiguous non-SIL span and
      ``lexlat``'s own per-phone escape price, with the span's own disambiguation symbol
      (``lexlat_k2``'s module docstring, ESCAPE).  ``escape = False`` compiles the STRICT lexicon
      instead -- a DISCLOSED DEVIATION from `SAE_4A_lexlat.md` Design 1, which fixes ESCAPE inside
      the marginal; ``summary.txt`` says which of the two was priced either way.  NEITHER graph is
      determinized: k2's ``determinize`` is tropical-only, so a determinized graph would
      under-count the log-semiring total this probe measures, and the escape loop is not
      determinizable at all.  ``summary.txt`` and ``build.json`` record ``determinized: False``
      for every rung of the ladder.
    * ``G`` -- the banked word trigram, converted from the CSR back-off automaton of the same npz
      (the ARPA itself lives in the producing job's cleaned work directory; the automaton carries
      every one of its 13,964,716 n-grams).  No end-of-sentence term, the bed's convention.

    THE PRUNING LADDER.  ``prune_ladder`` is an ORDERED tuple of thresholds in nats; the job tries
    them in order and keeps the FIRST that compiles inside ``attempt_hours`` and the job's memory
    allocation.  ``0.0`` -- the full banked trigram -- must be its first entry, so a run that
    succeeds at once used the banked LM unchanged.  The rung actually used, the wall clock, the
    peak RSS and the resulting n-gram counts are written to ``build.json`` and ``summary.txt``:
    the deviation, if any, is recorded rather than hidden.  Each attempt is a SUBPROCESS, so an
    attempt killed by the OOM killer or by its own clock is caught and the next rung is tried.

    :param resources: ``LexiconTrieBuildJob``'s ``lexlat_resources.npz``, as a frozen path.
    :param env_python: the python of the k2 environment (``default_tools.K2_PYTHON_EXE``).
    :param d_min / recognizer_stride: the bed's own ``LatticeConfig`` values; the config asserts
        them against the model the probe runs on, they are never restated here.
    :param sil_prob: the optional-silence probability of the word boundary.
    :param escape: the Design's ESCAPE convention inside the graph (the default, and what route A
        has to price).  PORT: ``False`` (the strict lexicon) is cut and raises ``ValueError``.
    :param shuffled: build the graph over the seed-0 derangement (the phase's null); not used by
        the settling probe, available because the resource carries it.
    :param backoff_loops: where ``L``'s ``#0`` back-off self-loops sit
        (``lexlat_k2.BACKOFF_LOOP_PLACEMENTS``).  ``"all"`` is icefall's placement and the
        DEFAULT, so it is in ``__sis_hash_exclude__`` and every banked build keeps its hash;
        ``"word_boundary"`` puts the loops only where a word transition can start, which stops
        the log-semiring total from adding one copy per insertion position inside a word
        (``lexlat_k2._add_self_loops``, ``reports/audit_k2_overcount_2026-09-21.md`` section 3).
        A value other than the default is a DIFFERENT GRAPH and hashes as one.
    """

    __sis_version__ = 2
    #: ``backoff_loops`` is a NEW parameter; excluding it AT ITS DEFAULT keeps every banked build
    #: (``LexlatHLGBuildJob.rtX44PBJFNy1`` and the shuffled null) at the hash it already has,
    #: while a non-default value enters the hash and is therefore a new job.
    __sis_hash_exclude__ = {"backoff_loops": "all"}

    def __init__(
        self,
        *,
        name: str,
        resources: tk.Path,
        env_python: str,
        d_min: int,
        recognizer_stride: int,
        prune_ladder: Sequence[float] = (0.0,),
        sil_prob: float = 0.5,
        escape: bool = True,
        shuffled: bool = False,
        backoff_loops: str = "all",
        attempt_hours: float = 4.0,
        time_rqmt: float = 6.0,
        mem_rqmt: float = 200.0,
        cpu_rqmt: int = 16,
    ):
        super().__init__()
        if not escape:
            raise ValueError(
                "escape=False (the strict lexicon, no escape phone loop) is cut from the port: no "
                "in-scope graph compiles it")
        self.name = str(name)
        self.resources = resources
        self.env_python = str(env_python)
        self.d_min = int(d_min)
        self.recognizer_stride = int(recognizer_stride)
        self.prune_ladder = tuple(float(x) for x in prune_ladder)
        self.sil_prob = float(sil_prob)
        self.escape = bool(escape)
        self.shuffled = bool(shuffled)
        self.backoff_loops = str(backoff_loops)
        self.attempt_hours = float(attempt_hours)
        from ..model.lexlat_k2 import BACKOFF_LOOP_PLACEMENTS

        assert self.backoff_loops in BACKOFF_LOOP_PLACEMENTS, self.backoff_loops
        assert self.prune_ladder and self.prune_ladder[0] == 0.0, (
            "the first rung of the ladder is the FULL banked trigram (theta = 0); a run that "
            f"succeeds at once must have used it unchanged, got {self.prune_ladder}")
        assert list(self.prune_ladder) == sorted(self.prune_ladder), (
            f"the ladder is tried in order and must be increasing: {self.prune_ladder}")
        self.out_hlg = self.output_path("HLG.pt")
        self.out_stats = self.output_path("build.json")
        self.out_summary = self.output_path("summary.txt")
        #: Design of the dispatch 2026-09-21: 16 CPUs, 200 GB, 6 h (4 h per attempt)
        self.rqmt = {"cpu": int(cpu_rqmt), "mem": float(mem_rqmt), "time": float(time_rqmt)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json
        import shutil
        import subprocess as sp
        import time

        started = time.monotonic()
        attempts: List[dict] = []
        work_hlg = os.path.abspath("HLG.pt")
        work_json = os.path.abspath("build_attempt.json")
        chosen: Optional[dict] = None
        for theta in self.prune_ladder:
            for path in (work_hlg, work_json):
                if os.path.exists(path):
                    os.remove(path)
            cmd = [self.env_python, "-m", _INHOUSE_CHILD, "build-hlg",
                   "--resources", self.resources.get_path(),
                   "--out-hlg", work_hlg, "--out-json", work_json,
                   "--prune-theta", repr(float(theta)),
                   "--sil-prob", repr(self.sil_prob),
                   "--d-min", str(self.d_min),
                   "--recognizer-stride", str(self.recognizer_stride),
                   "--backoff-loops", self.backoff_loops]
            if self.shuffled:
                cmd.append("--shuffled")
            remaining = self.rqmt["time"] * 3600.0 * 0.95 - (time.monotonic() - started)
            if remaining < 300.0:
                attempts.append({"theta_nats": float(theta), "returncode": None,
                                 "note": "skipped: the job's wall clock is spent", "seconds": 0.0})
                print(f"[rung theta = {theta}] skipped, {remaining:.0f} s of wall left",
                      flush=True)
                break
            budget = max(60.0, min(self.attempt_hours * 3600.0, remaining))
            print(f"[rung theta = {theta}] {' '.join(cmd)}  (budget {budget / 3600:.2f} h)",
                  flush=True)
            t0 = time.monotonic()
            try:
                proc = sp.run(cmd, env=_child_env(), timeout=budget)
                rc = proc.returncode
                note = ""
            except sp.TimeoutExpired:
                rc, note = -1, f"killed after {budget / 3600:.2f} h"
            record = {"theta_nats": float(theta), "returncode": int(rc), "note": note,
                      "seconds": time.monotonic() - t0}
            if rc == 0 and os.path.exists(work_json) and os.path.exists(work_hlg):
                with open(work_json) as fh:
                    record["build"] = json.load(fh)
                attempts.append(record)
                chosen = record
                break
            print(f"[rung theta = {theta}] FAILED (rc {rc} {note}); trying the next rung",
                  flush=True)
            attempts.append(record)
        if chosen is None:
            raise RuntimeError(
                "no rung of the pruning ladder compiled an HLG inside the job's budget: "
                + json.dumps(attempts))
        shutil.move(work_hlg, self.out_hlg.get_path())
        build = chosen["build"]
        stats = {"name": self.name, "env_python": self.env_python,
                 "prune_ladder": list(self.prune_ladder), "attempt_hours": self.attempt_hours,
                 "backoff_loops": self.backoff_loops,
                 "attempts": attempts, "chosen_theta_nats": chosen["theta_nats"],
                 "elapsed_seconds": time.monotonic() - started, **build}
        with open(self.out_stats.get_path(), "w") as fh:
            json.dump(stats, fh, indent=2)
        prune = build.get("prune", {})
        with open(self.out_summary.get_path(), "w") as fh:
            fh.write(
                f"{self.name}  H . L . G for the k2 settling probe\n"
                f"k2 {build.get('k2')}  env {self.env_python}\n"
                f"H: {build['h']['states']} states, {build['h']['arcs']} arcs "
                f"(blank-free run-collapse, min_frames = {build['min_frames']} "
                f"= ceil(d_min {build['d_min']} / stride {build['recognizer_stride']}))\n"
                f"L: {build['l']['states']} states, {build['l']['arcs']} arcs "
                f"({build['n_pronunciations']} pronunciations, max disambig #"
                f"{build['max_disambig']}, sil_prob {build['sil_prob']})\n"
                + ("GRAPH PRICED: ESCAPE -- the Design's convention (one <unk> word-LM transition "
                   "per contiguous non-SIL span, plus the order-1 phone log probability + log 0.5 "
                   f"per escaped phone, in [{build.get('escape_price_min', float('nan')):.3f}, "
                   f"{build.get('escape_price_max', float('nan')):.3f}] nats; escape disambig "
                   f"#{build.get('escape_disambig')})\n"
                   if build.get("escape") else
                   "GRAPH PRICED: the STRICT LEXICON, with NO escape arcs -- a DISCLOSED "
                   "DEVIATION from `SAE_4A_lexlat.md` Design 1, which fixes ESCAPE inside the "
                   "marginal.  A read on this graph does not license route A.\n")
                + f"word arcs dropped from G: {build.get('dropped_words')}\n"
                + f"#0 back-off self-loops: {build.get('backoff_loops')}"
                + ("  -- icefall's placement: every L state with a non-eps outgoing token, so "
                   "one back-off route of a word transition can be consumed at SEVERAL positions "
                   "inside the word and the undeterminized graph's LOG-semiring total sums the "
                   "copies\n"
                   if build.get("backoff_loops", "all") == "all" else
                   "  -- only on the states a word transition can start from, so each back-off "
                   "route is consumed EXACTLY ONCE under the log semiring; the max-plus total is "
                   "unchanged\n")
                + f"determinized: {bool(build.get('determinized', False))}"
                + " -- k2's determinize is TROPICAL-ONLY (it merges the readings of a label "
                  "string by max), so a determinized graph would under-count the log-semiring "
                  "total this probe measures; the escape loop is not determinizable at all.  No "
                  "graph priced here is determinized and every score below is read off the "
                  "undeterminized graph.\n"
                + f"G: {build['g']['states']} states, {build['g']['arcs']} arcs "
                f"({build['n_1gram']} + {build['n_2gram']} + {build['n_3gram']} n-grams)\n"
                f"HLG: {build['hlg']['states']} states, {build['hlg']['arcs']} arcs\n"
                f"pruning: theta = {chosen['theta_nats']} nats"
                + ("  (the FULL banked trigram, nothing dropped)\n"
                   if chosen["theta_nats"] == 0.0 else
                   f"  (DISCLOSED DEVIATION: {prune.get('arcs_before')} -> "
                   f"{prune.get('arcs_after')} n-gram arcs)\n")
                + f"ladder tried: {[a['theta_nats'] for a in attempts]}\n"
                f"build: {build['seconds']:.1f} s, peak RSS {build['max_rss_gib']:.1f} GiB\n"
                + "".join(
                    f"  {s['stage']:<18s} {s['states']:>12d} states {s['arcs']:>12d} arcs "
                    f"{s['seconds']:8.1f} s  {s['max_rss_gib']:7.1f} GiB\n"
                    for s in build["stages"])
            )


#: The fraction of the job's memory allocation a rung's PREDICTED peak may reach before the rung
#: is skipped without composing.  The slack absorbs the prediction's error -- it extrapolates
#: linearly from a SINGLE reference build -- and the allocator's own overhead.
MEM_GUARD_FRACTION = 0.8


# ===================================================================================================
# the official graph
# ===================================================================================================
class LexlatOfficialHLGBuildJob(Job):
    """Compile ``H . L . G`` from the OFFICIAL LibriSpeech lexicon and ARPA, and save ``HLG.pt``.

    THE THREE FACTORS.  ``H`` is unchanged -- the bed's blank-free adjacent-run-collapse topology
    over the recognizer frames, 40 symbols including SIL, no blank, minimum duration
    ``ceil(d_min / recognizer_stride)``.  ``L`` and ``G`` are the official ones:

    * ``L`` -- openslr ``librispeech-lexicon.txt``, stress digits stripped, every phone mapped onto
      the bed's ``prior.PHONES`` (39 ARPAbet + SIL) and ASSERTED to be in it, entries with any
      unmappable phone dropped and COUNTED in ``build.json``, and the rest restricted to ``G``'s
      vocabulary.  ``<s>`` / ``</s>`` / ``<unk>`` / ``<UNK>`` are dropped AS WORDS.  SIL is
      optional between words at ``sil_prob``, the disambiguation is icefall's ``prepare_lang``, and
      the Design's ESCAPE phone loop is present with the PHASE'S OWN per-phone escape prices, read
      from ``escape_resources`` -- the same vector and the same one-``<unk>``-per-span transition
      the banked graph carries, so an official graph differs from it in ``L`` and ``G`` alone.
    * ``G`` -- the official ARPA, of any order, converted by ``lexlat_k2_official.parse_arpa_lm``
      into the CSR back-off automaton ``lexlat_k2.g_fsa`` reads: one arc per explicit n-gram, one
      epsilon (``#0``) back-off arc per state, every state final at score 0 (the bed's "no
      end-of-sequence term"), and the ``<s>`` / ``</s>`` arcs dropped.  At order 3 that converter
      reproduces ``lexlat.parse_arpa_word_lm``'s tables BIT FOR BIT
      (``test_lexlat_k2_official.test_order3_reader_is_the_banked_one``).

    NEITHER GRAPH IS DETERMINIZED, for ``lexlat_k2.compile_hlg``'s reason: k2's ``determinize`` is
    tropical-only, so it would under-count the log-semiring total the probe measures, and the
    escape loop is not determinizable at all.  ``build.json`` and ``summary.txt`` record
    ``determinized: False``.

    THE ARPA IS PARSED ONCE.  ``arpa-to-npz`` runs first and writes the CSR tables into the job's
    work directory; every rung of the pruning ladder re-reads that npz.  Its provenance -- file,
    order, per-order n-gram counts, vocabulary size, unknown-word spelling -- goes into
    ``lm.json`` and into ``build.json``.

    THE PRUNING LADDER is :class:`LexlatHLGBuildJob`'s rule verbatim: an ORDERED
    tuple of thresholds in nats, tried in order, the FIRST that compiles inside ``attempt_hours``
    and the job's memory allocation is kept, and ``0.0`` -- the full LM -- must be its first entry
    so that a run succeeding at once used the official LM unchanged.  Each attempt is a SUBPROCESS,
    so an attempt killed by the OOM killer or by its own clock is caught and the next rung tried.
    The criterion is ``lexlat_k2.prune_lm_tables``'s back-off gain, which drops n-gram ARCS and
    leaves the STATE set in place; ``lexlat_k2_official.compact_lm_tables`` then drops the states
    no surviving arc reaches and renumbers, so a rung shrinks ``G``'s ARC TOTAL -- n-gram arcs plus
    two per surviving state -- and not only its n-gram arcs.  ``build.json`` records states and
    arcs BEFORE and AFTER compaction at every rung.

    THE SIZE GUARD runs BEFORE the composition of every rung.  ``G``'s arc count is known once the
    tables are compacted, and the reference build
    (``LexlatHLGBuildJob.rtX44PBJFNy1/output/build.json``: 20.58 M ``G`` arcs, 143.87 M ``HLG``
    arcs, 26.04 GiB peak) fixes both ratios.  A rung whose PREDICTED peak exceeds
    ``0.8 * mem_rqmt`` is SKIPPED without composing and the next rung is tried, so a rung that
    cannot fit does not get the whole job killed by Slurm and take the ladder with it.  The
    prediction and the skip go into ``build.json``; the guard is a linear extrapolation from ONE
    build and is a scheduling device, never a measurement.  The rung used, the wall clock, the
    peak RSS and the resulting per-order counts are written out: the deviation, if any, is
    recorded.

    THE MAX-PLUS EQUIVALENCE CHECK IS NOT RUN HERE, and that is a stated choice, not an oversight.
    The source's ``test_lexlat_k2_official`` (not ported, nor are the two scorers it uses) runs it end to end on the identical code path -- an order-4 ARPA, a
    stress-marked multi-pronunciation lexicon, the escape loop -- against
    ``lexlat.string_best_segmentation`` plus ``lexlat_k2.sil_model_log_prob``.  Running it on a
    PRODUCTION graph would need a second full-size object beside the one being compiled (a
    ``lexlat.LexResources`` over the official automaton -- up to 84 M states for the full 4-gram,
    with its per-state unknown-word arc precomputed) inside the same allocation the pruning ladder
    exists to protect, and the equality is a property of the CONSTRUCTION, not of the LM's size.
    ``summary.txt`` says so for every graph it builds.

    :param name: the tag that names this graph in ``summary.txt``.
    :param arpa: the official ARPA (plain or ``.gz``), as a frozen path or a ``DownloadJob`` output.
    :param lexicon: openslr ``librispeech-lexicon.txt``.
    :param escape_resources: ``LexiconTrieBuildJob``'s ``lexlat_resources.npz`` -- read ONLY for
        the per-phone escape price vector and the inventory it is asserted against.
    :param env_python: the python of the k2 environment (``default_tools.K2_PYTHON_EXE``); it is in the job hash, so a
        rebuilt environment at a different path is a different measurement.
    :param d_min / recognizer_stride: the bed's own ``LatticeConfig`` values, which
        :class:`~.lexlat_k2_jobs.LexlatK2ProbeJob` asserts against the live model.
    :param escape: the Design's ESCAPE convention inside the graph (the default).  PORT: ``False``
        (the strict lexicon) is cut and raises ``ValueError``.
    :param backoff_loops: where ``L``'s ``#0`` back-off self-loops sit
        (``lexlat_k2.BACKOFF_LOOP_PLACEMENTS``), ``lexlat_k2.lexicon_to_fst``'s own parameter.
        ``"all"`` is icefall's placement and the DEFAULT, so it is in ``__sis_hash_exclude__`` and
        the three official builds of 2026-09-21 keep their hashes; ``"word_boundary"`` puts the
        loops only where a word transition can start, which stops the log-semiring total from
        adding one copy per insertion position inside a word (``lexlat_k2._add_self_loops``,
        ``reports/audit_k2_overcount_2026-09-21.md`` section 3, and the in-house rebuild
        ``LexlatHLGBuildJob.cdcxYJMjiYj5`` that amendment 9.5 passed on).  A value other than the
        default is a DIFFERENT GRAPH and hashes as one.
    :param shuffled: build the NULL -- the seed-0 derangement of the official word ->
        pronunciation assignment (``lexlat_k2_official.derange_official_prons``, which reuses
        ``lexlat.derange_pronunciations``, the in-house null's own function).  The same
        pronunciation multiset and the same trie shape; only which word a pronunciation ends at
        moves, so the null is Design 6's prior-weight control on the official lexicon.  ``False``
        is the DEFAULT and hash-excluded at it.
    """

    __sis_version__ = 1
    #: ``backoff_loops`` and ``shuffled`` are NEW parameters; excluding each AT ITS DEFAULT keeps
    #: the three finished official builds (``LexlatOfficialHLGBuildJob.GxCkDk90bQpT``,
    #: ``.NrghE6fnf5hc``, ``.YJsdBTcEQJz9``) at the hashes they already have, while a non-default
    #: value enters the hash and is therefore a new job.  ``LexlatHLGBuildJob`` does the same for
    #: the in-house graph (``lexlat_k2_jobs.py``), and the rule is one-directional: excluding a
    #: parameter that ALREADY existed would move every hash.
    __sis_hash_exclude__ = {"backoff_loops": "all", "shuffled": False}

    def __init__(
        self,
        *,
        name: str,
        arpa: tk.Path,
        lexicon: tk.Path,
        escape_resources: tk.Path,
        env_python: str,
        d_min: int,
        recognizer_stride: int,
        prune_ladder: Sequence[float] = (0.0,),
        sil_prob: float = 0.5,
        escape: bool = True,
        backoff_loops: str = "all",
        shuffled: bool = False,
        parse_hours: float = 4.0,
        attempt_hours: float = 4.0,
        time_rqmt: float = 6.0,
        mem_rqmt: float = 200.0,
        cpu_rqmt: int = 16,
    ):
        super().__init__()
        if not escape:
            raise ValueError(
                "escape=False (the strict lexicon, no escape phone loop) is cut from the port: no "
                "in-scope graph compiles it")
        self.name = str(name)
        self.arpa = arpa
        self.lexicon = lexicon
        self.escape_resources = escape_resources
        self.env_python = str(env_python)
        self.d_min = int(d_min)
        self.recognizer_stride = int(recognizer_stride)
        self.prune_ladder = tuple(float(x) for x in prune_ladder)
        self.sil_prob = float(sil_prob)
        self.escape = bool(escape)
        self.backoff_loops = str(backoff_loops)
        self.shuffled = bool(shuffled)
        self.parse_hours = float(parse_hours)
        self.attempt_hours = float(attempt_hours)
        from ..model.lexlat_k2 import BACKOFF_LOOP_PLACEMENTS

        assert self.backoff_loops in BACKOFF_LOOP_PLACEMENTS, self.backoff_loops
        assert self.prune_ladder and self.prune_ladder[0] == 0.0, (
            "the first rung of the ladder is the FULL official LM (theta = 0); a run that "
            f"succeeds at once must have used it unchanged, got {self.prune_ladder}")
        assert list(self.prune_ladder) == sorted(self.prune_ladder), (
            f"the ladder is tried in order and must be increasing: {self.prune_ladder}")
        self.out_hlg = self.output_path("HLG.pt")
        self.out_stats = self.output_path("build.json")
        self.out_lm = self.output_path("lm.json")
        self.out_summary = self.output_path("summary.txt")
        self.rqmt = {"cpu": int(cpu_rqmt), "mem": float(mem_rqmt), "time": float(time_rqmt)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json
        import shutil
        import subprocess as sp
        import time

        started = time.monotonic()
        env = _child_env()
        env["PYTHONPATH"] = _src_root()

        # --- (1) the ARPA, parsed ONCE into the CSR tables -------------------------------------
        lm_npz = os.path.abspath("lm_tables.npz")
        lm_json = os.path.abspath("lm.json")
        cmd = [self.env_python, "-m", _OFFICIAL_CHILD, "arpa-to-npz",
               "--arpa", self.arpa.get_path(),
               "--out-npz", lm_npz, "--out-json", lm_json]
        print(f"[arpa] {' '.join(cmd)}", flush=True)
        t0 = time.monotonic()
        proc = sp.run(cmd, env=env, timeout=self.parse_hours * 3600.0)
        if proc.returncode != 0 or not os.path.exists(lm_npz):
            raise RuntimeError(f"the ARPA parse exited with {proc.returncode}; see the log above")
        with open(lm_json) as fh:
            lm_record = json.load(fh)
        lm_record["parse_seconds"] = time.monotonic() - t0
        with open(self.out_lm.get_path(), "w") as fh:
            json.dump(lm_record, fh, indent=2)
        print(f"[arpa] order {lm_record['order']}, {lm_record['n_arcs']} arcs, "
              f"{lm_record['n_states']} states in {lm_record['parse_seconds']:.0f} s", flush=True)

        # --- (2) the pruning ladder, one subprocess per rung ------------------------------------
        # The ladder's budget clock starts HERE, after the parse.  The parse is charged separately
        # -- its seconds are logged above and in build.json -- and what the ladder may spend is the
        # job's wall allocation MINUS the parse, so the two together still fit inside time_rqmt and
        # Slurm never kills the job mid-rung.
        from .lexlat_k2_official import EXIT_SIZE_GUARD

        ladder_started = time.monotonic()
        ladder_budget = self.rqmt["time"] * 3600.0 * 0.95 - lm_record["parse_seconds"]
        print(f"[ladder] budget {ladder_budget / 3600:.2f} h "
              f"(= {self.rqmt['time']:.1f} h * 0.95 - {lm_record['parse_seconds']:.0f} s of parse)",
              flush=True)
        attempts: List[dict] = []
        work_hlg = os.path.abspath("HLG.pt")
        work_json = os.path.abspath("build_attempt.json")
        chosen: Optional[dict] = None
        for theta in self.prune_ladder:
            for path in (work_hlg, work_json):
                if os.path.exists(path):
                    os.remove(path)
            cmd = [self.env_python, "-m", _OFFICIAL_CHILD, "build-hlg",
                   "--lm-npz", lm_npz,
                   "--lexicon", self.lexicon.get_path(),
                   "--escape-resources", self.escape_resources.get_path(),
                   "--out-hlg", work_hlg, "--out-json", work_json,
                   "--prune-theta", repr(float(theta)),
                   "--sil-prob", repr(self.sil_prob),
                   "--d-min", str(self.d_min),
                   "--recognizer-stride", str(self.recognizer_stride),
                   # the guard's budget is a FRACTION of what Slurm granted this job, so a rung
                   # predicted not to fit is skipped instead of being killed with the job
                   "--mem-guard-gib", repr(MEM_GUARD_FRACTION * float(self.rqmt["mem"])),
                   "--backoff-loops", self.backoff_loops]
            if self.shuffled:
                cmd.append("--shuffled")
            remaining = ladder_budget - (time.monotonic() - ladder_started)
            if remaining < 300.0:
                attempts.append({"theta_nats": float(theta), "returncode": None,
                                 "note": "skipped: the ladder's wall clock is spent",
                                 "seconds": 0.0})
                print(f"[rung theta = {theta}] skipped, {remaining:.0f} s of wall left", flush=True)
                break
            budget = max(60.0, min(self.attempt_hours * 3600.0, remaining))
            print(f"[rung theta = {theta}] {' '.join(cmd)}  (budget {budget / 3600:.2f} h)",
                  flush=True)
            t0 = time.monotonic()
            try:
                proc = sp.run(cmd, env=env, timeout=budget)
                rc, note = proc.returncode, ""
            except sp.TimeoutExpired:
                rc, note = -1, f"killed after {budget / 3600:.2f} h"
            record = {"theta_nats": float(theta), "returncode": int(rc), "note": note,
                      "seconds": time.monotonic() - t0}
            if rc == 0 and os.path.exists(work_json) and os.path.exists(work_hlg):
                with open(work_json) as fh:
                    record["build"] = json.load(fh)
                attempts.append(record)
                chosen = record
                break
            if rc == EXIT_SIZE_GUARD and os.path.exists(work_json):
                # NOT a failure: the child priced the rung and declined to compose it
                with open(work_json) as fh:
                    skipped = json.load(fh)
                record["build"] = skipped
                record["skipped_by_size_guard"] = True
                guard = skipped.get("size_guard", {})
                print(f"[rung theta = {theta}] SKIPPED by the size guard: "
                      f"{guard.get('g_arcs')} G arcs predict "
                      f"{guard.get('predicted_hlg_arcs')} HLG arcs and "
                      f"{guard.get('predicted_peak_gib', float('nan')):.1f} GiB > "
                      f"{guard.get('mem_guard_gib', float('nan')):.1f} GiB; trying the next rung",
                      flush=True)
                attempts.append(record)
                continue
            print(f"[rung theta = {theta}] FAILED (rc {rc} {note}); trying the next rung",
                  flush=True)
            attempts.append(record)
        if chosen is None:
            raise RuntimeError(
                "no rung of the pruning ladder compiled an HLG inside the job's budget: "
                + json.dumps(attempts))
        shutil.move(work_hlg, self.out_hlg.get_path())
        build = chosen["build"]
        stats = {"name": self.name, "env_python": self.env_python,
                 "prune_ladder": list(self.prune_ladder), "attempt_hours": self.attempt_hours,
                 "backoff_loops": self.backoff_loops, "shuffled": self.shuffled,
                 "mem_guard_gib": MEM_GUARD_FRACTION * float(self.rqmt["mem"]),
                 "attempts": attempts, "chosen_theta_nats": chosen["theta_nats"],
                 "lm_provenance": lm_record,
                 "parse_seconds": lm_record["parse_seconds"],
                 "ladder_budget_seconds": ladder_budget,
                 "ladder_seconds": time.monotonic() - ladder_started,
                 "elapsed_seconds": time.monotonic() - started, **build}
        with open(self.out_stats.get_path(), "w") as fh:
            json.dump(stats, fh, indent=2)
        self._write_summary(build, chosen, attempts, lm_record)

    @staticmethod
    def _rung_line(a: dict) -> str:
        """One line per rung tried, saying what happened to it."""
        theta = a["theta_nats"]
        b = a.get("build", {})
        compact = b.get("compact", {})
        guard = b.get("size_guard", {})
        shape = (f"G {compact.get('g_arcs_before')} -> {compact.get('g_arcs_after')} arcs, "
                 f"{compact.get('states_before')} -> {compact.get('states_after')} states, "
                 f"predicted peak {guard.get('predicted_peak_gib', float('nan')):.1f} GiB"
                 if compact else "not reached")
        if a.get("skipped_by_size_guard"):
            budget = guard.get("mem_guard_gib", float("nan"))
            what = f"SKIPPED by the size guard (> {budget:.1f} GiB)"
        elif a.get("returncode") == 0:
            what = "BUILT"
        elif a.get("returncode") is None:
            what = a.get("note", "skipped")
        else:
            what = f"failed (rc {a.get('returncode')} {a.get('note', '')})".strip()
        return (f"  theta = {theta:<6g} {what}\n"
                f"      {shape}, {a.get('seconds', 0.0):.0f} s\n")

    def _write_summary(self, build, chosen, attempts, lm_record) -> None:
        prune = build.get("prune", {})
        compact = build.get("compact", {})
        guard = build.get("size_guard", {})
        lex = build.get("lexicon_stats", {})
        restrict = build.get("restrict_stats", {})
        order = int(build["order"])
        counts = build.get("ngram_counts", {})
        with open(self.out_summary.get_path(), "w") as fh:
            fh.write(
                f"{self.name}  H . L . G for the k2 cost probe, OFFICIAL LibriSpeech L and G\n"
                f"k2 {build.get('k2')}  env {self.env_python}\n"
                f"ARPA: {lm_record['arpa']}\n"
                f"  order {order}, vocabulary {build['vocab_size']}, unknown word "
                f"{lm_record['unk_word']!r}\n"
                f"  n-grams listed {lm_record['ngram_counts_listed']}\n"
                f"  n-grams kept   {counts}"
                f"  ({lm_record['n_unreachable']} dropped as unreachable, "
                f"{lm_record['n_unlisted_token']} with an unlisted word)\n"
                f"lexicon: {build['lexicon']}\n"
                f"  lines {lex.get('lines')}, kept {lex.get('kept')} after stress stripping; "
                f"dropped {lex.get('oov_phone')} for a phone outside the bed's "
                f"{build['n_phones']}-phone inventory, {lex.get('has_sil')} carrying SIL, "
                f"{lex.get('no_pron')} without a pronunciation, {lex.get('duplicate')} duplicates\n"
                f"  restricted to G: {restrict.get('oov_word')} entries dropped as out of the LM's "
                f"vocabulary, {restrict.get('dropped_marker')} dropped as markers "
                f"({build['dropped_words']}); {build['n_words_with_pronunciation']} words with a "
                f"pronunciation, {build['n_pronunciations']} pronunciations in L\n"
                f"  TOTAL lexicon entries dropped: {build['n_lexicon_entries_dropped']}\n"
                f"H: {build['h']['states']} states, {build['h']['arcs']} arcs "
                f"(blank-free run-collapse, min_frames = {build['min_frames']} "
                f"= ceil(d_min {build['d_min']} / stride {build['recognizer_stride']}))\n"
                f"L: {build['l']['states']} states, {build['l']['arcs']} arcs "
                f"(max disambig #{build['max_disambig']}, sil_prob {build['sil_prob']})\n"
                + ("GRAPH PRICED: ESCAPE -- the Design's convention, with the PHASE'S OWN escape "
                   "prices read from lexlat_resources.npz (one <unk> word-LM transition per "
                   "contiguous non-SIL span, plus the order-1 phone log probability + log 0.5 per "
                   f"escaped phone, in [{build.get('escape_price_min', float('nan')):.3f}, "
                   f"{build.get('escape_price_max', float('nan')):.3f}] nats; escape word "
                   f"{build.get('escape_word')!r}, escape disambig "
                   f"#{build.get('escape_disambig')})\n"
                   if build.get("escape") else
                   "GRAPH PRICED: the STRICT LEXICON, with NO escape arcs -- a DISCLOSED "
                   "DEVIATION from `SAE_4A_lexlat.md` Design 1, which fixes ESCAPE inside the "
                   "marginal.  A read on this graph does not license route A.\n")
                + f"#0 back-off self-loops: {build.get('backoff_loops', self.backoff_loops)}"
                + ("  (icefall's placement: every L state with a non-epsilon input token, so one "
                   "back-off route of a word transition is consumable at every position inside "
                   "the word and the undeterminized graph sums each as a separate path)\n"
                   if str(build.get("backoff_loops", self.backoff_loops)) == "all" else
                   "  (only where a word transition can start, so each back-off route is "
                   "consumed exactly once under the log semiring)\n")
                + (f"word -> pronunciation assignment: the seed-"
                   f"{(build.get('derangement') or {}).get('seed')} DERANGEMENT (this graph is "
                   f"the NULL: {(build.get('derangement') or {}).get('kept_by_homophony')} of "
                   f"{(build.get('derangement') or {}).get('n_words')} words kept a homophone of "
                   "their own pronunciation, the honest residual)\n"
                   if build.get("shuffled") else
                   "word -> pronunciation assignment: the official lexicon's own\n")
                + f"determinized: {bool(build.get('determinized', False))}"
                + " -- k2's determinize is TROPICAL-ONLY (it merges the readings of a label "
                  "string by max), so a determinized graph would under-count the log-semiring "
                  "total this probe measures; the escape loop is not determinizable at all.\n"
                + f"G: {build['g']['states']} states, {build['g']['arcs']} arcs\n"
                f"HLG: {build['hlg']['states']} states, {build['hlg']['arcs']} arcs\n"
                f"pruning: theta = {chosen['theta_nats']} nats"
                + ("  (the FULL official LM, nothing dropped)\n"
                   if chosen["theta_nats"] == 0.0 else
                   f"  (DISCLOSED DEVIATION: {prune.get('arcs_before')} -> "
                   f"{prune.get('arcs_after')} n-gram arcs)\n")
                + "  the criterion drops n-gram ARCS and leaves the STATE set in place; the "
                  "states no surviving arc reaches are then dropped by compact_lm_tables, which "
                  "renumbers and cannot move a score.\n"
                + f"compaction: {compact.get('states_before')} -> {compact.get('states_after')} "
                  f"states, {compact.get('ngram_arcs_before')} -> "
                  f"{compact.get('ngram_arcs_after')} n-gram arcs, "
                  f"G {compact.get('g_arcs_before')} -> {compact.get('g_arcs_after')} arcs "
                  f"(n-gram arcs + 2 per state - 1), {compact.get('seconds', 0.0):.0f} s\n"
                + f"size guard: {guard.get('g_arcs')} G arcs predicted "
                  f"{guard.get('predicted_hlg_arcs', float('nan')):.3g} HLG arcs and "
                  f"{guard.get('predicted_peak_gib', float('nan')):.1f} GiB against a budget of "
                  f"{guard.get('mem_guard_gib', float('nan')):.1f} GiB "
                  f"({MEM_GUARD_FRACTION} * {self.rqmt['mem']:.0f} GiB); the prediction is a "
                  f"ONE-POINT linear extrapolation from {guard.get('reference', {}).get('job')} "
                  f"and decides only which rungs are ATTEMPTED\n"
                + f"ladder tried: {[a['theta_nats'] for a in attempts]} of "
                  f"{list(self.prune_ladder)}\n"
                + "".join(self._rung_line(a) for a in attempts)
                + f"parse: {lm_record['parse_seconds']:.0f} s (charged separately; the ladder's "
                  f"clock starts after it)\n"
                + f"build: {build['seconds']:.1f} s, peak RSS {build['max_rss_gib']:.1f} GiB\n"
                + "".join(
                    f"  {s['stage']:<18s} {s['states']:>12d} states {s['arcs']:>12d} arcs "
                    f"{s['seconds']:8.1f} s  {s['max_rss_gib']:7.1f} GiB\n"
                    for s in build["stages"])
                + "\nMAX-PLUS EQUIVALENCE: not run on this graph, by design.  The equality of the "
                  "compiled graph's max-plus total with lexlat.string_best_segmentation plus the "
                  "silence model's constant is a property of the CONSTRUCTION and is checked end "
                  "to end on the identical code path -- an order-4 ARPA, a stress-marked "
                  "multi-pronunciation lexicon, the escape loop -- in test_lexlat_k2_official.  "
                  "Running it here would need a second full-size LexResources beside the graph "
                  "inside the allocation the pruning ladder exists to protect.\n")


#: the highest LM order at which the source's ``lexlat.string_log_sum`` / ``lexlat.string_best_segmentation``
#: (not ported) are an EXACT reference for the graph.  ``lexlat.lm_score`` takes the depth of its back-off walk
#: from ``LexResources.order``, which :func:`lexlat.load_resources` reads out of the npz, so the
#: reference is exact at every order whose ``order`` key this job writes.  It is a property of
#: ``lexlat.py``'s scorer, not a choice of this phase; the constant is kept so that the config and
#: the summary have ONE place to read it from if that ever regresses.
EXACT_REFERENCE_MAX_ORDER = 4


class LexlatOfficialResourcesJob(Job):
    """The official lexicon's trie + the official ARPA's CSR automaton as ONE ``lexlat`` npz (CPU).

    See the module doc for what is asserted against the graph's ``build.json`` and for the depth
    at which the scorers that read this file walk the back-off chain.  The job adds NO experimental constant: the
    ARPA, the lexicon, the escape prices and the pruning threshold are the GRAPH's own.

    :param name: the tag that names this resource in ``summary.txt``.
    :param build_json: the ``build.json`` of the :class:`LexlatOfficialHLGBuildJob`
        this npz is the CSR side of.  An INPUT: the graph is compiled first and its chosen pruning
        rung, order, vocabulary, state and n-gram counts, escape word, pronunciation count and
        ``max_disambig`` are read off it and asserted.
    :param arpa: the SAME official ARPA that build was given (its realpath is asserted against the
        ``build.json``'s provenance).
    :param lexicon: the SAME openslr ``librispeech-lexicon.txt``, likewise asserted.
    :param escape_resources: the in-house ``LexiconTrieBuildJob`` npz -- read ONLY for the RAW
        per-phone escape price column and the inventory it is asserted against, which is what makes
        the escape leg of the reference identical to the escape leg of the graph.
    :param derangement_seed: the null's seed; the default is
        ``lexlat_k2_official.DERANGEMENT_SEED`` (0), Design 6's own.
    :param time_rqmt / mem_rqmt / cpu_rqmt: sized from the finished builds' own measurements
        (``LexlatOfficialHLGBuildJob.NrghE6fnf5hc``: the full 4-gram ARPA parses in 397 s and the
        whole build peaked at 31.1 GiB; ``.GxCkDk90bQpT``: 16 s for the pruned 3-gram).  This job
        runs the parse, the prune and the compaction but NOT the composition, so those are upper
        bounds; the defaults keep the build job's own 200 GB and give the parse a wide clock.
    """

    __sis_version__ = 2

    def __init__(
        self,
        *,
        name: str,
        build_json: tk.Path,
        arpa: tk.Path,
        lexicon: tk.Path,
        escape_resources: tk.Path,
        derangement_seed: Optional[int] = None,
        time_rqmt: float = 4.0,
        mem_rqmt: float = 200.0,
        cpu_rqmt: int = 4,
    ):
        super().__init__()
        self.name = str(name)
        self.build_json = build_json
        self.arpa = arpa
        self.lexicon = lexicon
        self.escape_resources = escape_resources
        self.derangement_seed = None if derangement_seed is None else int(derangement_seed)
        self.out_resources = self.output_path("lexlat_resources.npz")
        self.out_json = self.output_path("resources.json")
        self.out_summary = self.output_path("summary.txt")
        self.rqmt = {"cpu": int(cpu_rqmt), "mem": float(mem_rqmt), "time": float(time_rqmt)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json
        import time

        import numpy as np

        from ..model import lexlat
        from ..model.lexlat_k2 import escape_prices, prune_lm_tables
        from ..phones import PHONE2ID, PHONES, SIL
        from . import lexlat_k2_official as O

        started = time.monotonic()
        seed = O.DERANGEMENT_SEED if self.derangement_seed is None else self.derangement_seed
        n_phones, sil_id = len(PHONES), PHONE2ID[SIL]
        with open(self.build_json.get_path()) as fh:
            build = json.load(fh)
        theta = float(build["chosen_theta_nats"])
        print(f"[graph] {build.get('name')}  order {build['order']}, theta {theta} nats, "
              f"backoff_loops {build.get('backoff_loops')}, shuffled {build.get('shuffled')}",
              flush=True)
        assert os.path.realpath(self.arpa.get_path()) == build["lm_provenance"]["arpa"], (
            "this is not the ARPA the graph was built from: "
            f"{os.path.realpath(self.arpa.get_path())} against {build['lm_provenance']['arpa']}")
        assert os.path.realpath(self.lexicon.get_path()) == build["lexicon"], (
            "this is not the lexicon the graph was built from: "
            f"{os.path.realpath(self.lexicon.get_path())} against {build['lexicon']}")

        # --- (1) G, through the build job's own three steps, in its own order -------------------
        t0 = time.monotonic()
        lm = O.parse_arpa_lm(self.arpa.get_path())
        parse_seconds = time.monotonic() - t0
        prune_stats: Dict[str, object] = {"theta_nats": theta}
        if theta > 0.0:
            lm, prune_stats = prune_lm_tables(lm, theta)
            print(f"[G] pruned at theta = {theta}: {prune_stats}", flush=True)
        lm, compact_stats = O.compact_lm_tables(lm)
        print(f"[G] compacted: {compact_stats}", flush=True)
        order = int(lm["order"])
        ngram_counts = {str(o): int(lm[f"n_{o}gram"]) for o in range(1, order + 1)}
        n_arcs = int(np.asarray(lm["arc_key"]).size)
        assert order == int(build["order"]), (order, build["order"])
        assert int(lm["n_words"]) == int(build["vocab_size"]), (lm["n_words"], build["vocab_size"])
        assert int(lm["n_states"]) == int(build["compact"]["states_after"]), (
            int(lm["n_states"]), build["compact"]["states_after"])
        assert n_arcs == int(build["compact"]["ngram_arcs_after"]), (
            n_arcs, build["compact"]["ngram_arcs_after"])
        assert ngram_counts == {str(k): int(v) for k, v in build["ngram_counts"].items()}, (
            ngram_counts, build["ngram_counts"])
        # THE INVENTORY IS THE GRAPH'S OWN, read off build.json.  The parsed LM tables carry NO
        # phone inventory: ``lexlat_k2_official.save_lm_npz`` STAMPS n_phones / sil_id onto the
        # npz it writes (the bed's constants, not a property of the ARPA), exactly as this job
        # stamps them onto its own npz below.  So the check that this resource prices the graph's
        # bed is against the build's record of the inventory it compiled H and the escape with.
        assert int(build["n_phones"]) == n_phones and int(build["sil_id"]) == sil_id, (
            "the graph was compiled for another phone inventory: "
            f"({build['n_phones']}, {build['sil_id']}) against ({n_phones}, {sil_id})")
        unk = int(lm["unk_word"])
        assert str(lm["words"][unk]) == str(build["escape_word"]), (
            str(lm["words"][unk]), build["escape_word"])

        # --- (2) L, the same pronunciations the graph's L was built from ------------------------
        entries, lex_stats = O.read_official_lexicon(self.lexicon.get_path())
        prons, restrict_stats = O.official_prons(entries, lm["words"])
        print(f"[L] lexicon: {lex_stats}; restricted to G: {restrict_stats}", flush=True)
        assert len(prons) == int(build["n_pronunciations"]), (
            len(prons), build["n_pronunciations"])
        _entries, max_disambig = O.add_lex_disambig(prons)
        assert int(max_disambig) == int(build["max_disambig"]), (
            max_disambig, build["max_disambig"])

        # --- (3) the trie, and the null's word-id column ----------------------------------------
        trie = O.build_trie_multi(prons, n_phones=n_phones)
        null_prons, null_stats = O.derange_official_prons(prons, lm["words"], seed=seed)
        trie_null = O.build_trie_multi(null_prons, n_phones=n_phones)
        for key in ("child", "word_start", "is_word_end"):
            assert np.array_equal(np.asarray(trie[key]), np.asarray(trie_null[key])), (
                f"the derangement moved the trie's {key}: it is not the prior-weight control")
        print(f"[trie] {int(trie['n_nodes'])} nodes, {int(trie['n_entries'])} entries; "
              f"null: {null_stats}", flush=True)

        # --- (4) the escape prices, the phase's own, RAW ----------------------------------------
        raw_escape = O.read_raw_escape_prices(
            self.escape_resources.get_path(), n_phones=n_phones, sil_id=sil_id)
        priced_here = escape_prices(raw_escape, n_phones=n_phones, sil_id=sil_id)
        priced_graph = O.read_escape_prices(
            self.escape_resources.get_path(), n_phones=n_phones, sil_id=sil_id)
        assert np.array_equal(priced_here, priced_graph), (
            "the escape price vector this npz implies is not the one the graph was compiled with")

        # --- (5) the npz, in lexlat.save_resources' own layout ----------------------------------
        lexlat.save_resources(
            self.out_resources.get_path(),
            trie=trie, lm=lm, escape_phone_log_prob=raw_escape,
            n_phones=n_phones, sil_id=sil_id, shuffled_word_id=trie_null["word_id"])

        # --- (6) read it back the way its consumers will ----------------------------------------
        import torch

        res = lexlat.load_resources(self.out_resources.get_path(), dtype=torch.float64)
        assert res.n_states == int(lm["n_states"]) and res.n_words == int(lm["n_words"])
        assert res.n_nodes == int(trie["n_nodes"])
        assert res.unk_word == unk and str(res.words[res.unk_word]) == str(build["escape_word"])
        null = lexlat.load_resources(
            self.out_resources.get_path(), dtype=torch.float64, shuffled=True)
        assert null.n_nodes == res.n_nodes
        # THE DEPTH THE SCORERS WILL WALK.  Without the npz's ``order`` key a reader falls back to
        # lexlat.DEFAULT_LM_ORDER and an order-4 resource is priced one level too shallow, which
        # is silent (log p = 0 for a word that only has a lower-order arc).  Assert, do not hope.
        assert res.order == order == null.order, (res.order, null.order, order)
        exact = order <= EXACT_REFERENCE_MAX_ORDER
        assert exact, (
            f"order {order} is above the depth lexlat's fixed-string scorers reach "
            f"({EXACT_REFERENCE_MAX_ORDER}); this npz would not be an exact reference")

        record = {
            "name": self.name,
            "graph_build_json": os.path.realpath(self.build_json.get_path()),
            "graph_name": build.get("name"),
            "graph_backoff_loops": build.get("backoff_loops"),
            "graph_shuffled": bool(build.get("shuffled", False)),
            "arpa": os.path.realpath(self.arpa.get_path()),
            "lexicon": os.path.realpath(self.lexicon.get_path()),
            "escape_resources": os.path.realpath(self.escape_resources.get_path()),
            "order": order,
            "theta_nats": theta,
            "prune": prune_stats,
            "compact": compact_stats,
            "ngram_counts": ngram_counts,
            "n_states": int(lm["n_states"]),
            "n_words": int(lm["n_words"]),
            "n_ngram_arcs": n_arcs,
            "begin_state": int(lm["begin_state"]),
            "unk_word": unk,
            "unk_word_spelling": str(lm["words"][unk]),
            "n_phones": n_phones,
            "sil_id": sil_id,
            "n_nodes": int(trie["n_nodes"]),
            "n_entries": int(trie["n_entries"]),
            "n_pronunciations": len(prons),
            "max_disambig": int(max_disambig),
            "lexicon_stats": lex_stats,
            "restrict_stats": restrict_stats,
            "derangement": null_stats,
            "derangement_seed": int(seed),
            "escape_price_min": float(min(p for p in priced_here if p > -1.0e29)),
            "escape_price_max": float(max(p for p in priced_here if p > -1.0e29)),
            "csr_reference_exact": bool(exact),
            "csr_reference_max_order": int(EXACT_REFERENCE_MAX_ORDER),
            "csr_reference_levels": int(res.order),
            "n_4gram_not_in_npz": (None if order < 4 else int(lm["n_4gram"])),
            "parse_seconds": parse_seconds,
            "elapsed_seconds": time.monotonic() - started,
        }
        with open(self.out_json.get_path(), "w") as fh:
            json.dump(record, fh, indent=2)
        with open(self.out_summary.get_path(), "w") as fh:
            fh.write(self._summary(record))
        print(open(self.out_summary.get_path()).read(), flush=True)

    def _summary(self, r: dict) -> str:
        lines: List[str] = [
            f"{self.name}  the OFFICIAL graph's CSR side as a lexlat resource npz "
            f"(trie + word LM + escape prices)",
            f"graph: {r['graph_name']}  (#0 back-off self-loops: {r['graph_backoff_loops']}, "
            f"shuffled {r['graph_shuffled']})",
            f"       build.json: {r['graph_build_json']}",
            f"ARPA:    {r['arpa']}",
            f"lexicon: {r['lexicon']}",
            f"escape prices (RAW column, the phase's own): {r['escape_resources']}",
            "",
            f"G: order {r['order']}, pruning theta {r['theta_nats']} nats (the rung THAT BUILD "
            f"chose, read off its build.json), {r['n_states']} states after compaction, "
            f"{r['n_ngram_arcs']} n-gram arcs, vocabulary {r['n_words']}",
            f"   n-grams kept: {r['ngram_counts']}",
            f"   unknown word: {r['unk_word_spelling']!r} (id {r['unk_word']}) -- the escape "
            f"span's word cost",
            f"L: {r['n_pronunciations']} pronunciations, max disambig #{r['max_disambig']}, "
            f"trie {r['n_nodes']} nodes / {r['n_entries']} entries",
            f"   escape price range [{r['escape_price_min']:.3f}, {r['escape_price_max']:.3f}] "
            f"nats per phone (order-1 phone log prob + log 0.5, SIL floored)",
            f"null: seed-{r['derangement_seed']} derangement, {r['derangement']} -- the trie's "
            f"child / word_start / is_word_end are bit-identical (asserted); only word_id moves",
            "",
            "ASSERTED AGAINST THE GRAPH'S build.json: order, vocabulary, states and n-gram arcs "
            "after compaction, per-order n-gram counts, the escape word's spelling, the "
            "pronunciation count and L's max_disambig.  The ARPA and the lexicon are asserted to "
            "be the very files that build read.  This npz therefore prices the same L and the "
            "same G as the graph; the #0 loop placement is a property of the GRAPH alone and "
            "does not enter a CSR score.",
            "",
        ]
        lines += [
            f"EXACT REFERENCE: yes at order {r['order']}.  lexlat.string_log_sum and "
            "lexlat.string_best_segmentation price a word transition through lexlat.lm_score, "
            f"whose back-off walk is {r['csr_reference_levels']} levels here -- the depth an "
            f"order-{r['order']} automaton needs.  The depth is not assumed: this npz stores its "
            "LM order, lexlat.load_resources reads it into LexResources.order, and this job "
            "asserted after the round trip that the reloaded resource reports it.",
        ]
        if r["n_4gram_not_in_npz"] is not None:
            lines += [
                f"n_4gram = {r['n_4gram_not_in_npz']} is recorded HERE and not in the npz: "
                "lexlat.save_resources' scalar list ends at n_3gram.  It is a count, read by "
                "nothing.",
            ]
        lines += [
            "",
            "This job measures nothing and decides nothing: it writes a resource.",
            f"parse {r['parse_seconds']:.0f} s, total {r['elapsed_seconds']:.0f} s",
        ]
        return "\n".join(lines) + "\n"


# ===================================================================================================
# the wiring
# ===================================================================================================
#: the graph kinds :func:`get_hlg` builds
HLG_KINDS = ("inhouse_3gram", "official_4gram", "official_3gram_1e7")

#: the ``#0`` back-off loop placement of every graph (``lexlat_k2.BACKOFF_LOOPS_WORD_BOUNDARY``)
_BACKOFF_LOOPS = "word_boundary"
#: the Design's escape convention and the build jobs' own optional-silence default
_ESCAPE = True
_SIL_PROB = 0.5
#: the build allocation (config_sae_4a_lexlat_k2_v1 ``BUILD_MEM_GB`` / ``BUILD_CPU``)
_BUILD_MEM_GB = 200.0
_BUILD_CPU = 16
#: the in-house ladder and budget (config_sae_4a_lexlat_k2_v1 ``PRUNE_LADDER``,
#: ``BUILD_ATTEMPT_HOURS``, ``BUILD_TIME_HOURS``); ``LexlatHLGBuildJob.cdcxYJMjiYj5`` settled at 0.0
_INHOUSE = {"prune_ladder": (0.0, 0.5, 2.0), "attempt_hours": 4.0, "time_rqmt": 6.0,
            "name": "lexlat_20/hlg_word_boundary", "null_name": "k2shuf_20/hlg_word_boundary",
            "theta": 0.0}
#: the official ARPA parse clock (config_sae_4a_lexlat_k2_official_v1 ``PARSE_HOURS``)
_PARSE_HOURS = 2.0
#: kind -> the official graph's tag, ladder, budget and the rung its banked build settled on
#: (config_sae_4a_lexlat_k2_official_v1 ``PRUNE_LADDER_4G`` / ``PRUNE_LADDER_3G``, ``BUILD_BUDGET``;
#: config_sae_4a_lexlat_k2_pack3_v1 ``THETA_4G`` / ``THETA_3G``: ``a3vpyag6WCBy`` 5.0, ``ViS3eo0AKqQr``
#: 0.0).  Only the 4-gram has a null (pack3 ``SHUFFLED_TAGS``).
_OFFICIAL = {
    "official_4gram": {"tag": "off4g_word_boundary",
                       "prune_ladder": (0.0, 0.5, 2.0, 5.0, 8.0, 12.0, 16.0),
                       "attempt_hours": 1.0, "time_rqmt": 8.0, "theta": 5.0, "null": True},
    "official_3gram_1e7": {"tag": "off3g_1e7_word_boundary", "prune_ladder": (0.0, 0.5, 2.0),
                           "attempt_hours": 4.0, "time_rqmt": 6.0, "theta": 0.0, "null": False},
}


def _expected_build(theta: float) -> Dict[str, object]:
    """The four ``build.json`` fields the lexicalised runtime asserts (pack_v1 ``EXPECTED_BUILD``)."""
    return {"backoff_loops": _BACKOFF_LOOPS, "escape": bool(_ESCAPE), "sil_prob": _SIL_PROB,
            "theta": float(theta)}


def get_hlg(kind: str, *, shuffled: bool = False) -> Dict[str, object]:
    """One graph as ``dict(hlg=..., stats=..., resources=..., expected_build=...)``.

    :param kind: one of :data:`HLG_KINDS`.
    :param shuffled: the seed-0 derangement null of that graph (``inhouse_3gram`` and
        ``official_4gram`` only).  ``expected_build`` is the treatment's: ``shuffled`` is the only
        ``build.json`` field allowed to differ.
    """
    from ..default_tools import K2_PYTHON_EXE
    from ..model.lexlat_k2 import BACKOFF_LOOPS_WORD_BOUNDARY, bed_min_duration
    from .word_lm import get_lexicon_trie, official_3gram_pruned_1e7_arpa, official_4gram_arpa, official_lexicon

    if kind not in HLG_KINDS:
        raise ValueError(f"unknown HLG kind {kind!r}; expected one of {HLG_KINDS}")
    assert BACKOFF_LOOPS_WORD_BOUNDARY == _BACKOFF_LOOPS
    d_min, stride = bed_min_duration()
    trie = get_lexicon_trie()

    if kind == "inhouse_3gram":
        name = _INHOUSE["null_name"] if shuffled else _INHOUSE["name"]
        job = LexlatHLGBuildJob(
            name=name,
            resources=trie.out_resources,
            env_python=K2_PYTHON_EXE,
            d_min=d_min,
            recognizer_stride=stride,
            prune_ladder=_INHOUSE["prune_ladder"],
            escape=_ESCAPE,
            shuffled=bool(shuffled),
            backoff_loops=_BACKOFF_LOOPS,
            attempt_hours=_INHOUSE["attempt_hours"],
            time_rqmt=_INHOUSE["time_rqmt"],
            mem_rqmt=_BUILD_MEM_GB,
            cpu_rqmt=_BUILD_CPU,
        )
        job.add_alias("sae/4a/lexlat_k2_pack/" + (f"hlg_shuffled_{_BACKOFF_LOOPS}" if shuffled
                                                  else f"hlg_{_BACKOFF_LOOPS}"))
        return {"hlg": job.out_hlg, "stats": job.out_stats, "resources": trie.out_resources,
                "expected_build": _expected_build(_INHOUSE["theta"])}

    spec = _OFFICIAL[kind]
    if shuffled and not spec["null"]:
        raise ValueError(f"{kind} has no null graph (the source funds none)")
    arpa = official_4gram_arpa() if kind == "official_4gram" else official_3gram_pruned_1e7_arpa()
    lexicon = official_lexicon()
    tag = spec["tag"]

    def _build(null: bool, leg: str) -> LexlatOfficialHLGBuildJob:
        job = LexlatOfficialHLGBuildJob(
            name=f"{tag}/{leg}",
            arpa=arpa,
            lexicon=lexicon,
            escape_resources=trie.out_resources,
            env_python=K2_PYTHON_EXE,
            d_min=d_min,
            recognizer_stride=stride,
            prune_ladder=spec["prune_ladder"],
            escape=_ESCAPE,
            backoff_loops=_BACKOFF_LOOPS,
            shuffled=null,
            parse_hours=_PARSE_HOURS,
            attempt_hours=spec["attempt_hours"],
            time_rqmt=spec["time_rqmt"],
            mem_rqmt=_BUILD_MEM_GB,
            cpu_rqmt=_BUILD_CPU,
        )
        job.add_alias(f"sae/4a/lexlat_k2_official/{tag}/{leg}")
        return job

    treatment = _build(False, "hlg")
    res = LexlatOfficialResourcesJob(
        name=f"{tag}/resources",
        build_json=treatment.out_stats,
        arpa=arpa,
        lexicon=lexicon,
        escape_resources=trie.out_resources,
    )
    res.add_alias(f"sae/4a/lexlat_k2_official/{tag}/resources")
    # the null takes the TREATMENT's resources npz, as in the source (pack3 ``graphs``)
    graph = _build(True, "hlg_shuffled") if shuffled else treatment
    return {"hlg": graph.out_hlg, "stats": graph.out_stats, "resources": res.out_resources,
            "expected_build": _expected_build(spec["theta"])}
