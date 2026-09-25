"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_genmarg_jobs.py (the sisyphus side:
``GenMargSampleJob``, ``GenMargSelectionJob``, ``eval_dataset``, ``_serializer``, ``_forward_job``,
``genmarg_reads``) with ``sae/emc/s1a_job.py`` (``select_utterances``).

The label-free generative reads of L2-1 (and L2-0): phi's held-out tau = 1 generative marginal
likelihood under the null recognizer, its posterior decode, and the L2-1 selection over a wave of
restarts.  The RETURNN side (``get_model``, the forward steps, the callbacks and every convention
of the jsons) is ``genmarg_steps``; the Viterbi pass is ``genmarg_decode``.

* :class:`GenMargSampleJob` -- the utterance set: ALL CV-holdout utterances (``segments``,
  ``n=None``) or the seed-0 sample of ``n`` tags with a non-empty gold string (``gold`` + ``split``).
* :func:`genmarg_reads` -- per phi checkpoint, one ``ReturnnForwardJobV2`` for statistic (a)
  (``genmarg.json``) and, with ``decode``, one for the posterior decode (``gendecode.json``,
  ``decode_raw.json``).
* :class:`GenMargSelectionJob` -- the L2-1 selection and the G4a.L2.2 verdict (:data:`SELECTION_RULE`).

Port changes:

* The source's graph-time pins become inputs: the bed's model args, dev dataset, batch and
  ``extern_data`` come from a ``training.config.build_train_config`` config at the bed's defaults
  (:func:`bed_from_train_config`; the source parsed D10e's written ``supphi_plain/returnn.config``
  from disk, whose model args are the bed's with the two init paths removed), the CV-holdout segment
  list is the ``cv_segments`` argument, and the RETURNN interpreter / root default to
  ``default_tools``.  The source's asserted prior path (``PhoneNgramPriorJob.RtzbESkOedsT``) is the
  caller's ``prior_npz`` now; the other bed assertions are kept.
* ``training_checkpoint`` / ``phi_c_checkpoint`` (pins of finished runs on disk) are cut: a phi is
  passed as a ``tk.Path`` or a checkpoint object with ``.path``.
* Statistic (b) (``GenerativeDecodeGapJob``) and the label-using ``GenDecodeReportJob`` are cut:
  ``genmarg_reads`` raises ``ValueError`` for ``gap=True``, ``report=True`` or ``"dev-other"``, and
  its defaults are ``datasets=("cv_holdout",)``, ``gap=False`` (the source's were both datasets and
  ``gap=True``).
* ``genmarg_reads`` also reads :data:`CV_DISJOINT` (the CV holdout minus every ladder fit item; the
  segment list is ``ladder.DisjointHoldoutSegmentsJob``'s, passed as ``disjoint_segments``).

Restored 2026-09-25 (G0.G, ``SAE_i6_P0.md``), LABEL-USING, REPORT ONLY -- :func:`dev_other_reads`:
the posterior decode of the D4 dev-other set (the seed-0 sample of :data:`SELECT_N` dev-other tags
with a non-empty gold string, :class:`GenMargSampleJob`) under the trigram prior and, report only,
under a uniform prior (:class:`UniformPhonePriorJob`), scored by :class:`DevOtherPhoneReadJob`
(direct PER, Hungarian PER, token and frame NMI(symbol, phone), E[d]).  QUARANTINE: it is a separate
builder that ``genmarg_reads`` cannot reach, its decode jsons carry ``dataset = "dev-other"`` which
:class:`GenMargSelectionJob` refuses, and no label-free selection may read any of its outputs.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from sisyphus import Job, Task, tk

from .genmarg_steps import FRAME_RATE_HZ, INIT_PATH_KEYS, RATE_BAND_HZ, _check_model_args

__all__ = [
    "GenMargSampleJob", "GenMargSelectionJob", "SELECTION_RULE", "genmarg_reads",
    "bed_from_train_config", "eval_dataset", "select_utterances", "READ_TEMPERATURE_SCHEDULE",
    "CV_DISJOINT", "LABEL_FREE_DATASETS", "DEV_OTHER", "PRIORS", "UniformPhonePriorJob",
    "DevOtherPhoneReadJob", "dev_other_reads", "align", "hungarian_map", "relabel", "nmi_bits",
    "corpus_per", "segment_frame_labels", "gold_frame_labels", "duration_means",
]

_PKG = __name__.rsplit(".", 2)[0]  # i6_experiments.users.wu.experiments.unsupervised_asr

#: amendment A5 / N6: "all 285 CV-holdout utterances, the 500-utterance D4 dev-other set"; the D4
#: set is the seed-0 order's first 500 (:func:`select_utterances`), the CV holdout is taken whole
SELECT_N = 500
#: the datasets this port reads (the source's second, ``"dev-other"``, is cut: module docstring)
DATASETS = ("cv_holdout",)
#: the realised counts the forward callbacks assert (cv.segments has 285 lines)
EXPECTED_UTTERANCES = {"cv_holdout": 285}
#: the CV holdout minus every ladder fit item (``ladder.DisjointHoldoutSegmentsJob``); its count is
#: derived by that job, so no count is asserted here
CV_DISJOINT = "cv_holdout_disjoint"
#: every label-free dataset ``genmarg_reads`` reads
LABEL_FREE_DATASETS = ("cv_holdout", CV_DISJOINT)
#: the LABEL-USING D4 dev-other read (:func:`dev_other_reads` only; module docstring)
DEV_OTHER = "dev-other"
#: the priors of the D4 decode: the bed's trigram (R1) and a uniform phone prior (R2, report only)
PRIORS = ("trigram", "uniform")
#: amendment A7: wave = 16 restarts + 4 nulls + 2 phi_c-initialised restarts + reruns of seeds 1, 2
N_RESTARTS = 16
N_NULLS = 4
N_PHI_C = 2
N_RERUNS = 2
#: amendment A7: the margin's floor, nats per frame
MARGIN_FLOOR = 0.01

FORWARD_TIME_H = 2
FORWARD_MEM_GB = 32
FORWARD_CPU = 4

#: the bed's model args as the genmarg read builds the model from them (D10e ``supphi_plain``'s
#: written ``get_model`` arguments with the two init paths removed); any other key is a model delta
#: the banked reads never carried and is refused
BED_MODEL_ARG_KEYS = (
    "temperature_schedule", "anchor_weight_schedule", "lam_agg", "count_ema_decay", "band",
    "prior_weight", "prior_npz_path", "eta_table_path", "lam_rate", "rate_rho_hz", "rate_fd_eps",
    "rate_fd_mode", "lattice_reduction", "lattice_checkpoint", "lattice_float64",
)
#: the ``temperature_schedule`` of the bed config the banked reads parsed (D10e ``supphi_plain``: 8
#: sub-epochs at tau 2.0).  Inert for every read here (the forward steps fix tau themselves; only the
#: train steps read the schedule); callers build the bed config for :func:`bed_from_train_config`
#: with it (``build_train_config(num_subepochs=8, temperature_schedule=READ_TEMPERATURE_SCHEDULE)``)
#: so the reads' hashed model args equal the banked ones.
READ_TEMPERATURE_SCHEDULE = (2.0,) * 8


def select_utterances(tags: Sequence[str], n: int, *, seed: int = 0) -> List[str]:
    """The first ``n`` of ``tags`` in the deterministic seed-``seed`` order (``s1a_job``'s rule)."""
    import numpy as np

    ordered = sorted(tags)
    perm = np.random.RandomState(int(seed)).permutation(len(ordered))
    return [ordered[i] for i in perm[:n]]


def _phi_path(phi) -> tk.Path:
    """A ``tk.Path``, or a checkpoint object's ``.path`` (``PtCheckpoint``); never a bare string."""
    if isinstance(phi, tk.Path):
        return phi
    path = getattr(phi, "path", None)
    assert isinstance(path, tk.Path), f"phi must be a tk.Path or carry one in .path, got {phi!r}"
    return path


# =================================================================================================
# the utterance sample
# =================================================================================================
class GenMargSampleJob(Job):
    """The utterance set as a segment list (module docstring): ``n=None`` takes EVERY tag of the
    source (the CV holdout, amendment A5 / N6), else the seed-0 sample of at most ``n`` tags.

    Exactly one source: ``segments`` (a tag-per-line file, the CV holdout's ``cv.segments``) or
    ``gold`` + ``split`` (``GoldPhonesJob`` json; the tags with a NON-EMPTY gold string, the D4
    selection of ``BlankfreeDecodeGapJob``).
    """

    def __init__(self, *, segments: Optional[tk.Path] = None, gold: Optional[tk.Path] = None,
                 split: Optional[str] = None, n: Optional[int] = SELECT_N, seed: int = 0):
        super().__init__()
        assert (segments is None) != (gold is None), "state exactly one of segments / gold"
        assert (gold is None) == (split is None), "gold needs its split"
        self.segments = segments
        self.gold = gold
        self.split = split
        self.n = None if n is None else int(n)
        self.seed = int(seed)
        self.out_segments = self.output_path("sample.segments")
        self.out_json = self.output_path("sample.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def select(tags: Sequence[str], n: Optional[int], seed: int) -> List[str]:
        if n is None:
            return sorted(tags)
        return select_utterances(list(tags), int(n), seed=int(seed))

    def run(self):
        if self.segments is not None:
            with open(self.segments.get_path()) as fh:
                tags = [ln.strip() for ln in fh if ln.strip()]
            source = os.path.realpath(self.segments.get_path())
        else:
            gold = json.load(open(self.gold.get_path()))[self.split]
            tags = [t for t, ph in gold.items() if ph]
            source = os.path.realpath(self.gold.get_path())
        assert len(set(tags)) == len(tags), "duplicate tags in the source"
        chosen = self.select(tags, self.n, self.seed)
        with open(self.out_segments.get_path(), "w") as fh:
            fh.write("\n".join(chosen) + "\n")
        record = {"source": source, "split": self.split, "eligible": len(tags),
                  "requested": self.n, "realised": len(chosen), "seed": self.seed,
                  "rule": ("every tag, sorted" if self.n is None else
                           "s1a_job.select_utterances: sorted tags, RandomState(seed).permutation, "
                           "first n")}
        with open(self.out_json.get_path(), "w") as fh:
            json.dump(record, fh, indent=1)
        print(json.dumps(record), flush=True)



# 6. the L2-1 selection reader and the G4a.L2.2 verdict
# =================================================================================================
# the L2-1 selection reader and the G4a.L2.2 verdict
# =================================================================================================
SELECTION_RULE = (
    "REGISTERED RULE (SAE_4A_lexlat_v2.md, pre-registered 2026-09-23, verbatim; kept as registered "
    "text, superseded by the amendments below):\n"
    "  L2-1 Selection: \"Best per-frame held-out tau = 1 marginal likelihood among the 16. Stage 2: "
    "the top 4 continue 4 more passes with the word graph on; re-selected by the held-out total "
    "including the k2 term (D13's sum).\"\n"
    "  G4a.L2.2 signal (L2-1 wave): clause \"selected stage-1 restart's held-out tau = 1 marginal "
    "likelihood minus the best null's\"; PASS \"> null spread (max minus min over the 4 nulls) = "
    "SIGNAL\"; FAIL \"else NO SIGNAL: EM under this prior finds nothing beyond the unit statistics "
    "the shuffled corpus keeps; L2-2 not funded; the phi-first route closes at this scale\".\n"
    "AMENDMENTS (SAE_4A_lexlat_v2.md, design review amendments 2026-09-23, before any job; "
    "verbatim):\n"
    "  A1: \"Stage 2 of L2-1 removed. [...] G4a.L2.3 reads the stage-1 selected phi.\"\n"
    "  A2: \"Every restart logs held-out expected non-SIL rate. A restart whose final held-out rate "
    "lies outside the bed's [5.80, 14.49] Hz is VOID for selection.\"\n"
    "  A7: \"G4a.L2.2 is a spend gate. Any segmental fit beats the within-utterance shuffle, so NO "
    "SIGNAL means L2-2 is not funded, not that the route fails. Margin = max(null range, identity "
    "band, 0.01 nats per frame); identity band = the larger |S| difference of seeds 1 and 2 against "
    "their exact reruns. Nulls are gated on their permuted holdout (E4 convention), real holdout "
    "reported. Wave = 16 restarts + 4 nulls + 2 phi_c-initialised restarts + reruns of seeds 1 and "
    "2 (6 nodes). Report only: selected restart minus best phi_c restart beyond the margin = BEYOND "
    "PRIVATE CODE, else NOT BEYOND. L2-2 is funded on SIGNAL AND at least one rt_r0 seed reading "
    "LIFT.\"\n"
    "  A8 (2026-09-23, after implementation, before any L2-1 job): \"Supersedes A2's 'expected "
    "rate' and A7's pooling sentence, which disagreed with each other and with the band's own "
    "definition: [5.80, 14.49] Hz is a greedy emitted rate per original second, 'never a posterior "
    "expectation' (SAE_4A_attrib.md S3b-R; c5 met an expected count with a diffuse posterior). Under "
    "the null recognizer the emitted sequence is phi's decode of the tau = 1 generative posterior "
    "(the genmarg decode). Rate = 50 x (sum of non-SIL tokens in that decode) / (sum of original 50 "
    "Hz frames), pooled over the CV holdout; A2's VOID, A3's probe PASS and NO ELIGIBLE RESTART read "
    "it, so the probe's scoring reads decode. The expected rate (both denominators) and the training "
    "monitor blankfree_nonsil_rate_retained_hz (retained frames, the sub-epoch's tau) are reported "
    "only [...]\"\n"
    "IMPLEMENTATION (the reading of the rule this job applies):\n"
    "  * S(json) = mean over the PAIRING SET of nll_tau1 / S_frames (per-frame held-out tau = 1 "
    "negative log marginal likelihood; LOWER S = HIGHER likelihood).  Pairing set = the CV-holdout "
    "utterances finite in EVERY gated json (16 restarts, 4 nulls on their permuted holdout, 2 phi_c "
    "restarts, 2 reruns); its size is printed.  The nulls' real-holdout jsons enter no clause and are "
    "reported over the pairing-set utterances finite in them.\n"
    "  * rate(restart) = the A8 EMITTED rate of the restart's FINAL checkpoint: its CV-holdout "
    "gendecode.json emitted_nonsil_rate.pooled_hz = 50 * sum(non-SIL tokens of the tau = 1 "
    "generative posterior decode) / sum(orig_length) over every decoded CV-holdout utterance.  A "
    "restart is VOID iff rate < 5.80 or rate > 14.49 Hz (bounds inclusive in the band).  The "
    "expected rate (50 * sum E[N_nonSIL] / sum orig_length over the pairing set) is printed for "
    "every json, REPORT ONLY.\n"
    "  * selected = the non-VOID restart with the lowest S (ties: first name in sorted order).  If "
    "every restart is VOID there is no selection and the verdict is NO ELIGIBLE RESTART.\n"
    "  * null range = max - min of the 4 null S (permuted holdout); identity band = max over the 2 "
    "reruns of |S(restart) - S(its exact rerun)|; margin = max(null range, identity band, 0.01).\n"
    "  * gap = S(best null) - S(selected) = the selected restart's held-out log-likelihood per frame "
    "minus the best (lowest-S) null's.  SIGNAL iff gap > margin; else NO SIGNAL.\n"
    "  * G4a.L2.2 IS A SPEND GATE: SIGNAL = L2-2 may be funded (A7 also requires at least one rt_r0 "
    "seed reading LIFT, which is read by L2-0 and is NOT an input of this job); NO SIGNAL = L2-2 is "
    "not funded.  NO SIGNAL does NOT close the phi-first route and says nothing about whether EM "
    "could work (route closure sits with G4a.L2.4).\n"
    "  * REFERENCE, REPORT ONLY (enters no gate): ref = S(best phi_c restart) - S(selected) = the "
    "selected random restart's held-out log-likelihood per frame minus the best of the 2 "
    "phi_c-initialised restarts'.  BEYOND PRIVATE CODE iff ref > margin, else NOT BEYOND.")


class GenMargSelectionJob(Job):
    """The L2-1 selection and the G4a.L2.2 verdict, as amended (:data:`SELECTION_RULE`, printed).

    REGISTERED RULE (verbatim, `SAE_4A_lexlat_v2.md`, pre-registered 2026-09-23; superseded as
    below): L2-1 Selection: "Best per-frame held-out tau = 1 marginal likelihood among the 16. Stage
    2: the top 4 continue 4 more passes with the word graph on; re-selected by the held-out total
    including the k2 term (D13's sum)."  G4a.L2.2: "selected stage-1 restart's held-out tau = 1
    marginal likelihood minus the best null's" | "> null spread (max minus min over the 4 nulls) =
    SIGNAL" | "else NO SIGNAL: [...] L2-2 not funded [...]".

    AMENDED (A1, A2, A7, A8, verbatim in :data:`SELECTION_RULE`): stage 2 is removed (no top-4
    list); a restart whose held-out EMITTED rate (A8: "Rate = 50 x (sum of non-SIL tokens in that
    decode) / (sum of original 50 Hz frames), pooled over the CV holdout"; the expected rate is
    report only) is outside [5.80, 14.49] Hz is VOID; margin = max(null range, identity band, 0.01
    nats per frame); nulls gated on their permuted holdout, real holdout reported; a report-only
    phi_c reference clause.

    G4a.L2.2 IS A SPEND GATE.  NO SIGNAL means "L2-2 not funded" -- a spending decision on this
    evidence -- and NOT "the phi-first route is closed"; it licenses no claim that EM would not have
    worked.  SIGNAL is necessary, not sufficient, for L2-2 (A7: also at least one rt_r0 seed reading
    LIFT, read elsewhere).

    :param restarts: ``{name: genmarg.json}`` of the 16 random restarts (CV holdout, real stream).
    :param nulls: ``{name: genmarg.json}`` of the 4 nulls on their PERMUTED holdout (the gated score).
    :param nulls_real: ``{name: genmarg.json}`` of the same 4 nulls on the REAL holdout (reported).
    :param phi_c: ``{name: genmarg.json}`` of the 2 phi_c-initialised restarts (real stream).
    :param reruns: ``{restart name: genmarg.json}`` of the exact reruns of seeds 1 and 2; each key must
        be the restart it reruns.
    :param restart_decodes: ``{name: gendecode.json}`` of the 16 restarts (CV holdout, real stream):
        the A8 emitted rate VOID reads.
    """

    def __init__(self, *, restarts: Dict[str, tk.Path], nulls: Dict[str, tk.Path],
                 nulls_real: Dict[str, tk.Path], phi_c: Dict[str, tk.Path],
                 reruns: Dict[str, tk.Path], restart_decodes: Dict[str, tk.Path], name: str = "",
                 n_restarts: int = N_RESTARTS, n_nulls: int = N_NULLS, n_phi_c: int = N_PHI_C,
                 n_reruns: int = N_RERUNS):
        super().__init__()
        self.restart_decodes = dict(restart_decodes)
        self.restarts = dict(restarts)
        self.nulls = dict(nulls)
        self.nulls_real = dict(nulls_real)
        self.phi_c = dict(phi_c)
        self.reruns = dict(reruns)
        self.name = str(name)
        self.n_restarts = int(n_restarts)
        self.n_nulls = int(n_nulls)
        self.n_phi_c = int(n_phi_c)
        self.n_reruns = int(n_reruns)
        self.out_json = self.output_path("selection.json")
        self.out_report = self.output_path("report.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def decide(*, restarts: Dict[str, dict], nulls: Dict[str, dict], nulls_real: Dict[str, dict],
               phi_c: Dict[str, dict], reruns: Dict[str, dict],
               restart_decodes: Dict[str, dict]) -> Dict[str, Any]:
        """The read on loaded ``genmarg.json`` / ``gendecode.json`` records (:data:`SELECTION_RULE`)."""
        assert set(nulls_real) == set(nulls), "every null needs its real-holdout json"
        assert set(restart_decodes) == set(restarts), "every restart needs its decode (A8 rate)"
        for k, dec in restart_decodes.items():
            assert dec["dataset"] == "cv_holdout" and dec["settings"]["shuffle_seed"] is None, k
            assert set(dec["per_utterance"]) == set(restarts[k]["per_utterance"]), (
                f"{k}: the decode and the marginal score different utterances")
            assert dec.get("emitted_nonsil_rate") is not None, f"{k}: no emitted rate in the decode"
        assert set(reruns) <= set(restarts), "a rerun key must name the restart it reruns"
        groups = {"restart": restarts, "null": nulls, "phi_c": phi_c, "rerun": reruns,
                  "null_real": nulls_real}
        gated = {(g, k): v for g in ("restart", "null", "phi_c", "rerun")
                 for k, v in groups[g].items()}
        every = {(g, k): v for g, recs in groups.items() for k, v in recs.items()}
        tag_sets = {key: set(rec["per_utterance"]) for key, rec in every.items()}
        first = next(iter(tag_sets.values()))
        assert all(t == first for t in tag_sets.values()), "the jsons score different utterances"
        for (g, k), rec in every.items():
            assert rec["dataset"] == "cv_holdout", ((g, k), rec["dataset"])
            shuffled = rec["settings"]["shuffle_seed"] is not None
            assert shuffled == (g == "null"), (
                f"{(g, k)}: nulls are gated on the permuted holdout, everything else on the real one")
            assert rec["summary"].get("expected_nonsil_rate") is not None, (
                f"{(g, k)}: no expected rate in the json")
        common = sorted(t for t in first
                        if all(not rec["per_utterance"][t]["impossible"] for rec in gated.values()))
        assert common, "no utterance is finite in every gated json"

        def value(rec, tags):
            return sum(rec["per_utterance"][t]["nll_tau1_per_frame"] for t in tags) / len(tags)

        def rate(rec, tags):
            e = sum(rec["per_utterance"][t]["expected_nonsil_tokens"] for t in tags)
            o = sum(rec["per_utterance"][t]["original_frames"] for t in tags)
            return FRAME_RATE_HZ * e / o

        lo, hi = RATE_BAND_HZ
        s_val = {g: {k: value(v, common) for k, v in sorted(groups[g].items())}
                 for g in ("restart", "null", "phi_c", "rerun")}
        expected_rates = {g: {k: rate(v, common) for k, v in sorted(groups[g].items())}
                          for g in ("restart", "null", "phi_c", "rerun")}
        emitted = {k: float(restart_decodes[k]["emitted_nonsil_rate"]["pooled_hz"])
                   for k in sorted(restarts)}
        real_tags = {k: [t for t in common if not v["per_utterance"][t]["impossible"]]
                     for k, v in nulls_real.items()}
        null_real = {k: {"s": value(v, real_tags[k]) if real_tags[k] else None,
                         "expected_rate_hz_report_only":
                             rate(v, real_tags[k]) if real_tags[k] else None,
                         "utterances": len(real_tags[k])} for k, v in sorted(nulls_real.items())}
        void = {k: not (lo <= r <= hi) for k, r in emitted.items()}
        eligible = sorted((k for k in s_val["restart"] if not void[k]),
                          key=lambda k: (s_val["restart"][k], k))
        null_vals = s_val["null"]
        best_null = min(null_vals, key=lambda k: (null_vals[k], k))
        null_range = max(null_vals.values()) - min(null_vals.values())
        identity = {k: abs(s_val["restart"][k] - s_val["rerun"][k]) for k in sorted(reruns)}
        identity_band = max(identity.values())
        margin = max(null_range, identity_band, MARGIN_FLOOR)
        best_phi_c = min(s_val["phi_c"], key=lambda k: (s_val["phi_c"][k], k))
        out = {
            "pairing_utterances": len(common), "utterances": len(first),
            "impossible_per_json": {f"{g}/{k}": rec["summary"]["n_impossible"]
                                    for (g, k), rec in every.items()},
            "s_per_frame": s_val, "emitted_rate_hz": emitted,
            "emitted_rate_utterances": {k: restart_decodes[k]["emitted_nonsil_rate"]["n"]
                                        for k in sorted(restarts)},
            "expected_rate_original_hz_report_only": expected_rates,
            "rate_band_hz": list(RATE_BAND_HZ),
            "void": void, "eligible": len(eligible),
            "null_shuffle_seeds": {k: v["settings"]["shuffle_seed"] for k, v in nulls.items()},
            "null_real_holdout": null_real,
            "best_null": best_null, "best_null_value": null_vals[best_null],
            "null_range": null_range, "identity": identity, "identity_band": identity_band,
            "margin_floor": MARGIN_FLOOR, "margin": margin,
            "best_phi_c": best_phi_c, "best_phi_c_value": s_val["phi_c"][best_phi_c],
        }
        if not eligible:
            out.update({"selected": None, "selected_value": None, "gap": None,
                        "verdict": "NO ELIGIBLE RESTART", "reference": None,
                        "reference_reading": None})
            return out
        selected = eligible[0]
        gap = null_vals[best_null] - s_val["restart"][selected]
        ref = s_val["phi_c"][best_phi_c] - s_val["restart"][selected]
        out.update({
            "selected": selected, "selected_value": s_val["restart"][selected],
            "selected_emitted_rate_hz": emitted[selected],
            "gap": gap, "verdict": "SIGNAL" if gap > margin else "NO SIGNAL",
            "reference": ref,
            "reference_reading": "BEYOND PRIVATE CODE" if ref > margin else "NOT BEYOND",
        })
        return out

    def run(self):
        def load(d):
            return {k: json.load(open(p.get_path())) for k, p in d.items()}

        inputs = {"restarts": self.restarts, "nulls": self.nulls, "nulls_real": self.nulls_real,
                  "phi_c": self.phi_c, "reruns": self.reruns, "restart_decodes": self.restart_decodes}
        for key, n in (("restarts", self.n_restarts), ("restart_decodes", self.n_restarts),
                       ("nulls", self.n_nulls),
                       ("nulls_real", self.n_nulls), ("phi_c", self.n_phi_c),
                       ("reruns", self.n_reruns)):
            assert len(inputs[key]) == n, (key, len(inputs[key]), n)
        out = self.decide(**{k: load(v) for k, v in inputs.items()})
        out.update({"name": self.name, "rule": SELECTION_RULE,
                    "inputs": {g: {k: os.path.realpath(p.get_path()) for k, p in d.items()}
                               for g, d in inputs.items()}})
        with open(self.out_json.get_path(), "w") as fh:
            json.dump(out, fh, indent=1)
        lines = [SELECTION_RULE, "",
                 f"pairing set: {out['pairing_utterances']} of {out['utterances']} utterances "
                 "(finite in every gated json)",
                 "held-out tau = 1 NLL per frame S (lower = higher likelihood); restarts: EMITTED "
                 f"non-SIL rate (A8, gated; band {RATE_BAND_HZ[0]}-{RATE_BAND_HZ[1]} Hz); every json: "
                 "expected rate (report only)"]
        for g in ("restart", "phi_c", "rerun", "null"):
            vals = out["s_per_frame"][g]
            for k in sorted(vals, key=lambda k: (vals[k], k)):
                extra = ""
                if g == "restart" and out["void"][k]:
                    extra = "  VOID (emitted rate outside the band)"
                if g == "null":
                    nr = out["null_real_holdout"][k]
                    extra = (f"  (shuffle seed {out['null_shuffle_seeds'][k]}; real holdout S "
                             f"{nr['s']} on {nr['utterances']}, reported only)")
                if g == "rerun":
                    extra = f"  (rerun of {k}; |dS| {out['identity'][k]:.6f})"
                emit = (f"emitted {out['emitted_rate_hz'][k]:.3f} Hz  " if g == "restart" else "")
                lines.append(f"  {g:8s} {k:24s} S {vals[k]:.6f}  {emit}expected (report only) "
                             f"{out['expected_rate_original_hz_report_only'][g][k]:.3f} Hz{extra}")
        lines += [
            f"eligible restarts: {out['eligible']} of {len(out['void'])}",
            f"selected restart: {out['selected']} (S {out['selected_value']})",
            f"best null (permuted holdout): {out['best_null']} (S {out['best_null_value']:.6f})",
            f"margin = max(null range {out['null_range']:.6f}, identity band "
            f"{out['identity_band']:.6f}, {MARGIN_FLOOR}) = {out['margin']:.6f} nats/frame",
            f"gap (best null S - selected S): {out['gap']}",
            f"G4a.L2.2 (spend gate): {out['verdict']}"
            + {"SIGNAL": " -- L2-2 may be funded if at least one rt_r0 seed reads LIFT (A7)",
               "NO SIGNAL": " -- L2-2 not funded (a spending decision, not a route closure)",
               "NO ELIGIBLE RESTART": " -- every restart VOID on the emitted-rate band"}[out["verdict"]],
            f"reference, report only: best phi_c {out['best_phi_c']} (S "
            f"{out['best_phi_c_value']:.6f}); best phi_c S - selected S = {out['reference']} -> "
            f"{out['reference_reading']}",
        ]
        with open(self.out_report.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


# =================================================================================================
# the graph builder
# =================================================================================================
def _extern_data(returnn_config) -> Dict[str, Any]:
    """The ``extern_data`` dict ``build_train_config`` writes as non-hashed code."""
    import ast

    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection

    found = []
    for item in returnn_config.python_prolog:
        if not isinstance(item, Collection):
            continue
        for obj in item.serializer_objects:
            code = getattr(obj, "code", None)
            if isinstance(code, str) and code.startswith("extern_data = "):
                found.append(ast.literal_eval(code[len("extern_data = "):].strip()))
    assert len(found) == 1, f"expected one extern_data block, got {len(found)}"
    return found[0]


def bed_from_train_config(returnn_config) -> Dict[str, Any]:
    """``model_args`` (init paths removed), ``dev``, ``batch_size``, ``max_seqs``, ``extern_data`` of
    the bed, from ``training.config.build_train_config``'s config at its default model arguments
    (the source's ``parse_bed_config`` of D10e's ``supphi_plain`` config; module docstring).

    The bed's dev dataset is the train stream's CV holdout (``dev_segments`` of that call); every
    model delta beyond the bed's own arguments is refused (a phi-first, k2 or duration key would
    build a model the banked reads never built).
    """
    from ..training.jobs import get_model_args

    c = returnn_config.config
    args = copy.deepcopy(dict(get_model_args(returnn_config)))
    for key in INIT_PATH_KEYS:
        args.pop(key, None)
    _check_model_args(args)
    extra = sorted(set(args) - set(BED_MODEL_ARG_KEYS))
    missing = sorted(set(BED_MODEL_ARG_KEYS) - set(args))
    assert not extra and not missing, (
        f"not the bed's model args: extra {extra}, missing {missing} (the genmarg read builds the "
        "bed's model and loads phi into it)")
    assert args["prior_weight"] == 1.0 and args["band"] == 25, args
    assert args["lattice_reduction"] == "matmul" and args["lattice_float64"] is True, args
    dev = copy.deepcopy(c["dev"])
    assert dev.get("class") == "MetaDataset" and dev["seq_order_control_dataset"] == "feats", dev
    return {"model_args": args, "dev": dev, "batch_size": copy.deepcopy(c["batch_size"]),
            "max_seqs": c["max_seqs"], "extern_data": _extern_data(returnn_config)}


def eval_dataset(bed_dev: Dict[str, Any], dataset: str, segments: tk.Path,
                 stream: Optional[Dict[str, Sequence[tk.Path]]] = None) -> Dict[str, Any]:
    """The bed's dev MetaDataset on ``dataset``'s VAD streams, filtered to ``segments``.

    The label-free datasets (:data:`LABEL_FREE_DATASETS`) are the bed's own dev stream (the train
    stream's HDFs).  ``"dev-other"`` (:func:`dev_other_reads` only) swaps the three sub-datasets'
    files for ``stream`` (``inputs.Inputs.dev_stream("dev-other")``: ``features`` / ``units`` /
    ``originals``); every other option of the bed's dev dataset is kept.
    """
    if dataset not in LABEL_FREE_DATASETS + (DEV_OTHER,):
        raise ValueError(f"dataset {dataset!r}: not one of {LABEL_FREE_DATASETS + (DEV_OTHER,)}")
    assert (stream is not None) == (dataset == DEV_OTHER), "a stream is stated for dev-other only"
    data = copy.deepcopy(bed_dev)
    for name in ("feats", "original", "units"):
        data["datasets"][name].pop("seq_list_filter_file", None)
    if stream is not None:
        for name, key in (("feats", "features"), ("units", "units"), ("original", "originals")):
            assert "files" in data["datasets"][name], (name, sorted(data["datasets"][name]))
            files = list(stream[key])
            assert files, f"empty {key} stream"
            data["datasets"][name]["files"] = files
    data["datasets"]["feats"]["seq_list_filter_file"] = segments
    return data


def _serializer(*, mode: str, phi, model_args, extern_data, shuffle_seed, name, dataset,
                expected_utterances):
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
    from i6_experiments.common.setups.serialization import NonhashedCode, PartialImport
    from returnn.util.pprint import pformat

    mod = f"{_PKG}.reverse_model.genmarg_steps"
    step, callback = {"marginal": ("genmarg_forward_step", "GenMargCallback"),
                      "decode": ("gendecode_forward_step", "GenDecodeCallback")}[mode]
    root = f"{_PKG}.reverse_model"
    return Collection(
        serializer_objects=[
            NonhashedCode(f"extern_data = {pformat(extern_data)}\n"),
            PartialImport(code_object_path=f"{mod}.genmarg_get_model", unhashed_package_root=root,
                          hashed_arguments={"phi_checkpoint": phi,
                                            "model_args": copy.deepcopy(model_args)},
                          unhashed_arguments={}, import_as="get_model"),
            PartialImport(code_object_path=f"{mod}.{step}", unhashed_package_root=root,
                          hashed_arguments={"shuffle_seed": shuffle_seed},
                          unhashed_arguments={}, import_as="forward_step"),
            PartialImport(code_object_path=f"{mod}.{callback}", unhashed_package_root=root,
                          hashed_arguments={"name": name, "dataset": dataset,
                                            "expected_utterances": expected_utterances},
                          unhashed_arguments={}, import_as="forward_callback"),
        ],
        make_local_package_copy=False,
        packages=None,
    )


def _forward_job(*, mode, phi, bed, data, shuffle_seed, name, dataset, expected_utterances=None,
                 returnn_python_exe=None, returnn_root=None):
    from i6_core.returnn.config import ReturnnConfig
    from i6_core.returnn.forward import ReturnnForwardJobV2

    from ..default_tools import RETURNN_EXE, RETURNN_ROOT

    config = {"forward_data": data, "batch_size": copy.deepcopy(bed["batch_size"]),
              "max_seqs": bed["max_seqs"], "sort_dataset": False}
    rcfg = ReturnnConfig(
        config=config, post_config={"backend": "torch", "watch_memory": True},
        python_epilog=[_serializer(mode=mode, phi=phi, model_args=bed["model_args"],
                                   extern_data=bed["extern_data"], shuffle_seed=shuffle_seed,
                                   name=name, dataset=dataset,
                                   expected_utterances=expected_utterances)])
    outputs = (["genmarg.json"] if mode == "marginal" else ["gendecode.json", "decode_raw.json"])
    return ReturnnForwardJobV2(
        model_checkpoint=None, returnn_config=rcfg,
        returnn_python_exe=RETURNN_EXE if returnn_python_exe is None else returnn_python_exe,
        returnn_root=RETURNN_ROOT if returnn_root is None else returnn_root,
        output_files=outputs, device="gpu",
        time_rqmt=FORWARD_TIME_H, mem_rqmt=FORWARD_MEM_GB, cpu_rqmt=FORWARD_CPU)


def genmarg_reads(phi, name: str, datasets: Sequence[str] = DATASETS,
                  shuffle: Optional[int] = None, *, bed: Dict[str, Any], cv_segments: tk.Path,
                  decode: bool = True, gap: bool = False, report: bool = False,
                  alias: Optional[str] = None, returnn_python_exe: Optional[tk.Path] = None,
                  returnn_root: Optional[tk.Path] = None,
                  disjoint_segments: Optional[tk.Path] = None) -> Dict[str, Dict[str, Any]]:
    """Statistic (a) [+ the posterior decode] of ``phi`` on the CV holdout (label-free).

    :param phi: a reverse-only or whole-model checkpoint: a ``tk.Path`` (a reverse-init job's
        output, an ``ExtractSubmoduleCheckpointJob`` slice) or a checkpoint object with ``.path``
        (``ReturnnTrainingJob.out_checkpoints[epoch]``, e.g. a probe's per-sub-epoch checkpoints).
    :param name: free text stamped into every output.
    :param datasets: ``("cv_holdout",)``: all utterances of the bed's CV holdout of the train stream;
        :data:`CV_DISJOINT` reads every utterance of ``disjoint_segments`` on the same stream (no
        count asserted: the segment job derives it).  ``"dev-other"`` raises (:func:`dev_other_reads`).
    :param shuffle: the ``permute_frames`` seed of a null restart's corpus (None = real stream).
        Its value is the caller's (the null corpus's); this module never chooses one.
    :param bed: :func:`bed_from_train_config` of the bed's config.
    :param cv_segments: the CV-holdout segment list (the bed's ``dev_segments``).
    :param decode: also build the posterior decode (its gendecode.json carries the A8 emitted rate,
        which every L2-1 rate clause reads).
    :param gap: / :param report: statistic (b) and the label-using report, cut in the port: True
        raises ``ValueError``.
    :param alias: alias prefix for the jobs (None = no alias).
    :param disjoint_segments: the :data:`CV_DISJOINT` segment list (required for that dataset only).
    :return: ``{dataset: {"sample", "marginal", "decode"}}`` (jobs, absent when not built).
    """
    if gap:
        raise ValueError("gap=True: statistic (b) (GenerativeDecodeGapJob) is not ported")
    if report:
        raise ValueError("report=True: GenDecodeReportJob (label-using) is not ported")
    unknown = sorted(set(datasets) - set(LABEL_FREE_DATASETS))
    if unknown:
        raise ValueError(f"datasets {unknown}: only {LABEL_FREE_DATASETS} are read here (the "
                         "label-using dev-other read is dev_other_reads)")
    if CV_DISJOINT in datasets and disjoint_segments is None:
        raise ValueError(f"{CV_DISJOINT!r} needs disjoint_segments (ladder.DisjointHoldoutSegmentsJob)")
    phi = _phi_path(phi)
    out: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        segments = cv_segments if dataset == "cv_holdout" else disjoint_segments
        sample = GenMargSampleJob(segments=segments, n=None)
        data = eval_dataset(bed["dev"], dataset, sample.out_segments)
        jobs: Dict[str, Any] = {"sample": sample}
        n_exp = EXPECTED_UTTERANCES.get(dataset)
        kw = dict(phi=phi, bed=bed, data=data, shuffle_seed=shuffle, name=name, dataset=dataset,
                  expected_utterances=n_exp, returnn_python_exe=returnn_python_exe,
                  returnn_root=returnn_root)
        jobs["marginal"] = _forward_job(mode="marginal", **kw)
        if decode:
            jobs["decode"] = _forward_job(mode="decode", **kw)
        if alias:
            for key, job in jobs.items():
                job.add_alias(f"{alias}/{dataset}/{key}")
        out[dataset] = jobs
    return out


# =================================================================================================
# LABEL-USING, REPORT ONLY: the D4 dev-other phone read (G0.G; module docstring, QUARANTINE)
# =================================================================================================
#: the Hungarian map's deletion target: a symbol mapped here is dropped from the hypothesis
DELETE = "<DELETE>"


def align(hyp: Sequence[str], ref: Sequence[str]) -> List[tuple]:
    """Levenshtein alignment of ``hyp`` against ``ref`` as ``(hyp token | None, ref token | None)``
    pairs in order, with ``analysis.per.edit_counts``' unit costs, tie rule (match / substitution
    before deletion before insertion) and backtrace, so its S + D + I is that function's."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    ops = [[0] * (m + 1) for _ in range(n + 1)]  # 0 = match, 1 = sub, 2 = del, 3 = ins
    for j in range(1, m + 1):
        ops[0][j] = 3
    for i in range(1, n + 1):
        cur = [prev[0] + 1] + [0] * m
        ops[i][0] = 2
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            c_sub = prev[j - 1] + (0 if same else 1)
            c_del = prev[j] + 1
            c_ins = cur[j - 1] + 1
            best = min(c_sub, c_del, c_ins)
            cur[j] = best
            ops[i][j] = 0 if (same and best == c_sub) else (1 if best == c_sub else (2 if best == c_del else 3))
        prev = cur
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        op = ops[i][j]
        if i > 0 and j > 0 and op in (0, 1):
            pairs.append((hyp[j - 1], ref[i - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and op == 2:
            pairs.append((None, ref[i - 1]))
            i -= 1
        else:
            pairs.append((hyp[j - 1], None))
            j -= 1
    return pairs[::-1]


def corpus_per(hyps: Dict[str, Sequence[str]], refs: Dict[str, Sequence[str]]) -> Dict[str, Any]:
    """(S + D + I) / N summed over ``refs``' tags, each with a fresh :func:`align`."""
    s = d = i = n = 0
    for tag in sorted(refs):
        for h, r in align(list(hyps[tag]), list(refs[tag])):
            s += h is not None and r is not None and h != r
            d += h is None
            i += r is None
        n += len(refs[tag])
    return {"per": (s + d + i) / n if n else None, "sub": s, "del": d, "ins": i, "ref_tokens": n}


def hungarian_map(pairs: Sequence[tuple], symbols: Sequence[str], phones: Sequence[str]) -> Dict[str, str]:
    """The one-to-one symbol -> phone | :data:`DELETE` map maximising the aligned matches.

    Gain of (symbol s, phone p) = the number of aligned pairs ``(s, p)`` (both sides present) in
    ``pairs`` (the identity-label alignment of the RAW decode, SIL kept, against the SIL-free gold);
    one extra zero-gain :data:`DELETE` column, so ``len(symbols)`` must be ``len(phones) + 1``
    (40 = 39 + 1): the square ``scipy.optimize.linear_sum_assignment(maximize=True)`` of the
    campaign's ``private_code.hungarian_labelling`` (``SAE/reports/impl_private_code_2026-09-20.md``).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    cols = list(phones) + [DELETE]
    assert len(symbols) == len(cols), (len(symbols), len(cols))
    si = {s: k for k, s in enumerate(symbols)}
    pi = {p: k for k, p in enumerate(phones)}
    gain = np.zeros((len(symbols), len(cols)), dtype=np.int64)
    for h, r in pairs:
        if h is not None and r is not None:
            gain[si[h], pi[r]] += 1
    rows, cs = linear_sum_assignment(gain, maximize=True)
    return {symbols[r]: cols[c] for r, c in zip(rows, cs)}


def relabel(tokens: Sequence[str], mapping: Dict[str, str]) -> List[str]:
    """``tokens`` through ``mapping``, :data:`DELETE` targets dropped (no re-collapse)."""
    return [mapping[t] for t in tokens if mapping[t] != DELETE]


def nmi_bits(pairs: Sequence[tuple]) -> Dict[str, Any]:
    """NMI = I(X; Y) / sqrt(H(X) H(Y)) in bits over the empirical joint of ``pairs`` ``(x, y)``
    (``None`` when either entropy is 0), with I, H(X), H(Y) and the pair count."""
    import math
    from collections import Counter

    n = len(pairs)
    if n == 0:
        return {"nmi": None, "mi_bits": None, "h_x_bits": None, "h_y_bits": None, "n": 0}
    joint = Counter(pairs)
    px = Counter(x for x, _ in pairs)
    py = Counter(y for _, y in pairs)
    h = lambda c: -sum(v / n * math.log2(v / n) for v in c.values())  # noqa: E731
    hx, hy = h(px), h(py)
    mi = sum(v / n * math.log2(v * n / (px[x] * py[y])) for (x, y), v in joint.items())
    mi = max(mi, 0.0)
    nmi = mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else None
    return {"nmi": nmi, "mi_bits": mi, "h_x_bits": hx, "h_y_bits": hy, "n": n}


def segment_frame_labels(segments: Sequence[Sequence[int]], n_frames: int) -> List[str]:
    """Per retained unit frame, the decoded symbol (``prior.PHONES`` name) of the segment covering it."""
    from ..phones import PHONES

    out: List[str] = []
    for k, start, dur in segments:
        assert int(start) == len(out), ("segments do not tile", start, len(out))
        out.extend([PHONES[int(k)]] * int(dur))
    assert len(out) == n_frames, (len(out), n_frames)
    return out


def gold_frame_labels(raw_index: Sequence[int], intervals: Sequence[tuple],
                      frame_rate_hz: float = FRAME_RATE_HZ) -> List[str]:
    """Per retained unit frame, the MFA gold phone at its centre ``(raw_index + 0.5) / frame_rate_hz``
    seconds (``raw_index`` = the frame's position in the unmasked 50 Hz stream, the VAD job's
    ``raw_index`` HDF); ``intervals`` = ``(canonical phone, start s, end s)``; a centre in no
    interval ``[start, end)`` is SIL."""
    import numpy as np

    from ..phones import SIL

    iv = sorted((float(s), float(e), p) for p, s, e in intervals)
    starts = np.array([s for s, _, _ in iv], dtype=np.float64)
    out = []
    for f in raw_index:
        c = (int(f) + 0.5) / frame_rate_hz
        k = int(np.searchsorted(starts, c, side="right")) - 1
        out.append(iv[k][2] if k >= 0 and c < iv[k][1] else SIL)
    return out


def duration_means(dur_logits) -> Dict[str, Any]:
    """E[d] of phi's duration table (``duration_prior.duration_law`` at ``ReverseConfig()``, the bed's
    reverse config): the mean over the 39 phone types, SIL's, and every type's."""
    from ..model.reverse import ReverseConfig
    from ..phones import ARPABET_39, PHONES, SIL
    from .duration_prior import duration_law, law_summary

    law = law_summary(duration_law(dur_logits, ReverseConfig()), PHONES)
    per_type = {p: law[p]["mean"] for p in PHONES}
    return {"phone_types_mean": sum(per_type[p] for p in ARPABET_39) / len(ARPABET_39),
            "sil": per_type[SIL], "per_type": per_type}


def _read_hdf_int_sequences(paths: Sequence[str]) -> Dict[str, Any]:
    import h5py
    import numpy as np

    rows: Dict[str, Any] = {}
    for path in paths:
        with h5py.File(path, "r") as fh:
            tags = [t.decode() if isinstance(t, bytes) else str(t) for t in fh["seqTags"][:]]
            lengths = fh["seqLengths"][:, 0]
            data = np.asarray(fh["inputs"][:]).reshape(-1)
            offset = 0
            for tag, length in zip(tags, lengths):
                assert tag not in rows, f"duplicate tag {tag}"
                rows[tag] = data[offset:offset + int(length)]
                offset += int(length)
            assert offset == len(data), (path, offset, len(data))
    return rows


class UniformPhonePriorJob(Job):
    """A ``model.prior.PhoneNgramPrior`` npz whose every conditional is uniform over the 40 symbols
    (log 1/40 in the unigram, bigram and trigram tables): R2's "uniform phone prior", loaded by the
    bed's model through ``prior_npz_path`` at the bed's ``prior_weight`` 1.0 (every token pays
    log 40 nats; the text prior carries no string preference)."""

    def __init__(self):
        super().__init__()
        self.out_prior = self.output_path("prior.npz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import numpy as np

        from ..model.prior import N_CTX, N_TYPES, PhoneNgramPrior

        v = -np.log(float(N_TYPES))
        prior = PhoneNgramPrior(np.full(N_TYPES, v), np.full((N_CTX, N_TYPES), v),
                                np.full((N_CTX * N_CTX, N_TYPES), v),
                                meta={"kind": "uniform", "log_p": float(v)})
        prior.save(self.out_prior.get_path())


class DevOtherPhoneReadJob(Job):
    """LABEL-USING, REPORT ONLY (QUARANTINE: module docstring).  R1 / R2's reads of one D4 dev-other
    posterior decode (``gendecode.json`` of :func:`dev_other_reads`) against the MFA gold.

    * DIRECT PER: per decoded utterance the decode's tokens with SIL dropped (the identity labelling:
      phone symbol -> its own name, SIL -> DELETE; no run-collapse, the decode emits segments), a
      Levenshtein (S + D + I) / N against the SIL-free gold (``analysis.per.edit_counts``' costs),
      pooled over the decoded utterances.  ``per_impossible_as_deletions`` adds every impossible
      utterance's gold as deletions.  ``adjacent_repeats`` / ``direct_per_collapsed`` state what a
      greedy-style run-collapse would change.
    * HUNGARIAN PER: the identity-label alignment of the RAW decode (SIL kept) against the gold gives
      the aligned (symbol, phone) counts; :func:`hungarian_map` picks the 1:1 symbol -> phone | DELETE
      map maximising the aligned matches (40 symbols x 39 phones + DELETE); the mapped decode
      (:func:`relabel`) is re-scored with a FRESH alignment.
    * TOKEN NMI(symbol, phone): :func:`nmi_bits` over the aligned (both sides present) pairs of that
      identity alignment (SIL a symbol, gold SIL-free).
    * FRAME NMI(symbol, phone) (when ``raw_index_hdfs`` and ``mfa_dir`` are given): per retained unit
      frame the decoded symbol (:func:`segment_frame_labels`) against the MFA phone at the frame's
      centre (:func:`gold_frame_labels`; SIL where no phone covers it), 40 x 40.
    * E[d]: phi's duration table (:func:`duration_means`), plus the decode's mean segment duration
      (non-SIL and SIL segments) beside it.
    """

    def __init__(self, *, gendecode: tk.Path, gold: tk.Path, sample_segments: tk.Path, phi: tk.Path,
                 prior: str, split: str = DEV_OTHER, raw_index_hdfs: Optional[Sequence[tk.Path]] = None,
                 mfa_dir: Optional[tk.Path] = None, mfa_glob: str = "data/dev_other-*.parquet"):
        super().__init__()
        assert (raw_index_hdfs is None) == (mfa_dir is None), "the frame read needs both inputs"
        self.gendecode = gendecode
        self.gold = gold
        self.sample_segments = sample_segments
        self.phi = phi
        self.prior = prior
        self.split = split
        self.raw_index_hdfs = None if raw_index_hdfs is None else list(raw_index_hdfs)
        self.mfa_dir = mfa_dir
        self.mfa_glob = mfa_glob
        self.out_json = self.output_path("phone_read.json")
        self.out_report = self.output_path("phone_read.txt")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def score(decodes: Dict[str, Optional[Sequence[str]]], gold: Dict[str, Sequence[str]]) -> Dict[str, Any]:
        """Direct PER, Hungarian PER and token NMI of ``decodes`` (tag -> raw tokens, SIL kept, or
        None when impossible) against ``gold`` (tag -> SIL-free phones)."""
        from ..phones import ARPABET_39, PHONES, SIL

        live = sorted(t for t, v in decodes.items() if v is not None)
        refs = {t: list(gold[t]) for t in live}
        raw = {t: list(decodes[t]) for t in live}
        identity = {p: p for p in ARPABET_39}
        identity[SIL] = DELETE
        direct = corpus_per({t: relabel(raw[t], identity) for t in live}, refs)
        pairs = [pr for t in live for pr in align(raw[t], refs[t])]
        mapping = hungarian_map(pairs, PHONES, ARPABET_39)
        hung = corpus_per({t: relabel(raw[t], mapping) for t in live}, refs)
        collapsed = {}
        repeats = 0
        for t in live:
            h = relabel(raw[t], identity)
            repeats += sum(a == b for a, b in zip(h, h[1:]))
            collapsed[t] = [x for k, x in enumerate(h) if k == 0 or x != h[k - 1]]
        dead = sorted(t for t, v in decodes.items() if v is None)
        n_all = direct["ref_tokens"] + sum(len(gold[t]) for t in dead)
        errors = direct["sub"] + direct["del"] + direct["ins"] + sum(len(gold[t]) for t in dead)
        return {
            "n_utterances": len(decodes), "n_decoded": len(live), "impossible_tags": dead,
            "direct": direct, "per_impossible_as_deletions": errors / n_all if n_all else None,
            "adjacent_repeats": repeats, "direct_per_collapsed": corpus_per(collapsed, refs)["per"],
            "hungarian": hung, "hungarian_map": mapping,
            "map_is_identity": all(mapping[s] == identity[s] for s in PHONES),
            "symbols_mapped_to_identity": sum(mapping[s] == identity[s] for s in PHONES),
            "token_nmi": nmi_bits([(h, r) for h, r in pairs if h is not None and r is not None]),
        }

    def run(self):
        import glob

        import pandas as pd

        from ..phones import PHONES, SIL, canonical_phone
        from .genmarg_steps import load_phi_state

        record = json.load(open(self.gendecode.get_path()))
        assert record["dataset"] == DEV_OTHER, record["dataset"]
        rows = record["per_utterance"]
        with open(self.sample_segments.get_path()) as fh:
            tags = [ln.strip() for ln in fh if ln.strip()]
        assert set(rows) == set(tags), "the decode does not cover exactly the D4 sample"
        gold = json.load(open(self.gold.get_path()))[self.split]
        decodes = {t: (None if rows[t]["impossible"] else rows[t]["tokens"]) for t in tags}
        out = {"schema": "sae-i6-g0g-devother-phone-read-v1", "label_using": True, "report_only": True,
               "prior": self.prior, "decode_name": record.get("name"),
               "gendecode": os.path.realpath(self.gendecode.get_path()),
               "phi_checkpoint": os.path.realpath(self.phi.get_path()),
               **self.score(decodes, gold)}

        state, meta = load_phi_state(self.phi.get_path())
        out["phi_load"] = meta
        out["e_d_table"] = duration_means(state["dur_logits"])
        seg = {"nonsil": [0, 0], "sil": [0, 0]}
        for t in tags:
            for k, _, d in (rows[t]["segments"] or []):
                key = "sil" if PHONES[int(k)] == SIL else "nonsil"
                seg[key][0] += int(d)
                seg[key][1] += 1
        out["e_d_decode"] = {k: (v[0] / v[1] if v[1] else None) for k, v in seg.items()}
        out["e_d_decode"]["segments"] = {k: v[1] for k, v in seg.items()}

        out["frame_nmi"] = None
        if self.raw_index_hdfs is not None:
            raw_index = _read_hdf_int_sequences([p.get_path() for p in self.raw_index_hdfs])
            files = sorted(glob.glob(os.path.join(self.mfa_dir.get_path(), self.mfa_glob)))
            assert files, f"no MFA parquet under {self.mfa_dir.get_path()}/{self.mfa_glob}"
            wanted, intervals = set(tags), {}
            for fp in files:
                df = pd.read_parquet(fp, columns=["id", "phonemes"])
                for r in df.itertuples():
                    if str(r.id) in wanted:
                        intervals[str(r.id)] = [(canonical_phone(p["phoneme"]), p["start"], p["end"])
                                                for p in r.phonemes]
            missing = sorted(wanted - set(intervals))
            assert not missing, f"{len(missing)} D4 tags without an MFA alignment, e.g. {missing[:3]}"
            pairs = []
            for t in tags:
                if rows[t]["impossible"]:
                    continue
                idx = raw_index[t]
                hyp = segment_frame_labels(rows[t]["segments"], len(idx))
                pairs.extend(zip(hyp, gold_frame_labels(idx, intervals[t])))
            out["frame_nmi"] = nmi_bits(pairs)
            out["frame_convention"] = ("retained 50 Hz unit frame; gold = MFA canonical phone at the "
                                       "frame centre (raw_index + 0.5) / 50 s, SIL where uncovered")

        with open(self.out_json.get_path(), "w") as fh:
            json.dump(out, fh, indent=1)
        f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        lines = [
            f"LABEL-USING, REPORT ONLY: D4 {self.split} phone read, prior = {self.prior}",
            f"decoded {out['n_decoded']} / {out['n_utterances']} (impossible {len(out['impossible_tags'])})",
            f"direct PER     {f(out['direct']['per'])}  (S {out['direct']['sub']} D {out['direct']['del']} "
            f"I {out['direct']['ins']} N {out['direct']['ref_tokens']})",
            f"Hungarian PER  {f(out['hungarian']['per'])}  (map is identity: {out['map_is_identity']}; "
            f"{out['symbols_mapped_to_identity']} / 40 symbols at their identity label)",
            f"token NMI      {f(out['token_nmi']['nmi'])}  (n {out['token_nmi']['n']})",
            f"frame NMI      {f(out['frame_nmi']['nmi']) if out['frame_nmi'] else 'not built'}",
            f"E[d] table     phones {f(out['e_d_table']['phone_types_mean'])}  SIL {f(out['e_d_table']['sil'])}",
            f"E[d] decode    non-SIL {f(out['e_d_decode']['nonsil'])}  SIL {f(out['e_d_decode']['sil'])}",
        ]
        with open(self.out_report.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


def dev_other_reads(phi, name: str, *, bed: Dict[str, Any], gold: tk.Path,
                    stream: Dict[str, Sequence[tk.Path]], raw_index_hdfs: Optional[Sequence[tk.Path]] = None,
                    mfa_dir: Optional[tk.Path] = None, priors: Sequence[str] = PRIORS,
                    alias: Optional[str] = None, returnn_python_exe: Optional[tk.Path] = None,
                    returnn_root: Optional[tk.Path] = None) -> Dict[str, Any]:
    """LABEL-USING, REPORT ONLY (QUARANTINE: module docstring).  ``phi``'s posterior decode of the D4
    dev-other set per prior, and its :class:`DevOtherPhoneReadJob`.

    :param phi: as :func:`genmarg_reads`'.
    :param bed: :func:`bed_from_train_config` of the bed's config; the ``"uniform"`` prior replaces
        only its ``prior_npz_path`` (by :class:`UniformPhonePriorJob`'s npz).
    :param gold: ``GoldPhonesJob.out_gold`` (the D4 sample and the reference).
    :param stream: ``Inputs.dev_stream("dev-other")``.
    :param raw_index_hdfs: / :param mfa_dir: the VAD job's dev-other ``raw_index`` HDFs and the MFA
        parquet dir (``data.gold.get_mfa_alignments("dev")``) for the frame NMI; both or neither.
    :param priors: a subset of :data:`PRIORS`.
    :return: ``{"sample", "uniform_prior" (when read), prior: {"decode", "report"}}``.
    """
    unknown = sorted(set(priors) - set(PRIORS))
    if unknown or not priors:
        raise ValueError(f"priors {list(priors)}: a non-empty subset of {PRIORS}")
    phi = _phi_path(phi)
    sample = GenMargSampleJob(gold=gold, split=DEV_OTHER, n=SELECT_N, seed=0)
    data = eval_dataset(bed["dev"], DEV_OTHER, sample.out_segments, stream=stream)
    out: Dict[str, Any] = {"sample": sample}
    named: Dict[str, Job] = {"sample": sample}
    for prior in priors:
        b = bed
        if prior == "uniform":
            out["uniform_prior"] = named["uniform_prior"] = UniformPhonePriorJob()
            args = copy.deepcopy(bed["model_args"])
            args["prior_npz_path"] = out["uniform_prior"].out_prior
            b = dict(bed, model_args=args)
        dec = _forward_job(mode="decode", phi=phi, bed=b, data=data, shuffle_seed=None,
                           name=f"{name}/{prior}", dataset=DEV_OTHER, expected_utterances=SELECT_N,
                           returnn_python_exe=returnn_python_exe, returnn_root=returnn_root)
        rep = DevOtherPhoneReadJob(gendecode=dec.out_files["gendecode.json"], gold=gold,
                                   sample_segments=sample.out_segments, phi=phi, prior=prior,
                                   raw_index_hdfs=raw_index_hdfs, mfa_dir=mfa_dir)
        out[prior] = {"decode": dec, "report": rep}
        named[f"{prior}/decode"], named[f"{prior}/report"] = dec, rep
    if alias:
        for key, job in named.items():
            job.add_alias(f"{alias}/{DEV_OTHER}/{key}")
    return out
