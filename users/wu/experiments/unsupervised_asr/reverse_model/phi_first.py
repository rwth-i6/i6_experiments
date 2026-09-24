"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/
config_sae_4a_lexlat_v2_em_v1.py, with ``sae/emc/blankfree_phifirst_jobs.py`` (``GRANULARITIES``,
``_rng``, ``permute_units``, ``PermutedUnitsHdfJob``).

L2-1 of SAE_4A_lexlat_v2.md, stage 1: phi-first EM with a null recognizer.  Label-free: no transcript
enters the training or the selection.

WHAT RUNS.  Theta is the NULL recognizer (``SaeBlankfreeModelV1(null_recognizer=True)`` with
``freeze_recognizer=True``: ``log_q = -log 40`` on every frame, never run, out of the optimizer);
phi trains alone by gradient on the lattice term under the frozen phone trigram (the bed's
``prior_npz``), the bed's objective otherwise unchanged (l_tau is then the path-space generative
objective up to the phi-independent ``(T'/tau) log V``, and tau = 1 is its exact E-step).  Agg and
rate have no gradient path under the null.

STAGE-1 RECIPE (:func:`stage1_config`)
  * theta null; phi at the bed's random init under RETURNN's ``random_seed = s``.  The same ``s`` is
    written as ``random_seed_offset = s`` on the train stream's order-controlling dataset (``feats``),
    so seed ``s`` also sets the training data order; a rerun is written with its original's seed and
    so has the same order.  Or phi from phi_c (:func:`phi_c_phi`) for the two phi_c restarts.
  * the bed's phone-trigram prior only: no k2, no lexicon, no stage 2.
  * tau per sub-epoch :data:`TAU_SCHEDULE` = ``[4, 1, 1, 1]`` (:func:`tau_schedule` for other lengths).
  * OPTIMISER: Adam at a CONSTANT phi learning rate 3e-3 with global-norm clip 5, no warmup and no
    decay (the supervised reverse fit's recipe, ``model.reverse.FitConfig``, at torch's default Adam
    betas (0.9, 0.999) and eps 1e-8).  Written as betas / eps on the bed's Adam, weight decay 0,
    ``learning_rates = [1e-4] * n`` times ``emc_param_groups``' phi multiplier 30.  Asserted on every
    written config.  The batch stays the bed's (88000 frames, 128 seqs).
  * DATA: the bed's own train stream in its own sub-epochs (``partition_epoch = 4``,
    ``laplace:.1000``), so :data:`NUM_SUBEPOCHS` = 4 sub-epochs are ONE pass; held-out = the bed's
    dev set (the CV holdout of the train stream), evaluated after every sub-epoch.
  * Weight decay is 0 (asserted): ``durfrz`` freezes the phone rows of phi's duration table by
    zeroing their gradient, which holds only without weight decay.

DURATION SETTINGS (A9): ``uniform`` (no duration keys; report only), ``durinit`` / ``durfrz`` (the
same config plus ``reverse_duration_prior`` = the ``prior.json`` of ``duration_prior.
BlankfreeDurationPriorMeanJob`` (:func:`duration_prior_json`) and ``reverse_duration_prior_mode`` =
``"init"`` / ``"freeze"``).  Nulls use the wave's setting; phi_c arms keep phi_c's own durations.
The wave's setting :data:`WAVE_DURATION_SETTING` and sub-epoch count :data:`WAVE_NUM_SUBEPOCHS` are
set by hand from the probe / A10 reads and recorded in the phase file; :func:`build_wave` refuses
while either is None.

NULLS.  The same recipe on the structure-destroyed corpus (:func:`null_corpus`,
:class:`PermutedUnitsHdfJob`, seed :data:`NULL_CORPUS_SEED`): every train shard's units permuted
within each utterance, so train AND the CV holdout are shuffled the same way.  The permutation is
``model.blankfree_permute.utterance_permutation(tag, n, seed)`` exactly, so the genmarg read with
``shuffle = NULL_CORPUS_SEED`` on the real stream scores the nulls' permuted holdout (the gated
score); the real holdout is scored as well (reported).

PROBE (:func:`build_probe`).  3 duration settings x seeds 1, 2, 4 sub-epochs each, every sub-epoch
checkpoint kept and scored on the CV holdout by ``genmarg.genmarg_reads`` (:func:`probe_scoring_reads`:
the held-out tau = 1 marginal and the posterior decode whose ``emitted_nonsil_rate.pooled_hz`` is the
A8 rate).

WAVE (:func:`build_wave`, A7).  16 restarts (seeds 1-16), 4 nulls (seeds 1-4), 2 restarts initialised
from phi_c (seeds 1, 2) and exact reruns of seeds 1 and 2, the final checkpoint kept;
:func:`selection_reads` builds the genmarg read of every arm, the decode of every restart, and
``genmarg.GenMargSelectionJob`` (the amended L2-1 selection and the G4a.L2.2 read).

Port changes:

* Inputs: the bed's ``build_train_config`` data dict (``training.arms``' ``data``) replaces the
  source's pinned streams, prior, eta table and flat init; phi_c's source checkpoint (``k2lat_20_x60``
  sub-epoch 60) is the caller's ``phi_c_source``; the genmarg reads take the bed from
  ``genmarg.bed_from_train_config(build_train_config(**data))`` and the CV holdout from
  ``data["dev_segments"]``.  :func:`duration_prior_json` builds the prior job from the same data dict
  at the bed's ``rate_rho_hz`` and ``LatticeConfig``'s frame rate (the source's
  ``prior_mean_job``'s arguments).
* The probe is a plain ``ReturnnTrainingJob`` (``training.jobs.train_arm``, the same mem / cpu /
  gpu_mem) instead of the instrumented ``PhiFirstProbeTrainingJob``; its step-time / GPU sampler, its
  ``probe.json`` summary and the ``torch_log_memory_usage`` post-config key that fed it are cut.  The
  probe reader (``PhiFirstProbeReadJob``), the A10 extension builder and its readers are not ported.
* The wave runs one ``train_arm`` job per arm instead of six 4-GPU ``PackedBlankfreeTrainJob`` nodes;
  the per-arm allocation is :func:`alloc_hours` (the pack's 1.5 h bar x 1.1 per 4 sub-epochs, scaled
  with the sub-epoch count; the source kept 1.65 h).  The exact reruns carry the inert config key
  :data:`RERUN_KEY` so that each is its own job, and every wave arm carries the inert config key
  :data:`WAVE_KEY` so that no wave arm is a probe run (:func:`build_wave`).
* ``PermutedUnitsHdfJob``'s ``"run"`` granularity (never registered) is cut: it raises ValueError.
* The builders return their jobs and register no outputs; the source config's ``sys.path`` prolog
  is dropped.
"""

from __future__ import annotations

import json
import os
import zlib
from typing import Any, Dict, Optional, Sequence

import numpy as np

from sisyphus import Job, Task, tk

__all__ = [
    "NUM_SUBEPOCHS", "TAU_SCHEDULE", "tau_schedule", "PROBE_SEEDS", "RESTART_SEEDS", "NULL_SEEDS",
    "PHI_C_SEEDS", "RERUN_SEEDS", "RERUN_KEY", "WAVE_KEY", "NULL_CORPUS_SEED", "NULL_GRANULARITY", "STAGE1_DELTA",
    "PHI_LR",
    "ADAM_BETAS", "ADAM_EPS", "GRAD_CLIP", "DURATION_SETTINGS", "PRIOR_MODE", "WAVE_DURATION_SETTING",
    "WAVE_NUM_SUBEPOCHS", "BAR_HOURS", "alloc_hours", "GRANULARITIES", "permute_units",
    "PermutedUnitsHdfJob", "null_corpus", "duration_prior_json", "phi_c_phi", "stage1_config",
    "arm_name", "probe_scoring_reads", "selection_reads", "build_probe", "build_wave",
]

ALIAS = "sae/4a/lexlat_v2/em"

# -- the stage-1 recipe -------------------------------------------------------------------------
#: sub-epochs per restart: the bed's partition_epoch 4, so one pass over train-clean-100 (A3)
NUM_SUBEPOCHS = 4
#: tau per sub-epoch, last entry held (A3)
TAU_SCHEDULE = [4.0, 1.0, 1.0, 1.0]
assert len(TAU_SCHEDULE) == NUM_SUBEPOCHS and TAU_SCHEDULE == [4.0, 1.0, 1.0, 1.0], TAU_SCHEDULE


def tau_schedule(num_subepochs: int) -> list:
    """tau per sub-epoch for a restart of ``num_subepochs``: 4 at sub-epoch 1, then 1 (A3; A10:
    "tau 4 at sub-epoch 1, then 1"); at 4 sub-epochs exactly :data:`TAU_SCHEDULE`."""
    n = int(num_subepochs)
    assert n >= 2, n
    return [4.0] + [1.0] * (n - 1)


assert tau_schedule(NUM_SUBEPOCHS) == TAU_SCHEDULE
#: seeds (A3, A7)
PROBE_SEEDS = (1, 2)
RESTART_SEEDS = tuple(range(1, 17))
NULL_SEEDS = tuple(range(1, 5))
PHI_C_SEEDS = (1, 2)
RERUN_SEEDS = (1, 2)
#: the reruns' inert, hashed config key (:func:`build_wave`): makes each rerun its own job
RERUN_KEY = "rerun_index"
#: the wave arms' inert, hashed config key (:func:`build_wave`): keeps a wave arm apart from the probe
#: run of the same seed and duration setting (at 4 sub-epochs their configs are otherwise equal)
WAVE_KEY = "wave_arm"
#: the permutation seed of the null corpus: 0, the precedent of every destroyed-structure control in
#: this campaign
NULL_CORPUS_SEED = 0
NULL_GRANULARITY = "unit"
#: the stage-1 delta against the bed's builder
STAGE1_DELTA = {"null_recognizer": True, "freeze_recognizer": True}
#: A3's phi optimiser: the supervised reverse fit's (``model.reverse.FitConfig``, torch Adam defaults)
PHI_LR = 3.0e-3
ADAM_BETAS = [0.9, 0.999]
ADAM_EPS = 1e-8
GRAD_CLIP = 5.0
#: A9's duration settings: uniform (as registered, report only) and the two prior modes
DURATION_SETTINGS = ("uniform", "durinit", "durfrz")
PRIOR_MODE = {"durinit": "init", "durfrz": "freeze"}
#: A9: the wave's duration setting, "durinit" or "durfrz", set from the probe read's verdict and
#: recorded in the phase file before the wave.  None = not decided: the wave does not build.
WAVE_DURATION_SETTING: Optional[str] = None
#: A10: the wave's sub-epoch count (a multiple of 4), set from the A10 read's verdict and recorded in
#: the phase file before the wave.  None = not decided: the wave does not build.
WAVE_NUM_SUBEPOCHS: Optional[int] = None

# -- time -----------------------------------------------------------------------------------------
#: G4a.L2.1's bar on a 4-sub-epoch restart's wall time (hours)
BAR_HOURS = 1.5
#: the source pack's shared-node time factor (``pack_jobs.SHARED_NODE_TIME_FACTOR``)
SHARED_NODE_TIME_FACTOR = 1.1


def alloc_hours(num_subepochs: int = NUM_SUBEPOCHS) -> float:
    """A restart's allocation for a run of ``num_subepochs`` sub-epochs: the banked 4-sub-epoch
    figure (the bar times the source pack's shared-node factor as the margin, 1.65 h) scaled
    linearly by ``num_subepochs / 4``, capped at ``training.jobs.TIME_RQMT`` (a longer run resumes
    from its last checkpoint in the next allocation).  1.65 h for the probe and a 4-sub-epoch wave.
    Port change: the source kept 1.65 h at any :data:`WAVE_NUM_SUBEPOCHS`."""
    from ..training.jobs import TIME_RQMT

    n = int(num_subepochs)
    assert n >= 1, n
    return min(BAR_HOURS * SHARED_NODE_TIME_FACTOR * n / NUM_SUBEPOCHS, TIME_RQMT)


# ----------------------------------------------------------------------------------------------
# the structure-destroyed corpus
# ----------------------------------------------------------------------------------------------

#: ``"unit"`` -- every unit frame of the utterance permuted independently (the REGISTERED null:
#: "units permuted within each utterance").  The source's ``"run"`` granularity (never registered)
#: is cut and raises ValueError.
GRANULARITIES = ("unit",)


def _check_granularity(granularity: str) -> None:
    if granularity == "run":
        raise ValueError("granularity 'run' (block permutation of unit runs) was never registered "
                         "and is not ported; the null is granularity 'unit'")
    assert granularity in GRANULARITIES, f"granularity is one of {GRANULARITIES}, not {granularity!r}"


def _rng(tag: str, seed: int) -> np.random.RandomState:
    """``blankfree_permute``'s idiom: a fixed stream per (seed, tag), drawn from no global RNG."""
    return np.random.RandomState((int(seed) ^ zlib.crc32(tag.encode("utf-8"))) & 0xFFFFFFFF)


def permute_units(units: np.ndarray, *, tag: str, seed: int, granularity: str) -> np.ndarray:
    """One utterance's unit sequence ``[T]`` with its order destroyed at ``granularity``.

    The multiset of units and the length are kept exactly; the result is a pure function of
    (``units``, ``tag``, ``seed``, ``granularity``).
    """
    _check_granularity(granularity)
    x = np.asarray(units)
    assert x.ndim == 1, x.shape
    n = int(x.shape[0])
    if n <= 1:
        return x.copy()
    rng = _rng(tag, seed)
    return x[rng.permutation(n)]


class PermutedUnitsHdfJob(Job):
    """Each units HDF shard, copied, with every utterance's units permuted (:func:`permute_units`).

    Only the ``inputs`` dataset is rewritten, sequence by sequence at the offsets ``seqLengths``
    states; ``seqTags``, ``seqLengths``, the attributes and every other field are the source's
    bytes, so lengths, tags (hence eta) and the file layout are unchanged.  The summary counts, per
    shard, the utterances, the frames, the utterances whose order actually changed and the share of
    unit positions that moved, and it asserts per utterance that the length and the unit multiset
    are kept.

    :param units_hdfs: the bed's units shards, in order.
    :param seed: the permutation seed; the permutation of an utterance is a function of it and the
        tag alone, so every shard, every restart and every re-run draw the same one.
    :param granularity: ``"unit"`` (the registered null); ``"run"`` raises ValueError (cut).
    """

    def __init__(self, *, units_hdfs: Sequence[tk.Path], seed: int, granularity: str = "unit"):
        _check_granularity(granularity)
        self.units_hdfs = list(units_hdfs)
        self.seed = int(seed)
        self.granularity = str(granularity)
        self.out_hdfs = [self.output_path(f"units.permuted.shard{k}.hdf")
                         for k in range(len(self.units_hdfs))]
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import shutil

        import h5py

        report = {"seed": self.seed, "granularity": self.granularity, "shards": []}
        for src, dst in zip(self.units_hdfs, self.out_hdfs):
            tmp = dst.get_path() + ".tmp"
            shutil.copyfile(src.get_path(), tmp)
            with h5py.File(tmp, "r+") as h:
                lengths = np.asarray(h["seqLengths"][:, 0], dtype=np.int64)
                tags = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in h["seqTags"][:]]
                data = np.asarray(h["inputs"][:])
                assert data.ndim == 1 and int(lengths.sum()) == int(data.shape[0]), (
                    data.shape, int(lengths.sum()))
                offsets = np.concatenate([[0], np.cumsum(lengths)])
                out = data.copy()
                changed = moved = 0
                for i, tag in enumerate(tags):
                    a, b = int(offsets[i]), int(offsets[i + 1])
                    seq = data[a:b]
                    new = permute_units(seq, tag=tag, seed=self.seed, granularity=self.granularity)
                    assert new.shape == seq.shape and np.array_equal(np.sort(new), np.sort(seq)), tag
                    changed += int(not np.array_equal(new, seq))
                    moved += int((new != seq).sum())
                    out[a:b] = new
                h["inputs"][...] = out
            os.replace(tmp, dst.get_path())
            report["shards"].append({
                "source": src.get_path(), "output": dst.get_path(), "utterances": len(tags),
                "frames": int(lengths.sum()), "utterances_reordered": changed,
                "frac_positions_changed": moved / max(1, int(lengths.sum())),
            })
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(report, f, indent=2)


# ----------------------------------------------------------------------------------------------
# the stage-1 configs
# ----------------------------------------------------------------------------------------------

def _data(data: Dict[str, Any]) -> Dict[str, Any]:
    from ..training.arms import ARM_DATA_KEYS

    missing = sorted(set(ARM_DATA_KEYS) - set(data))
    unknown = sorted(set(data) - set(ARM_DATA_KEYS))
    assert not missing and not unknown, f"data dict: missing {missing}, unknown {unknown}"
    return dict(data)


def _streams(data: Dict[str, Any], units: Optional[Sequence[tk.Path]] = None) -> Dict[str, Any]:
    """The bed's data dict, with the units shards replaced for a null (train AND dev: the bed's dev
    set is the train stream's CV holdout, so it reads the same shards)."""
    kw = _data(data)
    if units is not None:
        assert list(kw["train_units_hdfs"]) == list(kw["dev_units_hdfs"]), (
            "a null replaces the train stream's units shards, which the bed's dev set shares")
        assert len(units) == len(kw["train_units_hdfs"]), (len(units), len(kw["train_units_hdfs"]))
        kw["train_units_hdfs"] = list(units)
        kw["dev_units_hdfs"] = list(units)
    return kw


def null_corpus(data: Dict[str, Any], granularity: str = NULL_GRANULARITY, *,
                alias: Optional[str] = ALIAS) -> PermutedUnitsHdfJob:
    """The structure-destroyed units: every train shard (train and the CV holdout) permuted."""
    job = PermutedUnitsHdfJob(units_hdfs=_data(data)["train_units_hdfs"], seed=NULL_CORPUS_SEED,
                              granularity=granularity)
    if alias:
        job.add_alias(f"{alias}/null_corpus_{granularity}")
    return job


def duration_prior_json(data: Dict[str, Any], *, alias: Optional[str] = "sae/4a/lexlat_durprior") -> tk.Path:
    """A9's prior: the ``prior.json`` of D17's ``BlankfreeDurationPriorMeanJob`` from the bed's train
    stream lengths (label-free), at the bed's ``rate_rho_hz`` and the lattice's frame rate."""
    from ..model.lattice import LatticeConfig
    from ..training.config import build_train_config
    from ..training.jobs import get_model_args
    from .duration_prior import BlankfreeDurationPriorMeanJob

    kw = _data(data)
    rho_hz = get_model_args(build_train_config(**kw))["rate_rho_hz"]
    job = BlankfreeDurationPriorMeanJob(units_hdfs=kw["train_units_hdfs"],
                                        orig_length_hdfs=kw["train_original_hdfs"],
                                        rho_hz=rho_hz, frame_rate_hz=LatticeConfig().frame_rate_hz)
    if alias:
        job.add_alias(f"{alias}/prior_mean")
    return job.out_json


def phi_c_phi(phi_c_source, *, alias: Optional[str] = ALIAS) -> tk.Path:
    """phi_c's reverse block as a standalone checkpoint, loaded through ``reverse_checkpoint_path``
    (strict: missing and unexpected keys are errors), sliced by ``ExtractSubmoduleCheckpointJob``.

    :param phi_c_source: the whole-model checkpoint of phi_c (banked: ``k2lat_20_x60`` sub-epoch 60),
        a ``tk.Path`` or a checkpoint object with ``.path``.
    """
    from ..training.checkpoints import ExtractSubmoduleCheckpointJob
    from .genmarg import _phi_path

    job = ExtractSubmoduleCheckpointJob(checkpoint=_phi_path(phi_c_source), prefix="reverse.")
    if alias:
        job.add_alias(f"{alias}/phi_c_extract")
    return job.out_checkpoint


def _phi_lr_multiplier(cfg) -> float:
    """``emc_param_groups``' hashed ``reverse_lr_multiplier`` in the written config's prolog."""
    found = []
    for item in cfg.python_prolog:
        for obj in getattr(item, "serializer_objects", []):
            if getattr(obj, "import_as", None) == "emc_param_groups":
                found.append(obj.hashed_arguments["reverse_lr_multiplier"])
    assert len(found) == 1, found
    return float(found[0])


def stage1_config(seed: int, *, data: Dict[str, Any], units: Optional[Sequence[tk.Path]] = None,
                  reverse_init: Optional[tk.Path] = None, duration: str = "uniform",
                  num_subepochs: int = NUM_SUBEPOCHS):
    """One stage-1 restart's ``ReturnnConfig`` (module docstring, STAGE-1 RECIPE; ``duration`` one of
    :data:`DURATION_SETTINGS`).  ``num_subepochs`` sets the length of the two per-sub-epoch
    schedules, :func:`tau_schedule` and the constant learning rate; at the default 4 the config is
    the probe's.

    :param data: the bed's ``build_train_config`` data dict.
    :param units: the null corpus's units shards (``PermutedUnitsHdfJob.out_hdfs``), None = real.
    :param reverse_init: phi's init checkpoint (phi_c, :func:`phi_c_phi`), None = random init.
    """
    from ..model.reverse import FitConfig
    from ..training import config as base

    fit = FitConfig()
    assert (fit.lr, fit.grad_clip, fit.weight_decay) == (PHI_LR, GRAD_CLIP, 0.0), fit
    delta: Dict[str, Any] = dict(STAGE1_DELTA)
    assert duration in DURATION_SETTINGS, duration
    if reverse_init is not None:
        assert duration == "uniform", "phi_c arms keep phi_c's own durations (A9)"
        delta["reverse_checkpoint_path"] = reverse_init
    if duration != "uniform":
        delta["reverse_duration_prior"] = duration_prior_json(data)
        delta["reverse_duration_prior_mode"] = PRIOR_MODE[duration]
    cfg = base.build_train_config(
        **_streams(data, units), **delta, num_subepochs=int(num_subepochs),
        temperature_schedule=tau_schedule(num_subepochs),
        adam_betas=list(ADAM_BETAS), adam_eps=ADAM_EPS,
        random_seed=int(seed), random_seed_offset=int(seed),
    )
    c = cfg.config
    feats = c["train"]["dataset"]["datasets"]["feats"]
    assert feats["partition_epoch"] == 4 and feats["seq_ordering"] == "laplace:.1000", feats
    # A3: phi at a constant 3e-3 = the flat base LR x emc_param_groups' phi multiplier
    lrs = c["learning_rates"]
    mult = _phi_lr_multiplier(cfg)
    assert lrs == [base.THETA_LEARNING_RATE] * int(num_subepochs), lrs
    assert all(abs(lr * mult - PHI_LR) < 1e-15 for lr in lrs), (lrs, mult)
    assert c["gradient_clip_global_norm"] == GRAD_CLIP, c["gradient_clip_global_norm"]
    opt = c["optimizer"]
    # weight decay 0: durfrz's gradient hook freezes the phone rows only without it
    assert opt["class"] == "adam" and opt["weight_decay"] == 0.0, opt
    assert not any("weight_decay" in k and k != "weight_decay" for k in opt), opt
    assert opt["betas"] == list(ADAM_BETAS) and opt["eps"] == ADAM_EPS, opt
    for key in ("learning_rate_control", "learning_rate_warmup", "dynamic_learning_rate"):
        assert key not in c, key
    assert c["random_seed"] == int(seed), c["random_seed"]
    # A8: the seed also sets the training data order (module docstring, STAGE-1 RECIPE)
    meta = c["train"]["dataset"]
    assert meta["class"] == "MetaDataset" and meta["seq_order_control_dataset"] == "feats", meta
    assert feats["random_seed_offset"] == int(seed), feats
    return cfg


def arm_name(seed: int, kind: str = "em") -> str:
    """``em_sNN`` (restart), ``null_sNN``, ``phic_sNN`` (phi_c init), ``em_sNN_rerun``."""
    assert kind in ("em", "null", "phic", "rerun"), kind
    if kind == "rerun":
        return f"em_s{int(seed):02d}_rerun"
    return f"{kind}_s{int(seed):02d}"


# ----------------------------------------------------------------------------------------------
# the label-free reads
# ----------------------------------------------------------------------------------------------

def reads_bed(data: Dict[str, Any]) -> Dict[str, Any]:
    """``genmarg_reads``' ``bed`` and ``cv_segments`` for this bed: the bed's own config at its
    default model arguments (with the banked reads' inert ``READ_TEMPERATURE_SCHEDULE``) and its dev
    segment list (the CV holdout of the train stream)."""
    from ..training.config import build_train_config
    from .genmarg import READ_TEMPERATURE_SCHEDULE, bed_from_train_config

    kw = _data(data)
    cfg = build_train_config(**kw, num_subepochs=len(READ_TEMPERATURE_SCHEDULE),
                             temperature_schedule=list(READ_TEMPERATURE_SCHEDULE))
    return {"bed": bed_from_train_config(cfg), "cv_segments": kw["dev_segments"]}


def _marginal(phi, name: str, *, reads: Dict[str, Any], shuffle: Optional[int] = None,
              alias: Optional[str]) -> tk.Path:
    """Statistic (a) of ``phi`` on the CV holdout: the genmarg marginal job's ``genmarg.json``."""
    from .genmarg import genmarg_reads

    jobs = genmarg_reads(phi, name, datasets=("cv_holdout",), shuffle=shuffle, decode=False,
                         alias=alias, **reads)
    return jobs["cv_holdout"]["marginal"].out_files["genmarg.json"]


def _marginal_and_decode(phi, name: str, *, reads: Dict[str, Any], alias: Optional[str]) -> Dict[str, tk.Path]:
    """Statistic (a) and the posterior decode of ``phi`` on the (real) CV holdout: ``genmarg.json``
    and ``gendecode.json`` (whose ``emitted_nonsil_rate.pooled_hz`` is the A8 rate)."""
    from .genmarg import genmarg_reads

    jobs = genmarg_reads(phi, name, datasets=("cv_holdout",), decode=True, gap=False,
                         alias=alias, **reads)["cv_holdout"]
    assert set(jobs) == {"sample", "marginal", "decode"}, sorted(jobs)
    return {"genmarg.json": jobs["marginal"].out_files["genmarg.json"],
            "gendecode.json": jobs["decode"].out_files["gendecode.json"]}


def _probe_tag(setting: str, seed: int) -> str:
    """``sNN`` for the registered uniform probe, ``<setting>_sNN`` for the prior settings."""
    assert setting in DURATION_SETTINGS, setting
    return f"s{int(seed):02d}" if setting == "uniform" else f"{setting}_s{int(seed):02d}"


def probe_scoring_reads(probes: Dict[int, object], setting: str = "uniform", *, reads: Dict[str, Any],
                        alias: Optional[str] = ALIAS) -> Dict[int, Dict[int, Dict[str, tk.Path]]]:
    """Every probe sub-epoch checkpoint of one duration setting scored on the CV holdout: the
    held-out tau = 1 marginal (``genmarg.json``) and the posterior decode (``gendecode.json``).
    ``reads`` is :func:`reads_bed`.  Returns ``{seed: {sub-epoch: {file: path}}}``."""
    out: Dict[int, Dict[int, Dict[str, tk.Path]]] = {}
    for seed, job in sorted(probes.items()):
        out[seed] = {}
        for ep in range(1, NUM_SUBEPOCHS + 1):
            name = f"l21_probe_{_probe_tag(setting, seed)}_ep{ep}"
            out[seed][ep] = _marginal_and_decode(job.out_checkpoints[ep], name, reads=reads,
                                                 alias=f"{alias}/probe/reads/{name}" if alias else None)
    return out


def selection_reads(restarts: Dict[str, object], nulls: Dict[str, object],
                    references: Dict[str, object], last: int = NUM_SUBEPOCHS, *,
                    reads: Dict[str, Any], alias: Optional[str] = ALIAS) -> dict:
    """The G4a.L2.2 reads: each arm's final checkpoint (``out_checkpoints[last]``) scored on the CV
    holdout by ``genmarg_reads``, and ``GenMargSelectionJob`` over them (A7).

    Nulls are scored twice: on their PERMUTED holdout (``shuffle = NULL_CORPUS_SEED``, the gated
    score) and on the real one (reported).  ``references`` holds the phi_c restarts and the reruns,
    keyed by :func:`arm_name`; a rerun is handed to the selection under the restart it reruns.  Every
    restart is also decoded (``restart_decodes``): its emitted rate is what VOID reads (A8).
    """
    from .genmarg import GenMargSelectionJob

    last = int(last)
    base = f"{alias}/wave/reads" if alias else None

    def _al(a):
        return f"{base}/{a}" if base else None

    sel = {"restarts": {}, "nulls": {}, "nulls_real": {}, "phi_c": {}, "reruns": {},
           "restart_decodes": {}}
    for a, t in sorted(restarts.items()):
        r = _marginal_and_decode(t.out_checkpoints[last], f"l21_{a}", reads=reads, alias=_al(a))
        sel["restarts"][a] = r["genmarg.json"]
        sel["restart_decodes"][a] = r["gendecode.json"]
    for a, t in sorted(nulls.items()):
        sel["nulls"][a] = _marginal(t.out_checkpoints[last], f"l21_{a}_permuted", reads=reads,
                                    shuffle=NULL_CORPUS_SEED, alias=_al(f"{a}_permuted"))
        sel["nulls_real"][a] = _marginal(t.out_checkpoints[last], f"l21_{a}_real", reads=reads,
                                         alias=_al(f"{a}_real"))
    for s in PHI_C_SEEDS:
        a = arm_name(s, "phic")
        sel["phi_c"][a] = _marginal(references[a].out_checkpoints[last], f"l21_{a}", reads=reads,
                                    alias=_al(a))
    for s in RERUN_SEEDS:
        a = arm_name(s, "rerun")
        sel["reruns"][arm_name(s)] = _marginal(references[a].out_checkpoints[last], f"l21_{a}",
                                               reads=reads, alias=_al(a))
    job = GenMargSelectionJob(**sel, name="L2-1 wave")
    if alias:
        job.add_alias(f"{alias}/wave/selection")
    return {"reads": sel, "selection": job}


# ----------------------------------------------------------------------------------------------
# the probe and the wave
# ----------------------------------------------------------------------------------------------

def build_probe(data: Dict[str, Any], *, alias: Optional[str] = ALIAS) -> dict:
    """A3's probe per A9's duration setting: {uniform, durinit, durfrz} x seeds 1, 2, 4 sub-epochs
    each, every sub-epoch checkpoint kept and scored (:func:`probe_scoring_reads`).  The source's
    probe reader is not ported (module docstring)."""
    from ..training.jobs import train_arm

    alloc = alloc_hours()
    reads = reads_bed(data)
    probes: Dict[str, Dict[int, object]] = {}
    configs: Dict[str, Dict[int, object]] = {}
    scoring = {}
    for setting in DURATION_SETTINGS:
        probes[setting], configs[setting] = {}, {}
        for seed in PROBE_SEEDS:
            cfg = stage1_config(seed, data=data, duration=setting)
            tag = _probe_tag(setting, seed)
            job = train_arm(f"probe/{tag}/training", cfg, NUM_SUBEPOCHS,
                            keep_epochs=range(1, NUM_SUBEPOCHS + 1), time_rqmt=alloc,
                            alias_prefix=alias)
            probes[setting][seed], configs[setting][seed] = job, cfg
        scoring[setting] = probe_scoring_reads(probes[setting], setting, reads=reads, alias=alias)
    return {"probes": probes, "configs": configs, "reads": scoring, "tau": list(TAU_SCHEDULE),
            "alloc_hours": alloc}


def build_wave(data: Dict[str, Any], *, phi_c_source, duration: Optional[str] = None,
               num_subepochs: Optional[int] = None, alias: Optional[str] = ALIAS) -> dict:
    """The wave at ``duration`` (default :data:`WAVE_DURATION_SETTING`) and ``num_subepochs``
    (default :data:`WAVE_NUM_SUBEPOCHS`, A10); refuses while either is undecided.  One training job
    per arm (module docstring, port changes).

    The exact reruns ``em_sNN_rerun`` (:data:`RERUN_SEEDS`) are the restarts' configs, same seed,
    plus the config key :data:`RERUN_KEY` ``= 1``.  RETURNN never reads that key (no behaviour
    change); it only enters the job hash, so each rerun is a separate ``ReturnnTrainingJob`` and the
    identity band (``GenMargSelectionJob``) measures two independent trainings, as the source's
    separate "refs" pack did.  Without it Sisyphus would return the restart's own job.

    Every wave arm's config also carries :data:`WAVE_KEY` ``= 1``, inert in the same way.  In the
    source the probe (``phifirst_probe_training``) and the wave (``packed_blankfree_training`` packs)
    are separate trainings; at ``num_subepochs = 4`` the configs of restarts ``em_s01`` / ``em_s02``
    would otherwise equal the probe's seeds 1 / 2 of the same duration setting and merge with them
    (and the unhashed ``keep_epochs`` of whichever was built first would win).  Asserted: the 24 arms
    are 24 distinct jobs."""
    from ..training.jobs import train_arm

    duration = WAVE_DURATION_SETTING if duration is None else duration
    assert duration in ("durinit", "durfrz"), (
        f"WAVE_DURATION_SETTING = {duration!r}: the wave's duration setting (A9) is set from the probe "
        "read's verdict and recorded in the phase file before the wave; the wave does not build "
        "before that")
    n_sub = WAVE_NUM_SUBEPOCHS if num_subepochs is None else num_subepochs
    assert n_sub is not None and int(n_sub) >= NUM_SUBEPOCHS and int(n_sub) % 4 == 0, (
        f"WAVE_NUM_SUBEPOCHS = {n_sub!r}: the wave's sub-epoch count (A10, a multiple of 4) is set "
        "from the A10 read's verdict and recorded in the phase file before the wave; the wave does "
        "not build before that")
    n_sub = int(n_sub)
    corpus = null_corpus(data, NULL_GRANULARITY, alias=alias)
    phi_c = phi_c_phi(phi_c_source, alias=alias)
    alloc = alloc_hours(n_sub)
    fz = {"data": data, "duration": duration, "num_subepochs": n_sub}
    groups: Dict[str, Dict[str, object]] = {
        "restarts": {arm_name(s): stage1_config(s, **fz) for s in RESTART_SEEDS},
        "nulls": {arm_name(s, "null"): stage1_config(s, units=corpus.out_hdfs, **fz) for s in NULL_SEEDS},
        "references": {arm_name(s, "phic"): stage1_config(s, data=data, reverse_init=phi_c,
                                                          num_subepochs=n_sub) for s in PHI_C_SEEDS},
    }
    for s in RERUN_SEEDS:
        rerun = stage1_config(s, **fz)
        assert RERUN_KEY not in rerun.config, RERUN_KEY
        rerun.config[RERUN_KEY] = 1
        groups["references"][arm_name(s, "rerun")] = rerun
    assert sum(len(a) for a in groups.values()) == 24, {g: sorted(a) for g, a in groups.items()}
    for arms in groups.values():
        for cfg in arms.values():
            assert WAVE_KEY not in cfg.config, WAVE_KEY
            cfg.config[WAVE_KEY] = 1
    out: dict = {"null_corpus": corpus, "phi_c": phi_c, "tau": tau_schedule(n_sub), "duration": duration,
                 "num_subepochs": n_sub, "alloc_hours": alloc, "configs": groups, "restarts": {},
                 "nulls": {}, "references": {}}
    for group, arms in groups.items():
        for a, cfg in arms.items():
            out[group][a] = train_arm(f"wave/{a}/training", cfg, n_sub, keep_epochs=(n_sub,),
                                      time_rqmt=alloc, alias_prefix=alias)
    jobs = [j for g in ("restarts", "nulls", "references") for j in out[g].values()]
    assert len({j.job_id() for j in jobs}) == len(jobs) == 24, "two wave arms collapsed into one job"
    out["selection"] = selection_reads(out["restarts"], out["nulls"], out["references"], last=n_sub,
                                       reads=reads_bed(data), alias=alias)
    return out
