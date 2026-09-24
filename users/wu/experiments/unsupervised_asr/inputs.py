"""Every input of the phase-4a (EMC) setup, wired once from public raw sources.

One call of :func:`get_inputs` builds the whole input graph through the stage-1 getters and jobs and
returns an :class:`Inputs` that the entry points in ``config/`` consume.  Every getter here is
``lru_cache``-d, so one graph build creates each job once (sisyphus would also merge equal jobs by
hash; the cache keeps the aliases and registrations single).

Chain (the package module that owns each step in brackets):

* audio: openslr LibriSpeech FLAC -> 16 kHz Ogg Vorbis with the pinned ffmpeg 7.1.1 -> ogg zips for
  train-clean-100, dev-clean and dev-other (``data.librispeech.get_ogg_zip``);
* features: wav2vec 2.0 large-lv60 ``hidden_states[15]`` dumps by ``ReturnnForwardJobV2``, train in
  the four shipped shards, plus the 10 h seed and both dev sets (``w2v2.features``);
* units: the k-means unit store (``w2v2.units.get_units_store``);
* streams: joint rVAD masking of features and units (``data.vad.BlankfreeVadHdfJob``), checked
  against the banked totals ``BANKED_VAD_COUNTS`` (report-only under ``FFMPEG_PIN_ACCEPT``);
* ids / split: the sorted train-clean-100 ids (``data.librispeech.get_split_ids``) -> the seed-0 1 %
  CV holdout (``data.splits.CvHoldoutSplitJob``, 28,254 / 285);
* gold: MFA phones of dev-clean + dev-other (``data.gold.GoldPhonesJob``; PER reference only);
* speaker eta (``data.speaker.SpeakerEtaJob``) from the dumps' per-utterance statistics;
* phone prior: the Witten-Bell trigram on the unbiased ``SampleLinesJob`` window
  (``lm.phone_prior.get_phone_prior``);
* theta's flat init: ``training.init.FlatRecognizerInitJob(net_args=NET_ARGS, seed=0)``;
* the duration prior: ``BlankfreeDurationPriorMeanJob`` on the train stream
  (``reverse_model.phi_first.duration_prior_json``, the bed's own rho and frame rate);
* word graphs: :func:`get_graph` = ``lm.hlg.get_hlg(kind)``.

The ANALYSIS-ONLY seed inputs (they use transcripts: the 10 h seed's MFA gold, its split and its
target HDF) are a separate getter, :func:`get_seed_inputs`, so that no main-line entry point builds
them by accident.

The banked dev set of every arm is the TRAIN stream's HDFs under the CV-holdout segment list, so
``data["dev_*"]`` are the train shards (``training.arms`` module doc).  The dev-other / dev-clean
streams (:meth:`Inputs.dev_stream`) are the PER-read inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

from sisyphus import tk

__all__ = ["Inputs", "SeedInputs", "get_inputs", "get_seed_inputs", "get_graph", "DEV_SPLITS"]

#: the dev splits the VAD job masks and the PER reads score
DEV_SPLITS = ("dev-clean", "dev-other")


@dataclass(frozen=True)
class Inputs:
    """The main-line inputs (label-free except ``gold``, which only the PER reads use).

    :ivar data: ``training.build_train_config``'s data dict (keys ``training.arms.ARM_DATA_KEYS``).
    :ivar vad: the ``BlankfreeVadHdfJob`` (per-split feature / units / raw-index / orig-length HDFs).
    :ivar cv_split: the train-clean-100 ``CvHoldoutSplitJob``.
    :ivar gold: ``GoldPhonesJob.out_gold`` (dev-clean + dev-other MFA phones).
    :ivar eta_job: the ``SpeakerEtaJob``.
    :ivar prior_npz: the phone trigram prior.
    :ivar flat_checkpoint: theta's flat init.
    :ivar duration_prior: the duration prior's ``prior.json``.
    :ivar feature_dumps: ``w2v2.features.get_l15_feature_dumps()``.
    :ivar units_store: the packed unit store.
    :ivar ogg_zips: ``{subset: ogg zip}``.
    """

    data: Dict[str, Any]
    vad: Any
    cv_split: Any
    gold: tk.Path
    eta_job: Any
    prior_npz: tk.Path
    flat_checkpoint: tk.Path
    duration_prior: tk.Path
    feature_dumps: Dict[str, List[Any]]
    units_store: tk.Path
    ogg_zips: Dict[str, tk.Path]

    def dev_stream(self, split: str) -> Dict[str, List[tk.Path]]:
        """The rVAD-masked stream of ``split`` (``"dev-clean"`` / ``"dev-other"``):
        ``{"features", "originals", "units"}``, each a list of per-shard HDFs."""
        assert split in DEV_SPLITS, split
        return {
            "features": list(self.vad.out_feature_hdfs[split]),
            "originals": list(self.vad.out_orig_length_hdfs[split]),
            "units": list(self.vad.out_units_hdfs[split]),
        }


@dataclass(frozen=True)
class SeedInputs:
    """ANALYSIS ONLY (uses transcripts): the 10 h seed's MFA gold, ids, split and target HDF.

    ``seed_inputs`` is the dict ``reverse_model.ladder`` takes (``gold_json``, ``ids_json``,
    ``train_segments``, ``cv_segments``).
    """

    gold_job: Any
    split: Any
    targets: Any

    @property
    def seed_inputs(self) -> Dict[str, tk.Path]:
        return {
            "gold_json": self.gold_job.out_gold,
            "ids_json": self.gold_job.out_ids,
            "train_segments": self.split.out_train_segments,
            "cv_segments": self.split.out_cv_segments,
        }


@lru_cache(maxsize=None)
def get_inputs() -> Inputs:
    """Build (once) and return the main-line inputs."""
    from .data.gold import GoldPhonesJob, get_mfa_alignments
    from .data.librispeech import get_ogg_zip, get_split_ids
    from .data.speaker import SpeakerEtaJob
    from .data.splits import CvHoldoutSplitJob
    from .data.vad import BANKED_VAD_COUNTS, BlankfreeVadHdfJob
    from .default_tools import get_ffmpeg_pin_accept
    from .lm.phone_prior import get_phone_prior
    from .reverse_model.phi_first import duration_prior_json
    from .training.config import NET_ARGS
    from .training.init import FlatRecognizerInitJob
    from .w2v2.features import get_l15_feature_dumps
    from .w2v2.units import get_units_store

    ogg_zips = {s: get_ogg_zip(s) for s in ("train-clean-100",) + DEV_SPLITS}
    dumps = get_l15_feature_dumps()
    units_store = get_units_store()

    feature_hdfs = {s: [j.out_files["feats.hdf"] for j in dumps[s]] for s in ("train",) + DEV_SPLITS}
    # the pinned-ffmpeg audio is sample-identical to the banked audio (review_data), so the banked
    # totals are a real check: a mismatch raises after the manifest is written.  Under an ffmpeg accept
    # label (a different audio generation) the totals are expected to move: the job then only reports
    # them against the banked ones (``out_counts_report``) and does not raise.
    report_only = get_ffmpeg_pin_accept() is not None
    vad = BlankfreeVadHdfJob(
        ogg_zips={"train": [ogg_zips["train-clean-100"]], **{s: [ogg_zips[s]] for s in DEV_SPLITS}},
        feature_hdfs=feature_hdfs,
        units_store=units_store,
        expected_counts=BANKED_VAD_COUNTS,
        **({"counts_report_only": True} if report_only else {}),
    )
    vad.add_alias("sae/4a/data/vad")

    cv_split = CvHoldoutSplitJob(ids_source=get_split_ids("train-clean-100"), label_key=None)
    cv_split.add_alias("sae/4a/data/cv_split")

    gold = GoldPhonesJob(mfa_dir=get_mfa_alignments("dev"))
    gold.add_alias("sae/4a/data/gold_dev")

    eta = SpeakerEtaJob(
        fit_perutt=[j.out_files["perutt_stats.pkl"] for j in dumps["train"]],
        dev_perutt=[j.out_files["perutt_stats.pkl"] for s in DEV_SPLITS for j in dumps[s]],
        dev_tags_json=gold.out_gold,
    )
    eta.add_alias("sae/4a/data/speaker_eta")

    flat = FlatRecognizerInitJob(net_args=NET_ARGS, seed=0)
    flat.add_alias("sae/4a/init/flat_s0")

    train = {
        "feature_hdfs": list(vad.out_feature_hdfs["train"]),
        "units_hdfs": list(vad.out_units_hdfs["train"]),
        "original_hdfs": list(vad.out_orig_length_hdfs["train"]),
    }
    data = {
        "train_feature_hdfs": train["feature_hdfs"],
        "train_units_hdfs": train["units_hdfs"],
        "train_original_hdfs": train["original_hdfs"],
        # the banked dev set: the train stream under the CV-holdout segments
        "dev_feature_hdfs": list(train["feature_hdfs"]),
        "dev_units_hdfs": list(train["units_hdfs"]),
        "dev_original_hdfs": list(train["original_hdfs"]),
        "train_segments": cv_split.out_train_segments,
        "dev_segments": cv_split.out_cv_segments,
        "prior_npz": get_phone_prior(),
        "eta_npz": eta.out_eta,
        "flat_checkpoint": flat.out_checkpoint,
    }
    return Inputs(
        data=data,
        vad=vad,
        cv_split=cv_split,
        gold=gold.out_gold,
        eta_job=eta,
        prior_npz=data["prior_npz"],
        flat_checkpoint=flat.out_checkpoint,
        duration_prior=duration_prior_json(data),
        feature_dumps=dumps,
        units_store=units_store,
        ogg_zips=ogg_zips,
    )


@lru_cache(maxsize=None)
def get_seed_inputs() -> SeedInputs:
    """ANALYSIS ONLY (uses transcripts): the seed gold (``SeedGoldPhonesJob``), its seed-0 1 % split
    (2,821 / 28) and its target HDF (``PhoneTargetHdfJob(label_key=None)``), as the source built them
    (``config_sae_4a_s0b_inits_v1``)."""
    from .data.gold import PhoneTargetHdfJob, SeedGoldPhonesJob, get_mfa_alignments, get_seed_10h_ids
    from .data.splits import CvHoldoutSplitJob

    gold = SeedGoldPhonesJob(seed_ids=get_seed_10h_ids(), mfa_dir=get_mfa_alignments("train-clean-100"))
    gold.add_alias("sae/4a/analysis_only/seed/gold")
    split = CvHoldoutSplitJob(ids_source=gold.out_ids, label_key=None)
    split.add_alias("sae/4a/analysis_only/seed/cv_split")
    targets = PhoneTargetHdfJob(labels_json=gold.out_gold, ids_json=gold.out_ids, label_key=None)
    targets.add_alias("sae/4a/analysis_only/seed/targets")
    return SeedInputs(gold_job=gold, split=split, targets=targets)


@lru_cache(maxsize=None)
def get_graph(kind: str, shuffled: bool = False) -> Dict[str, Any]:
    """The k2 word graph ``{"hlg", "stats", "resources", "expected_build"}`` of ``kind``
    (``lm.hlg.HLG_KINDS``: ``inhouse_3gram``, ``official_4gram``, ``official_3gram_1e7``)."""
    from .lm.hlg import get_hlg

    return get_hlg(kind, shuffled=shuffled)
