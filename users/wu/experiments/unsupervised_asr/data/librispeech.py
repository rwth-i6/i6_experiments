"""New in the port (replaces the train-clean-100 HF Ogg dataset of speech-llm c49559ce
src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/data/huggingface.py and the
parquet-based ``LibriSpeechSplitIdsJob`` of src/speech_llm/sae/emc/feature_dump.py; the audio loading
follows i6_experiments 5207c8adf users/wu/experiments/posterior_hmm/data/common.py).

LibriSpeech as the phase-4a setup reads it -- a deliberate change of audio source (brief change
2026-09-23): the banked runs read ``TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb`` (the
``openslr/librispeech_asr`` parquet re-encoded to Ogg Vorbis ``-q 3``); the port reads the i6
LibriSpeech corpora of ``i6_experiments.common.datasets.librispeech``:

* :func:`get_bliss_corpus` -- the FLAC bliss corpora of ``get_bliss_corpus_dict(audio_format="flac")``
  (openslr downloads, ``DownloadLibriSpeechCorpusJob``) transcoded to 16 kHz Ogg Vorbis by
  ``BlissChangeEncodingJob`` with the arguments ``get_bliss_corpus_dict(audio_format="ogg")`` uses
  (``output_format="ogg"``, ``codec="libvorbis"``, ``sample_rate=16000``; :data:`.ffmpeg_pin.ENCODE_KWARGS`),
  plus ``ffmpeg_binary`` = :func:`get_checked_ffmpeg_binary` and ``hash_binary=True``.  The common
  getter runs whatever ``ffmpeg`` is on the job PATH and hashes none; the encode is the only step
  where the binary enters the audio, so the pin is placed on that job;
* :func:`get_checked_ffmpeg_binary` -- ``settings.py``'s required ``FFMPEG_BINARY``
  (``default_tools.get_ffmpeg_binary``) after :class:`.ffmpeg_pin.FfmpegPinCheckJob` has checked on
  dev-other that it reproduces the banked audio; the encode jobs depend on the check;
* :func:`get_ogg_zip` -- those corpora as RETURNN ogg zips, built as posterior_hmm's ``get_zip``
  (``BlissToOggZipJob``, ``no_conversion=True``: the Ogg files are copied, no ffmpeg call) with
  ``default_tools.SAE_PYTHON_EXE`` / ``RETURNN_ROOT``; read by the L15 forward (``OggZipDataset``)
  and by the joint rVAD job (:mod:`.ogg_zip`);
* :class:`LibriSpeechSplitIdsJob` / :func:`get_split_ids` -- the SORTED bare utterance ids of one
  subset, read off the zip's segment names ``<corpus>/<utt>/<utt>`` (:func:`.ogg_zip.utt_id_of_segment`).
  The CV holdout, the seed selection, the unit-store coverage and gold join on these ids;
* :func:`get_train_shard_ids` -- the four shipped train-clean-100 shard lists of the banked L15 dump
  (``L15FeatureHdfJob.e2athsQ218Og``): shard k holds HF split rows ``[k::4]``
  (``TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb`` ``train``), listed in the banked HDF order
  (sorted within the shard), with the sha256 of each file in :data:`TRAIN_SHARD_IDS_SHA256`.

Audio: with ffmpeg 7.1.1 (the conda build that wrote the banked HF Ogg) and the i6 command line
(``-c:a libvorbis -ar 16000``, libvorbis default quality 3), the re-encode of the openslr FLAC
matches the banked decode at the float decode floor (review check, 159/159 dev-other utterances,
135.6-136.4 dB).  A different ffmpeg build changes most waveforms (another cluster's Lavc61.3.100:
median 62 dB, minimum 19.9 dB on dev-other).
"""

from __future__ import annotations

import os
from functools import cache
from typing import Dict, List

from sisyphus import Job, Task, tk

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict

from .. import default_tools
from .ogg_zip import read_ogg_zip_index, utt_id_of_segment

__all__ = [
    "LIBRISPEECH_SUBSETS",
    "EXPECTED_UTTS",
    "get_checked_ffmpeg_binary",
    "get_bliss_corpus",
    "get_ogg_zip",
    "LibriSpeechSplitIdsJob",
    "get_split_ids",
    "TRAIN_SHARD_IDS_FILES",
    "TRAIN_SHARD_IDS_SHA256",
    "TRAIN_SHARD_NUM_UTTS",
    "get_train_shard_ids",
]

#: alias prefix of the corpus jobs (posterior_hmm's "corpora" prefix)
_OUTPUT_PREFIX = "corpora"

#: the LibriSpeech subsets phase 4a reads
LIBRISPEECH_SUBSETS = ("train-clean-100", "dev-clean", "dev-other")

# Expected utterance counts (dev/test: setup report §1; train-clean-100: AvStatesJob.Dsynh5MqmgjY
# ``train 28539 utts``, the CV holdout's 28,539).
EXPECTED_UTTS = {
    "train-clean-100": 28539,
    "dev-clean": 2703,
    "dev-other": 2864,
    "test-clean": 2620,
    "test-other": 2939,
}


def _check_subset(subset: str) -> None:
    if subset not in LIBRISPEECH_SUBSETS:
        raise ValueError(f"subset must be one of {LIBRISPEECH_SUBSETS}, got {subset!r}")


def _flac_corpus(subset: str) -> tk.Path:
    """The FLAC bliss corpus of one LibriSpeech subset (the common getter)."""
    return get_bliss_corpus_dict(audio_format="flac", output_prefix=_OUTPUT_PREFIX)[subset]


@cache
def get_checked_ffmpeg_binary() -> tk.Path:
    """The pinned ffmpeg after its run-time check (:class:`.ffmpeg_pin.FfmpegPinCheckJob` on the
    dev-other FLAC corpus): the check job's ``out_ffmpeg_binary`` wrapper.  ``settings.py``'s
    ``FFMPEG_PIN_ACCEPT`` (``default_tools.get_ffmpeg_pin_accept``) is passed as ``accept_label``."""
    from .ffmpeg_pin import FfmpegPinCheckJob

    job = FfmpegPinCheckJob(ffmpeg_binary=default_tools.get_ffmpeg_binary(), corpus_file=_flac_corpus("dev-other"),
                            accept_label=default_tools.get_ffmpeg_pin_accept())
    job.add_alias(os.path.join(_OUTPUT_PREFIX, "LibriSpeech", "ffmpeg_pin_check"))
    return job.out_ffmpeg_binary


@cache
def get_bliss_corpus(subset: str) -> tk.Path:
    """The 16 kHz Ogg Vorbis bliss corpus of one LibriSpeech subset, encoded by the pinned, checked ffmpeg."""
    from i6_core.audio.encoding import BlissChangeEncodingJob

    from .ffmpeg_pin import ENCODE_KWARGS

    _check_subset(subset)
    job = BlissChangeEncodingJob(
        corpus_file=_flac_corpus(subset),
        **ENCODE_KWARGS,
        ffmpeg_binary=get_checked_ffmpeg_binary(),
        hash_binary=True,
    )
    job.add_alias(os.path.join(_OUTPUT_PREFIX, "LibriSpeech", "ogg_conversion_pinned_ffmpeg", subset))
    return job.out_corpus


@cache
def get_ogg_zip(subset: str) -> tk.Path:
    """The RETURNN ogg zip of one LibriSpeech subset (seq tags ``<subset>/<utt>/<utt>``)."""
    from i6_core.returnn.oggzip import BlissToOggZipJob

    job = BlissToOggZipJob(
        bliss_corpus=get_bliss_corpus(subset),
        no_conversion=True,  # the corpus is already Ogg: the files are copied
        returnn_python_exe=default_tools.SAE_PYTHON_EXE,
        returnn_root=default_tools.RETURNN_ROOT,
    )
    job.add_alias(os.path.join(_OUTPUT_PREFIX, "LibriSpeech", "ogg_zip", subset))
    return job.out_ogg_zip


class LibriSpeechSplitIdsJob(Job):
    """{subset: [utt id, ...]} (sorted bare ids) off the LibriSpeech ogg zips' segment names.

    :param ogg_zips: ``{subset: ogg zip}`` (:func:`get_ogg_zip`); the source took ``hf_home`` and
        read the parquet ``id`` column.  Subsets listed in :data:`EXPECTED_UTTS` are count-checked.
    """

    def __init__(self, *, ogg_zips: Dict[str, tk.Path]):
        super().__init__()
        self.ogg_zips = {s: ogg_zips[s] for s in sorted(ogg_zips)}
        self.out_ids = self.output_path("split_ids.json")
        self.out_per_split = {s: self.output_path(f"ids.{s}.json") for s in self.ogg_zips}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json

        out: Dict[str, List[str]] = {}
        for s, zip_path in self.ogg_zips.items():
            ids = [utt_id_of_segment(e["seq_name"]) for e in read_ogg_zip_index(zip_path.get_path())]
            assert len(set(ids)) == len(ids), f"{s}: duplicate ids"
            if s in EXPECTED_UTTS:
                assert len(ids) == EXPECTED_UTTS[s], f"{s}: {len(ids)} ids, expected {EXPECTED_UTTS[s]}"
            out[s] = sorted(ids)
            with open(self.out_per_split[s].get_path(), "w") as fh:
                json.dump(out[s], fh)
            print(f"{s}: {len(ids)} ids", flush=True)
        with open(self.out_ids.get_path(), "w") as fh:
            json.dump(out, fh)


@cache
def _split_ids_job() -> LibriSpeechSplitIdsJob:
    job = LibriSpeechSplitIdsJob(ogg_zips={s: get_ogg_zip(s) for s in LIBRISPEECH_SUBSETS})
    job.add_alias("datasets/LibriSpeech/split_ids")
    return job


def get_split_ids(split: str) -> tk.Path:
    """Sorted json id list of ``"train-clean-100"``, ``"dev-clean"`` or ``"dev-other"``."""
    _check_subset(split)
    return _split_ids_job().out_per_split[split]


#: the banked train-clean-100 shards of the L15 dump (``L15FeatureHdfJob.e2athsQ218Og``
#: ``feats.train-clean-100.shard{k}.hdf`` tags; = HF ``train`` rows ``[k::4]`` of ``OYvh9012Pgkb``,
#: sorted within the shard), one id per line in the banked HDF order
TRAIN_SHARD_IDS_FILES = tuple(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), f"train_clean_100_shard{k}_ids.txt") for k in range(4)
)
TRAIN_SHARD_IDS_SHA256 = (
    "e8af8370f965973e0c14de0a5f7db73089f6372db538907620c4ff80abaa804f",
    "b18b233b2f20243ce601029a081367b07c00a6204c74823521186779368877e8",
    "db1f8583be18e884158e6730054fb9aeb7c964f7c92858335f8c7724bdef3c1f",
    "e89139bcfd59a533110c773f4ccb8951eba3774ee87a24bd0cc4190b91d03755",
)
TRAIN_SHARD_NUM_UTTS = (7135, 7135, 7135, 7134)


def get_train_shard_ids(k: int) -> tk.Path:
    """The shipped id list of banked train-clean-100 shard ``k`` (one id per line, banked order)."""
    if k not in range(len(TRAIN_SHARD_IDS_FILES)):
        raise ValueError(f"shard must be in 0..{len(TRAIN_SHARD_IDS_FILES) - 1}, got {k}")
    return tk.Path(
        TRAIN_SHARD_IDS_FILES[k],
        hash_overwrite=f"unsupervised_asr_train_clean_100_shard{k}_ids_{TRAIN_SHARD_IDS_SHA256[k][:16]}",
    )
