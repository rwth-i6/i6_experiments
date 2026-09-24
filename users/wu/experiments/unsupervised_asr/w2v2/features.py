"""New in the port (replaces speech-llm c49559ce src/speech_llm/sae/av_states.py ``AvStatesJob`` with
``tap="encoder"``, ``av_checkpoint=None``, and src/speech_llm/sae/emc/feature_dump.py ``L15FeatureHdfJob``).

The frozen layer-15 feature store of phase 4a: wav2vec2-large-lv60 ``hidden_states[15]``, 1024-d
float16 at 50 Hz, one RETURNN HDF per dump, written by a plain i6_core ``ReturnnForwardJobV2``
whose RETURNN-side ``get_model`` / ``forward_step`` / callback live in :mod:`.forward`.

The source computed the same tensor in two steps: ``AvStatesJob`` (a child script that built the
whole ``SpeechLmV2`` including a Qwen3 decoder it never ran, and pickled ``{tag: float16 [T, 1024]}``)
and ``L15FeatureHdfJob`` (a byte-for-byte repack of those pickles into HDFs).  Here one forward
writes the HDF directly; the model-side preprocessing is the source's (float32 waveform, batch 1,
per-utterance waveform normalisation, float32 encoder).

Audio source (brief change 2026-09-23, deliberate): the source decoded the HF Ogg ``-q 3`` dataset
(``TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb``) with ``datasets.Audio``; the port reads the
i6 LibriSpeech ogg zips (:func:`..data.librispeech.get_ogg_zip`) through the common
``OggZipDataset`` + ``AudioRawDatastream`` setup of posterior_hmm (raw features, no peak
normalisation, no pre-emphasis) inside a ``MetaDataset``.  The zips' Ogg files are encoded from the
openslr FLAC by the pinned ffmpeg 7.1.1 (:func:`..data.librispeech.get_bliss_corpus`), which
reproduces the banked HF Ogg decode at the float floor (dev-other subset check), so the features are
expected to match the banked HDFs up to forward nondeterminism.  Seq tags ``<corpus>/<utt>/<utt>``
are stored as bare ids.

Each dump also writes what the source's ``AvStatesJob`` wrote beside the states: the per-utterance
``{tag: float32 [2, D]}`` mean/std pickle (``SpeakerEtaJob`` reads row 0) and the global float32
``[2, D]`` mean/std over the dumped frames (``QuantizeStatesJob`` standardizes with it).

Which utterances, in which order (:class:`L15ForwardSeqOrderJob`):

* train-clean-100 shard k of 4: the banked shard (``AvStatesJob.Dsynh5MqmgjY`` rows ``[k::4]`` of
  the HF split row order), from the shipped list :func:`..data.librispeech.get_train_shard_ids`, in
  its order (the banked HDF order, sorted within the shard), so shard membership and order equal
  the banked dumps;
* dev-clean, dev-other: the whole subset zip, sorted (``L15FeatureHdfJob.6ChpQYsQh1VI``);
* the 10 h seed: ``seed_10h_ids.txt`` in its row order out of the train-clean-100 zip (the ``train``
  split of ``AvStatesJob.c4Ak1rACchRC``); its global statistics and frames are the k-means fit input.

Pinned: :data:`W2V2_REVISION` of :data:`W2V2_REPO_ID`, with the sha256 of each file loaded
(:data:`W2V2_SHA256`).
"""

from __future__ import annotations

import os
from functools import cache
from typing import Dict, List, Optional, Tuple

from sisyphus import Job, Task, tk

from .. import default_tools
from ..data.hf_hub import DownloadHuggingFaceSnapshotJob
from .forward import AUDIO_KEY, L15_DIM

__all__ = [
    "W2V2_REPO_ID",
    "W2V2_REVISION",
    "W2V2_SHA256",
    "L15_DIM",
    "L15_FRAME_HZ",
    "L15_OUTPUT_FILES",
    "NUM_TRAIN_SHARDS",
    "get_w2v2_model_dir",
    "L15ForwardSeqOrderJob",
    "build_l15_forward_config",
    "l15_forward_job",
    "get_l15_feature_dumps",
]

_alias_prefix = "sae/4a/feats"

# -------------------------------------------------------------------------------------------------
# pinned model
# -------------------------------------------------------------------------------------------------
W2V2_REPO_ID = "facebook/wav2vec2-large-lv60"
W2V2_REVISION = "0cde644b64dac88d8416bec1c92a4099b850ba0b"
W2V2_SHA256: Dict[str, str] = {
    "config.json": "89e251162bc1f1dc66cb1d68f5ce3782b3d4bf2e10709bd867122bca47102b9b",
    "preprocessor_config.json": "c403ce09975b90dff0dd8302c42d422e9de1f166cd7772df23490069893cb0cf",
    "pytorch_model.bin": "1d095321ee151b9cbeeb2f325d104bf1813b795c9bbbee58efcde34f701d0386",
}

L15_FRAME_HZ = 50.0
#: the output files of every dump (RFJv2 ``out_files`` keys)
L15_OUTPUT_FILES = ("feats.hdf", "perutt_stats.pkl", "global_stats.npy", "feats.stats.txt")
#: train-clean-100 is dumped as the 4 banked shards (rows [k::4] of the HF Ogg train split,
#: AvStatesJob.Dsynh5MqmgjY), read from the shipped lists (data.librispeech.get_train_shard_ids)
NUM_TRAIN_SHARDS = 4

# batch 1 as the source: ``max_seqs=1`` alone decides the batching; the sample budget only has to
# exceed the longest utterance (LibriSpeech < 36 s) so that no warning path is taken.
_MAX_SEQS = 1
_BATCH_SIZE_SAMPLES = 16_000 * 60

# rqmt of the source's sharded encoder dump (AvStatesJob, shards > 1)
_RQMT = {"time": 8, "mem": 64, "cpu": 4, "gpu_mem": 24}


@cache
def get_w2v2_model_dir() -> tk.Path:
    """``from_pretrained``-ready directory of the pinned wav2vec2-large-lv60 checkpoint."""
    job = DownloadHuggingFaceSnapshotJob(
        repo_id=W2V2_REPO_ID, repo_type="model", revision=W2V2_REVISION, files=W2V2_SHA256,
        hf_home=default_tools.HF_HOME,
    )
    job.add_alias(f"models/wav2vec2-large-lv60_{W2V2_REVISION[:8]}")
    return job.out_dir


# -------------------------------------------------------------------------------------------------
# which utterances, in which order
# -------------------------------------------------------------------------------------------------
class L15ForwardSeqOrderJob(Job):
    """The utterance subset and forward order of one dump, as RETURNN dataset files.

    ``out_seq_list`` (one zip seq tag per line) is the ``OggZipDataset`` ``segment_file``;
    ``out_seq_order`` is a python dict literal ``{seq tag: rank}`` over the selected tags for
    ``seq_ordering="sorted"`` + ``seq_order_seq_lens_file`` (a stable argsort of the rank), so the
    forward visits exactly the selected tags in the chosen order.  Selection and order are on the
    bare utterance ids (:func:`..data.ogg_zip.utt_id_of_segment`).

    :param ogg_zip: one LibriSpeech ogg zip (:func:`..data.librispeech.get_ogg_zip`).
    :param ids: a json list or a text file (one bare id per line) of the utterances to dump.
    :param shard: ``(k, n)``: dump every n-th utterance of the sorted ids starting at k instead
        (exactly one of ``ids`` / ``shard``; ``(0, 1)`` is the whole zip).
    :param order: ``"sorted"`` (id order) or ``"given"`` (the order of ``ids``; sorted for ``shard``).
    """

    def __init__(self, *, ogg_zip: tk.Path, ids: Optional[tk.Path] = None,
                 shard: Optional[Tuple[int, int]] = None, order: str = "sorted"):
        super().__init__()
        if (ids is None) == (shard is None):
            raise ValueError("give exactly one of ids / shard")
        if shard is not None:
            k, n = shard
            if not (n >= 1 and 0 <= k < n):
                raise ValueError(f"invalid shard {shard}")
            shard = (int(k), int(n))
        if order not in ("sorted", "given"):
            raise ValueError(f"order must be 'sorted' or 'given', got {order!r}")
        self.ogg_zip = ogg_zip
        self.ids = ids
        self.shard = shard
        self.order = order
        self.out_seq_list = self.output_path("seq_list.txt")
        self.out_seq_order = self.output_path("seq_order.py")
        self.out_stats = self.output_path("seq_order.stats.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json

        from ..data.ogg_zip import read_ogg_zip_index, utt_id_of_segment

        seq_names = [e["seq_name"] for e in read_ogg_zip_index(self.ogg_zip.get_path())]
        seq_of = {utt_id_of_segment(t): t for t in seq_names}
        assert len(seq_of) == len(seq_names), "duplicate utterance ids in the zip"
        if self.shard is not None:
            k, n = self.shard
            chosen = sorted(seq_of)[k::n]
        else:
            path = self.ids.get_path()
            with open(path) as fh:
                text = fh.read()
            chosen = json.loads(text) if path.endswith(".json") else [l.strip() for l in text.splitlines() if l.strip()]
            chosen = [str(t) for t in chosen]
            assert len(set(chosen)) == len(chosen), "duplicate ids requested"
            missing = set(chosen) - set(seq_of)
            assert not missing, f"{len(missing)} requested ids not in the zip, e.g. {sorted(missing)[:3]}"
        assert chosen, "empty selection"
        if self.order == "sorted":
            chosen = sorted(chosen)

        with open(self.out_seq_list.get_path(), "w") as fh:
            fh.write("\n".join(seq_of[u] for u in chosen) + "\n")
        with open(self.out_seq_order.get_path(), "w") as fh:
            fh.write(repr({seq_of[u]: i for i, u in enumerate(chosen)}) + "\n")
        lines = [
            f"ogg zip = {self.ogg_zip.get_path()} ({len(seq_names)} utts)",
            f"selection = {'every %d-th sorted id from %d' % (self.shard[1], self.shard[0]) if self.shard else self.ids.get_path()}"
            f"   order = {self.order}",
            f"selected = {len(chosen)}   first = {chosen[:3]}",
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


# -------------------------------------------------------------------------------------------------
# the forward
# -------------------------------------------------------------------------------------------------
def _forward_serializer(model_args: Dict, extern_data: Dict, callback_args: Dict):
    """``Collection`` for the L15 forward (no local package copy: the recipe dir is on sys.path)."""
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
    from i6_experiments.common.setups.serialization import Import, NonhashedCode, PartialImport
    from returnn.util.pprint import pformat

    pkg = __package__.rsplit(".", 1)[0]
    return Collection(
        serializer_objects=[
            NonhashedCode(f"extern_data = {pformat(extern_data)}\n"),
            PartialImport(
                code_object_path=f"{pkg}.w2v2.forward.get_model",
                unhashed_package_root=pkg,
                hashed_arguments=model_args,
                unhashed_arguments={},
                import_as="get_model",
            ),
            Import(
                code_object_path=f"{pkg}.w2v2.forward.l15_forward_step",
                unhashed_package_root=pkg,
                import_as="forward_step",
            ),
            PartialImport(
                code_object_path=f"{pkg}.w2v2.forward.L15FeatureCallback",
                unhashed_package_root=pkg,
                hashed_arguments=callback_args,
                unhashed_arguments={},
                import_as="forward_callback",
            ),
        ],
        make_local_package_copy=False,
        packages=None,
    )


def build_l15_forward_config(
    *,
    ogg_zip: tk.Path,
    seq_list: tk.Path,
    seq_order: tk.Path,
    hf_model_dir: tk.Path,
    expected_num_seqs: int = 0,
):
    """ReturnnConfig of one L15 dump: ``MetaDataset`` over one ``OggZipDataset`` with raw audio."""
    from i6_core.returnn.config import ReturnnConfig
    from i6_experiments.common.setups.returnn.datasets import MetaDataset, OggZipDataset
    from i6_experiments.common.setups.returnn.datastreams.audio import AudioRawDatastream, ReturnnAudioRawOptions

    # the source's waveform: no peak normalisation, no pre-emphasis
    audio = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=False, preemphasis=None),
    )
    zip_dataset = OggZipDataset(
        files=[ogg_zip],
        audio_options=audio.as_returnn_audio_opts(),
        segment_file=seq_list,
        seq_ordering="sorted",
        additional_options={"seq_order_seq_lens_file": seq_order},
    )
    data = MetaDataset(
        data_map={AUDIO_KEY: ("zip_dataset", "data")},
        datasets={"zip_dataset": zip_dataset},
        seq_order_control_dataset="zip_dataset",
    )
    extern_data = {AUDIO_KEY: audio.as_returnn_extern_data_opts()}
    config = {
        "forward_data": data.as_returnn_opts(),
        "batch_size": _BATCH_SIZE_SAMPLES,
        "max_seqs": _MAX_SEQS,
    }
    post_config = {"backend": "torch", "watch_memory": True}
    return ReturnnConfig(
        config=config,
        post_config=post_config,
        python_epilog=[
            _forward_serializer(
                {"hf_model_dir": hf_model_dir, "encoder_layer": 15},
                extern_data,
                {"feature_dim": L15_DIM, "expected_num_seqs": int(expected_num_seqs)},
            )
        ],
    )


def l15_forward_job(
    *,
    name: str,
    subset: str,
    ids: Optional[tk.Path] = None,
    shard: Optional[Tuple[int, int]] = None,
    order: str = "sorted",
    expected_num_seqs: int = 0,
):
    """The ``ReturnnForwardJobV2`` of one dump (outputs :data:`L15_OUTPUT_FILES`)."""
    from i6_core.returnn.forward import ReturnnForwardJobV2

    from ..data.librispeech import get_ogg_zip

    ogg_zip = get_ogg_zip(subset)
    seq = L15ForwardSeqOrderJob(ogg_zip=ogg_zip, ids=ids, shard=shard, order=order)
    seq.add_alias(f"{_alias_prefix}/{name}/seq_order")
    job = ReturnnForwardJobV2(
        model_checkpoint=None,
        returnn_config=build_l15_forward_config(
            ogg_zip=ogg_zip, seq_list=seq.out_seq_list, seq_order=seq.out_seq_order,
            hf_model_dir=get_w2v2_model_dir(), expected_num_seqs=expected_num_seqs,
        ),
        returnn_python_exe=default_tools.RETURNN_EXE,
        returnn_root=default_tools.RETURNN_ROOT,
        output_files=list(L15_OUTPUT_FILES),
        device="gpu",
        time_rqmt=_RQMT["time"],
        mem_rqmt=_RQMT["mem"],
        cpu_rqmt=_RQMT["cpu"],
    )
    job.rqmt["gpu_mem"] = _RQMT["gpu_mem"]
    job.add_alias(f"{_alias_prefix}/{name}/forward")
    return job


@cache
def get_l15_feature_dumps() -> Dict[str, List]:
    """The phase-4a dumps: ``{"train": [4 jobs], "dev-clean": [job], "dev-other": [job], "seed": [job]}``."""
    from ..data.gold import SEED_10H_NUM_UTTS, get_seed_10h_ids
    from ..data.librispeech import EXPECTED_UTTS, TRAIN_SHARD_NUM_UTTS, get_train_shard_ids

    assert len(TRAIN_SHARD_NUM_UTTS) == NUM_TRAIN_SHARDS
    assert sum(TRAIN_SHARD_NUM_UTTS) == EXPECTED_UTTS["train-clean-100"]
    dumps = {
        "train": [
            l15_forward_job(
                name=f"train.shard{k}", subset="train-clean-100", ids=get_train_shard_ids(k), order="given",
                expected_num_seqs=TRAIN_SHARD_NUM_UTTS[k],
            )
            for k in range(NUM_TRAIN_SHARDS)
        ],
        "seed": [
            l15_forward_job(
                name="seed_10h", subset="train-clean-100", ids=get_seed_10h_ids(), order="given",
                expected_num_seqs=SEED_10H_NUM_UTTS,
            )
        ],
    }
    for s in ("dev-clean", "dev-other"):
        dumps[s] = [
            l15_forward_job(name=s, subset=s, shard=(0, 1), order="sorted",
                            expected_num_seqs=EXPECTED_UTTS[s])
        ]
    for s, jobs in dumps.items():
        for k, job in enumerate(jobs):
            tk.register_output(f"{_alias_prefix}/{s}.shard{k}.stats.txt", job.out_files["feats.stats.txt"])
    return dumps


