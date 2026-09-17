"""
Continue training the FZJ text-injection winner with GlowTTS+GL **TTS audio** added.

Why: the winner was trained on LS-960 audio + LM text injected as frozen per-phoneme mean-log-mel
(it has never heard synthetic speech). The old RZ CTC ``phrWiFtUTwmK`` instead got LS-960 + 1.3 TB of
GlowTTS+Griffin-Lim audio. Measured on the DLM hypothesis pass (random 20k, seed 0, same augmentation):

    real LS-960 audio:  winner 2.22%  vs  old CTC 5.02%
    GlowTTS audio:      winner 14.53% vs  old CTC 10.25%

i.e. the winner is 2.3x better on real speech and 1.4x worse on exactly the audio it never saw.
Paper 2 (2604.26514) claims the pseudo-speech encoder does *not* replace TTS (pure CTC 4.95 vs 4.15);
this run asks whether TTS audio still adds anything **on top of** text injection.

How, in one sentence: keep ``_train_tts_encoder`` exactly as the winner ran it and only replace the
``asr`` sub-dataset of its ``CombinedDataset`` with the LS+TTS ``DistributeFilesDataset`` that
Albert's own RZ recipe uses (``denoising_lm_2024.sis_recipe.tts_data``), initialising from the
winner's ``epoch.038.pt``.

Three things this deliberately does NOT do:

- It does not call ``tts_data.get_asr_tts_extended_task``: that asserts ``len(tts_files) == 750`` and
  would drag all 750 zips into the graph as missing inputs. We hold 100 on FZJ (see NUM_TTS_FILES).
- It does not touch ``tts_data.get_tts_oggzips`` / ``NUM_CORPUS_FILES``. The DLM hypothesis pipeline
  in ``dlm_on_winner.py`` depends on ``len(get_tts_oggzips()) == 750`` (only the length is used), and
  mutating either would re-hash the hypothesis bundles that are running. We build our own ``tk.Path``
  list with the **identical** ``hash_overwrite`` pinning instead, so the zips hash the same as on RZ
  regardless of where they live.
- It does not set ``distrib_shard_files``/``sharding_fix``. Those belong to the Loquacious path.
  The winner's convention on FZJ (see ``dlm_on_winner.train_paper_best_dlm_4gpu``) is that RETURNN DDP
  ranks each iterate the *full* sub-epoch partition with a per-rank seed offset and no sharding; we
  keep that, so the LS:TTS ratio each rank sees is the intended 1:1.
"""

from __future__ import annotations

import unittest.mock
from functools import partial
from typing import Any, Dict, List

from sisyphus import tk

import returnn.frontend as rf


# The winner's own checkpoint, epoch 38 (the ``recog_results_best`` epoch), reached via Albert's
# imported training job rather than a literal path, so a re-import cannot silently point elsewhere.
WINNER_TRAIN_JOB = "i6_core/returnn/training/ReturnnTrainingJob.8iFbool3x3TU"
WINNER_EPOCH = 38

# TTS ogg zips pushed RZ -> FZJ (rsync from RZ login23-2; JSC forbids outgoing SSH from JUPITER).
# 100 of the 750 parts: at TTS_FILES_PER_SUBEPOCH per sub-epoch that is exactly NEP sub-epochs, and
# 180 GB / 100 inodes instead of 1.35 TB. Transfer more parts and raise both numbers to train longer.
TTS_BASE_DIR = "/e/project1/spell/koch13/lm_tts_2024"
NUM_TTS_FILES = 100
TTS_FILES_PER_SUBEPOCH = 10  # ~1000 h vs LS-960's 960 h -> the 1:1 ratio tts_data.py documents
NEP = NUM_TTS_FILES // TTS_FILES_PER_SUBEPOCH  # sub-epochs; one full pass over the transferred TTS data

# Fine-tune schedule. The winner ran peak 5e-3 over 38 sub-epochs and ended at 1e-6; RETURNN's
# import_model_train_epoch1 restarts the epoch counter, optimizer and LR schedule at 1, so re-entering
# at the winner's peak would undo it. 5e-4 = 1/10 of the original peak.
PEAK_LR = 5e-4

_COMBINED_MAIN_NAME = "LS ASR + Text(spm+phon)"  # _train_tts_encoder's DatasetConfigStatic main_name


def tts_oggzips() -> List[tk.Path]:
    """
    The transferred TTS ogg zips, pinned to the same hashes ``tts_data.get_tts_oggzips`` gives them on RZ.

    The ``hash_overwrite`` is what makes this location-independent: without it the Sisyphus hash would
    embed ``/e/project1/...`` and nothing would be comparable to the RZ runs.
    """
    return [
        tk.Path(
            f"{TTS_BASE_DIR}/lm_data_part{i}_ogg.zip",
            hash_overwrite=f"rossenbach/tts_decoder_asr/ctc_rnnt_standalone_2024/ls_lm_data/{i}",
        )
        for i in range(1, NUM_TTS_FILES + 1)
    ]


def sub_epoch_dataset_no_filecache(
    files: List[Any], *, base_opts: Dict[str, Any], multi_proc_dataset: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    ``tts_data._get_distribute_files_dataset_for_epoch`` minus the ``CachedFile`` wrapping.

    Why we cannot use the original here (measured 2026-09-17, first launch died on it):
    **JUPITER compute nodes have no local disk.** ``df`` on a running job shows only
    ``LiveOS_rootfs 96G`` for ``/`` and ``/var/tmp`` (RAM-backed) plus a 239 G ``/dev/shm`` tmpfs, and
    ``$TMPDIR`` is unset -- so RETURNN's ``FileCache`` default ``$TMPDIR/$USER/returnn/file_cache``
    lands in ``/var/tmp`` and every cached byte costs node memory. The run filled it to **85.5 GB** and
    died with ``We cannot free enough space``, because with 4 DDP ranks and no file sharding each rank
    randomly picks its *own* 10 zips per sub-epoch: 4 x 10 x 1.8 GB + 16.2 GB of LS-960 ~= 88 GB, which
    is what the cache reported.

    Caching buys nothing here anyway: the winner's own training reads these very ogg zips straight off
    the project filesystem -- its ``use_cache_manager: True`` logs ``Cache manager: Error occurred,
    using local file`` on every rank -- and it trained at 1.16 h/epoch that way. So we read directly too,
    at zero node-RAM cost, and keep the per-rank file randomisation.

    Kept byte-for-byte from the original otherwise: same path concatenation order (LS first), same
    ``AbstractPath`` resolution, same ``MultiProcDataset`` wrapping.
    """
    from sisyphus.job_path import AbstractPath
    from i6_experiments.users.zeyer.datasets.utils import multi_proc as mp_ds_utils

    opts = base_opts.copy()
    assert opts["class"] == "OggZipDataset"
    files = opts["path"] + files
    files = [fn.get_path() if isinstance(fn, AbstractPath) else fn for fn in files]
    assert all(isinstance(fn, str) for fn in files)
    opts["path"] = files  # no CachedFile: see docstring

    if multi_proc_dataset is not None:
        opts = mp_ds_utils.multi_proc_dataset_opts(opts, **multi_proc_dataset)

    return opts


def _add_tts_to_asr_branch(combined: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rewrite a ``CombinedDataset``'s ``asr`` sub-dataset into ``DistributeFilesDataset`` over the TTS zips,
    each sub-epoch building one ``OggZipDataset`` over ``[3 LS-960 zips] + [TTS_FILES_PER_SUBEPOCH TTS zips]``.

    The per-sub-epoch builder is :func:`sub_epoch_dataset_no_filecache`, which is
    ``tts_data._get_distribute_files_dataset_for_epoch`` minus the ``CachedFile`` wrapping -- see its
    docstring for why the original cannot be used on JUPITER (no local disk; the first launch died
    filling a RAM-backed ``/var/tmp``).
    """
    datasets = dict(combined["datasets"])
    asr = dict(datasets["asr"])

    # _train_tts_encoder wraps the OggZip in MultiProcDataset; the DFD needs the bare OggZip as base_opts
    # and re-applies MPD *inside* each sub-epoch dataset (mp must wrap the ogg decode, not the DFD).
    if asr["class"] == "MultiProcDataset":
        mp_opts = {k: v for k, v in asr.items() if k not in ("class", "dataset")}
        base_opts = dict(asr["dataset"])
    else:
        mp_opts = None
        base_opts = asr
    assert base_opts["class"] == "OggZipDataset", f"unexpected asr sub-dataset {base_opts['class']!r}"

    # The TTS zips store their audio under out.ogg/; content_name makes one OggZipDataset read both
    # layouts. use_cache_manager is dropped as in tts_data: on FZJ it only logs
    # "Cache manager: Error occurred, using local file" anyway (the winner's own run does exactly that).
    base_opts = dict(base_opts)
    base_opts.pop("use_cache_manager", None)
    base_opts["content_name"] = "out.ogg"

    datasets["asr"] = {
        "class": "DistributeFilesDataset",
        "files": tts_oggzips(),
        "get_sub_epoch_dataset": partial(
            sub_epoch_dataset_no_filecache, base_opts=base_opts, multi_proc_dataset=mp_opts
        ),
        "partition_epoch": NEP,
        "seq_ordering": "random",
    }
    out = dict(combined)
    out["datasets"] = datasets
    return out


class _PatchAsrBranchWithTts:
    """
    Context manager: while active, the ``DatasetConfigStatic`` that ``_train_tts_encoder`` builds for its
    train data gets its ``asr`` branch replaced by :func:`_add_tts_to_asr_branch`.

    ``_train_tts_encoder`` imports ``DatasetConfigStatic`` *inside* the function
    (exp2026_05_28_tts_encoder_fzj.py:3605), so patching the attribute on its defining module is enough and
    nothing else in the process is affected. We guard on ``main_name`` and on the dataset actually being a
    ``CombinedDataset`` with an ``asr`` branch, and assert afterwards that we rewrote **exactly one** --
    a silent zero would train the plain winner again under a new name, which is the expensive failure here.
    """

    def __init__(self):
        self.count = 0
        self._patch = None

    def __enter__(self):
        from returnn_common.datasets_old_2022_10 import interface as _iface

        real = _iface.DatasetConfigStatic

        def _wrapped(*args, **kwargs):
            ds = kwargs.get("train_dataset")
            if (
                kwargs.get("main_name") == _COMBINED_MAIN_NAME
                and isinstance(ds, dict)
                and ds.get("class") == "CombinedDataset"
                and "asr" in ds.get("datasets", {})
            ):
                kwargs = dict(kwargs, train_dataset=_add_tts_to_asr_branch(ds))
                self.count += 1
            return real(*args, **kwargs)

        self._patch = unittest.mock.patch.object(_iface, "DatasetConfigStatic", _wrapped)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        if exc[0] is None:
            assert self.count == 1, f"expected to rewrite exactly 1 train dataset, rewrote {self.count}"
        return False


def winner_checkpoint(winner_model):
    """
    The winner's epoch-38 checkpoint, taken from the ``ModelWithCheckpoints`` its own ``_train_tts_encoder``
    call returned -- so the dependency edge is on the real training job and cannot drift to a stale path.

    :param winner_model: the return value of ``fzj_dlm._train_winner`` (i.e. of ``_train_tts_encoder``)
    """
    ckpt = winner_model.get_epoch(WINNER_EPOCH).checkpoint
    assert winner_model.definition is not None  # a ModelWithCheckpoints, not a bare path
    job_id = ckpt.path.creator._sis_id() if ckpt.path.creator is not None else None
    assert job_id == WINNER_TRAIN_JOB, f"winner checkpoint comes from {job_id!r}, expected {WINNER_TRAIN_JOB!r}"
    return ckpt


def train_winner_plus_tts(*, prefix: str, winner_model, name: str = "winner-plusTts-nEp10-lr5e-4"):
    """
    The winner's own ``_train_tts_encoder`` call (``_sa == 50``), verbatim except for:

    - ``nep`` 38 -> NEP and ``peak_lr`` 5e-3 -> PEAK_LR (this is a fine-tune, not a fresh run),
    - ``import_model_train_epoch1`` = the winner's epoch-38 checkpoint,
    - the ``asr`` branch extended with TTS audio by :class:`_PatchAsrBranchWithTts`.

    Everything else -- pseudo-encoder tables, specaug 50, Muon, packed tensors, CUDA graphs, the text
    branch and its epoch split -- is unchanged, so the comparison against the winner is clean.
    """
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon
    from i6_experiments.users.zeyer.datasets.hf_librispeech_mfa_alignments import (
        get_mfa_phone_mean_logmel_table,
        get_mfa_phone_duration_table,
    )
    from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_tts_encoder

    _sa = 50
    with _PatchAsrBranchWithTts():
        return _train_tts_encoder(
            name,
            prefix=prefix,
            with_ctc_lm_recog=True,
            text_train_epoch_split=75,
            batch_size_audio_frames=70_000,
            batch_size_phon=6_000,
            max_phon_len=300,
            asr_logmel=True,
            pseudo_speech_enc=True,
            pseudo_enc_frozen_table=get_mfa_phone_mean_logmel_table().out_mean_table,
            pseudo_enc_duration_table=get_mfa_phone_duration_table().out_duration_table,
            pseudo_enc_duration_sigma=0.45,
            pseudo_enc_duration_scale=0.7,
            pseudo_enc_max_len_factor=10,
            train_seq_ordering="random",
            pseudo_enc_lerp=True,
            pseudo_enc_blank_duration_range=(0, 0),
            pseudo_enc_specaug_max_width=6,
            single_stream=True,
            interleave_gumbel_scale=1.0,
            glow_tts_add_silence_between_words=0.15,
            base_lr=1.0,
            peak_lr=PEAK_LR,
            nep=NEP,
            behavior_version=29,  # packed tensors need >= 29
            pseudo_enc_frontend_concat=True,
            extra_config_updates={
                "optimizer.class": rf.build_dict(Muon)["class"],
                "packed_tensors": True,
                "torch_distributed": {"reduce_type": "grad_explicit"},
                "batch_size": None,
                "packed_batch_size": {"data": 11_200_000, "classes": 5_000, "phonemes": 6_000},
                "batching": "random",
                "torch_cuda_graph": {
                    "batch_size_bound": 500,
                    "dim_capacity": {"data": 312_000, "classes": 80, "phonemes": 300},
                    "warmup_steps": 0,
                    "compile": True,
                },
                "optimizer.weight_decay": 0.027,  # 0.01 / 0.370
                "specaugment_num_spatial_mask_factor": _sa,
                "specaugment_steps": (1850, 5550, 9250),  # (5000, 15000, 25000) * 0.370
                # Init from the winner. train_v4 deep-merges `config` before it would set this itself,
                # so this is the supported route -- _train_tts_encoder exposes no `init_params=`.
                # NOTE this restarts epoch counter / optimizer / LR at 1 (returnn/engine/base.py:156-192):
                # it is "initialise from the winner", not an optimizer-state resume.
                "import_model_train_epoch1": winner_checkpoint(winner_model),
            },
            extra_config_deletes=["optimizer.epsilon"],
        )


def describe_train_dataset():
    """
    Console helper: print the rewritten ``asr`` branch without building anything.

    Verifies the two things that are easy to get silently wrong -- that the TTS zips carry RZ's hashes,
    and that the per-sub-epoch dataset really is LS-960 + TTS in one OggZipDataset.
    """
    files = tts_oggzips()
    print(f"TTS zips: {len(files)}, partition_epoch {NEP} -> {len(files) // NEP} per sub-epoch")
    print(f"  first: {files[0].get_path()}\n  hash_overwrite: {files[0].hash_overwrite}")
    fake_asr = {
        "class": "MultiProcDataset",
        "num_workers": 4,
        "buffer_size": 10,
        "dataset": {"class": "OggZipDataset", "path": ["LS1", "LS2", "LS3"], "use_cache_manager": True},
    }
    out = _add_tts_to_asr_branch({"class": "CombinedDataset", "datasets": {"asr": fake_asr, "text": {}}})
    asr = out["datasets"]["asr"]
    print(f"  asr branch -> {asr['class']}, partition_epoch {asr['partition_epoch']}")
    sub = asr["get_sub_epoch_dataset"](["TTS_a", "TTS_b"])
    print(f"  sub-epoch dataset: {sub['class']}, inner paths = {len(sub['dataset']['path'])} (3 LS + 2 TTS)")
