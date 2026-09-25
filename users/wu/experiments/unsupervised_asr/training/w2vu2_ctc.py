"""New in the port: the SAE section 1d self-training student (wav2vec-U 2.0 stage 3), trained by fairseq
itself through i6_core's ``FairseqHydraTrainingJob``.

Ported from the live ``w2vu2/selftrain.py`` (``LibriStAudioJob``, ``Wav2Vec2CtcFinetuneJob``) of the
reference setup.  The banked production run is ``Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0``: fairseq 0.12.2
``audio_finetuning`` + ``wav2vec_ctc`` + ``ctc``, the ``vox_100h.yaml`` fine-tuning config with the
section 1d overrides, 28,539 train-clean-100 utterances with the GAN teacher's pseudo phone labels
(``GanPseudoLabelJob.xjn6QnNqwEEH``), 4 GPUs, the last checkpoint.

* :class:`FairseqAudioManifestJob` -- one LibriSpeech ogg zip (``data.librispeech.get_ogg_zip``) to
  FLAC files plus a fairseq manifest ``<name>.tsv`` and the utterance ids ``<name>.uid`` (same row
  order).  Production's ``LibriStAudioJob`` wrote the decoded float32 waveform of the Ogg Vorbis HF
  dataset with ``soundfile.write(..., format="FLAC")`` (PCM_16); this job does the same with the
  waveform :mod:`..data.ogg_zip` decodes (RETURNN's ``OggZipDataset`` decode).  i6_core's
  ``CreateManifestJob`` is not used: it lists files in ``glob`` order (file-system dependent), while
  the manifest row order feeds fairseq's seeded batch order and the ``.uid`` join.
* :class:`FairseqCtcDataJob` -- the fairseq data dir (``train.tsv`` / ``train.phn`` /
  ``dict.phn.txt``) from a manifest, the pseudo labels and the dev gold, as production's
  ``Wav2Vec2CtcFinetuneJob._build_data_dir``: utterances with an empty label are dropped, the
  dictionary is the sorted union of the label phones and the dev gold phones, each ``"<phone> 1"``.
* :data:`PRODUCTION_FINETUNE_CONFIG` / :func:`build_ctc_training_job` -- the fairseq hydra config
  production wrote (``vox_100h.yaml`` without its ``hydra`` block, plus the section 1d overrides),
  with only the paths as arguments; ``max_update`` / ``max_epoch`` / ``save_interval`` go through the
  job arguments, as ``FairseqHydraConfig`` requires.
* :func:`get_fairseq_w2v2_lv60_checkpoint` -- the fairseq-format LV-60 pretrained checkpoint
  (``model.w2v_path``), the same URL production fetched, sha256-pinned to the banked file.

The pseudo labels are the teacher's collapsed, silence-free phone strings, json
``{"labels": {utt_id: "P1 P2 ..."}}`` (production ``GanPseudoLabelJob.out_labels``); the dev gold is
``data.gold.GoldPhonesJob`` json ``{split: {utt_id: [phones]}}``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from sisyphus import Job, Task, tk

from i6_core.fairseq.training import FairseqHydraConfig, FairseqHydraTrainingJob

__all__ = [
    "FAIRSEQ_W2V2_LV60_URL",
    "FAIRSEQ_W2V2_LV60_SHA256",
    "get_fairseq_w2v2_lv60_checkpoint",
    "FairseqAudioManifestJob",
    "FairseqCtcDataJob",
    "PRODUCTION_FINETUNE_CONFIG",
    "MAX_UPDATE",
    "MAX_EPOCH",
    "SAVE_INTERVAL",
    "TRAIN_RQMT",
    "build_ctc_training_job",
    "get_last_checkpoint",
]

#: production ``FetchFairseqW2v2Job`` (``selftrain._W2V2_LV60_URL``)
FAIRSEQ_W2V2_LV60_URL = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt"
#: sha256 of the banked ``FetchFairseqW2v2Job.ZNW3mVCvsT6v/output/wav2vec_vox_new.pt`` (3,174,007,860 bytes)
FAIRSEQ_W2V2_LV60_SHA256 = "9b0748fbd4c725ff62266e3b9544cf948d117bc7fa2dc49528184de547736844"


def get_fairseq_w2v2_lv60_checkpoint() -> tk.Path:
    """The fairseq wav2vec 2.0 Large (LV-60k) pretrained checkpoint without fine-tuning head."""
    from i6_core.tools.download import DownloadJob

    return DownloadJob(
        url=FAIRSEQ_W2V2_LV60_URL, target_filename="wav2vec_vox_new.pt", checksum=FAIRSEQ_W2V2_LV60_SHA256
    ).out_file


def _read_lines(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _read_tsv(path: str) -> Tuple[str, List[Tuple[str, str]]]:
    """fairseq manifest -> (root, [(relpath, n_samples), ...]) in file order."""
    with open(path) as f:
        root = f.readline().strip()
        rows = [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]
    return root, rows


class FairseqAudioManifestJob(Job):
    """One ogg zip -> ``audio/<name>/<utt_id>.flac`` + ``<name>.tsv`` + ``<name>.uid`` under ``out_dir``.

    ``<name>.tsv`` is fairseq's manifest (first line the root ``<out_dir>/audio``, then
    ``<name>/<utt_id>.flac<TAB><n_samples>``); ``<name>.uid`` lists the utterance ids in the same
    order (zip order).  The FLAC is ``soundfile.write(path, wav, 16000, format="FLAC")`` of the float32
    waveform, production's ``LibriStAudioJob._write_split`` call.
    """

    def __init__(self, *, ogg_zip: tk.Path, name: str):
        """
        :param ogg_zip: a LibriSpeech RETURNN ogg zip (``data.librispeech.get_ogg_zip(subset)``)
        :param name: the split name of the manifest (``train``, ``dev-clean``, ``dev-other``)
        """
        super().__init__()
        self.ogg_zip = ogg_zip
        self.name = name

        self.out_dir = self.output_path("manifest", directory=True)
        self.rqmt = {"cpu": 4, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import soundfile as sf

        from ..data.ogg_zip import iter_ogg_zip_audio

        root = self.out_dir.get_path()
        audio_root = os.path.join(root, "audio")
        os.makedirs(os.path.join(audio_root, self.name), exist_ok=True)
        rows = []
        for uid, wav in iter_ogg_zip_audio(self.ogg_zip.get_path()):
            rel = f"{self.name}/{uid}.flac"
            sf.write(os.path.join(audio_root, rel), wav, 16000, format="FLAC")
            rows.append((uid, rel, len(wav)))
        assert rows, f"empty ogg zip {self.ogg_zip}"
        assert len({uid for uid, _, _ in rows}) == len(rows), "duplicate utterance ids in the ogg zip"
        with open(os.path.join(root, f"{self.name}.tsv"), "w") as t, open(
            os.path.join(root, f"{self.name}.uid"), "w"
        ) as u:
            print(audio_root, file=t)
            for uid, rel, n in rows:
                print(f"{rel}\t{n}", file=t)
                print(uid, file=u)
        print(f"{self.name}: {len(rows)} utts", flush=True)


class FairseqCtcDataJob(Job):
    """The fairseq ``audio_finetuning`` data dir of the CTC student: ``train.tsv``, ``train.phn``,
    ``dict.phn.txt`` (production ``Wav2Vec2CtcFinetuneJob._build_data_dir``)."""

    def __init__(self, *, manifest_dir: tk.Path, labels: tk.Path, gold: tk.Path, name: str = "train"):
        """
        :param manifest_dir: :class:`FairseqAudioManifestJob` ``out_dir`` of the train split
        :param labels: pseudo labels, json ``{"labels": {utt_id: "P1 P2 ..."}}``
        :param gold: ``GoldPhonesJob`` json ``{split: {utt_id: [phones]}}``; its phones join the dictionary
            so the dev references never map to ``<unk>``
        :param name: the manifest name inside ``manifest_dir``
        """
        super().__init__()
        self.manifest_dir = manifest_dir
        self.labels = labels
        self.gold = gold
        self.name = name

        self.out_data_dir = self.output_path("data", directory=True)
        self.out_dict_phn = self.output_path("data/dict.phn.txt")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        root, rows = _read_tsv(os.path.join(self.manifest_dir.get_path(), f"{self.name}.tsv"))
        uids = _read_lines(os.path.join(self.manifest_dir.get_path(), f"{self.name}.uid"))
        assert len(uids) == len(rows), (len(uids), len(rows))
        with open(self.labels.get_path()) as f:
            labels = json.load(f)["labels"]

        kept, phns, vocab = [], [], set()
        for (relpath, size), uid in zip(rows, uids):
            seq = labels.get(uid, "").strip()
            if not seq:  # empty teacher transcript -> drop the utterance
                continue
            kept.append((relpath, size))
            phns.append(seq)
            vocab.update(seq.split())
        assert kept, "no utterance with a non-empty label"
        with open(self.gold.get_path()) as f:
            gold = json.load(f)
        for d in gold.values():
            for seq in d.values():
                vocab.update(seq)

        data = self.out_data_dir.get_path()
        os.makedirs(data, exist_ok=True)
        with open(os.path.join(data, "train.tsv"), "w") as t:
            print(root, file=t)
            for relpath, size in kept:
                print(f"{relpath}\t{size}", file=t)
        with open(os.path.join(data, "train.phn"), "w") as p:
            p.write("\n".join(phns) + "\n")
        with open(self.out_dict_phn.get_path(), "w") as dct:
            for sym in sorted(vocab):
                print(f"{sym} 1", file=dct)
        print(f"data: {len(kept)}/{len(rows)} train utts kept, |vocab|={len(vocab)}", flush=True)


#: The fairseq hydra config production passed to ``fairseq_cli.hydra_train`` (its ``config_ft/finetune.yaml``):
#: fairseq 0.12.2 ``examples/wav2vec/config/finetuning/vox_100h.yaml`` without the ``hydra`` block, with
#: the section 1d settings of ``Wav2Vec2CtcFinetuneJob._write_config`` applied.  The paths
#: (``task.data``, ``model.w2v_path``, ``checkpoint.save_dir``) are filled by :func:`build_ctc_training_job`
#: and the job; ``optimization.max_update`` / ``max_epoch`` and ``checkpoint.save_interval`` are job arguments.
PRODUCTION_FINETUNE_CONFIG: Dict[str, Dict[str, Any]] = {
    "common": {"fp16": True, "log_format": "json", "log_interval": 200},
    "checkpoint": {
        "no_epoch_checkpoints": True,
        "best_checkpoint_metric": "wer",
        "save_interval_updates": 2500,
    },
    "task": {"_name": "audio_finetuning", "normalize": True, "labels": "phn"},
    "dataset": {
        "num_workers": 6,
        "max_tokens": 1280000,
        "skip_invalid_size_inputs_valid_test": True,
        "valid_subset": "train",  # no labelled dev; disable_validation skips it
        "disable_validation": True,
    },
    "distributed_training": {"ddp_backend": "legacy_ddp", "distributed_world_size": 4},
    "criterion": {"_name": "ctc", "zero_infinity": True},
    "optimization": {"lr": [3e-05], "sentence_avg": True, "update_freq": [5]},
    "optimizer": {"_name": "adam", "adam_betas": "(0.9,0.98)", "adam_eps": 1e-08},
    "lr_scheduler": {"_name": "tri_stage", "phase_ratio": [0.1, 0.4, 0.5], "final_lr_scale": 0.05},
    "model": {
        "_name": "wav2vec_ctc",
        "apply_mask": True,
        "mask_prob": 0.5,
        "mask_channel_prob": 0.5,
        "mask_channel_length": 64,
        "layerdrop": 0.1,
        "activation_dropout": 0.1,
        "feature_grad_mult": 0.0,
        "freeze_finetune_updates": 8000,
    },
}
#: production ``optimization.max_update``
MAX_UPDATE = 40000
#: fairseq's default (0 = no epoch limit), as in the production run
MAX_EPOCH = 0
#: fairseq's default ``checkpoint.save_interval``, as in the production run
SAVE_INTERVAL = 1
#: production ``Wav2Vec2CtcFinetuneJob`` rqmt (4 GPUs, 80 GB each, 60 GB, 16 CPUs, 11.5 h), in
#: ``FairseqHydraTrainingJob``'s per-GPU convention for ``cpu`` and ``mem``
TRAIN_RQMT = {"gpu": 4, "gpu_mem": 80, "cpu": 4, "mem": 15, "time": 11.5}


def build_ctc_training_job(
    *,
    data_dir: tk.Path,
    w2v_path: tk.Path,
    fairseq_python_exe: tk.Path,
    fairseq_root: tk.Path,
    rqmt: Optional[Dict[str, Any]] = None,
) -> FairseqHydraTrainingJob:
    """The section 1d CTC fine-tune as a ``FairseqHydraTrainingJob`` with production's config.

    :param data_dir: :class:`FairseqCtcDataJob` ``out_data_dir``
    :param w2v_path: the fairseq LV-60 pretrained checkpoint (:func:`get_fairseq_w2v2_lv60_checkpoint`)
    :param fairseq_python_exe: python of the fairseq 0.12.2 env
    :param fairseq_root: dir holding ``fairseq_cli/hydra_train.py`` (put first on the child's PYTHONPATH)
    :param rqmt: resource override (not hashed); default :data:`TRAIN_RQMT`
    """
    import copy

    config = copy.deepcopy(PRODUCTION_FINETUNE_CONFIG)
    config["task"]["data"] = data_dir
    config["model"]["w2v_path"] = w2v_path
    return FairseqHydraTrainingJob(
        FairseqHydraConfig(config),
        max_epoch=MAX_EPOCH,
        max_update=MAX_UPDATE,
        save_interval=SAVE_INTERVAL,
        keep_epochs=[],  # checkpoints are named checkpoint_last.pt / checkpoint_<ep>_<upd>.pt, not checkpoint<ep>.pt
        rqmt=dict(rqmt if rqmt is not None else TRAIN_RQMT),
        fairseq_python_exe=fairseq_python_exe,
        fairseq_root=fairseq_root,
    )


def get_last_checkpoint(job: FairseqHydraTrainingJob) -> tk.Path:
    """``checkpoint_last.pt`` of the run, the checkpoint production decoded (no dev, no selection)."""
    return job.out_checkpoint_dir.join_right("checkpoint_last.pt")
