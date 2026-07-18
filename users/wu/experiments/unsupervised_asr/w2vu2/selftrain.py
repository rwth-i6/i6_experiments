"""SAE §1d — no-Kaldi self-training of the §1c winning GAN, the published w2v-U stage-3 recipe.

The paper's self-training is (1) pseudo-label, (2) HMM-GMM realign, (3) CTC fine-tune the encoder.
Kaldi is unavailable on this aarch64 cluster (no binaries, no module, multi-day to build), so the
HMM stage is dropped and its label-cleaning role is approximated by the CTC student's own capacity +
SpecAugment (a phone-LM decode lever can be layered on later). What remains is the paper's own
stage 3, run as fairseq's reference `audio_finetuning` + `wav2vec_ctc` (no reimplementation):

  GanPseudoLabelJob      teacher (§1c GAN) greedy phones on LS-100 train features   [w2vu GPU]
  LibriStAudioJob        HF arrays -> flac + fairseq .tsv manifests (train/dev)      [speech_llm CPU]
  FetchFairseqW2v2Job    the fairseq-format LV-60 pretrained ckpt (w2v_path)         [login, online]
  Wav2Vec2CtcFinetuneJob fairseq-hydra-train, phn labels, SpecAugment, last ckpt     [w2vu GPU]  (added below)
  Wav2Vec2CtcDecodeJob   viterbi decode dev -> per-split PER vs the 0.214 GAN-init   [w2vu GPU]  (added below)

Everything is kept sil-free and collapsed, exactly the §1c PER convention, so the self-trained PER is
directly comparable to the GAN-init number.
"""

from __future__ import annotations

import os
import re
import subprocess as sp
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from sisyphus import Job, Task, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import W2VU_PYTHON, assert_w2vu_env

_EVAL_WORKER = os.path.join(os.path.dirname(__file__), "eval_per.py")

# fairseq-format wav2vec2-Large (LV-60k) pretrained, no fine-tune head -- the w2v_path vox_100h.yaml
# expects. This is the fairseq twin of the HF `facebook/wav2vec2-large-lv60` we dump features with.
_W2V2_LV60_URL = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt"
_W2V2_LV60_BYTES = 3174007860


class FetchFairseqW2v2Job(Job):
    """Download the fairseq-format LV-60 pretrained checkpoint (needed as the CTC fine-tune base).

    mini_task -> runs on the short/login engine, the only side with internet (settings.py). Size is
    asserted so a truncated download fails here instead of deep inside fairseq's checkpoint loader.
    """

    def __init__(self, *, url: str = _W2V2_LV60_URL, expected_bytes: int = _W2V2_LV60_BYTES):
        super().__init__()
        self.url = url
        self.expected_bytes = expected_bytes
        self.out_ckpt = self.output_path("wav2vec_vox_new.pt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        dst = self.out_ckpt.get_path()
        sp.check_call(["wget", "-q", "-O", dst, self.url])
        got = os.path.getsize(dst)
        assert got == self.expected_bytes, f"size {got} != expected {self.expected_bytes}"
        print(f"fetched {self.url} -> {dst} ({got} bytes)", flush=True)


class GanPseudoLabelJob(Job):
    """§1d stage 1: the §1c GAN teacher's greedy phone transcript for every LS-100 train utt.

    Same collapsed sil-free viterbi decode the §1c PER used (eval_per.dump_labels), keyed by utt id so
    LibriStAudioJob's manifest can be joined to it. worker under speech_llm, GPU forward under w2vu.
    """

    requires_env = "w2vu"

    def __init__(
        self,
        *,
        checkpoint: tk.Path,   # teacher GAN checkpoint (the ppl-selected s0 checkpoint_best)
        data_dir: tk.Path,     # GAN task.data (holds train.npy/.lengths/.ids)
        text_data: tk.Path,    # GAN task.text_data (dict.txt) -> generator vocab
        split: str = "train",  # which feats dump to label
        python_exe: tk.Path = W2VU_PYTHON,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.data_dir = data_dir
        self.text_data = text_data
        self.split = split
        self.python_exe = python_exe

        self.out_labels = self.output_path("labels.json")  # {"labels": {id: "p1 p2 ..."}, ...}
        self.rqmt = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        assert_w2vu_env(self.python_exe)
        recipe = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
        args = [
            os.fspath(self.python_exe), _EVAL_WORKER, "--dump-labels",
            "--ckpt", self.checkpoint.get_path(),
            "--data", self.data_dir.get_path(),
            "--text-data", self.text_data.get_path(),
            "--feats", os.path.join(self.data_dir.get_path(), f"{self.split}.npy"),
            "--out", self.out_labels.get_path(),
        ]
        print("RUN:", " ".join(args), flush=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([recipe, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        sp.check_call(args, env=env)


class LibriStAudioJob(Job):
    """HF audio arrays -> flac files + fairseq `.tsv` manifests, once, reused across ST iterations.

    fairseq's FileAudioDataset reads real audio files, but our LS lives in an on-disk HF dataset
    (Arrow arrays, no files), so train + dev are materialised to flac here. The `dev` split is routed
    into dev-clean / dev-other by the gold id->split map so each can be scored separately; utts with no
    gold are dropped (unscorable). Label-agnostic on purpose: pseudo-labels change every self-training
    round, the audio does not, so `.phn` files are written by the consumers, not here.

    Runs under the default (speech_llm) env: `datasets` + `soundfile` are kept out of the w2vu env.
    """

    def __init__(
        self,
        *,
        hf_data_dir: tk.Path,          # DatasetDict on disk with "train" and "dev" splits
        gold: tk.Path,                 # GoldPhonesJob json {split: {id: [phones]}} -> dev routing
        train_split: str = "train",
        dev_split: str = "dev",
        limit: Optional[int] = None,   # cap utts per split (smoke)
    ):
        super().__init__()
        self.hf_data_dir = hf_data_dir
        self.gold = gold
        self.train_split = train_split
        self.dev_split = dev_split
        self.limit = limit

        self.out_dir = self.output_path("audio", directory=True)  # holds audio/ + *.tsv + *.uid
        self.rqmt = {"cpu": 4, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _write_split(self, ds, split_name, ids_wanted, root, limit=None):
        """Materialise one (sub)split -> flac under root/<split_name>/, return [(id, nframes)]."""
        import numpy as np
        import soundfile as sf

        sub_dir = os.path.join(root, split_name)
        os.makedirs(sub_dir, exist_ok=True)
        rows = []
        for ex in ds:
            uid = str(ex["id"])
            if ids_wanted is not None and uid not in ids_wanted:
                continue
            wav = np.asarray(ex["audio"]["array"], dtype=np.float32)
            fp = os.path.join(sub_dir, f"{uid}.flac")
            sf.write(fp, wav, 16000, format="FLAC")
            rows.append((uid, len(wav)))
            if limit and len(rows) >= limit:
                break
        return rows

    def _emit(self, name, rows, root):
        """Write <name>.tsv (fairseq manifest: root header + relpath\\tnframes) and <name>.uid."""
        with open(os.path.join(root, f"{name}.tsv"), "w") as t, \
             open(os.path.join(root, f"{name}.uid"), "w") as u:
            print(os.path.join(root, "audio"), file=t)
            for uid, n in rows:
                # relpath is under the audio/ root; sub dir = train or dev
                sub = "train" if name == "train" else "dev"
                print(f"{sub}/{uid}.flac\t{n}", file=t)
                print(uid, file=u)
        print(f"{name}: {len(rows)} utts", flush=True)

    def run(self):
        import json

        from datasets import Audio, load_from_disk

        root = self.out_dir.get_path()
        audio_root = os.path.join(root, "audio")
        os.makedirs(audio_root, exist_ok=True)

        dd = load_from_disk(self.hf_data_dir.get_path())
        train_ds = dd[self.train_split].cast_column("audio", Audio(sampling_rate=16000))
        dev_ds = dd[self.dev_split].cast_column("audio", Audio(sampling_rate=16000))

        with open(self.gold.get_path()) as f:
            gold = json.load(f)
        id2split = {uid: s for s, d in gold.items() for uid in d}  # dev-clean / dev-other

        train_rows = self._write_split(train_ds, "train", None, audio_root, limit=self.limit)
        self._emit("train", train_rows, root)

        # one pass over dev (never limited -- decode needs the full, bucketed dev), split by gold
        wanted = set(id2split)
        dev_rows = self._write_split(dev_ds, "dev", wanted, audio_root)
        buckets: Dict[str, list] = {}
        for uid, n in dev_rows:
            buckets.setdefault(id2split[uid], []).append((uid, n))
        for s, rows in sorted(buckets.items()):
            self._emit(s, rows, root)


def _fairseq_dir(python_exe: tk.Path) -> str:
    out = sp.check_output(
        [os.fspath(python_exe), "-c",
         "import fairseq, os; print(os.path.dirname(fairseq.__file__))"], text=True)
    return out.strip()


def _read_tsv(path: str) -> Tuple[str, List[Tuple[str, str]]]:
    """fairseq manifest -> (root, [(relpath, size), ...]) preserving order."""
    with open(path) as f:
        root = f.readline().strip()
        rows = [tuple(l.rstrip("\n").split("\t")) for l in f if l.strip()]
    return root, rows


def _read_lines(path: str) -> List[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def _torch_unsafe_load_env() -> Dict[str, str]:
    """Env that lets torch 2.6 load a fairseq checkpoint again.

    torch 2.6 flipped `torch.load`'s default to `weights_only=True`; the 2020 fairseq LV-60 ckpt (and
    any wav2vec_ctc checkpoint that embeds its `argparse.Namespace` args) then fails to unpickle.
    fairseq's `load_checkpoint_to_cpu` calls `torch.load` without `weights_only`, so this env var
    overrides it back to False. Safe here: the checkpoints are the official fairseq download and our
    own training, not untrusted input.
    """
    env = dict(os.environ)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return env


class Wav2Vec2CtcFinetuneJob(Job):
    """§1d stage 3: CTC fine-tune wav2vec2-lv60 on the GAN pseudo phone labels (fairseq reference).

    fairseq's own `audio_finetuning` + `wav2vec_ctc` + `ctc`, driven by the `vox_100h.yaml` config we
    fine-tune wav2vec2-Large with -- no reimplementation. Adaptations from that reference, all forced
    by the unsupervised setting:

      labels ltr -> phn            teacher labels are phones, not letters
      valid_subset -> train        no labelled dev exists; disable_validation skips the forward pass,
      + disable_validation         so no checkpoint is ever *selected* by a label (selection honesty):
                                   only checkpoint_last.pt is written and that is what we decode.
      max_update 80k -> ~40k, freeze 10k -> ~8k   budget-reduced single-node pass; the trajectory is
                                   read off the interval checkpoints (as in §1c), so a cut-short run
                                   still yields a self-trained PER.

    `task.normalize=true` is mandatory: the LV-60 checkpoint was pretrained normalized and
    wav2vec2_asr.py asserts train/finetune agree. Blank = index 0 (<s>); dict.phn.txt lists only real
    phones (fairseq prepends <s>/<pad>/</s>/<unk> at 0-3). Empty pseudo-labels are dropped (utt removed
    from both manifest and labels) so CTC never sees an all-blank target.
    """

    requires_env = "w2vu"

    def __init__(
        self,
        *,
        audio_dir: tk.Path,        # LibriStAudioJob.out_dir (train.tsv/.uid + audio/)
        train_labels: tk.Path,     # GanPseudoLabelJob.out_labels (json {id: "phones"})
        gold: tk.Path,             # GoldPhonesJob json -> dict must also cover dev gold phones
        w2v_path: tk.Path,         # FetchFairseqW2v2Job.out_ckpt (fairseq LV-60 pretrained)
        max_update: int = 40000,
        freeze_finetune_updates: int = 8000,
        save_interval_updates: int = 2500,
        ngpu: int = 4,
        update_freq: int = 5,
        max_tokens: int = 1280000,
        lr: float = 3e-5,
        python_exe: tk.Path = W2VU_PYTHON,
        time_rqmt: float = 11.5,
        gpu_mem: int = 80,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.train_labels = train_labels
        self.gold = gold
        self.w2v_path = w2v_path
        self.max_update = max_update
        self.freeze_finetune_updates = freeze_finetune_updates
        self.save_interval_updates = save_interval_updates
        self.ngpu = ngpu
        self.update_freq = update_freq
        self.max_tokens = max_tokens
        self.lr = lr
        self.python_exe = python_exe

        self.out_dir = self.output_path("train", directory=True)
        self.out_last = self.output_path("train/checkpoint_last.pt")
        self.out_dict = self.output_path("dict.phn.txt")
        self.out_log = self.output_path("train.log")
        self.rqmt = {"gpu": ngpu, "gpu_mem": gpu_mem, "mem": 60, "time": time_rqmt, "cpu": 4 * ngpu}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _build_data_dir(self) -> str:
        """Assemble the fairseq data dir: filtered train.tsv + train.phn + dict.phn.txt."""
        import json

        root, rows = _read_tsv(os.path.join(self.audio_dir.get_path(), "train.tsv"))
        uids = _read_lines(os.path.join(self.audio_dir.get_path(), "train.uid"))
        assert len(uids) == len(rows), (len(uids), len(rows))
        labels = json.load(open(self.train_labels.get_path()))["labels"]

        kept, phns, vocab = [], [], set()
        for (relpath, size), uid in zip(rows, uids):
            seq = labels.get(uid, "").strip()
            if not seq:                       # empty teacher transcript -> drop the utt entirely
                continue
            kept.append((relpath, size))
            phns.append(seq)
            vocab.update(seq.split())

        # dict must also cover the dev gold phones (else they decode as <unk> and inflate PER)
        gold = json.load(open(self.gold.get_path()))
        for d in gold.values():
            for seq in d.values():
                vocab.update(seq)

        data = os.path.abspath("data")
        os.makedirs(data, exist_ok=True)
        with open(os.path.join(data, "train.tsv"), "w") as t:
            print(root, file=t)
            for relpath, size in kept:
                print(f"{relpath}\t{size}", file=t)
        with open(os.path.join(data, "train.phn"), "w") as p:
            p.write("\n".join(phns) + "\n")
        with open(os.path.join(data, "dict.phn.txt"), "w") as dct:
            for sym in sorted(vocab):
                print(f"{sym} 1", file=dct)

        # persist the dict as a job output too (the decode job must use this exact index map)
        with open(self.out_dict.get_path(), "w") as dct:
            for sym in sorted(vocab):
                print(f"{sym} 1", file=dct)
        print(f"data: {len(kept)}/{len(rows)} train utts kept, |vocab|={len(vocab)}", flush=True)
        return data

    def _write_config(self, fs: str, data: str) -> str:
        src = os.path.join(fs, "examples", "wav2vec", "config", "finetuning", "vox_100h.yaml")
        with open(src) as f:
            cfg = yaml.safe_load(f)
        cfg.pop("hydra", None)

        settings = {
            "task.data": data,
            "task.normalize": True,
            "task.labels": "phn",
            "model.w2v_path": self.w2v_path.get_path(),
            "model.freeze_finetune_updates": self.freeze_finetune_updates,
            "dataset.valid_subset": "train",       # no labelled dev; disable_validation skips it
            "dataset.disable_validation": True,
            "optimization.max_update": self.max_update,
            "optimization.lr": [self.lr],
            "optimization.update_freq": [self.update_freq],
            "dataset.max_tokens": self.max_tokens,
            "distributed_training.distributed_world_size": self.ngpu,
            "checkpoint.save_dir": self.out_dir.get_path(),
            "checkpoint.save_interval_updates": self.save_interval_updates,
            "checkpoint.no_epoch_checkpoints": True,
        }
        for k, v in sorted(settings.items()):
            node = cfg
            *parents, leaf = k.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = v

        assert "???" not in yaml.safe_dump(cfg), "unfilled MISSING (???) left in the finetune config"
        out = os.path.abspath("config_ft")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "finetune.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return out

    def run(self):
        assert_w2vu_env(self.python_exe)
        fs = _fairseq_dir(self.python_exe)
        data = self._build_data_dir()
        cfg_dir = self._write_config(fs, data)

        args = [
            os.fspath(self.python_exe), "-m", "fairseq_cli.hydra_train",
            f"--config-dir={cfg_dir}", "--config-name=finetune",
        ]
        print("RUN:", " ".join(args), flush=True)
        with open(self.out_log.get_path(), "w") as log:
            sp.check_call(args, stdout=log, stderr=sp.STDOUT, env=_torch_unsafe_load_env())


class Wav2Vec2CtcDecodeJob(Job):
    """Viterbi-decode a fine-tuned wav2vec_ctc checkpoint on dev -> per-split PER, vs the 0.214 GAN-init.

    fairseq's NEW `examples.speech_recognition.new.infer` viterbi path is pure-torch argmax (no
    flashlight): argmax -> unique_consecutive -> drop blank, i.e. the same greedy collapse §1c used.
    `common_eval.post_process=none` keeps phones space-separated so the reported "WER" is a phone
    error rate. References are the MFA gold phones; the dict is the fine-tune's own dict.phn.txt so the
    model's output indices line up.
    """

    requires_env = "w2vu"

    def __init__(
        self,
        *,
        audio_dir: tk.Path,        # LibriStAudioJob.out_dir ({split}.tsv/.uid + audio/)
        checkpoint: tk.Path,       # Wav2Vec2CtcFinetuneJob.out_last
        gold: tk.Path,             # GoldPhonesJob json -> dev references
        dict_phn: tk.Path,         # Wav2Vec2CtcFinetuneJob.out_dict (same index map as training)
        splits: Sequence[str] = ("dev-clean", "dev-other"),
        max_tokens: int = 1100000,
        python_exe: tk.Path = W2VU_PYTHON,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.checkpoint = checkpoint
        self.gold = gold
        self.dict_phn = dict_phn
        self.splits = list(splits)
        self.max_tokens = max_tokens
        self.python_exe = python_exe

        self.out_per = self.output_path("per.json")
        self.rqmt = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _build_data_dir(self) -> str:
        import json
        import shutil

        gold = json.load(open(self.gold.get_path()))
        data = os.path.abspath("data")
        os.makedirs(data, exist_ok=True)
        shutil.copyfile(self.dict_phn.get_path(), os.path.join(data, "dict.phn.txt"))
        for s in self.splits:
            src_tsv = os.path.join(self.audio_dir.get_path(), f"{s}.tsv")
            uids = _read_lines(os.path.join(self.audio_dir.get_path(), f"{s}.uid"))
            os.symlink(src_tsv, os.path.join(data, f"{s}.tsv"))
            with open(os.path.join(data, f"{s}.phn"), "w") as p:
                for uid in uids:
                    p.write(" ".join(gold[s][uid]) + "\n")
        return data

    def run(self):
        assert_w2vu_env(self.python_exe)
        import json

        fs = _fairseq_dir(self.python_exe)
        data = self._build_data_dir()
        conf = os.path.join(fs, "examples", "speech_recognition", "new", "conf")

        pers = {}
        for s in self.splits:
            resdir = os.path.abspath(f"decode_{s}")
            os.makedirs(resdir, exist_ok=True)
            args = [
                os.fspath(self.python_exe), "-m", "examples.speech_recognition.new.infer",
                f"--config-dir={conf}", "--config-name=infer",
                "task=audio_finetuning", f"task.data={data}", "task.labels=phn",
                "decoding.type=viterbi", "common_eval.post_process=none",
                f"common_eval.path={self.checkpoint.get_path()}",
                f"dataset.gen_subset={s}", f"dataset.max_tokens={self.max_tokens}",
                "distributed_training.distributed_world_size=1",
                f"common_eval.results_path={resdir}", f"decoding.results_path={resdir}",
                f"hydra.run.dir={resdir}",
            ]
            print("RUN:", " ".join(args), flush=True)
            out = sp.check_output(args, stderr=sp.STDOUT, text=True, env=_torch_unsafe_load_env())
            print(out, flush=True)
            m = re.findall(r"Word error rate:\s*([0-9.]+)", out)
            assert m, f"no 'Word error rate:' in infer output for {s}"
            pers[s] = float(m[-1]) / 100.0

        with open(self.out_per.get_path(), "w") as f:
            json.dump(pers, f, indent=2)
        print("PER:", json.dumps(pers, indent=2), flush=True)
