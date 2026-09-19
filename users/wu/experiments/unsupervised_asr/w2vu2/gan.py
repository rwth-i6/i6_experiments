"""SAE §1c — the wav2vec-U 2.0 GAN, run as fairseq's reference implementation.

We do not reimplement the GAN. `fairseq-hydra-train` is invoked with fairseq's own `w2vu2.yaml`
plus the overrides below, so every documented paper-vs-code divergence (gradient-penalty norm taken
over the *time* axis, perplexity-space diversity `(V-PPL)/V`, mean-reduction smoothness, BCE with
flipped labels, 1:1 G/D alternation) is inherited from the reference rather than re-derived.

Overrides forced by our encoder (BEST-RQ 512-d @ 25 Hz vs wav2vec2-Large 1024-d @ 50 Hz):

  input_dim              1024 -> 512
  generator_stride          3 -> 2     25/2 = 12.5 Hz output; theirs is 50/3 = 16.7 Hz
  generator_kernel          9 -> 5     time-matched: 9 frames @50 Hz = 180 ms ~ 5 frames @25 Hz
  target_downsample_rate    2 -> 1     our .km is pre-aligned to the 25 Hz encoder rate (features.py)

The stride is the one deviation with published evidence behind it: Table 1 shows generator *output*
rate is what governs convergence (14-16.7 Hz converges; 25-28 Hz gives >100 PER), and our measured
phone rate is ~11.9 Hz (SAE_0a: ~2.1 frames/phone @25 Hz), so stride 2 -> 12.5 Hz is closer to
ground truth than the paper's own 16.7 Hz. Stride 1 (=25 Hz) sits exactly on their divergent
configuration and is expected to fail; it is kept only as a labelled negative control.

`generator_batch_norm` (gamma init) is the single item the paper calls "critical for convergence"
(30 for wav2vec2-Large, 35 for XLSR-53 -- i.e. it tracks the feature distribution, so it must be
swept for BEST-RQ rather than inherited; SAE_PLAN §1c says {20, 30, 40}).
"""

from __future__ import annotations

import os
import re
import subprocess as sp
import threading
import time
from typing import Any, Dict, Optional

import yaml
from sisyphus import Job, Task, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import W2VU_PYTHON, assert_w2vu_env

_alias_prefix = "sae/1c"

# SAE §4a step 4: the fairseq user module that adds the reverse term (a package dir, so that the
# `sys.path.insert(parent)` fairseq does for a user dir exposes nothing else).
REV_USER_DIR = os.path.join(os.path.dirname(__file__), "userdir", "w2vu_rev")

# phi's optimizer, inherited verbatim from the registered blank-free bed
# (emc_train_jobs.PHI_LEARNING_RATE / EMC_ADAM_BETAS / EMC_ADAM_EPS / EMC_WEIGHT_DECAY, which are
# themselves §1c's generator-group values at phi's own learning rate).
REV_PHI_LR = 3.0e-3
REV_PHI_ADAM_BETAS = [0.5, 0.98]
REV_PHI_ADAM_EPS = 1e-06
REV_PHI_WEIGHT_DECAY = 0.0


def w2vu2_overrides(
    *,
    input_dim: int = 512,
    generator_stride: int = 2,
    generator_kernel: int = 5,
    generator_batch_norm: int = 30,
    target_downsample_rate: int = 1,
    gradient_penalty: float = 1.0,
    smoothness_weight: float = 1.5,
    code_penalty: float = 3.0,
    mmi_weight: float = 0.5,
    seed: int = 0,
    max_update: int = 150_000,
) -> Dict[str, Any]:
    """The (model, optimization, checkpoint) knobs we set; everything else stays at w2vu2.yaml."""
    return {
        "model.input_dim": input_dim,
        "model.generator_stride": generator_stride,
        "model.generator_kernel": generator_kernel,
        "model.generator_batch_norm": generator_batch_norm,
        "model.target_downsample_rate": target_downsample_rate,
        "model.gradient_penalty": gradient_penalty,
        "model.smoothness_weight": smoothness_weight,
        "model.code_penalty": code_penalty,
        "model.mmi_weight": mmi_weight,
        "common.seed": seed,
        "optimization.max_update": max_update,
    }


class FairseqW2vu2TrainJob(Job):
    """One GAN run = fairseq-hydra-train, spawned under the `w2vu` env python (py3.9 + torch 2.6+cu126).

    worker_wrapper always runs the worker itself under the speech_llm python, so fairseq is an explicit
    subprocess; `requires_env` is what stops it inheriting speech_llm's LD_LIBRARY_PATH.
    """

    requires_env = "w2vu"  # class attr -> not an __init__ arg -> not hashed (settings.py::worker_wrapper)

    # SAE §4a step 4 parameters. Conditional exclusion (Job.hash): a parameter left at the value
    # below is dropped from the hash, so every job that existed before this delta keeps its dir name,
    # while any arm that actually sets one gets a new hash. Verified by census, not by reasoning.
    __sis_hash_exclude__ = {
        "lam_rev": None,
        "rev_units_dir": None,
        "rev_batch_utts": None,
        "rev_frozen_phi": None,
    }

    def __init__(
        self,
        *,
        data_dir: tk.Path,
        text_data: tk.Path,
        kenlm_path: tk.Path,
        overrides: Dict[str, Any],
        aux_target_postfix: str = "km",
        python_exe: tk.Path = W2VU_PYTHON,
        time_rqmt: float = 11.5,
        gpu_mem: int = 80,
        lam_rev: Optional[float] = None,
        rev_units_dir: Optional[tk.Path] = None,
        rev_batch_utts: Optional[int] = None,
        rev_frozen_phi: Optional[tk.Path] = None,
    ):
        """``lam_rev`` > 0 switches the run to the reverse-term user module (SAE §4a step 4).

        :param lam_rev: weight of ``L_rev`` in the generator loss. ``None`` and ``0.0`` both run the
            reproduction's own config, byte for byte -- no phi, no unit field, fairseq's own
            ``unpaired_audio_text`` / ``wav2vec_u``. ``0.0`` differs from ``None`` only in the job
            hash, which is what makes the weight-0 rerun a separate job dir.
        :param rev_units_dir: ``W2vu2RevUnitsJob`` output dir ({split}.rev500, {split}.eta.npy).
        :param rev_batch_utts: b, the utterances per generator update the term is computed on
            (``None`` / 0 = the whole batch, which reproduces the full term).
        :param rev_frozen_phi: a blank-free checkpoint to load phi from; phi is then never updated
            and the run declares no ``reverse`` optimizer group.
        """
        super().__init__()
        self.data_dir = data_dir
        self.text_data = text_data
        self.kenlm_path = kenlm_path
        self.overrides = dict(overrides)
        self.aux_target_postfix = aux_target_postfix
        self.python_exe = python_exe
        self.lam_rev = lam_rev
        self.rev_units_dir = rev_units_dir
        self.rev_batch_utts = rev_batch_utts
        self.rev_frozen_phi = rev_frozen_phi
        if self._rev_active():
            assert rev_units_dir is not None, "lam_rev > 0 needs rev_units_dir"

        self.out_dir = self.output_path("train", directory=True)
        self.out_best = self.output_path("train/checkpoint_best.pt")
        self.out_last = self.output_path("train/checkpoint_last.pt")
        self.out_log = self.output_path("train.log")
        self.rqmt = {"gpu": 1, "gpu_mem": gpu_mem, "mem": 60, "time": time_rqmt, "cpu": 8}

    def _rev_active(self) -> bool:
        return self.lam_rev is not None and float(self.lam_rev) > 0

    def tasks(self):
        # resume="run": fairseq restarts from checkpoint_last.pt in the save dir, so a run that hits
        # the allocation limit continues in the next allocation instead of starting over.
        yield Task("run", resume="run", rqmt=self.rqmt)

    def _fairseq_dir(self) -> str:
        out = sp.check_output(
            [os.fspath(self.python_exe), "-c", "import fairseq, os; print(os.path.dirname(fairseq.__file__))"],
            text=True,
        )
        return out.strip()

    def _write_config(self, fs: str) -> str:
        """Materialize fairseq's w2vu2.yaml with our settings merged in, and return its dir.

        Two things are done to the reference yaml, both necessary:

        1. The `hydra:` block is dropped. It is FAIR-cluster infrastructure -- a submitit launcher,
           `/checkpoint/$USER` sweep dirs, `partition: devlab,learnlab,learnfair`,
           `constraint: volta32gb`. Hydra rejects `hydra.launcher.submitit_folder` outright without
           the submitit plugin ("Key 'submitit_folder' not in 'BasicLauncherConf'"), and *installing*
           that plugin would be worse: hydra would submit a second Slurm job from inside the one
           sisyphus already allocated. Sisyphus owns scheduling; hydra only builds the model.
        2. Every setting is written into the yaml rather than passed as a `k=v` CLI override, because
           hydra's append rules here are subtle enough to get wrong: fairseq registers `common`,
           `dataset`, `optimization` ... as typed dataclasses, so `common.seed` exists even though the
           yaml omits it (a `+` there fails with "item is already at"), while `model` stays an
           untyped dict until `_name: wav2vec_u` resolves, so the real field
           `model.target_downsample_rate` needs `+` (without it: "not in struct"). Writing the
           merged config sidesteps the distinction and leaves the exact config in the job dir.

        Derived from the installed yaml at runtime rather than vendored, so model/task/optimization
        stay pinned to the reference we claim to reproduce.
        """
        src = os.path.join(fs, "examples", "wav2vec", "unsupervised", "config", "gan", "w2vu2.yaml")
        with open(src) as f:
            cfg = yaml.safe_load(f)
        cfg.pop("hydra", None)

        # fairseq 0.12.2 ships a w2vu2.yaml its own release cannot load: both optimizer groups set
        # `amsgrad: false`, but the released FairseqAdamConfig has no such field (the configs come
        # from a branch whose dataclass did) -> "Key 'amsgrad' not in 'FairseqAdamConfig'". Dropping
        # it is a no-op, not a deviation: fairseq's Adam already defaults amsgrad=False
        # (optim/adam.py:144, and `group.get("amsgrad", False)`). Asserted, so that a future config
        # actually asking for amsgrad fails loudly instead of being silently ignored.
        for group in cfg.get("optimizer", {}).get("groups", {}).values():
            opt = group.get("optimizer", {})
            if "amsgrad" in opt:
                assert opt["amsgrad"] is False, f"amsgrad={opt['amsgrad']} cannot be honoured"
                opt.pop("amsgrad")

        settings = dict(self.overrides)
        settings.update({
            "task.data": self.data_dir.get_path(),
            "task.text_data": self.text_data.get_path(),
            "task.kenlm_path": self.kenlm_path.get_path(),
            "task.aux_target_postfix": self.aux_target_postfix,
            "common.user_dir": os.path.join(fs, "examples", "wav2vec", "unsupervised"),
            "checkpoint.save_dir": self.out_dir.get_path(),
            "distributed_training.distributed_world_size": 1,
        })
        if self._rev_active():
            settings.update(self._rev_settings())
        for k, v in sorted(settings.items()):
            node = cfg
            *parents, leaf = k.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = v

        assert "???" not in yaml.safe_dump(cfg), "unfilled MISSING (???) left in the config"

        out = os.path.abspath("config_gan")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "w2vu2.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return out

    def _rev_settings(self) -> Dict[str, Any]:
        """The SAE §4a step-4 delta on top of the reproduction's config, and nothing else.

        The user dir moves to the `w2vu_rev` package, which registers fairseq's own task and model
        first and then the two subclasses; task and model names move to those subclasses; phi gets
        its own composite-optimizer group unless it is frozen (fairseq's composite optimizer asserts
        that the declared groups are exactly the `param_group` tags it finds, and frozen parameters
        never reach it).

        `model.rev_phi_seed` is the run's own `common.seed`: phi's cold initialization follows the
        arm's seed, the way every other initialization in the run does.
        """
        seed = int(self.overrides.get("common.seed", 0))
        s: Dict[str, Any] = {
            "common.user_dir": REV_USER_DIR,
            "task._name": "unpaired_audio_text_rev",
            "task.rev_units_dir": self.rev_units_dir.get_path(),
            "model._name": "wav2vec_u_rev",
            "model.lam_rev": float(self.lam_rev),
            "model.rev_batch_utts": int(self.rev_batch_utts or 0),
            "model.rev_phi_seed": seed,
        }
        if self.rev_frozen_phi is not None:
            s["model.rev_frozen_phi"] = self.rev_frozen_phi.get_path()
        else:
            s.update({
                "optimizer.groups.reverse.lr": [REV_PHI_LR],
                "optimizer.groups.reverse.lr_float": None,
                "optimizer.groups.reverse.optimizer._name": "adam",
                "optimizer.groups.reverse.optimizer.adam_betas": REV_PHI_ADAM_BETAS,
                "optimizer.groups.reverse.optimizer.adam_eps": REV_PHI_ADAM_EPS,
                "optimizer.groups.reverse.optimizer.weight_decay": REV_PHI_WEIGHT_DECAY,
                "optimizer.groups.reverse.lr_scheduler._name": "fixed",
                "optimizer.groups.reverse.lr_scheduler.warmup_updates": 0,
            })
        return s

    def run(self):
        assert_w2vu_env(self.python_exe)
        fs = self._fairseq_dir()
        cfg_dir = self._write_config(fs)

        # No `-m`: hydra multirun exists to fan out a sweep, which is sisyphus' job here. In multirun
        # hydra also instantiates a launcher, which is what drags the stripped block back in.
        args = [
            os.fspath(self.python_exe), "-m", "fairseq_cli.hydra_train",
            f"--config-dir={cfg_dir}", "--config-name=w2vu2",
        ]
        print("RUN:", " ".join(args), flush=True)
        # append, not truncate: with resume="run" a second allocation continues the same training
        # from checkpoint_last.pt and its log must not overwrite the first one's.
        with open(self.out_log.get_path(), "a") as log:
            sp.check_call(args, stdout=log, stderr=sp.STDOUT)


class FairseqW2vu2ProfileJob(FairseqW2vu2TrainJob):
    """A short GAN run whose cost is measured: seconds per update and peak GPU memory.

    Same code path, config and data as ``FairseqW2vu2TrainJob`` -- only ``optimization.max_update``
    is short (the config sets it) and the run is instrumented. This is the profile SAE_4A_attrib.md
    requires before step 4 is funded: 100 updates at lam_rev = 1.0 with b = 160 (all) and b = 16, and
    at lam_rev = 0, so the projected wall time of a 150k-update run can be compared to the 0.24 s per
    update of the reproduction.

    ``sec_per_update`` is read off fairseq's own ``ups`` meter (updates per second over the last log
    interval, i.e. the average the run itself reports); ``peak_gpu_mib`` is the maximum of
    ``nvidia-smi`` samples taken every 2 s in this process, and ``peak_alloc_gb`` is
    ``total - min(gb_free)`` over the log records, i.e. torch's own ``max_memory_allocated``.

    The run logs every ``log_interval`` updates (10, against the reproduction's 100), so a
    100-update profile yields a speed series at 10, 20, ... 100 instead of one end-of-run number.
    ``sec_per_update_warm`` is the mean of ``1 / ups`` over the records from ``WARMUP_UPDATES`` on
    -- the records are equally spaced, so that mean is the window's total time per update, with the
    warmup updates (CUDA/cuDNN autotuning, first lattice allocation) left out.
    """

    # fairseq 0.12.2 writes `[<ts>][train_inner][INFO] - {json}`; older releases wrote
    # `<ts> | INFO | train_inner | {json}`. Match the tag and take the trailing object, so the
    # parse does not depend on the separator of the installed release.
    _TRAIN_INNER_RE = re.compile(r"train_inner.*?(\{.*\})\s*$")

    _RECORD_KEYS = ("num_updates", "ups", "wps", "wall", "train_wall", "gb_free",
                    "loss", "loss_rev", "rev_per_frame", "rev_batch_utts")

    # The speed series is averaged from this update on; earlier updates carry the warmup the
    # 150k-update projection must not inherit.
    WARMUP_UPDATES = 20

    def __init__(self, *, max_update_cap: int = 2000, log_interval: int = 10, **kwargs):
        """:param log_interval: updates between fairseq ``train_inner`` records (the speed series).

        The reproduction logs every 100 updates, i.e. a 100-update profile would report a single
        number covering the warmup as well; 10 gives 9 records in updates 20..100 to average.
        """
        super().__init__(**kwargs)
        updates = int(self.overrides.get("optimization.max_update", 0))
        assert 0 < updates <= max_update_cap, (
            f"a profile job must be short: optimization.max_update = {updates}"
        )
        self.log_interval = int(log_interval)
        assert self.log_interval > 0
        self.overrides["common.log_interval"] = self.log_interval
        self.out_profile = self.output_path("profile.json")
        self.rqmt = dict(self.rqmt)
        self.rqmt["time"] = 1

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)  # a profile is re-run from scratch, never resumed

    @staticmethod
    def _gpu_totals() -> float:
        out = sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
        )
        return float(out.strip().splitlines()[0])

    @classmethod
    def _parse_train_inner(cls, log_path: str) -> list:
        """fairseq's ``train_inner`` json records in a train.log, oldest first."""
        import json

        records = []
        with open(log_path) as fh:
            for line in fh:
                m = cls._TRAIN_INNER_RE.search(line.rstrip())
                if not m:
                    continue
                try:
                    rec = json.loads(m.group(1))
                except ValueError:
                    continue
                if "ups" in rec and "num_updates" in rec:
                    records.append({k: rec[k] for k in rec if k in cls._RECORD_KEYS})
        return records

    def run(self):
        import json

        stop = threading.Event()
        peak = [0.0]

        def sample():
            while not stop.wait(2.0):
                try:
                    out = sp.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        text=True,
                    )
                    peak[0] = max(peak[0], max(float(x) for x in out.strip().splitlines()))
                except Exception as e:  # a failed sample must never kill the run
                    print(f"nvidia-smi sample failed: {e}", flush=True)

        watcher = threading.Thread(target=sample, daemon=True)
        watcher.start()
        t0 = time.time()
        try:
            super().run()
        finally:
            stop.set()
            watcher.join(timeout=10)
        wall = time.time() - t0

        records = self._parse_train_inner(self.out_log.get_path())
        assert records, f"no train_inner json record with 'ups' in {self.out_log.get_path()}"
        last = records[-1]
        warm = [r for r in records if float(r["num_updates"]) >= self.WARMUP_UPDATES]
        total_mib = self._gpu_totals()
        gb_free = [float(r["gb_free"]) for r in records if "gb_free" in r]
        profile = {
            "lam_rev": self.lam_rev,
            "rev_batch_utts": self.rev_batch_utts,
            "max_update": int(self.overrides.get("optimization.max_update", 0)),
            "num_updates": int(float(last["num_updates"])),
            "sec_per_update": 1.0 / float(last["ups"]),
            "ups": float(last["ups"]),
            "log_interval": self.log_interval,
            "warmup_updates": self.WARMUP_UPDATES,
            "n_records_warm": len(warm),
            "sec_per_update_warm": (
                sum(1.0 / float(r["ups"]) for r in warm) / len(warm) if warm else None
            ),
            "job_wall_sec": wall,
            "peak_gpu_mib": peak[0],
            "gpu_total_mib": total_mib,
            "peak_alloc_gb": (total_mib / 1024.0 - min(gb_free)) if gb_free else None,
            "records": records,
        }
        with open(self.out_profile.get_path(), "w") as fh:
            json.dump(profile, fh, indent=2)
        print(json.dumps({k: v for k, v in profile.items() if k != "records"}, indent=2), flush=True)
