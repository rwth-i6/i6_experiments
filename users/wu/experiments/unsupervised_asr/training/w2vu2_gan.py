"""Ported from i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/gan.py
(``w2vu2_overrides``, ``FairseqW2vu2TrainJob._write_config``) and w2vu2/pipeline.py
(``build_sae_1c_gan`` with ``encoder=W2V2_LV60_L15``, ``sil_probs=(0.5,)``, ``grid=seed_grid(5)``:
config/sae_1c_w2v2_pilot.py).

Section 1c: the wav2vec-U 2.0 GAN, run as fairseq's reference implementation (fairseq 0.12.2,
``examples/wav2vec/unsupervised``, ``config/gan/w2vu2.yaml``), five seeds, the best seed selected by
the unsupervised ``weighted_lm_ppl``.  The GAN is not reimplemented: every run is i6_core's
``FairseqHydraTrainingJob`` (``<w2vu python> <fairseq_root>/fairseq_cli/hydra_train.py --config-dir
<job>/output --config-name fairseq_hydra_config.yaml checkpoint.save_dir=<job>/output/checkpoints``),
with the software of :mod:`..w2vu2_tools`.

The config (:func:`w2vu2_gan_config`)
-------------------------------------
fairseq's ``w2vu2.yaml`` at the tag, with the source's two edits and its settings:

1. The ``hydra:`` block is dropped (FAIR-cluster submitit launcher; hydra rejects it without the
   plugin, and sisyphus owns scheduling).
2. The amsgrad workaround (source gan.py:194): fairseq 0.12.2 ships a ``w2vu2.yaml`` its own release
   cannot load -- both optimizer groups set ``amsgrad: false``, but the released ``FairseqAdamConfig``
   has no such field ("Key 'amsgrad' not in 'FairseqAdamConfig'").  The key is asserted False and
   dropped; fairseq's Adam defaults to amsgrad=False, so this is a no-op.
3. The settings of the source's W2V2_LV60_L15 arm: ``model.input_dim`` 1024,
   ``model.generator_stride`` 3, ``model.generator_kernel`` 9, ``model.generator_batch_norm`` 30,
   ``model.target_downsample_rate`` 1, ``model.gradient_penalty`` 1.0, ``model.smoothness_weight`` 1.5,
   ``model.code_penalty`` 3.0, ``model.mmi_weight`` 0.5, ``common.seed`` (0..4),
   ``optimization.max_update`` 150000; the paths ``task.data``, ``task.text_data``,
   ``task.kenlm_path``, ``common.user_dir`` (the tag's ``examples/wav2vec/unsupervised``);
   ``task.aux_target_postfix`` km; ``distributed_training.distributed_world_size`` 1.

Port changes, none of which changes a value fairseq sees (checked against the source's resolved
config in its ``train.log``; see the tests):

* The yaml is vendored (:data:`W2VU2_YAML`, the tag's file verbatim) instead of being read from the
  installed package at run time: i6_core writes the config from the graph-time dict, and the
  fairseq root is a job output.  A test asserts the vendored text equals the tag's file.
* i6_core owns ``optimization.max_update``, ``optimization.max_epoch`` and
  ``checkpoint.save_interval`` (``max_update``/``max_epoch``/``save_interval`` of the job, written
  into the yaml unhashed); they are removed from the hashed dict and passed as 150000, 0 (fairseq's
  default, the source never set it) and 1000 (the yaml's value).  ``checkpoint.save_dir`` is i6_core's
  ``checkpoints/`` dir (a command line override), so it is dropped from the dict too.
* ``lr_scheduler: pass_through`` is written as ``{"_name": "pass_through"}`` (i6_core's
  ``FairseqHydraConfig.check_consistency`` calls ``.get`` on every top-level value, which fails on a
  string); fairseq resolves both to the same node (the source's own resolved config prints
  ``'lr_scheduler': {'_name': 'pass_through'}``).
* The yaml is dumped by i6_core with sorted keys; the source kept the file order.  Hydra composes
  by key, not by order.

Selection (:class:`W2vu2GanSelectJob`)
--------------------------------------
Best of the five seeds by ``weighted_lm_ppl`` (the paper's protocol and the source's pilot config:
"Selection = best-of-5 seeds by weighted_lm_ppl"): per seed, fairseq's own best validation value
``extra_state["best"]`` of ``checkpoint_last.pt`` (the value its ``checkpoint_best.pt`` was saved
at, ``best_checkpoint_metric: weighted_lm_ppl``, lower is better); the argmin seed's
``checkpoint_best.pt`` is the selected GAN.  No label is read.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sisyphus import Job, Task, tk

__all__ = [
    "W2VU2_YAML",
    "GAN_SEEDS",
    "GAN_MAX_UPDATE",
    "GAN_SAVE_INTERVAL",
    "GAN_RQMT",
    "w2vu2_base_config",
    "w2vu2_overrides",
    "w2vu2_gan_config",
    "build_w2vu2_gan",
    "W2vu2GanSelectJob",
    "W2vu2Gan",
    "get_w2vu2_gan",
]

#: ``examples/wav2vec/unsupervised/config/gan/w2vu2.yaml`` of fairseq v0.12.2, verbatim.
W2VU2_YAML = r"""# @package _group_

common:
  fp16: false
  fp16_no_flatten_grads: true
  log_format: json
  log_interval: 100
  tensorboard_logdir: tb
  reset_logging: false
  suppress_crashes: false

checkpoint:
  save_interval: 1000
  save_interval_updates: 1000
  no_epoch_checkpoints: true
  best_checkpoint_metric: weighted_lm_ppl
  save_dir: .

distributed_training:
  distributed_world_size: 1

task:
  _name: unpaired_audio_text
  data: ???
  text_data: ???
  labels: phn
  sort_by_length: false
  unfiltered: false
  max_length: null
  append_eos: false
  kenlm_path: ???
  aux_target_postfix: km

dataset:
  num_workers: 6
  batch_size: 160
  skip_invalid_size_inputs_valid_test: true
  valid_subset: valid
  validate_interval: 1000
  validate_interval_updates: 1000

criterion:
  _name: model
  log_keys:
    - accuracy_dense
    - accuracy_token
    - temp
    - code_ppl

optimization:
  max_update: 150000
  clip_norm: 5.0
  lr: [0]

optimizer:
  _name: composite
  groups:
    generator:
      lr: [0.00005]
      lr_float: null
      optimizer:
        _name: adam
        adam_betas: [0.5,0.98]
        adam_eps: 1e-06
        weight_decay: 0
        amsgrad: false
      lr_scheduler:
        _name: fixed
        warmup_updates: 0
    discriminator:
      lr: [ 0.0003 ]
      lr_float: null
      optimizer:
        _name: adam
        adam_betas: [0.5,0.98]
        adam_eps: 1e-06
        weight_decay: 0.0001
        amsgrad: false
      lr_scheduler:
        _name: fixed
        warmup_updates: 0

lr_scheduler: pass_through

model:
  _name: wav2vec_u

  discriminator_dim: 384
  discriminator_depth: 2
  discriminator_kernel: 8
  discriminator_linear_emb: false
  discriminator_causal: true
  discriminator_max_pool: false
  discriminator_act_after_linear: false
  discriminator_dropout: 0.0
  discriminator_weight_norm: false

  generator_stride: 3
  generator_kernel: 9
  generator_bias: false
  generator_dropout: 0.1
  generator_batch_norm: 30
  generator_residual: true

  smoothness_weight: 1.5
  smoothing: 0
  smoothing_one_sided: false
  gumbel: false
  hard_gumbel: false
  gradient_penalty: 1.0
  code_penalty: 3.0
  temp: [ 2,0.1,0.99995 ]
  input_dim: 1024
  mmi_weight: 0.5
  target_dim: 64

  segmentation:
    type: JOIN
    mean_pool_join: false
    remove_zeros: false


hydra:
  job:
    config:
      override_dirname:
        kv_sep: ':'
        item_sep: '__'
        exclude_keys:
          - run_config
          - distributed_training.distributed_port
          - common.user_dir
          - task.data
          - task.kenlm_path
          - task.text_data
          - model.generator_layers
          - task.labels
          - task.force_model_seed
  sweep:
    dir: /checkpoint/${env:USER}/${env:PREFIX}/${hydra.job.config_name}/${hydra.job.override_dirname}
    subdir: ${hydra.job.num}
  launcher:
    submitit_folder: ${hydra.sweep.dir}
    timeout_min: 3000
    cpus_per_task: 10
    gpus_per_node: 1
    tasks_per_node: 1
    mem_gb: 120
    nodes: 1
    name: ${env:PREFIX}_${hydra.job.config_name}
    partition: devlab,learnlab,learnfair,scavenge
    comment: intern_endding_soon
    constraint: volta32gb
    max_num_timeout: 30
"""

#: ``common.user_dir``, relative to the fairseq root (the source: ``<fairseq>/examples/wav2vec/unsupervised``)
FAIRSEQ_USER_DIR = "examples/wav2vec/unsupervised"
#: the source's ``seed_grid(5)``
GAN_SEEDS = (0, 1, 2, 3, 4)
#: ``optimization.max_update`` (the source's ``w2vu2_overrides`` default and the yaml's value)
GAN_MAX_UPDATE = 150_000
#: ``checkpoint.save_interval`` of the yaml (epochs; ``save_interval_updates`` 1000 does the saving)
GAN_SAVE_INTERVAL = 1000
#: the source's ``FairseqW2vu2TrainJob`` rqmt (``gpu_mem`` 80, ``time_rqmt`` 11.5), except ``mem``: 100, not
#: the source's 60.  The source's 60 was never enforced (JUPITER job 968302: ``--exclusive``, sacct
#: AllocTRES ``mem=0``), so its success is no evidence that 60 suffices.  Measured need (review
#: reports/review_gan_port_2026-09-25.md, F1, from s0's usage.run.1 and sacct): a file working set of about
#: 42 GB that fairseq mmaps and keeps hot (train.npy 31.6 GB + text train.bin/.idx 7.0 GB + valid.npy
#: 3.3 GB; shared between the 6+6 persistent DataLoader workers, charged once) plus up to about 45-57 GB
#: of anonymous memory (main process about 45 GiB at exit, about 2 GiB per worker at fork); the 307.7 GiB
#: sisyphus peak is a per-process RSS sum that counts the shared mmap pages once per worker.  Under a
#: cgroup-enforced 60 GB the train.npy pages would be re-read every epoch, or the job OOM-killed.
#: rqmt is not hashed (FairseqHydraTrainingJob.hash).
GAN_RQMT = {"gpu": 1, "gpu_mem": 80, "mem": 100, "time": 11.5, "cpu": 8}


def w2vu2_base_config() -> Dict[str, Any]:
    """:data:`W2VU2_YAML` with the source's two edits: ``hydra`` dropped, ``amsgrad`` asserted False
    and dropped (module docstring)."""
    import yaml

    cfg = yaml.safe_load(W2VU2_YAML)
    cfg.pop("hydra", None)
    for group in cfg.get("optimizer", {}).get("groups", {}).values():
        opt = group.get("optimizer", {})
        if "amsgrad" in opt:
            assert opt["amsgrad"] is False, f"amsgrad={opt['amsgrad']} cannot be honoured"
            opt.pop("amsgrad")
    return cfg


def w2vu2_overrides(
    *,
    input_dim: int,
    generator_stride: int,
    generator_kernel: int,
    generator_batch_norm: int = 30,
    target_downsample_rate: int = 1,
    gradient_penalty: float = 1.0,
    smoothness_weight: float = 1.5,
    code_penalty: float = 3.0,
    mmi_weight: float = 0.5,
    seed: int = 0,
) -> Dict[str, Any]:
    """The source's ``w2vu2_overrides`` without ``optimization.max_update`` (an i6_core job argument
    here).  The encoder geometry has no default: the source's defaults were its BEST-RQ arm's."""
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
    }


def w2vu2_gan_config(
    *,
    data_dir: tk.Path,
    text_data: tk.Path,
    kenlm_path: tk.Path,
    overrides: Dict[str, Any],
    user_dir: tk.Path,
    aux_target_postfix: str = "km",
):
    """The ``FairseqHydraConfig`` of one GAN run (module docstring)."""
    from i6_core.fairseq.training import FairseqHydraConfig

    cfg = w2vu2_base_config()
    # owned by FairseqHydraTrainingJob (max_update / max_epoch / save_interval) or its command line
    # (checkpoint.save_dir); asserted at the yaml's values so that a changed yaml fails here
    assert cfg["optimization"].pop("max_update") == GAN_MAX_UPDATE
    assert cfg["checkpoint"].pop("save_interval") == GAN_SAVE_INTERVAL
    assert cfg["checkpoint"].pop("save_dir") == "."
    # i6_core's check_consistency needs a dict at the top level (module docstring)
    assert cfg["lr_scheduler"] == "pass_through"
    cfg["lr_scheduler"] = {"_name": "pass_through"}

    settings = dict(overrides)
    settings.update({
        "task.data": data_dir,
        "task.text_data": text_data,
        "task.kenlm_path": kenlm_path,
        "task.aux_target_postfix": aux_target_postfix,
        "common.user_dir": user_dir,
        "distributed_training.distributed_world_size": 1,
    })
    for k, v in sorted(settings.items()):
        node = cfg
        *parents, leaf = k.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = v

    def _missing(x) -> bool:
        if isinstance(x, dict):
            return any(_missing(v) for v in x.values())
        if isinstance(x, list):
            return any(_missing(v) for v in x)
        return x == "???"

    assert not _missing(cfg), "unfilled MISSING (???) left in the config"
    return FairseqHydraConfig(cfg)


def build_w2vu2_gan(
    *,
    seed: int,
    data_dir: tk.Path,
    text_data: tk.Path,
    kenlm_path: tk.Path,
    fairseq_python_exe: Optional[tk.Path] = None,
    fairseq_root: Optional[tk.Path] = None,
):
    """One seed of the W2V2_LV60_L15 GAN as ``FairseqHydraTrainingJob``.

    Outputs used downstream: ``out_checkpoint_dir`` (``checkpoints/``, holding ``checkpoint_best.pt``,
    ``checkpoint_last.pt`` and ``checkpoint_<epoch>_<update>.pt`` every 1000 updates) and
    ``out_fairseq_hydra_yaml``.
    """
    from i6_core.fairseq.training import FairseqHydraTrainingJob

    from ..w2vu2_tools import get_fairseq_root, get_w2vu_python

    root = fairseq_root if fairseq_root is not None else get_fairseq_root()
    config = w2vu2_gan_config(
        data_dir=data_dir,
        text_data=text_data,
        kenlm_path=kenlm_path,
        overrides=w2vu2_overrides(input_dim=1024, generator_stride=3, generator_kernel=9, seed=seed),
        user_dir=root.join_right(FAIRSEQ_USER_DIR),
    )
    return FairseqHydraTrainingJob(
        config,
        max_epoch=0,
        max_update=GAN_MAX_UPDATE,
        save_interval=GAN_SAVE_INTERVAL,
        keep_epochs=[],
        rqmt=dict(GAN_RQMT),
        fairseq_python_exe=fairseq_python_exe if fairseq_python_exe is not None else get_w2vu_python(),
        fairseq_root=root,
    )


class W2vu2GanSelectJob(Job):
    """Best-of-seeds by ``weighted_lm_ppl`` (module docstring), CPU, in-process.

    :param checkpoint_dirs: ``{seed: FairseqHydraTrainingJob.out_checkpoint_dir}``.
    :param hydra_yamls: ``{seed: FairseqHydraTrainingJob.out_fairseq_hydra_yaml}``.

    Outputs: ``out_checkpoint`` (symlink to the selected seed's ``checkpoint_best.pt``),
    ``out_hydra_yaml`` (symlink to its ``fairseq_hydra_config.yaml``), ``out_selection``
    (``selection.json``: per seed the best ``weighted_lm_ppl`` and its update, the selected seed) and
    the output var ``out_seed``.
    """

    def __init__(self, *, checkpoint_dirs: Dict[int, tk.Path], hydra_yamls: Dict[int, tk.Path]):
        super().__init__()
        assert sorted(checkpoint_dirs) == sorted(hydra_yamls), (sorted(checkpoint_dirs), sorted(hydra_yamls))
        self.checkpoint_dirs = {int(s): checkpoint_dirs[s] for s in sorted(checkpoint_dirs)}
        self.hydra_yamls = {int(s): hydra_yamls[s] for s in sorted(hydra_yamls)}

        self.out_checkpoint = self.output_path("checkpoint_best.pt")
        self.out_hydra_yaml = self.output_path("fairseq_hydra_config.yaml")
        self.out_selection = self.output_path("selection.json")
        self.out_seed = self.output_var("seed")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def read_best(checkpoint_dir: str) -> Dict[str, Any]:
        """fairseq's best ``weighted_lm_ppl`` of one run (``checkpoint_last.pt`` ``extra_state``), and
        the update of its ``checkpoint_best.pt``."""
        import os

        import torch

        last = torch.load(os.path.join(checkpoint_dir, "checkpoint_last.pt"), map_location="cpu", weights_only=True)
        best = torch.load(os.path.join(checkpoint_dir, "checkpoint_best.pt"), map_location="cpu", weights_only=True)
        value = float(last["extra_state"]["best"])
        assert float(best["extra_state"]["val_loss"]) == value, (checkpoint_dir, best["extra_state"], value)
        return {
            "weighted_lm_ppl": value,
            "best_update": int(best["optimizer_history"][-1]["num_updates"]),
            "last_update": int(last["optimizer_history"][-1]["num_updates"]),
        }

    def run(self):
        import json
        import os

        per_seed = {s: self.read_best(d.get_path()) for s, d in self.checkpoint_dirs.items()}
        seed = min(per_seed, key=lambda s: (per_seed[s]["weighted_lm_ppl"], s))
        os.symlink(os.path.join(self.checkpoint_dirs[seed].get_path(), "checkpoint_best.pt"),
                   self.out_checkpoint.get_path())
        os.symlink(self.hydra_yamls[seed].get_path(), self.out_hydra_yaml.get_path())
        with open(self.out_selection.get_path(), "w") as f:
            json.dump({"metric": "weighted_lm_ppl", "rule": "argmin over seeds",
                       "per_seed": {str(s): v for s, v in per_seed.items()}, "selected_seed": seed}, f, indent=2)
        self.out_seed.set(seed)
        print(json.dumps({s: v["weighted_lm_ppl"] for s, v in per_seed.items()}), "->", seed, flush=True)


class W2vu2Gan:
    """What :func:`get_w2vu2_gan` wires: the data, the per-seed trainings and the selection."""

    def __init__(self, *, data, text_data, kenlm_path, trainings, selection):
        self.data = data                    # data.w2vu2_features.W2vu2FeatureDataJob
        self.text_data = text_data          # lm.w2vu2_text.FairseqTextDataJob
        self.kenlm_path = kenlm_path        # tk.Path, the phone 4-gram binary
        self.trainings = trainings          # {seed: FairseqHydraTrainingJob}
        self.selection = selection          # W2vu2GanSelectJob


def get_w2vu2_gan(seeds: Sequence[int] = GAN_SEEDS) -> W2vu2Gan:
    """The section 1c graph: audio data, text data, phone LM, one GAN per seed, the selection.

    Aliases follow the source's (``sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s<seed>``).
    """
    from ..data.w2vu2_features import get_w2vu2_feature_data
    from ..lm.w2vu2_text import get_w2vu2_phone_lm, get_w2vu2_text_data

    data = get_w2vu2_feature_data()
    text = get_w2vu2_text_data()
    kenlm = get_w2vu2_phone_lm()
    prefix = "sae/1c/w2v2_lv60_l15/gan_l15_sil0.5"
    trainings: Dict[int, Any] = {}
    for seed in seeds:
        job = build_w2vu2_gan(seed=seed, data_dir=data.out_dir, text_data=text.out_dir, kenlm_path=kenlm)
        job.add_alias(f"{prefix}/s{seed}")
        trainings[seed] = job
    select = W2vu2GanSelectJob(
        checkpoint_dirs={s: j.out_checkpoint_dir for s, j in trainings.items()},
        hydra_yamls={s: j.out_fairseq_hydra_yaml for s, j in trainings.items()},
    )
    select.add_alias(f"{prefix}/select")
    return W2vu2Gan(data=data, text_data=text, kenlm_path=kenlm, trainings=trainings, selection=select)
