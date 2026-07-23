"""Sisyphus job: AudexDuplex Stage-0 (frozen-backbone audio warmup).

Renders the ``moshi_family.audex.finetune_launcher`` YAML and runs it under the same torchrun + PYTHONPATH
machinery as the other finetunes (``FinetuneAdapter`` + ``launch_training``), mirroring ``RLFinetune``.
Single GPU: the Audex-2B trunk is frozen and only the fresh Mimi audio embeddings + depformer train
(``AudexDuplexLM.freeze_backbone``). ``synthetic=True`` is the dependency-free GPU smoke (random codes,
no Mimi/data); ``synthetic=False`` reuses the Moshi Mimi pipeline on ``train_data`` with the text row
neutralised to the Audex pad (see ``audex/train_data.py``).

The Audex-2B checkpoint is snapshot-downloaded lazily by ``AudexDuplexLM.from_pretrained`` on the compute
node (HF cache on hpcwork), so no separate download job is needed.
"""

import json

from sisyphus import Job, Task

from .finetune import (
    FinetuneAdapter,
    finetune_completed_fraction,
    launch_training,
    prepare_run_dir,
)


def _render_audex_config(job: "AudexDuplexFinetune") -> str:
    """Render the ``moshi_family.audex.finetune_launcher`` YAML (Stage-0 audio-warmup knobs)."""
    train_data = job.train_data.get() if job.train_data is not None else ""
    init_checkpoint = job.init_checkpoint.get() if job.init_checkpoint is not None else ""
    curriculum_json = json.dumps(job.curriculum) if job.curriculum is not None else "null"
    return f"""# AudexDuplex {job.stage} finetune config (moshi_family.audex.finetune_launcher schema)
out_dir: "{job.out_rundir.get()}"
stage: "{job.stage}"
train_data: "{train_data}"
init_checkpoint: "{init_checkpoint}"
mimi_hf_repo: "{job.mimi_hf_repo}"
synthetic: {str(job.synthetic).lower()}
max_steps: {job.max_steps}
grad_accum: {job.grad_accum}
per_gpu_batch: {job.per_gpu_batch}
lr: {job.lr}
lora_rank: {job.lora_rank}
lora_scaling: {job.lora_scaling}
text_replay_frac: {job.text_replay_frac}
curriculum_json: '{curriculum_json}'
warmup_steps: {job.warmup_steps}
grad_clip: 1.0
duration_sec: {job.duration_sec}
save_every: {job.save_every}
log_every: 10
seed: {job.seed}
text_pad_id: {job.text_pad_id}
sample_every: {job.sample_every}
unfreeze_text_readout: {job.unfreeze_text_readout}
text_pad_weight: {job.text_pad_weight}
"""


def _audex_render_shim(job: "AudexDuplexFinetune", _batch_size: int, _max_steps: int) -> str:
    """Adapter ``render_config`` shim (module-level fn so the pickled job stays picklable). Stage-0 sizes
    nothing from the data, so the args are ignored."""
    return _render_audex_config(job)


AUDEX_STAGE0_ADAPTER = FinetuneAdapter(
    name="audex_stage0",
    batch_size=1,
    render_config=_audex_render_shim,
    launcher_module="moshi_family.audex.finetune_launcher",
    fork_module="moshi_family",
)


class AudexDuplexFinetune(Job):
    """Stage-0 frozen-backbone audio warmup for AudexDuplex (Audex-2B trunk frozen; train the Mimi audio
    embeddings + depformer). Renders the launcher YAML + runs it via torchrun on a single GPU."""

    __sis_hash_exclude__ = {
        "warmup_steps": 100,
        "grad_accum": 8,
        "per_gpu_batch": 1,
        "save_every": 500,
        "duration_sec": 20.0,
        "text_pad_id": 0,
        "rqmt": None,
        # Stage-1 knobs default to Stage-0 behaviour so the existing Stage-0 job hash is unchanged; only
        # a non-default `stage`/`init_checkpoint` (Stage-1) forks a distinct job.
        "stage": "stage0",
        "init_checkpoint": None,
        "lora_rank": 16,
        "lora_scaling": 2.0,
        "text_replay_frac": 0.0,
        "curriculum": None,
        # Bump to force a clean re-train after an unhashed lib fix (e.g. the moshi_loss text-pad bug);
        # excluded at 1 so existing jobs keep their hash, a higher value forks a fresh job + cascades eval.
        "code_version": 1,
        "sample_every": 100,
        "unfreeze_text_readout": False,
        "text_pad_weight": 0.5,
    }

    def __init__(
        self,
        *,
        venv_python_path,
        train_data=None,
        stage: str = "stage0",
        init_checkpoint=None,
        lora_rank: int = 16,
        lora_scaling: float = 2.0,
        text_replay_frac: float = 0.0,
        curriculum=None,
        code_version: int = 1,
        sample_every: int = 100,
        unfreeze_text_readout: bool = False,
        text_pad_weight: float = 0.5,
        mimi_hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        synthetic: bool = False,
        max_steps: int = 2000,
        grad_accum: int = 8,
        per_gpu_batch: int = 1,
        lr: float = 1e-4,
        warmup_steps: int = 100,
        duration_sec: float = 20.0,
        save_every: int = 500,
        seed: int = 0,
        text_pad_id: int = 0,
        rqmt: dict | None = None,
    ):
        self.adapter = AUDEX_STAGE0_ADAPTER
        self.venv_python_path = venv_python_path
        self.train_data = train_data
        self.stage = stage
        self.init_checkpoint = init_checkpoint
        self.lora_rank = lora_rank
        self.lora_scaling = lora_scaling
        self.text_replay_frac = text_replay_frac
        self.curriculum = curriculum
        self.code_version = code_version
        self.sample_every = sample_every
        self.unfreeze_text_readout = unfreeze_text_readout
        self.text_pad_weight = text_pad_weight
        self.mimi_hf_repo = mimi_hf_repo
        self.synthetic = synthetic
        self.max_steps = max_steps
        self.grad_accum = grad_accum
        self.per_gpu_batch = per_gpu_batch
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.duration_sec = duration_sec
        self.save_every = save_every
        self.seed = seed
        self.text_pad_id = text_pad_id
        self.out_config = self.output_path("config.yaml")
        self.out_rundir = self.output_path("run_dir", directory=True)
        # Single GPU: 2B frozen trunk + depformer training fits comfortably; give it room + a long wall.
        self.rqmt = rqmt or {"gpu": 1, "cpu": 6, "mem": 64, "time": 8}

    def tasks(self):
        yield Task("write_config", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return finetune_completed_fraction(self, self.adapter)

    def write_config(self):
        prepare_run_dir(self.out_rundir.get())
        with open(self.out_config, "w") as f:
            f.write(_render_audex_config(self))

    def run(self):
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        launch_training(self, self.adapter)
