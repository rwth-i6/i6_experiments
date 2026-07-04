"""Sisyphus job for GRPO interactivity-alignment post-training (arXiv 2606.11167).

A thin Job that mirrors ``SpeechFinetune`` but for the RL trainer: it writes the
``moshi_family.rl.launcher`` YAML config and runs it under the same torchrun + PYTHONPATH machinery
(``finetune.launch_training``), so the lib (``moshi_family``) is on the path and the HF cache is wired
exactly as the SFT path. Unlike ``SpeechFinetune`` it needs **no arrow ``train_data``** (the GRPO data
is the per-axis ``SegmentSource``, not a teacher-forcing dataset), so it skips the duration/epoch
sizing; ``train_data`` is optional and only consumed by a real ``SeamlessSegmentSource``.

The saved adapter lands in the fork layout (``checkpoints/checkpoint_<step>/consolidated/
lora.safetensors``), so an RL-tuned model evaluates through the unchanged
``moshi_family_backend_spec(lora_rank=...)`` -- same overlay path as every other LoRA finetune here.
See ``projects/2026-01-speech-llm/rl_alignment.md``.
"""

from __future__ import annotations

from sisyphus import Job, Task

from .finetune import (
    FinetuneAdapter,
    finetune_completed_fraction,
    launch_training,
    prepare_run_dir,
)


def _render_rl_config(job: "RLFinetune") -> str:
    """Render the ``moshi_family.rl.launcher`` YAML (GRPO knobs; paper defaults)."""
    train_data = job.train_data.get() if job.train_data is not None else ""
    return f"""# GRPO interactivity-alignment config (moshi_family.rl.launcher schema)
hf_repo_id: "{job.hf_repo_id}"
out_dir: "{job.out_rundir.get()}"
segment_source: "{job.segment_source}"
train_data: "{train_data}"
vad_backend: "{job.vad_backend}"
quality_asr: "{job.quality_asr}"
quality_asr_model: "{job.quality_asr_model or ""}"
max_steps: {job.max_steps}
grad_accum: {job.grad_accum}
num_samples: {job.num_samples}
capture_s: {job.capture_s}
lora_rank: {job.lora_rank}
lora_scaling: 2.0
lr: {job.lr}
warmup_steps: {job.warmup_steps}
grad_clip: 2.0
context_max_s: {job.context_max_s}
kl_beta: {job.kl_beta}
clip_eps: 0.2
save_every: {job.save_every}
seed: {job.seed}
"""


def _rl_render_shim(job: "RLFinetune", _batch_size: int, _max_steps: int) -> str:
    """Adapter ``render_config`` shim (must be a module-level fn, not a lambda -- the job is pickled by
    Sisyphus and a lambda is unpicklable). RL sizes nothing from the data, so the args are ignored."""
    return _render_rl_config(job)


# The launcher_module + fork_module are all ``launch_training`` reads off the adapter; the config is
# rendered by the job itself (no arrow sizing).
RL_LIB_ADAPTER = FinetuneAdapter(
    name="rl_grpo",
    batch_size=1,
    render_config=_rl_render_shim,
    launcher_module="moshi_family.rl.launcher",
    fork_module="moshi_family",
)


class RLFinetune(Job):
    """GRPO post-training of a moshi_family model (LoRA policy over a frozen base).

    The adapter is fixed (``RL_LIB_ADAPTER``) and set inside ``__init__`` so it never enters the hash;
    the run is identified by its GRPO knobs. ``segment_source="synthetic"`` + ``vad_backend="energy"``
    is the dependency-free smoke configuration (no real training signal, no ``silero-vad``)."""

    __sis_hash_exclude__ = {
        "warmup_steps": 100,
        "grad_accum": 1,
        "capture_s": 12.0,
        "context_max_s": 30.0,
        "kl_beta": 0.01,
        "lora_rank": 128,
        "train_data": None,
        "quality_judge": None,
        "quality_asr": "whisper",
        "quality_asr_model": None,
        "rqmt": None,
    }

    def __init__(
        self,
        *,
        venv_python_path,
        hf_repo_id: str = "kyutai/moshiko-pytorch-bf16",
        segment_source: str = "synthetic",
        train_data=None,
        vad_backend: str = "silero",
        quality_judge: str | None = None,
        quality_asr: str = "whisper",
        quality_asr_model: str | None = None,
        max_steps: int = 3200,
        grad_accum: int = 1,
        num_samples: int = 16,
        capture_s: float = 12.0,
        lora_rank: int = 128,
        lr: float = 2e-7,
        warmup_steps: int = 100,
        context_max_s: float = 30.0,
        kl_beta: float = 0.01,
        save_every: int = 800,
        seed: int = 0,
        rqmt: dict | None = None,
    ):
        self.adapter = RL_LIB_ADAPTER
        self.venv_python_path = venv_python_path
        self.hf_repo_id = hf_repo_id
        self.segment_source = segment_source
        self.train_data = train_data
        self.vad_backend = vad_backend
        # Content-quality reward: when ``quality_judge`` is set, run() co-launches that vLLM judge on a
        # 2nd GPU and the trainer scores turn/interruption replies with ASR->LLM relevance (paper sec. 3).
        self.quality_judge = quality_judge
        self.quality_asr = quality_asr
        self.quality_asr_model = quality_asr_model
        self.max_steps = max_steps
        self.grad_accum = grad_accum
        self.num_samples = num_samples
        self.capture_s = capture_s
        self.lora_rank = lora_rank
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.context_max_s = context_max_s
        self.kl_beta = kl_beta
        self.save_every = save_every
        self.seed = seed
        self.out_config = self.output_path("config.yaml")
        self.out_rundir = self.output_path("run_dir", directory=True)
        # GRPO is heavier per step than SFT (G rollouts + 3 forwards); give it more memory/time. The
        # quality reward adds a co-launched vLLM judge -> a 2nd GPU + more CPU/mem for ASR + the client.
        if rqmt is not None:
            self.rqmt = rqmt
        elif quality_judge is not None:
            self.rqmt = {"gpu": 2, "cpu": 8, "mem": 80, "time": 23}
        else:
            self.rqmt = {"gpu": 1, "cpu": 6, "mem": 48, "time": 23}

    def tasks(self):
        yield Task("write_config", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return finetune_completed_fraction(self, self.adapter)

    def write_config(self):
        prepare_run_dir(self.out_rundir.get())
        with open(self.out_config, "w") as f:
            f.write(_render_rl_config(self))

    def run(self):
        # Reduce CUDA fragmentation OOMs (LoRA forward hit 93/93GB); the torchrun subprocess inherits this.
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if not self.quality_judge:
            launch_training(self, self.adapter)
            return
        # Quality reward ON: reuse the shared retrieval seam to co-launch the vLLM judge on GPU0 and pin
        # the trainer to GPU1; the judge url/model reach the launcher via RL_JUDGE_BASE_URL/RL_JUDGE_MODEL
        # (launch_training inherits os.environ for its torchrun subprocess).
        import os

        from .inference_harness import run_with_optional_retrieval

        self.retrieval_llm = self.quality_judge  # keys the GPU split in run_with_optional_retrieval

        def _drive(extra_args, extra_env):
            if extra_env:
                os.environ.update(extra_env)
            a = list(extra_args)
            if "--llm_base_url" in a:
                os.environ["RL_JUDGE_BASE_URL"] = a[a.index("--llm_base_url") + 1]
                os.environ["RL_JUDGE_MODEL"] = a[a.index("--llm_model_name") + 1]
            launch_training(self, self.adapter)

        run_with_optional_retrieval(self, _drive)
