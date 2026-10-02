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
    render_launcher_fix_keys,
    render_train_perf_keys,
)


def _render_knowledge_keys(job: "RLFinetune") -> str:
    """Knowledge-GRPO keys (``moshi_family.rl.launcher._build_knowledge``), rendered only for
    ``segment_source="knowledge"`` (PersonaPlex, or base Moshi since 2026-10-02) -- every earlier config renders exactly
    as before."""
    if getattr(job, "segment_source", None) != "knowledge":
        return ""
    keys = f"""family: "{job.family}"
reward: "knowledge"
reward_transcript: "{job.reward_transcript}"
rows_per_call: {job.rows_per_call}
context_frames: {job.context_frames}
probe_every: {job.probe_every}
probe_samples: {job.probe_samples}
log_every: {job.log_every}
skip_zero_signal: {str(bool(job.skip_zero_signal)).lower()}
advantage_split: {str(bool(getattr(job, "advantage_split", False))).lower()}
loss_norm: "{getattr(job, "loss_norm", "tokens")}"
brevity_weight: {getattr(job, "brevity_weight", 0.25)}
lr_schedule: "{getattr(job, "lr_schedule", "cosine")}"
garbled_weight: {getattr(job, "garbled_weight", 0.0)}
scale_std: {str(bool(getattr(job, "scale_std", True))).lower()}
questions_per_episode: {getattr(job, "questions_per_episode", 1)}
answer_end_pad_s: {getattr(job, "answer_end_pad_s", 1.0)}
"""
    return keys + (f'init_lora: "{job.init_lora.get()}"\n' if getattr(job, "init_lora", None) is not None else "")


def _render_selfplay_keys(job: "RLFinetune") -> str:
    """PersonaPlex self-play GRPO keys (``moshi_family.rl.launcher._build_personaplex_selfplay``), rendered only for
    ``segment_source="selfplay"``."""
    return f"""family: "{job.family}"
selfplay_policy: "{job.selfplay_policy}"
reward_transcript: "{job.reward_transcript}"
judge_rubric: "{job.out_judge_rubric.get()}"
convs_per_call: {job.convs_per_call}
conversation_s: {job.conversation_s}
conversation_s_start: {job.conversation_s_start if job.conversation_s_start is not None else job.conversation_s}
context_frames: {job.context_frames}
probe_every: {job.probe_every}
probe_samples: {job.probe_samples}
log_every: {job.log_every}
skip_zero_signal: {str(bool(job.skip_zero_signal)).lower()}
loss_norm: "{job.loss_norm}"
lr_schedule: "{job.lr_schedule}"
"""


def _render_rl_config(job: "RLFinetune") -> str:
    """Render the ``moshi_family.rl.launcher`` YAML (GRPO knobs; paper defaults)."""
    if getattr(job, "segment_source", None) == "knowledge":
        train_data = job.out_knowledge_manifest.get()
    elif getattr(job, "segment_source", None) == "selfplay":
        train_data = job.out_selfplay_manifest.get()
    else:
        train_data = job.train_data.get() if job.train_data is not None else ""
    return (
        f"""# GRPO interactivity-alignment config (moshi_family.rl.launcher schema)
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
group_micro_batch: {job.group_micro_batch}
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
        + (
            _render_selfplay_keys(job)
            if getattr(job, "segment_source", None) == "selfplay"
            else _render_knowledge_keys(job)
        )
        + render_launcher_fix_keys(getattr(job, "launcher_fixes", None), ("fp32_master_weights",), who="RL GRPO")
        + render_train_perf_keys(
            # Training-speed keys (finetune.RL_FAST_TRAIN_PERF; runs.fast_train_hparams), only when set.
            # depformer_skip_masked = text-only log-prob forwards; no loader, so no prefetch.
            getattr(job, "train_perf", None) or {},
            ("contiguous_step_inputs", "fused_causal_attention", "fused_lora_add", "depformer_skip_masked"),
            who="RL GRPO",
        )
    )


def _rl_render_shim(job: "RLFinetune", _batch_size: int, _max_steps: int) -> str:
    """Adapter ``render_config`` shim (must be a module-level fn, not a lambda -- the job is pickled by
    Sisyphus and a lambda is unpicklable). RL sizes nothing from the data, so the args are ignored."""
    return _render_rl_config(job)


# The launcher_module + pythonpath_package are all ``launch_training`` reads off the adapter; the config is
# rendered by the job itself (no arrow sizing).
RL_LIB_ADAPTER = FinetuneAdapter(
    name="rl_grpo",
    batch_size=1,
    render_config=_rl_render_shim,
    launcher_module="moshi_family.rl.launcher",
    pythonpath_package="moshi_family",
)


class RLFinetune(Job):
    """GRPO post-training of a moshi_family model (LoRA policy over a frozen base).

    The adapter is fixed (``RL_LIB_ADAPTER``) and set inside ``__init__`` so it never enters the hash;
    the run is identified by its GRPO knobs. ``segment_source="synthetic"`` + ``vad_backend="energy"``
    is the dependency-free smoke configuration (no real training signal, no ``silero-vad``)."""

    __sis_hash_exclude__ = {
        "warmup_steps": 100,
        "grad_accum": 2,
        "group_micro_batch": 4,
        "capture_s": 12.0,
        "context_max_s": 30.0,
        "kl_beta": 0.01,
        "lora_rank": 128,
        "train_data": None,
        "quality_judge": None,
        "quality_asr": "whisper",
        "quality_asr_model": None,
        "rqmt": None,
        # Launcher setup fixes (finetune.LAUNCHER_FIX_PRESETS["rl_grpo"]; runs.launcher_fix_hparams):
        # excluded at None so every existing job keeps its hash; new runs pass the policy's dict.
        "launcher_fixes": None,
        # Training-speed switches (finetune.RL_FAST_TRAIN_PERF; runs.fast_train_hparams): excluded at
        # None so every existing job keeps its hash; new runs pass the policy's dict.
        "train_perf": None,
        # PersonaPlex knowledge GRPO (2026-09-29, rl_alignment.md "Knowledge GRPO"): excluded at their defaults, so
        # every existing (base-Moshi interactivity) job keeps its hash.
        "family": "moshi",
        "knowledge_questions": None,
        "knowledge_audio": None,
        "knowledge_exclude": None,
        "n_probe": 64,
        "reward_transcript": "monologue",
        "rows_per_call": 32,
        "context_frames": 750,
        "probe_every": 0,
        "probe_samples": 4,
        "log_every": 10,
        "skip_zero_signal": False,
        "judge_max_model_len": None,
        "advantage_split": False,
        "loss_norm": "tokens",
        "brevity_weight": 0.25,
        "lr_schedule": "cosine",
        "garbled_weight": 0.0,
        "scale_std": True,
        "init_lora": None,
        "questions_per_episode": 1,
        "answer_end_pad_s": 1.0,
        # PersonaPlex self-play GRPO (2026-10-01, rl_alignment.md "Self-play persona GRPO"): excluded at their defaults
        "selfplay_scenarios": None,
        "selfplay_policy": "shared",
        "conversation_s": 30.0,
        "conversation_s_start": None,
        "convs_per_call": 16,
        "judge_rubric": "spj2_judge_v1",
        "n_probe_scenarios": 16,
    }

    @classmethod
    def hash(cls, parsed_args):
        """A self-play run's judge rubric TEXT is part of its hash (as ``selfplay_bench.SelfPlayJudgeV2``); every other
        job hashes exactly as before."""
        if parsed_args.get("segment_source") != "selfplay":
            return super(RLFinetune, cls).hash(parsed_args)
        from .selfplay_bench import judge_rubric_digest

        return super(RLFinetune, cls).hash(
            {**parsed_args, "judge_rubric_digest": judge_rubric_digest(parsed_args["judge_rubric"])}
        )

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
        grad_accum: int = 2,
        num_samples: int = 16,
        group_micro_batch: int = 4,
        capture_s: float = 12.0,
        lora_rank: int = 128,
        lr: float = 2e-7,
        warmup_steps: int = 100,
        context_max_s: float = 30.0,
        kl_beta: float = 0.01,
        save_every: int = 800,
        seed: int = 0,
        rqmt: dict | None = None,
        launcher_fixes: dict | None = None,
        train_perf: dict | None = None,
        family: str = "moshi",
        knowledge_questions=None,
        knowledge_audio=None,
        knowledge_exclude=None,
        n_probe: int = 64,
        reward_transcript: str = "monologue",
        rows_per_call: int = 32,
        context_frames: int = 750,
        probe_every: int = 0,
        probe_samples: int = 4,
        log_every: int = 10,
        skip_zero_signal: bool = False,
        judge_max_model_len: int | None = None,
        advantage_split: bool = False,
        loss_norm: str = "tokens",
        brevity_weight: float = 0.25,
        lr_schedule: str = "cosine",
        garbled_weight: float = 0.0,
        scale_std: bool = True,
        init_lora=None,
        questions_per_episode: int = 1,
        answer_end_pad_s: float = 1.0,
        selfplay_scenarios=None,
        selfplay_policy: str = "shared",
        conversation_s: float = 30.0,
        conversation_s_start: float | None = None,
        convs_per_call: int = 16,
        judge_rubric: str = "spj2_judge_v1",
        n_probe_scenarios: int = 16,
    ):
        """Knowledge GRPO (``segment_source="knowledge"``, PersonaPlex or base Moshi): ``knowledge_questions`` (an
        ``LLMPreprocess`` out_hf) + ``knowledge_audio`` (their TTS, storage hf) are the prompts, minus any question in
        ``knowledge_exclude`` (e.g. the benchmark's validation split); ``n_probe`` of them are the fixed held-out probe
        (``probe_samples`` rollouts each every ``probe_every`` steps). ``reward_transcript``: what the judge reads
        (``monologue`` / ``asr``). ``rows_per_call``: rollouts per engine call (several questions' groups);
        ``context_frames``: the PersonaPlex KV context (any value >= the frames a rollout steps gives identical
        outputs). ``judge_max_model_len``: the co-launched vLLM judge's context (short prompts).

        Self-play GRPO (``family="personaplex"``, ``segment_source="selfplay"``): ``selfplay_scenarios`` (a
        ``selfplay_bench.SelfPlayTrainScenarioSet`` dataset) are the prompts, ``n_probe_scenarios`` of them the fixed
        probe; conversations of ``conversation_s_start`` (default ``conversation_s``) growing linearly to
        ``conversation_s`` over the run (the probe always at ``conversation_s``), ``convs_per_call`` per engine call, G = ``num_samples`` per
        scenario; the reward is the ``judge_rubric`` (spj2) judge, ``quality_judge`` serving it. ``selfplay_policy``:
        ``"shared"`` (one LoRA plays both sides) or ``"two_lora"`` (one per side)."""
        assert family in ("moshi", "personaplex"), family
        # PersonaPlex runs knowledge or self-play GRPO; base Moshi the interactivity GRPO or (2026-10-02) knowledge GRPO
        assert family == "moshi" or segment_source in ("knowledge", "selfplay"), (family, segment_source)
        assert segment_source != "selfplay" or family == "personaplex", (family, segment_source)
        self.judge_rubric_payload = None
        if segment_source == "selfplay":
            # imported here, in the manager: the worker running write_config cannot import selfplay_bench (its import
            # chain needs i6_core); the payload is pickled with the job and is not hashed (the hash has the digest)
            from .selfplay_bench import (
                SPJ2_JSON_SCHEMA,
                SPJ2_METRICS,
                SPJ2_REASON_CHARS,
                SPJ2_RUBRICS,
                judge_rubric_digest,
            )

            assert judge_rubric in SPJ2_RUBRICS, judge_rubric
            # the benchmark's rubric, schema and metric names, verbatim: the reward cannot drift from spj2
            self.judge_rubric_payload = {
                "name": judge_rubric,
                "digest": judge_rubric_digest(judge_rubric),
                "text": SPJ2_RUBRICS[judge_rubric],
                "schema": SPJ2_JSON_SCHEMA,
                "metrics": list(SPJ2_METRICS),
                "reason_chars": SPJ2_REASON_CHARS,
            }
            assert selfplay_scenarios is not None and quality_judge, "self-play GRPO needs scenarios and the judge"
            assert judge_rubric in SPJ2_RUBRICS, judge_rubric
            assert selfplay_policy in ("shared", "two_lora"), selfplay_policy
            assert grad_accum % 2 == 0, "self-play: grad_accum counts (scenario, side) groups"
            assert convs_per_call % num_samples == 0, (convs_per_call, num_samples)
            assert init_lora is None or selfplay_policy == "shared", "init_lora with two_lora"
        self.selfplay_scenarios = selfplay_scenarios
        self.selfplay_policy = selfplay_policy
        self.conversation_s = conversation_s
        self.conversation_s_start = conversation_s_start
        self.convs_per_call = convs_per_call
        self.judge_rubric = judge_rubric
        self.n_probe_scenarios = n_probe_scenarios
        if segment_source == "knowledge":
            assert knowledge_questions is not None and knowledge_audio is not None and quality_judge, (
                "knowledge GRPO needs questions, their audio and the judge"
            )
        self.family = family
        self.knowledge_questions = knowledge_questions
        self.knowledge_audio = knowledge_audio
        self.knowledge_exclude = knowledge_exclude
        self.n_probe = n_probe
        self.reward_transcript = reward_transcript
        self.rows_per_call = rows_per_call
        self.context_frames = context_frames
        self.probe_every = probe_every
        self.probe_samples = probe_samples
        self.log_every = log_every
        self.skip_zero_signal = skip_zero_signal
        self.judge_max_model_len = judge_max_model_len
        # knowledge GRPO: split advantage (correct z-scored + brevity mean-centred x 0.25) and Dr. GRPO's constant
        # per-sequence loss normaliser ("constant"; "tokens" = the paper's 1/N)
        self.advantage_split = advantage_split
        assert loss_norm in ("tokens", "constant"), loss_norm
        self.loss_norm = loss_norm
        # knowledge reward: weight of the brevity component (-min(words, 40) / 40 on correct rollouts); 0 = correctness only
        self.brevity_weight = brevity_weight
        # after the warmup: "cosine" (to 0, as SFT) or "constant"
        assert lr_schedule in ("cosine", "constant"), lr_schedule
        self.lr_schedule = lr_schedule
        # knowledge reward: weight of the garbled-reply penalty component (-1 per token-soup rollout); 0 = off
        self.garbled_weight = garbled_weight
        # knowledge GRPO: False = Dr. GRPO's unscaled advantage (correct only mean-centred, not divided by the group std)
        assert scale_std or advantage_split, "scale_std=False needs advantage_split"
        self.scale_std = scale_std
        # knowledge GRPO from an SFT adapter: a consolidated lora.safetensors (same rank/scaling), merged into the
        # frozen base (= pi_ref); checkpoints then hold the combined adapter at rank 2 x lora_rank
        assert init_lora is None or family == "personaplex", "init_lora: PersonaPlex only"
        self.init_lora = init_lora
        self.overlay_lora_rank = lora_rank * (2 if init_lora is not None else 1)
        # knowledge GRPO: questions per rollout (multi-question episodes; > 1 needs advantage_split, monologue,
        # brevity 0) and the text silence that ends a reply
        assert questions_per_episode == 1 or (advantage_split and reward_transcript == "monologue"), "episodes"
        self.questions_per_episode = questions_per_episode
        self.answer_end_pad_s = answer_end_pad_s
        self.adapter = RL_LIB_ADAPTER
        self.launcher_fixes = launcher_fixes
        self.train_perf = train_perf
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
        self.group_micro_batch = group_micro_batch
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
        if segment_source == "knowledge":
            self.out_knowledge_manifest = self.output_path("knowledge_manifest.json")
        if segment_source == "selfplay":
            self.out_selfplay_manifest = self.output_path("selfplay_manifest.json")
            self.out_judge_rubric = self.output_path("judge_rubric.json")
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
        if self.segment_source == "knowledge":
            import json

            manifest = {
                "questions": self.knowledge_questions.get(),
                "audio": self.knowledge_audio.get(),
                "exclude_questions": self.knowledge_exclude.get() if self.knowledge_exclude is not None else None,
                "n_probe": self.n_probe,
                "seed": self.seed,
            }
            with open(self.out_knowledge_manifest, "w") as f:
                json.dump(manifest, f, indent=1)
        if self.segment_source == "selfplay":
            import json

            manifest = {
                "scenarios": self.selfplay_scenarios.get(),
                "n_probe": self.n_probe_scenarios,
                "seed": self.seed,
            }
            with open(self.out_selfplay_manifest, "w") as f:
                json.dump(manifest, f, indent=1)
            with open(self.out_judge_rubric, "w") as f:
                json.dump(self.judge_rubric_payload, f, indent=1)
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
        self.retrieval_llm_max_model_len = getattr(self, "judge_max_model_len", None)

        def _drive(extra_args, extra_env):
            if extra_env:
                os.environ.update(extra_env)
            a = list(extra_args)
            if "--llm_base_url" in a:
                os.environ["RL_JUDGE_BASE_URL"] = a[a.index("--llm_base_url") + 1]
                os.environ["RL_JUDGE_MODEL"] = a[a.index("--llm_model_name") + 1]
            launch_training(self, self.adapter)

        run_with_optional_retrieval(self, _drive)
