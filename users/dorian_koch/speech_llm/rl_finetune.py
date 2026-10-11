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
    if getattr(job, "segment_source", None) not in ("knowledge", "multitask"):
        return ""
    keys = f"""family: "{job.family}"
reward: "{getattr(job, "knowledge_reward", "knowledge")}"
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
    if getattr(job, "ramble_weight", 0.0) or getattr(job, "long_capture_s", 0.0):  # absent at the defaults
        keys += (
            f"ramble_weight: {job.ramble_weight}\nlong_capture_s: {job.long_capture_s}\n"
            f"long_capture_every: {job.long_capture_every}\n"
        )
    if getattr(job, "knowledge_judge_version", "v1") != "v1":  # absent at the default (2026-10-11)
        keys += f'knowledge_judge_version: "{job.knowledge_judge_version}"\n'
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
""" + _render_partner_key(job)


def _render_multitask_key(job: "RLFinetune") -> str:
    """Multi-task GRPO (``segment_source="multitask"``): the knowledge keys above plus ONE JSON line (valid YAML) with
    the trainer's axis order and the self-play builder's own keys (``moshi_family.rl.launcher`` multi-task branch)."""
    import json

    if getattr(job, "segment_source", None) != "multitask":
        return ""
    sp = job.multitask["selfplay"]
    mt = {
        "axis_cycle": list(job.multitask["axis_cycle"]),
        "selfplay": {
            "train_data": job.out_selfplay_manifest.get(),
            "judge_rubric": job.out_judge_rubric.get(),
            "selfplay_policy": "shared",
            "convs_per_call": job.convs_per_call,
            "conversation_s": job.conversation_s,
            "conversation_s_start": job.conversation_s_start
            if job.conversation_s_start is not None
            else job.conversation_s,
            "num_samples": sp["num_samples"],
            "probe_samples": sp["probe_samples"],
            "reward_transcript": sp["reward_transcript"],
        },
    }
    return f"multitask: {json.dumps(mt, sort_keys=True)}\n"


def _render_partner_key(job: "RLFinetune") -> str:
    """``init_lora`` and ``selfplay_partner`` (``moshi_family.rl.partner_channel.DEFAULT_SPEC``, one JSON line = valid
    YAML), each rendered only when set -- every earlier self-play config renders exactly as before."""
    import json

    out = f'init_lora: "{job.init_lora.get()}"\n' if getattr(job, "init_lora", None) is not None else ""
    spec = getattr(job, "selfplay_partner", None)
    if not spec:
        return out
    spec = {k: (v.get() if hasattr(v, "get_path") else v) for k, v in spec.items()}
    return out + f"selfplay_partner: {json.dumps(spec, sort_keys=True)}\n"


def _render_rl_config(job: "RLFinetune") -> str:
    """Render the ``moshi_family.rl.launcher`` YAML (GRPO knobs; paper defaults)."""
    if getattr(job, "segment_source", None) in ("knowledge", "multitask"):
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
quality_asr_num_beams: {getattr(job, "quality_asr_num_beams", 5)}
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
            else _render_knowledge_keys(job) + _render_multitask_key(job)
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
        # reward-side Whisper beam width (2026-10-11; moshi_family.rl.quality.ASR_NUM_BEAMS): excluded at 5 = every
        # earlier ASR arm; 1 = greedy (asr_bench 2026-10-05)
        "quality_asr_num_beams": 5,
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
        # ramble penalty + long reply windows (2026-10-02): excluded at their defaults (off)
        "ramble_weight": 0.0,
        "long_capture_s": 0.0,
        "long_capture_every": 0,
        "scale_std": True,
        "init_lora": None,
        "questions_per_episode": 1,
        # instruction-following GRPO (2026-10-03, moshi_family.rl.instruct_reward): excluded at the default
        "knowledge_reward": "knowledge",
        "answer_end_pad_s": 1.0,
        # broad-spectrum knowledge pools (2026-10-11, knowledge_broad_rl.md): excluded at their defaults (TriviaQA only)
        "knowledge_sources": None,
        "knowledge_base_category": None,
        "knowledge_balance": False,
        "n_probe_per_category": 0,
        "knowledge_judge_version": "v1",
        # PersonaPlex self-play GRPO (2026-10-01, rl_alignment.md "Self-play persona GRPO"): excluded at their defaults
        "selfplay_scenarios": None,
        "selfplay_policy": "shared",
        "conversation_s": 30.0,
        "conversation_s_start": None,
        "convs_per_call": 16,
        "judge_rubric": "spj2_judge_v1",
        "n_probe_scenarios": 16,
        # recipe-controlled self-play reward (2026-10-02): excluded at None = the spj2_judge_v1 reward of ppx_rl_sp_mono
        "selfplay_reward": None,
        # what each self-play side hears of its partner (2026-10-02): excluded at None = the sampled codes directly
        "selfplay_partner": None,
        # multi-task GRPO (2026-10-03): excluded at None
        "multitask": None,
    }

    @classmethod
    def hash(cls, parsed_args):
        """A self-play run's judge rubric TEXT is part of its hash (as ``selfplay_bench.SelfPlayJudgeV2``); every other
        job hashes exactly as before. The experiment ``card`` is never hashed."""
        parsed_args = {k: v for k, v in parsed_args.items() if k != "card"}
        if parsed_args.get("segment_source") not in ("selfplay", "multitask"):
            return super(RLFinetune, cls).hash(parsed_args)
        from .selfplay_bench import judge_rubric_digest

        rubric = (parsed_args.get("selfplay_reward") or {}).get("rubric", parsed_args["judge_rubric"])
        return super(RLFinetune, cls).hash({**parsed_args, "judge_rubric_digest": judge_rubric_digest(rubric)})

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
        quality_asr_num_beams: int = 5,
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
        ramble_weight: float = 0.0,
        long_capture_s: float = 0.0,
        long_capture_every: int = 0,
        scale_std: bool = True,
        init_lora=None,
        questions_per_episode: int = 1,
        knowledge_reward: str = "knowledge",
        answer_end_pad_s: float = 1.0,
        knowledge_sources: tuple | None = None,
        knowledge_base_category: str | None = None,
        knowledge_balance: bool = False,
        n_probe_per_category: int = 0,
        knowledge_judge_version: str = "v1",
        selfplay_scenarios=None,
        selfplay_policy: str = "shared",
        conversation_s: float = 30.0,
        conversation_s_start: float | None = None,
        convs_per_call: int = 16,
        judge_rubric: str = "spj2_judge_v1",
        n_probe_scenarios: int = 16,
        selfplay_reward: dict | None = None,
        selfplay_partner: dict | None = None,
        multitask: dict | None = None,
        card: dict | None = None,
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
        ``"shared"`` (one LoRA plays both sides) or ``"two_lora"`` (one per side).

        Multi-task GRPO (``segment_source="multitask"``, 2026-10-03; ``moshi_family.rl.multitask``): every step rolls
        out knowledge AND self-play groups on one shared LoRA, in the order ``multitask["axis_cycle"]`` (len =
        ``grad_accum``). The knowledge part takes the knowledge args (``num_samples``, ``probe_samples``,
        ``reward_transcript``, ``quality_judge``); the self-play part the self-play args plus ``multitask["selfplay"]``
        = ``{"num_samples", "probe_samples", "reward_transcript"}`` and its own judge ``multitask["selfplay_judge"]``
        (``selfplay_judge_max_model_len``). run() serves the two judges on the first two GPUs, the trainer on the
        last: ``rqmt`` gpu >= 3."""
        from .experiment_card import check_card

        # experiment card (speech_llm.experiment_card): not hashed, written by write_config
        self.card = check_card(card)
        if self.card is not None:
            self.out_card = self.output_path("experiment_card.json")
            self.out_card_md = self.output_path("experiment_card.md")
        assert family in ("moshi", "personaplex"), family
        # PersonaPlex runs knowledge or self-play GRPO; base Moshi the interactivity GRPO or (2026-10-02) knowledge GRPO
        assert family == "moshi" or segment_source in ("knowledge", "selfplay", "multitask"), (family, segment_source)
        assert segment_source not in ("selfplay", "multitask") or family == "personaplex", (family, segment_source)
        self.judge_rubric_payload = None
        if segment_source in ("selfplay", "multitask"):
            # imported here, in the manager: the worker running write_config cannot import selfplay_bench (its import
            # chain needs i6_core); the payload is pickled with the job and is not hashed (the hash has the digest)
            from .selfplay_bench import (
                SPJ2_REASON_CHARS,
                SPJ2_RUBRICS,
                judge_rubric_digest,
                rubric_json_schema,
                rubric_text_metrics,
            )

            assert judge_rubric in SPJ2_RUBRICS, judge_rubric
            # selfplay_reward (the recipe's reward spec): "rubric" (spj2 or selfplay_bench.SP_REWARD_RUBRICS) + the
            # keys of moshi_family.rl.selfplay_reward.DEFAULT_REWARD_SPEC (components, weights, transcript format /
            # gap / source text, judge sampling), shipped to the launcher in the rubric JSON's "reward"
            spec = dict(selfplay_reward or {})
            rubric_name = spec.pop("rubric", judge_rubric)
            assert selfplay_reward is None or judge_rubric == "spj2_judge_v1", "name the rubric in selfplay_reward"
            unknown = set(spec) - {"components", "weights", "transcript_format", "transcript_gap_s", "transcript_overlap_words",
                                   "transcript_source", "judge_temperature", "judge_max_tokens", "judge_attempts",
                                   "rate_penalty"}  # fmt: skip
            assert not unknown, f"selfplay_reward: unknown keys {sorted(unknown)}"
            text, metrics = rubric_text_metrics(rubric_name)
            for c, m in (spec.get("components") or {}).items():
                assert all(x in metrics for x in ([m["shared"]] if "shared" in m else [m["a"], m["b"]])), (c, m)
            # default: the benchmark's rubric, schema and metric names, verbatim (the reward cannot drift from spj2)
            self.judge_rubric_payload = {
                "name": rubric_name,
                "digest": judge_rubric_digest(rubric_name),
                "text": text,
                "schema": rubric_json_schema(metrics),
                "metrics": list(metrics),
                "reason_chars": SPJ2_REASON_CHARS,
                **({"reward": spec} if spec else {}),
            }
            assert selfplay_scenarios is not None and quality_judge, "self-play GRPO needs scenarios and the judge"
            assert judge_rubric in SPJ2_RUBRICS, judge_rubric
            assert selfplay_policy in ("shared", "two_lora"), selfplay_policy
            if segment_source == "selfplay":
                assert grad_accum % 2 == 0, "self-play: grad_accum counts (scenario, side) groups"
                assert convs_per_call % num_samples == 0, (convs_per_call, num_samples)
            assert init_lora is None or selfplay_policy == "shared", "init_lora with two_lora"
        self.selfplay_scenarios = selfplay_scenarios
        self.selfplay_policy = selfplay_policy
        self.conversation_s = conversation_s
        self.conversation_s_start = conversation_s_start
        self.convs_per_call = convs_per_call
        self.judge_rubric = judge_rubric
        self.selfplay_reward = selfplay_reward
        if selfplay_partner is not None:
            # moshi_family.rl.partner_channel.DEFAULT_SPEC (validated in full by the launcher); noise_pool = a
            # DnsNoisePool output (tk.Path: a dependency of this job)
            assert segment_source == "selfplay", "selfplay_partner: self-play only"
            unknown = set(selfplay_partner) - {
                "mimi_roundtrip",
                "augment",
                "noise_pool",
                "noise_ref_dbfs",
                "seed",
                "span_mask",
            }
            assert not unknown, f"selfplay_partner: unknown keys {sorted(unknown)}"
            assert selfplay_partner.get("mimi_roundtrip") or not selfplay_partner.get("augment"), (
                "augment needs roundtrip"
            )
        self.selfplay_partner = selfplay_partner
        self.n_probe_scenarios = n_probe_scenarios
        if segment_source == "multitask":
            assert set(multitask or {}) <= {"axis_cycle", "selfplay", "selfplay_judge", "selfplay_judge_max_model_len"}
            sp = multitask["selfplay"]
            assert set(sp) == {"num_samples", "probe_samples", "reward_transcript"}, sorted(sp)
            cyc = list(multitask["axis_cycle"])
            n_sp = cyc.count("selfplay")
            assert len(cyc) == grad_accum and n_sp % 2 == 0 and cyc.count("knowledge") + n_sp == len(cyc), cyc
            assert convs_per_call % sp["num_samples"] == 0, (convs_per_call, sp)
            # one model, one batch: the two generators must step the same number of rows (moshi_family.rl.multitask)
            assert 2 * convs_per_call == rows_per_call, (convs_per_call, rows_per_call)
            assert multitask["selfplay_judge"] and selfplay_policy == "shared" and init_lora is None, multitask
            assert selfplay_partner is None, "multi-task: no partner channel"
        else:
            assert multitask is None, "multitask: segment_source='multitask' only"
        self.multitask = multitask
        if segment_source in ("knowledge", "multitask"):
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
        # knowledge reward: weight of the ramble penalty (-1 per rollout the judge flags as off-topic or looping); 0 = off
        self.ramble_weight = ramble_weight
        # knowledge GRPO: every long_capture_every-th step samples a long_capture_s reply window (0 = off). The KV
        # context must hold lead-in + question (<= 20 s) + that window, or the long rollouts run on a sliding window.
        assert (long_capture_s > 0) == (long_capture_every > 0), (long_capture_s, long_capture_every)
        assert long_capture_s == 0 or context_frames >= (long_capture_s + 20.0) * 12.5, (long_capture_s, context_frames)
        self.long_capture_s = long_capture_s
        self.long_capture_every = long_capture_every
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
        # the judge of the question->reply rollouts: "knowledge" (exact answer, TriviaQA) or "instruct" (1-5 rubric
        # against a reference reply; the prompts are spoken requests, speech_llm.instruct_prompts)
        assert knowledge_reward in ("knowledge", "instruct"), knowledge_reward
        self.knowledge_reward = knowledge_reward
        self.answer_end_pad_s = answer_end_pad_s
        # broad-spectrum pools (moshi_family.rl.knowledge_segments): further (category, questions, audio) sets next to
        # knowledge_questions/_audio (labelled knowledge_base_category), category-balanced draws, a fixed probe shard of
        # n_probe_per_category per category, and the correctness judge's prompt version (knowledge_reward.JUDGE_SYSTEMS)
        if knowledge_sources is not None:
            knowledge_sources = tuple(dict(x) for x in knowledge_sources)
            for x in knowledge_sources:
                assert set(x) == {"category", "questions", "audio"}, sorted(x)
            assert knowledge_base_category, "knowledge_sources: name the main pool's category (knowledge_base_category)"
            assert (
                len({x["category"] for x in knowledge_sources} | {knowledge_base_category})
                == len(knowledge_sources) + 1
            )
        assert not (knowledge_balance or n_probe_per_category) or knowledge_sources, (
            "balance/probe shard: broad pools only"
        )
        assert knowledge_judge_version in ("v1", "v2_broad"), knowledge_judge_version
        assert knowledge_judge_version == "v1" or knowledge_reward == "knowledge", knowledge_judge_version
        self.knowledge_sources = knowledge_sources
        self.knowledge_base_category = knowledge_base_category
        self.knowledge_balance = bool(knowledge_balance)
        self.n_probe_per_category = int(n_probe_per_category)
        self.knowledge_judge_version = knowledge_judge_version
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
        assert int(quality_asr_num_beams) >= 1, quality_asr_num_beams
        self.quality_asr_num_beams = int(quality_asr_num_beams)
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
        if segment_source in ("knowledge", "multitask"):
            self.out_knowledge_manifest = self.output_path("knowledge_manifest.json")
        if segment_source in ("selfplay", "multitask"):
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
        # resume="run": a SLURM time-limit kill is rescheduled by the manager; the launcher resumes from the latest
        # complete checkpoint (rl/launcher.py resume_from_checkpoint). Tasks are not hashed.
        yield Task("run", resume="run", rqmt=self.rqmt)

    def completed_fraction(self):
        return finetune_completed_fraction(self, self.adapter)

    def write_config(self):
        prepare_run_dir(self.out_rundir.get())
        if getattr(self, "card", None) is not None:
            from .experiment_card import write_card

            write_card(self, self.out_card.get_path(), self.out_card_md.get_path())
        if self.segment_source in ("knowledge", "multitask"):
            import json

            manifest = {
                "questions": self.knowledge_questions.get(),
                "audio": self.knowledge_audio.get(),
                "exclude_questions": self.knowledge_exclude.get() if self.knowledge_exclude is not None else None,
                "n_probe": self.n_probe,
                "seed": self.seed,
            }
            if getattr(self, "knowledge_sources", None):  # broad pools only: older manifests stay byte-identical
                manifest.update(
                    sources=[
                        {"category": x["category"], "questions": x["questions"].get(), "audio": x["audio"].get()}
                        for x in self.knowledge_sources
                    ],
                    base_category=self.knowledge_base_category,
                    balance=self.knowledge_balance,
                    probe_per_category=self.n_probe_per_category,
                )
            with open(self.out_knowledge_manifest, "w") as f:
                json.dump(manifest, f, indent=1)
        if self.segment_source in ("selfplay", "multitask"):
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
        text = _render_rl_config(self)
        # a hashed knob the launcher never sees trains silently without it (2026-10-02: init_lora was not rendered for
        # self-play, so a run "from p6" would have started from released PersonaPlex)
        for key in ("init_lora", "selfplay_partner"):
            if getattr(self, key, None) is not None:
                assert f"\n{key}: " in text, f"{key} is set but not rendered into the launcher config"
        with open(self.out_config, "w") as f:
            f.write(text)

    def _run_two_judges(self):
        """Multi-task GRPO: the knowledge judge (``quality_judge``) on the first visible GPU, the self-play judge on the
        second, the trainer on the last; the launcher reads RL_JUDGE_* and RL_SP_JUDGE_*."""
        import os

        from .common import vllm_server

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(",")
        assert len(visible) >= 3, f"multi-task GRPO needs 3 GPUs (two judges + the trainer), got {visible}"
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        mt = self.multitask
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible[0]
            with vllm_server(self.quality_judge, max_model_len=self.judge_max_model_len) as k_url:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible[1]
                with vllm_server(mt["selfplay_judge"], max_model_len=mt.get("selfplay_judge_max_model_len")) as s_url:
                    os.environ.update(
                        CUDA_VISIBLE_DEVICES=visible[-1],
                        RL_JUDGE_BASE_URL=k_url,
                        RL_JUDGE_MODEL=self.quality_judge,
                        RL_SP_JUDGE_BASE_URL=s_url,
                        RL_SP_JUDGE_MODEL=mt["selfplay_judge"],
                    )
                    launch_training(self, self.adapter)
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def run(self):
        # Reduce CUDA fragmentation OOMs (LoRA forward hit 93/93GB); the torchrun subprocess inherits this.
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if not self.quality_judge:
            launch_training(self, self.adapter)
            return
        if getattr(self, "multitask", None):
            self._run_two_judges()
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
