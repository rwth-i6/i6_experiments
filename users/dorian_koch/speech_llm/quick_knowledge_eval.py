"""Fast, small "knowledge" eval you can run *during* a training run to catch factual-recall
collapse early -- instead of discovering it 17h + a full LLM-judged benchmark later.

Motivation (2026-07-27): the A7-mix arm trained to a healthy loss (0.47) yet answered trivia by
confidently confabulating ("War Requiem -> William Henry Harrison"), scoring ~9% vs base 17.7%.
The train loss showed nothing; only an *accuracy* probe reveals it. So every run should carry a
cheap accuracy eval, not just a val-loss.

Design -- imitate the real knowledge benchmark at a fraction of the cost:

    LoadRawDataset -> LLMPreprocess -> SubsampleDataset(n=QUICK_N)  [SHARED cache w/ full bench]
      -> ChatterboxSingleSpeakerInference (TTS the questions, cached)
      -> SpeechInference(mode="knowledge", single shard)            [the model answers, in speech]
      -> MonologueTranscription (read the inner-monologue <i>.txt, NO Whisper)
      -> AliasMatchGrading (normalized alias containment, NO vLLM judge -> login-node, instant)

The two expensive *tail* stages of the full pipeline (Whisper ASR + a gemma-4-31B LLM judge) are
replaced by free ones. The first two data stages are constructed identically to
``knowledge_benchmark_py`` so they hit the SAME Sisyphus hash and reuse its cache (no recompute).

Modularity:
  * ``normalize_answer`` / ``alias_match``  -- pure functions, importable & unit-tested; could later
    drive an in-training-loop probe without any Sisyphus.
  * ``AliasMatchGrading``                   -- a drop-in, GPU-free replacement for ``LLMGrading``
    (identical eval_results.jsonl / summary.json schema, so every downstream printer/ResultNotify
    works unchanged).
  * ``quick_knowledge_eval_py``             -- assembles one small eval for a checkpoint.
  * ``attach_knowledge_evals``        -- ONE call attaches evals at several checkpoint steps,
    giving a accuracy-vs-step curve *during* the run.

The alias-match accuracy is an APPROXIMATE proxy for monitoring a trend, not a reportable number --
the LLM-judged full benchmark stays the metric of record. See ``check_quick_knowledge_eval.py``.
"""

from __future__ import annotations

import json
import os
import re
import string

from sisyphus import Job, Task, tk

from .knowledge_benchmark import (
    LoadRawDataset,
    LLMPreprocess,
    SubsampleDataset,
    ChatterboxSingleSpeakerInference,
    MonologueTranscription,
    WhisperTranscription,
    sharded_knowledge_inference,
    resolve_lora,
    resolve_personaplex_weights,
    resolve_audex_weights,
    MOSHI_BACKEND,
)
from .result_notify import notify_result
from .speech_inference import SAVE_EVERY as _SAVE_EVERY
from .speech_inference import impossible_checkpoint_reason

# Defaults chosen so a single eval is small enough to finish in a few minutes on one GPU but large
# enough that a real collapse (base ~18% -> ~9%) clears the sampling noise of ~sqrt(p(1-p)/n).
QUICK_N = 64
# A seed distinct from the full benchmark's (42) so this is its own cached subset of the SAME
# held-out validation split (disjoint from the train split the finetunes use).
QUICK_SEED = 1234


# ---------------------------------------------------------------------------
# Scoring (pure, importable, unit-tested)
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation + articles, collapse whitespace.
    A space is kept at both ends so substring tests respect word boundaries."""
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f" {s} "


def alias_match(prediction: str, answer: str, aliases) -> tuple[int, int]:
    """Return ``(binary_correct, quality_score)`` by normalized alias *containment*.

    A prediction is correct if any of {answer} + aliases appears as a whole-word substring of the
    (verbose) prediction -- the model typically says "It's New York City. In 2020..." so exact
    match is too strict; substring on the article/punctuation-normalized strings is the right test.

    quality: 5 if the prediction is essentially just the answer, 3 if the answer is embedded in a
    longer reply, 1 if absent. This mirrors ``LLMGrading``'s 1-5 axis coarsely so summaries stay
    comparable in shape (not in exact value).
    """
    if isinstance(aliases, str):
        try:
            aliases = json.loads(aliases)
        except Exception:
            aliases = [aliases]
    aliases = aliases or []
    pred_n = normalize_answer(prediction)
    if pred_n.strip() == "":
        return 0, 1
    candidates = [answer, *aliases]
    for cand in candidates:
        if not cand:
            continue
        cand_n = normalize_answer(cand)
        core = cand_n.strip()
        if core and cand_n in pred_n:
            # essentially-just-the-answer if the reply is short and dominated by the match
            quality = 5 if len(pred_n.strip()) <= len(core) + 6 else 3
            return 1, quality
    return 0, 1


def summarize(eval_results: list[dict]) -> dict:
    """Aggregate eval rows into the same {category: {n, accuracy, avg_quality}, overall: {...}}
    schema ``LLMGrading`` emits, so downstream consumers are agnostic to which grader ran."""
    cats: dict[str, list] = {}
    for er in eval_results:
        cats.setdefault(er.get("category", "unknown"), []).append(er)
    summary: dict = {}
    for cat, items in cats.items():
        summary[cat] = {
            "n": len(items),
            "accuracy": sum(i["binary_correct"] for i in items) / len(items),
            "avg_quality": sum(i["quality_score"] for i in items) / len(items),
        }
    n = len(eval_results)
    summary["overall"] = {
        "n": n,
        "accuracy": sum(i["binary_correct"] for i in eval_results) / n if n else 0.0,
        "avg_quality": sum(i["quality_score"] for i in eval_results) / n if n else 0.0,
    }
    return summary


# ---------------------------------------------------------------------------
# Grading job (drop-in, GPU-free replacement for LLMGrading)
# ---------------------------------------------------------------------------


class AliasMatchGrading(Job):
    """Grade a transcriptions.jsonl (the same schema ``LLMGrading`` consumes) by normalized alias
    containment -- no vLLM judge. Login-node mini_task (free, seconds). Writes eval_results.jsonl +
    summary.json in the EXACT schema ``LLMGrading`` produces, so it is a drop-in for the tail of the
    benchmark when you want a fast, deterministic, judge-free number for monitoring."""

    def __init__(self, *, in_json: tk.Path):
        self.in_json = in_json
        self.out_eval = self.output_path("eval_results.jsonl")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        rows = []
        with open(str(self.in_json.get())) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        eval_results = []
        for r in rows:
            binary, quality = alias_match(r.get("transcription", ""), r["answer"], r.get("aliases", []))
            eval_results.append(
                {
                    "question": r["question"],
                    "reference": r["answer"],
                    "transcription": r.get("transcription", ""),
                    "category": r.get("category", "unknown"),
                    "binary_correct": binary,
                    "quality_score": quality,
                    "reasoning": "alias-match (normalized containment; approximate proxy, not LLM-judged)",
                }
            )

        with open(str(self.out_eval.get()), "w") as f:
            for er in eval_results:
                f.write(json.dumps(er) + "\n")
        summary = summarize(eval_results)
        with open(str(self.out_summary.get()), "w") as f:
            json.dump(summary, f, indent=2)
        n = summary["overall"]["n"]
        acc = summary["overall"]["accuracy"]
        print(f"[quick-knowledge] {n} graded, alias-match accuracy = {acc:.1%}", flush=True)


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def quick_knowledge_eval_py(
    *,
    tag: str,
    moshi_checkpoint: tk.Path | None = None,
    checkpoint_step: int | None = None,
    #: Overlay layout of ``moshi_checkpoint``: "lora" (adapter) or "full" (whole state dict).
    #: Must match the ARM; see resolve_lora. Default keeps existing LoRA hashes.
    moshi_overlay_kind: str = "lora",
    pplex_checkpoint: tk.Path | None = None,
    pplex_step: int | None = None,
    audex_checkpoint: tk.Path | None = None,
    audex_step: int | None = None,
    speech_backend=MOSHI_BACKEND,
    dataset_name: str = "triviaqa",
    split: str = "validation",
    llm_name: str = "google/gemma-4-31B-it",
    n: int = QUICK_N,
    seed: int = QUICK_SEED,
    code_version: int = 1,
    use_monologue: bool = True,
    notify: bool = True,
):
    """Assemble ONE small, fast knowledge eval for a checkpoint and register it under
    ``benchmark/quick/<tag>/``. Returns the ``AliasMatchGrading`` handle.

    Reuses the shared, cached data stages of the full benchmark (same ``LoadRawDataset`` /
    ``LLMPreprocess`` hashes -> no recompute), then a SMALL ``SubsampleDataset(n)`` + its TTS, a
    single-shard ``SpeechInference``, the free monologue read-out, and judge-free alias grading.

    ``moshi_checkpoint`` + ``checkpoint_step`` select a fine-tune LoRA (``None`` = base moshiko);
    ``pplex_*`` / ``audex_*`` select the other model families, mirroring ``knowledge_benchmark_py``.
    ``use_monologue`` reads the inner-monologue text (free, moshi-family only); set False to Whisper
    the reply audio instead (needed for backends without a monologue dump).
    """
    from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
        chatterbox_venv,
        moshi_venv,
        whisper_venv,
        make_speakers,
    )

    # 1-2. SHARED cached data stages (constructed identically to knowledge_benchmark_py so the hash
    # matches and nothing recomputes).
    raw = LoadRawDataset(dataset_name=dataset_name, split=split)
    preprocess = LLMPreprocess(in_hf=raw.out_hf, llm_name=llm_name)

    # 2b. Small reproducible subsample (own seed -> its own tiny cached artifact).
    data = SubsampleDataset(in_hf=preprocess.out_hf, n=n, seed=seed).out_hf
    tk.register_output(f"benchmark/quick/{tag}/sampled", data)

    # 3-4. Speaker voice + TTS (cached; shared across quick evals with the same n/seed).
    speakers = make_speakers()
    tts = ChatterboxSingleSpeakerInference(
        venv_python_path=chatterbox_venv(),
        in_hf=data,
        speaker_dir=speakers.out_dir,
    )
    tk.register_output(f"benchmark/quick/{tag}/tts_output", tts.out_dir)

    # 5. Model answers the questions (single shard -- small N; the model loads once).
    if audex_checkpoint is not None:
        lora_weights, lora_config = resolve_audex_weights(audex_checkpoint, audex_step), None
    elif pplex_checkpoint is not None:
        lora_weights, lora_config = resolve_personaplex_weights(pplex_checkpoint, pplex_step), None
    elif moshi_checkpoint is not None:
        lora_weights, lora_config = resolve_lora(moshi_checkpoint, checkpoint_step, moshi_overlay_kind)
    else:
        lora_weights, lora_config = None, None

    inference_venv = speech_backend.inference_venv() if speech_backend.inference_venv else moshi_venv()
    moshi_out = sharded_knowledge_inference(
        num_shards=1,
        venv_python_path=inference_venv,
        in_dir=tts.out_dir,
        lora_weights=lora_weights,
        lora_config=lora_config,
        offline_script=(speech_backend.offline_script or "moshi_offline_inference.py")
        if (speech_backend.offline_script or speech_backend.offline_module)
        else None,
        offline_module=speech_backend.offline_module,
        offline_extra_args=speech_backend.offline_extra_args,
        server=speech_backend.server,
        file_client=speech_backend.file_client,
        ws_url=speech_backend.ws_url,
        cloud_api=speech_backend.cloud_api,
        retrieval_llm=speech_backend.retrieval_llm,
        rqmt_override=speech_backend.rqmt_override,
        code_version=code_version,
    )
    tk.register_output(f"benchmark/quick/{tag}/moshi_output", moshi_out)

    # 6. Read out the answer text -- inner monologue (free) or Whisper the reply audio.
    if use_monologue:
        transcription = MonologueTranscription(in_dir=moshi_out, reference_data=data)
    else:
        transcription = WhisperTranscription(
            venv_python_path=whisper_venv(),
            in_dir=moshi_out,
            reference_data=data,
            model_size="large-v3-turbo",
        )
    tk.register_output(f"benchmark/quick/{tag}/transcription", transcription.out_json)

    # 7. Judge-free grading (login-node, instant).
    grading = AliasMatchGrading(in_json=transcription.out_json)
    tk.register_output(f"benchmark/quick/{tag}/eval_results", grading.out_eval)
    tk.register_output(f"benchmark/quick/{tag}/summary", grading.out_summary)

    if notify:
        notify_result(f"quick_{tag}", {"summary": grading.out_summary})

    return grading


#: How many checkpoints a derived track measures, final one included.
DEFAULT_EVAL_POINTS = 4

#: ``save_every`` as ``finetune.py`` renders it when a run does not override it. Owned by
#: ``speech_inference`` (next to the graph-build checkpoint guard) and re-exported here so the two
#: validators cannot disagree about the cadence.
SAVE_EVERY = _SAVE_EVERY

#: A step of ``None`` means "whatever the latest checkpoint is when this job runs" --
#: ``ResolveOverlayCheckpoint(step=None)`` resolves it. This is the ONLY way to name the final
#: checkpoint of a run whose length is not known at graph-build time (see below).
LATEST = None


def default_eval_steps(
    max_steps: int | None,
    save_every: int = SAVE_EVERY,
    *,
    points: int = DEFAULT_EVAL_POINTS,
) -> tuple:
    """Checkpoint steps to benchmark for a training run: a cadence, PLUS the final step, always.

    Derived from the run's own ``max_steps`` so it cannot drift out of sync with it.

    ``max_steps=None`` means the run is sized at RUN time -- ``resolve_max_steps`` computes it from
    the dataset when ``num_epochs`` is set, so the recipe genuinely cannot know the final step (e.g.
    ``ft_v3_r8`` at ``num_epochs=1`` ends at step 3871). In that case the only honest answer is
    ``(LATEST,)``: name the final checkpoint by reference, not by number. A caller that wants
    intermediates for such a run must list them explicitly, accepting that a step past the run's
    actual end has no checkpoint.

    Every emitted step is one that will EXIST: intermediates are snapped down to ``save_every``
    multiples and anything at or past ``max_steps`` is dropped in favour of ``max_steps`` itself
    (which ``run_training`` always writes). So a 120-step run yields just its final checkpoint
    rather than a 500 that never happens.
    """
    assert save_every > 0, save_every
    if max_steps is None:
        return (LATEST,)
    assert max_steps > 0, max_steps
    out = {max_steps}
    for k in range(1, max(points, 1)):
        snapped = int((max_steps * k / max(points, 1)) // save_every) * save_every
        if 0 < snapped < max_steps:
            out.add(snapped)
    return tuple(sorted(out))


def attach_knowledge_evals(
    *,
    run_tag: str,
    max_steps: int | None,
    save_every: int = SAVE_EVERY,
    moshi_checkpoint: tk.Path | None = None,
    pplex_checkpoint: tk.Path | None = None,
    audex_checkpoint: tk.Path | None = None,
    steps=None,
    speech_backend=MOSHI_BACKEND,
    n: int = QUICK_N,
    **kwargs,
):
    """Attach a fast knowledge eval at each of ``steps`` for one training run, giving it a
    accuracy-vs-step curve. ONE call to give any run the standing eval routine.
    Returns ``{step: grading_handle}``.

    Each eval depends on the training job's ``out_rundir``, so the whole track fires **after training
    finishes**, not as each checkpoint lands -- Sisyphus dependencies are job-level. (This docstring
    claimed the latter for months; it was never true.)

    ``steps`` defaults to :func:`default_eval_steps` over the run's own ``max_steps``. Pass it
    explicitly only to override the cadence -- the final checkpoint is enforced either way.

    **The final checkpoint MUST be measured.** Hardcoded step lists silently rot: the a8-4gpu track
    said ``(1000, 2000, 3000)`` while the run was lengthened to 3,750, so the end-of-epoch
    checkpoint -- the point of the run -- would have gone unmeasured, and that run has no in-loop
    probe to fall back on (2026-08-05). ``max_steps=None`` (a run sized from ``num_epochs`` at run
    time) can only satisfy this with :data:`LATEST` in ``steps``, since no number is knowable here.

    Every listed step must also be a checkpoint that will exist: a multiple of ``save_every`` below
    ``max_steps``, ``max_steps`` itself, or :data:`LATEST`.
    """
    if steps is None:
        steps = default_eval_steps(max_steps, save_every)
    steps = tuple(steps)
    assert steps, f"{run_tag}: empty checkpoint evals -- the run would produce no curve at all"
    covers_final = (LATEST in steps) or (max_steps is not None and max_steps in steps)
    assert covers_final, (
        f"{run_tag}: checkpoint evals {steps} does not measure the FINAL checkpoint "
        f"(max_steps={max_steps}). A track that stops short leaves the end of the run unmeasured, "
        f"which is how a lengthened run silently loses its last datapoint. Add "
        f"{max_steps if max_steps is not None else 'LATEST'} to steps, or let steps default."
        + (
            " This run's length is resolved at RUN time (num_epochs), so LATEST is the only way "
            "to name its final checkpoint."
            if max_steps is None
            else ""
        )
    )
    for s in steps:
        # The same rule ResolveOverlayCheckpoint applies to every checkpoint reference it mints;
        # applying it here too names the TRACK in the message rather than one resolved path.
        reason = impossible_checkpoint_reason(s, max_steps=max_steps, save_every=save_every)
        assert reason is None, f"{run_tag}: track step {s} {reason}"
    handles = {}
    for step in steps:
        step_tag = f"{run_tag}_s{step}" if step is not None else f"{run_tag}_final"
        handles[step] = quick_knowledge_eval_py(
            tag=step_tag,
            moshi_checkpoint=moshi_checkpoint,
            checkpoint_step=step,
            pplex_checkpoint=pplex_checkpoint,
            pplex_step=step,
            audex_checkpoint=audex_checkpoint,
            audex_step=step,
            speech_backend=speech_backend,
            n=n,
            **kwargs,
        )
    return handles


# ---------------------------------------------------------------------------
# In-training-loop probe set (question wavs + gold answers packaged for the trainer)
# ---------------------------------------------------------------------------


class BuildKnowledgeProbeSet(Job):
    """Package a held-out knowledge probe set the TRAINER can consume in-loop: one ``probe.jsonl``
    aligning each TTS'd question wav to its gold answer. Login-node mini_task (free).

    ``clip i = dataset row i = <i>.wav`` (the benchmark's contract), so we glob ``tts_dir`` for
    ``<i>.wav`` and look up row ``i`` of ``dataset`` for answer/aliases/category. Each line:
    ``{"i", "wav": <abs Lustre path>, "answer", "aliases", "category", "question"}``. The absolute
    wav path points into the (persistent) TTS job output, so the trainer reads it directly -- no copy.
    The moshi_family launcher's in-loop probe reads this file (``knowledge_probe_data``)."""

    def __init__(self, *, tts_dir: tk.Path, dataset: tk.Path):
        self.tts_dir = tts_dir
        self.dataset = dataset
        self.out_jsonl = self.output_path("probe.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import glob

        from datasets import load_from_disk

        ds = load_from_disk(str(self.dataset.get()))
        tts_dir = str(self.tts_dir.get())
        wavs = sorted(
            glob.glob(os.path.join(tts_dir, "*.wav")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
        n = 0
        with open(str(self.out_jsonl.get()), "w") as f:
            for w in wavs:
                i = int(os.path.splitext(os.path.basename(w))[0])
                if i >= len(ds):
                    continue
                row = ds[i]
                f.write(
                    json.dumps(
                        {
                            "i": i,
                            "wav": os.path.realpath(w),
                            "answer": row["answer"],
                            "aliases": row.get("aliases", []),
                            "category": row.get("category", "unknown"),
                            "question": row.get("question", ""),
                        }
                    )
                    + "\n"
                )
                n += 1
        assert n > 0, f"no <i>.wav found in {tts_dir} -- probe set would be empty"
        print(f"[probe-set] wrote {n} probe entries -> {self.out_jsonl.get()}", flush=True)


def knowledge_probe_set_py(
    *,
    n: int = QUICK_N,
    seed: int = QUICK_SEED,
    dataset_name: str = "triviaqa",
    split: str = "validation",
    llm_name: str = "google/gemma-4-31B-it",
    tag: str = "default",
) -> tk.Path:
    """Materialize a held-out knowledge probe set (``probe.jsonl``) for the in-training-loop probe and
    return its path -- feed it to ``SpeechFinetune(knowledge_probe_data=...)`` (MOSHI_LIB_ADAPTER).

    Reuses the SAME cached data stages as ``quick_knowledge_eval_py`` (identical ``SubsampleDataset``
    (n, seed) + TTS hashes), so a probe set costs only the tiny packaging job on top of stages the
    quick-eval track already built. The subsample is drawn from the ``validation`` split (held out
    from every finetune's train data), so it is a fair recall probe.
    """
    from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
        chatterbox_venv,
        make_speakers,
    )

    raw = LoadRawDataset(dataset_name=dataset_name, split=split)
    preprocess = LLMPreprocess(in_hf=raw.out_hf, llm_name=llm_name)
    data = SubsampleDataset(in_hf=preprocess.out_hf, n=n, seed=seed).out_hf
    speakers = make_speakers()
    tts = ChatterboxSingleSpeakerInference(
        venv_python_path=chatterbox_venv(),
        in_hf=data,
        speaker_dir=speakers.out_dir,
    )
    probe = BuildKnowledgeProbeSet(tts_dir=tts.out_dir, dataset=data)
    tk.register_output(f"benchmark/probe_set/{tag}/probe_jsonl", probe.out_jsonl)
    return probe.out_jsonl
