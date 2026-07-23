from glob import glob
import json
from pathlib import Path
from typing import List, Sequence, Tuple
from sisyphus import Job, Task, tk
import os
import sys
import subprocess
from .common import vllm_server
from .speech_inference import SpeechInference
from .inference_harness import FDB_TASK_MAP
from i6_experiments.users.dorian_koch.jobs.venv import CreateVenv
from functools import lru_cache

from .moshi_client import MoshiFileClient, _ws_url, moshi_server
from .speech_backends import MOSHI_BACKEND

# `moshified_fdb_v1_v15` lives at recipe/moshified_fdb_v1_v15 (a sibling of i6_experiments).
# Sisyphus's RecipeFinder resolves recipe imports relative to the cwd, but a worker's cwd can
# change during run() (e.g. the moshi server block below), which breaks the lazy imports of
# moshified_fdb_v1_v15. Pin the recipe dir on sys.path so those imports work regardless of cwd.
import i6_experiments as _i6_experiments

_recipe_dir = os.path.dirname(os.path.dirname(_i6_experiments.__file__))
if _recipe_dir not in sys.path:
    sys.path.insert(0, _recipe_dir)

# {tag -> {task -> eval_job}}, populated as a side effect of every fdb_benchmark_py call so a
# late aggregation (fdb_latency_histograms_from_registry) can wire a cross-model comparison
# job without threading eval-job handles through each scattered call site.
_FDB_REGISTRY: dict = {}

# Model provenance for every FDB-derived visualization (CLAUDE.md rule: always show what WE trained
# vs. a released checkpoint we only ran inference on). "ours" = weights/adapter we trained
# (finetune / LoRA / RL); "hf" = a released HuggingFace checkpoint. Single source of truth --
# extend when a new tag is benchmarked. Second field = the base the weights sit on, for captions.
#: Canonical provenance map: benchmark tag -> ("ours" | "hf", base model).
#:
#: "ours" = a checkpoint WE trained (finetune / LoRA / RL adapter); "hf" = a released checkpoint we
#: only ran inference on. Every visualization must render the two differently, so a reader can never
#: mistake a baseline for our contribution.
#:
#: **Add an entry whenever you add a benchmark tag.** `_origin_map` raises on an unknown tag rather
#: than silently omitting it -- an omitted tag used to render with no provenance marker at all,
#: which is exactly the mislabelling this map exists to prevent.
FDB_MODEL_ORIGIN = {
    # --- released HuggingFace checkpoints (baselines) ---------------------------------------
    "moshi_base": ("hf", "kyutai/moshiko"),
    "moshi_family": ("hf", "kyutai/moshiko (our lib)"),
    "moshirag": ("hf", "kyutai/moshika-rag"),
    "moshirag_family": ("hf", "kyutai/moshika-rag (our lib)"),
    "moshika_rl_seamless_hf": ("hf", "kyutai/moshika-rl-seamless"),
    "personaplex": ("hf", "nvidia/personaplex-7b-v1"),
    "personaplex_family": ("hf", "nvidia/personaplex-7b-v1 (our lib)"),
    "personaplex_b1": ("hf", "nvidia/personaplex-7b-v1"),
    "personaplex_b8": ("hf", "nvidia/personaplex-7b-v1"),
    "unmute": ("hf", "kyutai Unmute"),
    "kame_family": ("hf", "KAME (our lib)"),
    "flm_audio": ("hf", "FLM-Audio"),
    # Cloud APIs. The "hf" bucket really means "external baseline, not trained by us" -- these are
    # not HuggingFace checkpoints, but they belong on the same side of the ours/theirs split.
    "gemini": ("hf", "Google Gemini (cloud API)"),
    "openai": ("hf", "OpenAI Realtime (cloud API)"),
    # --- checkpoints we trained --------------------------------------------------------------
    # TriviaQA-synthetic LoRA finetunes of moshiko (the knowledge-degradation investigation).
    "moshi_ft": ("ours", "LoRA on kyutai/moshiko"),
    "moshi_ft_v2": ("ours", "LoRA on kyutai/moshiko"),
    "moshi_ft_v3": ("ours", "LoRA on kyutai/moshiko"),
    "moshi_ft_v3_r16": ("ours", "LoRA r16 on kyutai/moshiko"),
    "moshi_ft_v3_r8": ("ours", "LoRA r8 on kyutai/moshiko"),
    "moshi_ft_a2_lowlr": ("ours", "LoRA on kyutai/moshiko (lr 1e-6)"),
    # A0-null is the pipeline control: 0 training steps, so the adapter is an identity and this
    # should reproduce moshi_base exactly. Marked "ours" because it goes through our train/save/load
    # path -- it is a check on our tooling, not a released checkpoint.
    "moshi_ft_a0_null": ("ours", "identity LoRA on kyutai/moshiko (0-step control)"),
    # A4: same hyper-params as A2, but trained on real Fisher conversations instead of
    # TriviaQA-synthetic -- the corpus is the only variable between the two.
    "moshi_ft_a4_fisher": ("ours", "LoRA on kyutai/moshiko (Fisher)"),
    # A6: same hparams as A2, TriviaQA reshaped for turn count / answer length -- the corpus-shape arm.
    "moshi_ft_a6_mix": ("ours", "LoRA on kyutai/moshiko (reshaped TriviaQA mix)"),
    "moshi_lib_ft": ("ours", "LoRA on kyutai/moshiko (our lib)"),
    "moshi_lib_smoke": ("ours", "LoRA on kyutai/moshiko (smoke test)"),
    "moshirag_lib_ft": ("ours", "LoRA on kyutai/moshika-rag (our lib)"),
    # RL-finetuned adapters.
    "moshi_rl": ("ours", "RL LoRA on kyutai/moshiko"),
    "moshi_rl_seamless": ("ours", "RL LoRA on kyutai/moshiko"),
    "moshi_rl_seamless_ckpt1400": ("ours", "RL LoRA on kyutai/moshiko"),
    # PersonaPlex finetunes.
    "personaplex_ft": ("ours", "finetune of nvidia/personaplex-7b-v1"),
    "personaplex_lib_ft": ("ours", "finetune of nvidia/personaplex-7b-v1 (our lib)"),
}


def _origin_map(labels):
    """``{label -> "ours" | "hf"}`` for the given benchmark tags.

    Raises on an unknown tag instead of dropping it: a missing entry would render that model with no
    provenance marker, letting a checkpoint we trained pass for a released baseline (or vice versa).
    Fix by adding the tag to :data:`FDB_MODEL_ORIGIN`, not by loosening this check.
    """
    unknown = sorted(set(labels) - set(FDB_MODEL_ORIGIN))
    assert not unknown, (
        f"no provenance entry for benchmark tag(s) {unknown} -- add them to FDB_MODEL_ORIGIN "
        f"in fdb.py ('ours' for a checkpoint we trained, 'hf' for a released one) so they are "
        f"marked correctly in plots/tables"
    )
    return {lab: FDB_MODEL_ORIGIN[lab][0] for lab in labels}


@lru_cache
def fdb_asr_venv():
    """Dedicated venv for the Full-Duplex-Bench NeMo ASR scoring (asr.py).

    The manager .venv pins NeMo 2.0.0 (forced by transformers>=5.5.0), which lacks
    transcribe(timestamps=True). This isolated venv installs NeMo >=2.2 so asr.py
    runs unmodified (upstream-style), invoked as a subprocess.
    """
    venv = CreateVenv(
        packages=[
            [
                "torch",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            ["nemo_toolkit[asr]>=2.2", "soundfile"],
        ],
        hash_overwrite="fdb_asr_venv_v1",
    )
    tk.register_output("venv/fdb_asr", venv.out_env_path)
    return venv.out_python_path


@lru_cache
def fdb_eval_venv():
    """Dedicated venv for the Full-Duplex-Bench non-LLM scoring (the fork's evaluate.py).

    The fork eval scripts (backchannel / pause / turn-taking) need silero-vad + scipy +
    tqdm on top of torch/torchaudio/soundfile, and load wavs via soundfile (no torchcodec).
    Kept separate from the manager .venv so its torch/torchaudio churn (which made
    torchaudio.load hard-require torchcodec) can't break scoring. Versions are pinned to
    the manager .venv at port time so metrics stay identical; CPU wheels suffice since the
    eval runs as a login-node mini_task. Invoked as a subprocess via out_python_path.
    """
    venv = CreateVenv(
        packages=[
            [
                "torch==2.11.0",
                "torchaudio==2.11.0",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            ["silero-vad==6.2.1", "scipy==1.17.1", "numpy==2.2.6", "tqdm==4.67.3", "soundfile"],
        ],
        hash_overwrite="fdb_eval_venv_v1",
    )
    tk.register_output("venv/fdb_eval", venv.out_env_path)
    return venv.out_python_path


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


class FullDuplexBenchEval_Evaluation(Job):
    def __init__(
        self,
        *,
        fdb_task: str,
        in_audios: tk.Path,
        hf_model: str = "openai/gpt-oss-120b",
        eval_venv_python: tk.AbstractPath | None = None,
    ):
        self.fdb_task = fdb_task

        self.in_audios = in_audios
        # Venv python used to run the fork's evaluate.py as a subprocess for the non-LLM
        # tasks (hash-excluded: only affects how scoring runs, not the produced result).
        self.eval_venv_python = eval_venv_python
        self.out_log = self.output_path("evaluation_output.txt")
        self.out_eval = self.output_path("eval.json")
        self.needs_llm_inference = FDB_TASK_MAP.get(self.fdb_task, self.fdb_task) in [
            "user_interruption",
            "behavior",
            "general_before_after",
        ]
        if self.needs_llm_inference:
            self.rqmt = {
                "gpu": 1,
                "cpu": 2,
                "mem": 16,
                "time": 2,
            }
            self.hf_model = hf_model
        else:
            self.rqmt = None

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        # Hash the eval venv only when it is actually used (non-LLM tasks); excluding it
        # when None keeps the expensive LLM (user_interruption) eval job hashes stable.
        if d.get("eval_venv_python") is None:
            d.pop("eval_venv_python", None)
        d["__version"] = 5
        return super().hash(d)

    def tasks(self):
        if self.rqmt is None:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        import moshified_fdb_v1_v15

        mapped_task = FDB_TASK_MAP.get(self.fdb_task, self.fdb_task)

        if not self.needs_llm_inference:
            # Non-LLM scoring: run the fork's evaluate.py as a subprocess in a venv with a
            # working audio stack (fdb_asr_venv). Avoids importing the fork into the manager
            # .venv, whose torchaudio now hard-requires torchcodec (absent here and on the
            # login node). The fork loads wavs via soundfile, so no torchcodec is needed.
            assert self.eval_venv_python is not None, "eval_venv_python required for non-LLM scoring"
            evaluate_py = os.path.join(os.path.dirname(moshified_fdb_v1_v15.__file__), "evaluation", "evaluate.py")
            cmd = [
                self.eval_venv_python.get(),
                evaluate_py,
                "--task",
                mapped_task,
                "--root_dir",
                self.in_audios.get_path(),
                "--out-json",
                self.out_eval.get_path(),
            ]
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            with open(self.out_log, "w", encoding="utf-8") as f:
                subprocess.run(cmd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)
            return

        # LLM-judged task (user_interruption): needs the vLLM server, so run in-process in
        # the manager .venv. eval_user_interruption does not load audio via torchaudio, so
        # torchcodec is not involved on this path.
        import moshified_fdb_v1_v15.evaluation.evaluate as moshified_fdb_v1_v15_evaluate
        from moshified_fdb_v1_v15.evaluation.evaluate import main as evaluate_main

        sys.argv = [
            "evaluate.py",
            "--task",
            mapped_task,
            "--root_dir",
            self.in_audios.get_path(),
        ]
        with open(self.out_log, "w", encoding="utf-8") as f:
            tee = Tee(sys.stdout, f)
            sys.stdout = tee
            sys.path.append(str(Path(moshified_fdb_v1_v15_evaluate.__file__).parent))
            with vllm_server(self.hf_model) as llm_url:
                sys.argv += ["--llm-api-url", llm_url, "--llm-model", self.hf_model]
                os.environ["OPENAI_API_KEY"] = "fake_key_for_vllm"
                res = evaluate_main()
            with open(self.out_eval, "w", encoding="utf-8") as f_eval:
                json.dump(res, f_eval, indent=4)
            sys.stdout = sys.__stdout__


def moshified_fdb_eval(
    fdb_task: str,
    backend=MOSHI_BACKEND,
    lora_weights=None,
    lora_config=None,
    asr_venv_python=None,
    server_venv_python=None,
    unmute_llm=None,
    code_version=1,
    seed=None,
):
    # Cloud realtime backends (Gemini/OpenAI) run as a login-node mini_task.
    infer = SpeechInference(
        mode="fdb",
        fdb_task=fdb_task,
        code_version=code_version,
        server=backend.server,
        hf_repo=backend.server_hf_repo,
        seed=seed,
        file_client=backend.file_client,
        ws_url=backend.ws_url,
        # FDB uses the offline driver only for server-less (end-to-end) backends; Moshi &
        # Unmute (both have a server) stay on the realtime streaming path. SpeechInference
        # infers offline-vs-server from offline_script/module presence, so leave them None
        # for served backends.
        offline_script=backend.offline_script if backend.server is None else None,
        offline_module=backend.offline_module if backend.server is None else None,
        # The seed reaches the two inference paths differently: served backends (Moshi/Unmute)
        # take it via SpeechInference(seed=) -> moshi_server; the offline driver takes it as a CLI
        # arg. Inject --seed into the offline args here so one fdb_benchmark_py(seed=) knob covers both.
        offline_extra_args=(
            (*backend.offline_extra_args, "--seed", str(int(seed)))
            if (backend.server is None and seed is not None)
            else (backend.offline_extra_args if backend.server is None else ())
        ),
        # RAG retrieval (MoshiRAG) only applies to the offline/server-less path.
        retrieval_llm=backend.retrieval_llm if backend.server is None else None,
        lora_weights=lora_weights,
        lora_config=lora_config,
        asr_venv_python=asr_venv_python,
        venv_python_path=server_venv_python,
        unmute_llm=unmute_llm,
        cloud_api=backend.cloud_api,
    )
    # rqmt is not hashed: apply the backend resource override by mutating the built job
    # (server-less/offline backends only; served backends keep the default rqmt).
    if backend.server is None and backend.rqmt_override is not None:
        infer.rqmt = {**infer.rqmt, **backend.rqmt_override}
    needs_llm = FDB_TASK_MAP.get(fdb_task, fdb_task) in [
        "user_interruption",
        "behavior",
        "general_before_after",
    ]
    _eval = FullDuplexBenchEval_Evaluation(
        fdb_task=fdb_task,
        in_audios=infer.out_dir,
        eval_venv_python=None if needs_llm else fdb_eval_venv(),
    )
    return _eval.out_eval


FDB_TASKS = [
    "icc_backchannel",
    "candor_turn_taking",
    "candor_pause_handling",
    "synthetic_pause_handling",
    "synthetic_user_interruption",
]


def fdb_benchmark_py(
    tag: str = "moshi_base",
    moshi_checkpoint: tk.Path | None = None,
    checkpoint_step: int | None = None,
    pplex_checkpoint: tk.Path | None = None,
    pplex_step: int | None = None,
    server_venv_python: tk.AbstractPath | None = None,
    backend=MOSHI_BACKEND,
    unmute_llm: str | None = None,
    code_version: int = 1,
    seed: int | None = None,
    tasks=None,
):
    """Full Duplex Bench eval over all tasks for one model, namespaced by ``tag``.

    ``moshi_checkpoint`` is a ``MoshiFinetune.out_rundir`` to eval a fine-tuned (LoRA)
    model via the moshi server; ``None`` evals the base ``kyutai/moshiko``. Registered
    under ``fdb/<tag>/<task>/...`` so base and finetuned runs coexist.
    """
    # Local import avoids a module-load cycle (knowledge_benchmark imports fdb-side helpers).
    from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import (
        resolve_lora,
        resolve_personaplex_weights,
    )

    # PersonaPlex finetune = a partial state_dict overlay (trained_heads.safetensors, no config);
    # Moshi finetune = a LoRA adapter. Both ride the same per-run lora_weights/lora_config seam.
    if pplex_checkpoint is not None:
        lora_weights, lora_config = resolve_personaplex_weights(pplex_checkpoint, pplex_step), None
    else:
        lora_weights, lora_config = resolve_lora(moshi_checkpoint, checkpoint_step)

    asr_venv_python = fdb_asr_venv()
    eval_jobs: dict = {}
    # ``tasks`` lets a caller (e.g. the seed-variance sweep) run a subset -- e.g. only
    # candor_turn_taking, the one task that logs per-turn latencies -- to bound GPU cost.
    for t in FDB_TASKS if tasks is None else tasks:
        bench = moshified_fdb_eval(
            fdb_task=t,
            code_version=code_version,
            backend=backend,
            lora_weights=lora_weights,
            lora_config=lora_config,
            asr_venv_python=asr_venv_python,
            server_venv_python=server_venv_python,
            unmute_llm=unmute_llm,
            seed=seed,
        )
        tk.register_output(f"fdb/{tag}/{t}/eval", bench)
        tk.register_output(f"fdb/{tag}/{t}/audio", bench.creator.in_audios)
        bench.creator.add_alias(f"fdb/{tag}/{t}")
        eval_jobs[t] = bench.creator
    # Return {task -> FullDuplexBenchEval_Evaluation} so callers can wire follow-on jobs
    # (e.g. fdb_latency_histograms_py) onto each task's out_log. Was None before; additive.
    _FDB_REGISTRY[tag] = eval_jobs
    return eval_jobs


class FDBLatencyHistogram(Job):
    """Overlaid histogram of Full-Duplex-Bench take-turn *latencies* across one or more runs.

    Parses the per-turn ``the latency is <float|None>`` lines the fork's turn-taking evaluator
    prints into each ``FullDuplexBenchEval_Evaluation`` log (``out_log``). Negative latency =
    the model starts its turn BEFORE the reference speaker finishes (early / interrupting);
    positive = it waits after. ``None`` = the model did not take the turn (TOR=0): counted for
    the take-turn rate, excluded from the latency distribution.

    Only ``candor_turn_taking`` emits these lines today (other tasks log TOR only); an empty
    series is rendered as an empty labelled bar rather than crashing, so the job is safe to
    point at any task. Reads existing logs only -> a cheap login-node mini_task that never
    re-runs the (expensive, GPU) eval it summarizes.
    """

    def __init__(self, *, eval_logs: dict, task: str, origin: dict | None = None, clip_s: float = 15.0, bins: int = 48):
        # eval_logs: {label -> evaluation_output.txt Path}. Insertion order = legend order + hash.
        # origin: {label -> "ours"|"hf"} provenance, surfaced in every plot + the stats json so a
        # reader always sees what we trained vs. a released checkpoint (see FDB_MODEL_ORIGIN).
        self.eval_logs = eval_logs
        self.task = task
        self.origin = origin or {}
        self.clip_s = clip_s
        self.bins = bins
        self.out_png = self.output_path("latency_hist.png")
        self.out_stats = self.output_path("latency_stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _parse(self, path: str):
        import re

        pat = re.compile(r"the latency is (None|-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)")
        lat: list = []
        n_none = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = pat.search(line)
                if not m:
                    continue
                if m.group(1) == "None":
                    n_none += 1
                else:
                    lat.append(float(m.group(1)))
        return lat, n_none

    def _prov(self, label):
        """(marker, is_ours, short_text) for a label's provenance; neutral dot if unknown."""
        o = self.origin.get(label)
        if o == "ours":
            return "●", True, "ours"
        if o == "hf":
            return "○", False, "HF"
        return "·", False, ""

    def run(self):
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        series: dict = {}
        stats: dict = {}
        for label, log in self.eval_logs.items():
            lat, n_none = self._parse(log.get_path())
            series[label] = lat
            n_total = len(lat) + n_none
            core = [v for v in lat if abs(v) <= self.clip_s]
            stats[label] = {
                "n_rounds": n_total,
                "n_took_turn": len(lat),
                "n_no_turn": n_none,
                "take_turn_rate": (len(lat) / n_total) if n_total else 0.0,
                "latency_mean": float(np.mean(lat)) if lat else None,
                "latency_median": float(np.median(lat)) if lat else None,
                "latency_std": float(np.std(lat)) if lat else None,
                "latency_min": min(lat) if lat else None,
                "latency_max": max(lat) if lat else None,
                "frac_early_lt0": (sum(v < 0 for v in lat) / len(lat)) if lat else None,
                "n_outliers_gt_clip_s": len(lat) - len(core),
                "origin": self.origin.get(label),
            }
        # Guard (see fdb.md "seed-plumbing bug"): >=2 runs whose latency series are byte-identical is
        # the tell-tale of a seeding failure -- e.g. a wrapper monkeypatch that never applied, so every
        # "seed" ran the same fixed RNG. Distinct runs/seeds of a sampled model must not collapse to
        # identical numbers. Warn loudly so it can never ship silently again (not a hard raise: a truly
        # greedy/deterministic backend legitimately produces identical series across seeds).
        if len(series) >= 2 and len({tuple(v) for v in series.values()}) == 1:
            print(
                "\n" + "!" * 78 + "\n"
                f"!! FDBLatencyHistogram[{self.task}]: all {len(series)} runs "
                f"({', '.join(series)}) produced\n"
                "!! BIT-IDENTICAL latency series -- seeding almost certainly NOT applied\n"
                "!! (unless this backend is intentionally greedy/deterministic). Check the seed path.\n"
                + "!" * 78
                + "\n",
                flush=True,
            )

        with open(self.out_stats.get_path(), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        # Pick the readable view for the model count: overlaid histograms for a small A/B/C
        # comparison, a sorted ridgeline once there are enough models that overlaid bars muddy.
        if len(series) <= 3:
            self._plot_overlay(series, stats, plt, np)
        else:
            self._plot_ridgeline(series, stats, plt, np)

    def _plot_overlay(self, series, stats, plt, np):
        edges = np.linspace(-self.clip_s, self.clip_s, self.bins + 1)
        colors = plt.get_cmap("tab10").colors
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, (label, lat) in enumerate(series.items()):
            core = [v for v in lat if abs(v) <= self.clip_s]
            s = stats[label]
            n_out = s["n_outliers_gt_clip_s"]
            med = s["latency_median"]
            mark, _is_ours, ptxt = self._prov(label)
            tag = f" [{ptxt}]" if ptxt else ""
            leg = f"{mark} {label}{tag}  (n={s['n_took_turn']}, take={s['take_turn_rate']:.0%}"
            leg += f", med={med:.2f}s" if med is not None else ", med=n/a"
            leg += f", +{n_out} |lat|>{self.clip_s:g}s)" if n_out else ")"
            c = colors[i % len(colors)]
            ax.hist(core, bins=edges, alpha=0.5, color=c, edgecolor="black", linewidth=0.4, label=leg)
            if med is not None:
                ax.axvline(max(-self.clip_s, min(self.clip_s, med)), color=c, linestyle="--", linewidth=1.6)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("take-turn latency (s)    [<0 = model starts before ref speaker ends; >0 = waits]")
        ax.set_ylabel("count")
        ax.set_title(
            f"FDB {self.task}: take-turn latency distribution  (|lat| <= {self.clip_s:g}s shown; dashed = median)"
            f"\n● trained by us   ·   ○ HuggingFace checkpoint",
            fontsize=11,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.out_png.get_path(), dpi=130)
        plt.close(fig)

    def _plot_ridgeline(self, series, stats, plt, np):
        # Stacked per-model KDE densities, sorted by median (eager/interrupting at top -> patient at
        # bottom); each ridge scaled to its own peak so the SHAPE compares regardless of how many
        # turns the model took. Fill colour is diverging on the median (red=eager, blue=patient).
        grid = np.linspace(-self.clip_s, self.clip_s, 240)
        bw = 0.6
        order = sorted(
            series,
            key=lambda k: (stats[k]["latency_median"] is None, stats[k]["latency_median"] or 0.0),
        )
        cmap = plt.get_cmap("coolwarm_r")
        fig, ax = plt.subplots(figsize=(11, 0.62 * len(order) + 2.4))
        for row, label in enumerate(order):
            base = len(order) - row  # top row highest
            lat = np.array([v for v in series[label] if abs(v) <= self.clip_s], float)
            if len(lat):
                dens = np.zeros_like(grid)
                for x in lat:
                    dens += np.exp(-0.5 * ((grid - x) / bw) ** 2)
                dens /= dens.max() if dens.max() > 0 else 1.0
            else:
                dens = np.zeros_like(grid)
            s = stats[label]
            med = s["latency_median"]
            col = cmap((np.clip(med if med is not None else 0.0, -3.0, 3.0) + 3.0) / 6.0)
            ax.fill_between(grid, base, base + dens * 1.7, color=col, alpha=0.8, linewidth=0.6, edgecolor="black")
            mark, is_ours, _ptxt = self._prov(label)
            ax.annotate(
                f"{mark} {label}",
                (-self.clip_s - 0.3, base + 0.12),
                ha="right",
                va="bottom",
                fontsize=9,
                family="monospace",
                fontweight="bold" if is_ours else "normal",
                color="black" if is_ours else "0.45",
                annotation_clip=False,
            )
            rail = f"n={s['n_took_turn']}  {s['take_turn_rate']:.0%}"
            rail += f"  med={med:+.2f}s" if med is not None else ""
            ax.annotate(
                rail,
                (self.clip_s + 0.3, base + 0.12),
                ha="left",
                va="bottom",
                fontsize=8,
                family="monospace",
                color="0.35",
                annotation_clip=False,
            )
            if med is not None:
                ax.plot([max(-self.clip_s, min(self.clip_s, med))], [base], "o", color="black", ms=3.2)
        ax.axvline(0, color="black", linewidth=1.1)
        ax.set_yticks([])
        ax.set_xlim(-self.clip_s, self.clip_s)
        ax.set_ylim(0.4, len(order) + 2.2)
        ax.set_xlabel("take-turn latency (s)    [<0 = starts before ref speaker ends; >0 = waits]")
        ax.set_title(
            f"FDB {self.task}: take-turn latency distributions (per model, sorted by median)"
            f"\n● trained by us   ·   ○ HuggingFace checkpoint",
            fontsize=11,
        )
        ax.grid(True, axis="x", alpha=0.3)
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)
        fig.savefig(self.out_png.get_path(), dpi=130, bbox_inches="tight")
        plt.close(fig)


def fdb_latency_histograms_py(tag, runs, tasks=("candor_turn_taking",), origin=None, clip_s=15.0):
    """Wire an FDBLatencyHistogram comparing take-turn latency across runs.

    ``runs``: ``{label -> fdb_benchmark_py(...) return value}`` (each a ``{task -> eval_job}``
    dict). For each task in ``tasks`` shared by the runs, register an overlaid histogram under
    ``fdb_latency/<tag>/<task>/``. Cheap mini_task -- parses already-produced eval logs, so it
    re-runs only when the label set / clip changes, never the eval itself.
    """
    for task in tasks:
        logs = {label: jobs[task].out_log for label, jobs in runs.items() if task in jobs}
        if not logs:
            continue
        hist = FDBLatencyHistogram(eval_logs=logs, task=task, origin=origin or _origin_map(logs), clip_s=clip_s)
        tk.register_output(f"fdb_latency/{tag}/{task}/hist.png", hist.out_png)
        tk.register_output(f"fdb_latency/{tag}/{task}/stats.json", hist.out_stats)
        hist.add_alias(f"fdb_latency/{tag}/{task}")


def fdb_latency_histograms_from_registry(hist_tag, model_tags, tasks=("candor_turn_taking",), clip_s=15.0):
    """Wire a cross-model FDBLatencyHistogram from tags already benchmarked via fdb_benchmark_py.

    Reads ``_FDB_REGISTRY`` (populated as a side effect of each ``fdb_benchmark_py`` call), so it
    needs no handles threaded through the call sites -- just list the model tags to compare, in the
    order you want them labelled. Tags absent from the registry (never wired, or gated off this
    run) are skipped, so a blocked/missing model can never block the histogram. Call it AFTER the
    fdb_benchmark_py calls it references (registry is populated at config-load time).
    """
    runs = {tag: _FDB_REGISTRY[tag] for tag in model_tags if tag in _FDB_REGISTRY}
    for task in tasks:
        logs = {tag: jobs[task].out_log for tag, jobs in runs.items() if task in jobs}
        if len(logs) < 2:
            continue
        hist = FDBLatencyHistogram(eval_logs=logs, task=task, origin=_origin_map(logs), clip_s=clip_s)
        tk.register_output(f"fdb_latency/{hist_tag}/{task}/hist.png", hist.out_png)
        tk.register_output(f"fdb_latency/{hist_tag}/{task}/stats.json", hist.out_stats)
        hist.add_alias(f"fdb_latency/{hist_tag}/{task}")
