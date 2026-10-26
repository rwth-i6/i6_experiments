"""Held-out teacher-forced loss of PersonaPlex checkpoints (user, 2026-09-25: "it would be really good to have more
held out eval losses").

  PersonaHeldoutSet      a FIXED window set in the training table's schema (``moshi_family.personaplex.heldout_set``):
                         ``heldout`` = JRE windows no PersonaPlex arm trained on, or ``train_sample`` = seeded rows of
                         a training table; both read with the minimal prompt, the row's voice clip, the clean user
                         stream and no channel swap
  PersonaHeldoutLoss     one checkpoint (or base) on one or more sets: per window exactly the launcher's
                         ``loss_step`` (``moshi_family.personaplex.heldout_loss``), eval mode, the run's own loss
                         settings from its ``config.yaml``; mean over windows + 95% bootstrap CI
  PersonaHeldoutSummary  merges many PersonaHeldoutLoss outputs into one JSON (run, step, windows_seen, set, metrics,
                         CI; plus held-out minus train-sample per checkpoint with a bootstrap CI)
  PersonaHeldoutTurnPos  several checkpoints on the held-out set: text word CE and assistant semantic CE by position in
                         the assistant's current turn (0-5 / 5-10 / 10-20 / 20+ s, outside; ``heldout_turnpos``)

``checkpoint_path(run, kind, step)`` is how an eval waits for a checkpoint of a run that is STILL TRAINING: a
``tk.Path`` inside the run's work dir whose availability is "the checkpoint's ``consolidated/config.json`` exists"
(both launchers write it after the weights; the full-FT weights go tmp -> rename), instead of Sisyphus' default
"the creating job has finished". ``ResolveOverlayCheckpoint`` takes the whole ``run_dir`` and so waits for the end
of the run; a p5 checkpoint saved at hour 1.4 would otherwise be evaluated at hour 12. The step is checked at graph
build with ``assert_checkpoint_will_exist`` (a never-to-exist step would wait forever).
"""

from __future__ import annotations

import json
import os
import subprocess

from sisyphus import Job, Task, tk

from .common import worker_env
from .label_free_hash import hash_bytes, label_free_hash, values_canon  # noqa: F401
from .persona_prompts import split_train_episodes
from .speech_inference import assert_checkpoint_will_exist

#: Weights file of each overlay kind in ``run_dir/checkpoints/checkpoint_<step>/consolidated/``.
CHECKPOINT_WEIGHTS = {"lora": "lora.safetensors", "full": "model.safetensors"}


def _consolidated_complete(path) -> bool:
    """Availability of a checkpoint weights path: the weights AND the ``config.json`` written after them."""
    p = path.get_path()
    return os.path.isfile(p) and os.path.isfile(os.path.join(os.path.dirname(p), "config.json"))


def _file_exists(path) -> bool:
    return os.path.isfile(path.get_path())


def checkpoint_path(run, overlay_kind: str, step: int) -> tk.Path:
    """The weights of ``run`` (a ``SpeechFinetune``) at ``step``, available as soon as that checkpoint is complete --
    also while the run is still training. Same hash as any other path of the run's output at that location."""
    assert overlay_kind in CHECKPOINT_WEIGHTS, overlay_kind
    assert_checkpoint_will_exist(run.out_rundir, overlay_kind, step)
    rel = f"run_dir/checkpoints/checkpoint_{int(step):06d}/consolidated/{CHECKPOINT_WEIGHTS[overlay_kind]}"
    return tk.Path(rel, creator=run, available=_consolidated_complete)


def run_config_path(run) -> tk.Path:
    """The run's rendered ``config.yaml`` (written by its ``write_config`` task), available once it exists."""
    return tk.Path("config.yaml", creator=run, available=_file_exists)


def _progress(job) -> dict | None:
    from sisyphus import global_settings as gs

    try:
        return json.load(open(os.path.join(job._sis_path(gs.JOB_WORK_DIR), "progress.json")))
    except (OSError, ValueError):
        return None


class PersonaHeldoutSet(Job):
    """A fixed PersonaPlex evaluation window set (``heldout_set``'s docstring). ``mode="heldout"``: ``n`` seeded
    windows of the pool (``windows`` table rows whose id is in no ``exclude_ids`` table and whose episode is in no
    ``exclude_contexts`` table, with a voice clip in ``voice``, and -- with ``exclude_split`` -- not a train episode of
    that episode split). ``mode="train_sample"``: ``n`` seeded rows of
    ``train`` (an ``AttachPersonaPrompts`` output). Output ``dataset`` (+ ``summary.json``: pool counts, ids)."""

    def __init__(
        self,
        *,
        mode: str,
        venv_python_path: tk.Path,
        n: int = 400,
        seed: int = 0,
        windows: tk.Path | None = None,
        context: tk.Path | None = None,
        voice: tk.Path | None = None,
        exclude_contexts: list | None = None,
        exclude_ids: list | None = None,
        train: tk.Path | None = None,
        exclude_split: dict | None = None,
    ):
        assert mode in ("heldout", "train_sample"), mode
        if mode == "heldout":
            assert windows is not None and context is not None and voice is not None and exclude_ids
            assert train is None
        else:
            assert train is not None and windows is None and context is None and voice is None
            assert exclude_split is None
        if exclude_split is not None:
            assert set(exclude_split) == {"fraction", "seed", "heldout"}, exclude_split
            exclude_split = {
                "fraction": float(exclude_split["fraction"]),
                "seed": int(exclude_split["seed"]),
                "heldout": sorted(set(exclude_split["heldout"])),
            }
        #: A later run's EPISODE split (``persona_prompts.split_train_episodes``: fraction, seed, forced held-out
        #: episodes): its train episodes are excluded too, so the set is held out for that run as well (p6).
        self.exclude_split = exclude_split
        self.mode = mode
        self.venv_python_path = venv_python_path
        self.n = int(n)
        self.seed = int(seed)
        self.windows, self.context, self.voice = windows, context, voice
        self.exclude_contexts = list(exclude_contexts or [])
        self.exclude_ids = list(exclude_ids or [])
        self.train = train
        self.out_dataset = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 24, "time": 1})

    def run(self):
        out = os.path.dirname(self.out_dataset.get_path())
        cmd = [self.venv_python_path.get(), "-m", "moshi_family.personaplex.heldout_set", self.mode, "--out", out]
        cmd += ["--n", str(self.n), "--seed", str(self.seed)]
        if self.mode == "heldout":
            cmd += ["--windows", self.windows.get_path(), "--context", self.context.get_path()]
            cmd += ["--voice", self.voice.get_path()]
            cmd += ["--exclude_contexts", *[p.get_path() for p in self.exclude_contexts]]
            cmd += ["--exclude_ids", *[p.get_path() for p in self.exclude_ids]]
            if self.exclude_split is not None:
                from datasets import load_from_disk

                eps = sorted(set(load_from_disk(self.context.get_path()).select_columns(["episode_id"])["episode_id"]))
                s = self.exclude_split
                train_eps = split_train_episodes(eps, fraction=s["fraction"], seed=s["seed"], heldout=set(s["heldout"]))
                path = os.path.abspath("exclude_split_episodes.json")
                with open(path, "w") as f:
                    json.dump(sorted(train_eps), f)
                print(
                    f"[heldout-set] excluding {len(train_eps)} of {len(eps)} episodes (split {s['fraction']}, seed {s['seed']})"
                )
                cmd += ["--exclude_episodes_file", path]
        else:
            cmd += ["--train", self.train.get_path()]
        env = worker_env(offline=False)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)


class PersonaHeldoutLoss(Job):
    """Teacher-forced loss of one PersonaPlex checkpoint on fixed window sets (``heldout_loss``'s docstring).

    ``sets``: ``{name: PersonaHeldoutSet.out_dataset}``. ``train_config``: the config.yaml whose loss / text-stream
    settings apply (the run's own; for base PersonaPlex the config of the family it is compared with).
    ``overlay``: checkpoint weights (``checkpoint_path``) or None = base; ``lora_rank`` None for a full finetune.
    Output dir: ``results.json`` (per set: n, per metric mean + ci95; ``meta``), ``per_window.<set>.jsonl``."""

    def __init__(
        self,
        *,
        sets: dict,
        train_config: tk.Path,
        venv_python_path: tk.Path,
        overlay: tk.Path | None = None,
        lora_rank: int | None = None,
        hf_repo: str = "nvidia/personaplex-7b-v1",
        seed: int = 0,
    ):
        assert sets, "no window set"
        assert not (overlay is None and lora_rank is not None), "a LoRA rank needs its adapter"
        self.sets = dict(sorted(sets.items()))
        self.train_config = train_config
        self.venv_python_path = venv_python_path
        self.overlay = overlay
        self.lora_rank = None if lora_rank is None else int(lora_rank)
        self.hf_repo = hf_repo
        self.seed = int(seed)
        self.out_dir = self.output_path("eval", directory=True)
        # One 7B in bf16 (~16 GB) + a B=1 forward of <= ~900 frames. A full-finetune overlay is read onto the GPU
        # beside it (~15 GB transient). Measured 2026-09-25 (see the backlog): minutes per checkpoint.
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 64, "time": 2, "gpu_mem_gb": 80}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        d = _progress(self)
        try:
            return min(d["done"] / d["total"], 1.0)
        except (TypeError, KeyError, ZeroDivisionError):
            return None

    def info(self):
        d = _progress(self)
        return None if d is None else f"{d.get('done')}/{d.get('total')} windows"

    def run(self):
        env = worker_env(offline=True)
        cmd = [self.venv_python_path.get(), "-u", "-m", "moshi_family.personaplex.heldout_loss"]
        cmd += ["--train_config", self.train_config.get_path(), "--out", self.out_dir.get_path()]
        cmd += ["--seed", str(self.seed)]
        for name, path in self.sets.items():
            cmd += ["--set", f"{name}={path.get_path()}"]
        if self.overlay is not None:
            cmd += ["--overlay", self.overlay.get_path()]
            if self.lora_rank is not None:
                cmd += ["--lora_rank", str(self.lora_rank)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)


class PersonaHeldoutTurnPos(Job):
    """Held-out CE by position within the assistant's current turn, 0-5 / 5-10 / 10-20 / 20+ s and outside a turn
    (``moshi_family.personaplex.heldout_turnpos``; user, 2026-09-25: "Moshi training was unstable for turns longer
    than 10 s"). Text word CE and assistant semantic CE, frame-pooled per bucket, 95% cluster bootstrap over windows.

    ``heldout``: one ``PersonaHeldoutSet.out_dataset``. ``train_config``: the config.yaml whose loss / text-stream
    settings apply (one for every checkpoint of the job). ``evals``: ``{label: (weights, lora_rank)}`` -- weights a
    ``checkpoint_path`` or None = base, lora_rank None for a full finetune. Output dir: ``results.json`` (per label,
    metric, bucket: mean, ci95, n_frames, n_windows) and ``per_window.<label>.jsonl`` (per-bucket sums / counts, for
    paired comparisons)."""

    def __init__(
        self,
        *,
        heldout: tk.Path,
        train_config: tk.Path,
        venv_python_path: tk.Path,
        evals: dict,
        gap_s: float = 1.0,
        seed: int = 0,
    ):
        assert evals, "no checkpoint"
        for label, (weights, rank) in evals.items():
            assert "=" not in label and "@" not in label, label
            assert not (weights is None and rank is not None), f"{label}: a LoRA rank needs its adapter"
        self.heldout = heldout
        self.train_config = train_config
        self.venv_python_path = venv_python_path
        self.evals = {k: (w, None if r is None else int(r)) for k, (w, r) in sorted(evals.items())}
        self.gap_s = float(gap_s)
        self.seed = int(seed)
        self.out_dir = self.output_path("eval", directory=True)
        # PersonaHeldoutLoss's per checkpoint (one 7B in bf16, a full-FT overlay read beside it), 1 h per 2 checkpoints
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 64, "time": 1 + len(self.evals) // 2, "gpu_mem_gb": 80}

    @classmethod
    def hash(cls, parsed_args):
        """Run labels are not hashed for a new job (``label_free_hash``)."""
        return label_free_hash(
            cls,
            parsed_args,
            canon=lambda a: {**a, "evals": values_canon(a["evals"])},
            base_hash=lambda a: super(PersonaHeldoutTurnPos, cls).hash(a),
            labels=lambda a: sorted(a["evals"]),
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        d = _progress(self)
        try:
            return min(d["done"] / d["total"], 1.0)
        except (TypeError, KeyError, ZeroDivisionError):
            return None

    def info(self):
        d = _progress(self)
        return None if d is None else f"{d.get('done')}/{d.get('total')} window evals"

    def run(self):
        env = worker_env(offline=True)
        cmd = [self.venv_python_path.get(), "-u", "-m", "moshi_family.personaplex.heldout_turnpos"]
        cmd += ["--train_config", self.train_config.get_path(), "--set", self.heldout.get_path()]
        cmd += ["--out", self.out_dir.get_path(), "--gap", str(self.gap_s), "--seed", str(self.seed)]
        for label, (weights, rank) in self.evals.items():
            spec = "base" if weights is None else weights.get_path() + ("" if rank is None else f"@{rank}")
            cmd += ["--eval", f"{label}={spec}"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)


class PersonaHeldoutSummary(Job):
    """One JSON over many ``PersonaHeldoutLoss`` outputs. ``entries``: list of ``{"run", "step", "windows_seen",
    "eval": PersonaHeldoutLoss.out_dir}``. Output ``summary.json``: ``rows`` (run, step, windows_seen, set, n,
    metrics {mean, ci95}) and ``gaps`` (per checkpoint, held-out minus train-sample mean of every metric, 95%
    bootstrap CI resampling each set's windows independently), sorted by run then step."""

    def __init__(self, *, entries: list, gap_sets: tuple = ("heldout", "train_sample"), seed: int = 0):
        self.entries = [dict(e) for e in entries]
        self.gap_sets = tuple(gap_sets)
        self.seed = int(seed)
        self.out_summary = self.output_path("summary.json")

    @classmethod
    def hash(cls, parsed_args):
        """The entries' ``run`` names are not hashed for a new job (``label_free_hash``)."""
        return label_free_hash(
            cls,
            parsed_args,
            canon=lambda a: {
                **a,
                "entries": sorted(hash_bytes({k: v for k, v in dict(e).items() if k != "run"}) for e in a["entries"]),
            },
            base_hash=lambda a: super(PersonaHeldoutSummary, cls).hash(a),
            labels=lambda a: [dict(e).get("run") for e in a["entries"]],
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        write_summary(self.entries, self.out_summary.get_path(), gap_sets=self.gap_sets, seed=self.seed)


def _per_window(eval_dir: str, set_name: str) -> list[dict]:
    with open(os.path.join(eval_dir, f"per_window.{set_name}.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_summary(entries: list, out_path: str, *, gap_sets=("heldout", "train_sample"), seed: int = 0) -> dict:
    """The summary JSON of ``PersonaHeldoutSummary`` (also usable on partial results: entries whose eval dir has no
    ``results.json`` yet are listed under ``missing``). ``entries[i]["eval"]`` may be a path string or a tk.Path."""
    import numpy as np

    rows, gaps, missing = [], [], []
    for e in sorted(entries, key=lambda e: (e["run"], int(e["step"]))):
        d = e["eval"].get_path() if hasattr(e["eval"], "get_path") else str(e["eval"])
        res_path = os.path.join(d, "results.json")
        if not os.path.isfile(res_path):
            missing.append({"run": e["run"], "step": int(e["step"])})
            continue
        res = json.load(open(res_path))
        for set_name, r in sorted(res["sets"].items()):
            rows.append(
                {
                    "run": e["run"],
                    "step": int(e["step"]),
                    "windows_seen": e.get("windows_seen"),
                    "set": set_name,
                    "n": r["n"],
                    "metrics": {m: {"mean": v["mean"], "ci95": v["ci95"]} for m, v in r["metrics"].items()},
                }
            )
        a, b = gap_sets
        if a in res["sets"] and b in res["sets"]:
            wa, wb = _per_window(d, a), _per_window(d, b)
            g = {}
            for k, m in enumerate(sorted(res["sets"][a]["metrics"])):
                xa = np.asarray([r[m] for r in wa if r.get(m) is not None], dtype=np.float64)
                xb = np.asarray([r[m] for r in wb if r.get(m) is not None], dtype=np.float64)
                if len(xa) < 2 or len(xb) < 2:
                    continue
                rng = np.random.default_rng([seed, k])
                da = xa[rng.integers(0, len(xa), (2000, len(xa)))].mean(1)
                db = xb[rng.integers(0, len(xb), (2000, len(xb)))].mean(1)
                diff = da - db
                g[m] = {
                    "mean": float(xa.mean() - xb.mean()),
                    "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                }
            gaps.append(
                {"run": e["run"], "step": int(e["step"]), "windows_seen": e.get("windows_seen"), f"{a}_minus_{b}": g}
            )
    out = {"rows": rows, "gaps": gaps, "missing": missing}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out
