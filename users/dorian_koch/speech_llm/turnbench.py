"""TurnBench (backlog D4): an EXTERNAL turn-taking benchmark, run reuse-first.

TurnBench scores *when* a system speaks -- end-of-turn (EOT) and interruption (INT) onsets, by
recall / false-positive rate / median signed latency -- over two-channel human conversations. It is
the first externally comparable axis this project has for turn-taking, and it is exactly what
backlog A12 (self-aligned turn-start training) would move.

**Reuse-first, per the standing preference: we write no scorer and no metric.** The upstream repo
(`SesameAILabs/turnbench`, MIT) is vendored at a pinned commit and its own
``python -m turnbench.score`` is invoked verbatim. Their 16 published baselines ship their
``predictions-dev.json`` in-repo, so our harness can be validated against numbers we did not
produce before it is ever pointed at one of our models.

Two facts that shape the design, both measured rather than assumed (2026-09-15):

* **Scoring does NOT download the audio.** ``resolve_dataset(skip_audio=True)`` reads only the gold
  columns, column-projected over HTTP range requests -- the audio is ~99.96% of each shard. So the
  scoring path costs a few MB, not the 4.2 GB the full dataset implies. Running our *own* model
  through stage 1 is what needs the audio.
* **It therefore needs the internet, so scoring is a login-node ``mini_task``.** Compute nodes here
  are offline. The projected read is cached per (source, revision, columns), so repeat scoring is
  instant.

The dev dataset is gated; access is via the ``HF_TOKEN`` that ``settings.py`` injects into every
job env. Verified reachable 2026-09-15.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from sisyphus import Job, Task, tk

#: Upstream repo + the exact commit we vendor. Pinned because the SCORER is the measurement: an
#: unpinned clone would silently re-define the metric between two of our own runs.
TURNBENCH_REPO = "https://github.com/SesameAILabs/turnbench"
TURNBENCH_COMMIT = "76ccd045f121ccfa921abac2ad3107027e736911"

#: Files whose absence means the clone is not the repo we think it is. Checked after fetching so a
#: layout change upstream fails here, at vendoring time, rather than inside a scoring job.
_REQUIRED = (
    "turnbench/score.py",
    "turnbench/data.py",
    "baselines/moshi_vad/predictions-dev.json",
)

#: Their committed Moshi baseline. The harness-validation target: reproduce this system's published
#: dev numbers from its shipped predictions, using our vendored scorer, before trusting any number
#: we generate ourselves.
MOSHI_VAD_PREDICTIONS = "baselines/moshi_vad/predictions-dev.json"


class VendorTurnBench(Job):
    """Clone the TurnBench repo at a pinned commit into ``output/turnbench``."""

    def __init__(self, *, repo: str = TURNBENCH_REPO, commit: str = TURNBENCH_COMMIT):
        self.repo = repo
        self.commit = commit
        self.out_repo = self.output_path("turnbench", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)  # git needs the login node's internet

    def run(self):
        dest = self.out_repo.get_path()
        os.makedirs(dest, exist_ok=True)

        def git(*args):
            subprocess.run(["git", "-C", dest, *args], check=True)

        # Fetch the single pinned commit rather than cloning a branch: a branch clone is whatever
        # HEAD happens to be today, and the scorer IS the measurement.
        git("init", "-q")
        subprocess.run(
            ["git", "-C", dest, "remote", "add", "origin", self.repo],
            check=False,  # idempotent on a re-run
        )
        git("fetch", "-q", "--depth", "1", "origin", self.commit)
        git("checkout", "-q", "FETCH_HEAD")

        head = subprocess.run(
            ["git", "-C", dest, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        assert head == self.commit, f"vendored {head}, expected {self.commit}"
        for rel in _REQUIRED:
            assert os.path.isfile(os.path.join(dest, rel)), (
                f"{rel} missing from the vendored repo -- upstream layout changed, so the glue in "
                f"this module is describing a repo that no longer exists. Re-read it before pinning "
                f"a newer commit."
            )
        print(f"[turnbench] vendored {self.repo} @ {head}", flush=True)


class TurnBenchScore(Job):
    """Score one predictions file with the vendored ``turnbench.score``, verbatim.

    ``predictions_rel`` names a file inside the vendored repo (their shipped baselines);
    ``predictions_path`` is one we produced. Exactly one of the two.
    """

    def __init__(
        self,
        *,
        repo: tk.Path,
        venv_python: tk.Path,
        tag: str,
        predictions_rel: str | None = None,
        predictions_path: tk.Path | None = None,
        dataset: str | None = None,
    ):
        assert (predictions_rel is None) != (predictions_path is None), (
            "pass exactly one of predictions_rel (a shipped baseline) or predictions_path (ours)"
        )
        self.repo = repo
        self.venv_python = venv_python
        self.tag = tag
        self.predictions_rel = predictions_rel
        self.predictions_path = predictions_path
        self.dataset = dataset
        self.out_json = self.output_path("scores.json")
        self.out_log = self.output_path("score.log")

    def tasks(self):
        # mini_task: the scorer reads the gold columns over HTTP range requests, and compute nodes
        # here have no internet. It never touches audio, so this is light despite being a benchmark.
        yield Task("run", mini_task=True)

    def run(self):
        repo = self.repo.get_path()
        pred = os.path.join(repo, self.predictions_rel) if self.predictions_rel else self.predictions_path.get_path()
        assert os.path.isfile(pred), f"predictions file not found: {pred}"

        cmd = [self.venv_python.get(), "-m", "turnbench.score", pred]
        if self.dataset:
            cmd += ["--dataset", self.dataset]
        env = os.environ.copy()
        env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # rich wraps to the terminal width, and a wrapped table is unparseable. Pin it wide.
        env["COLUMNS"] = "240"
        env["TERM"] = "dumb"
        proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
        log = proc.stdout + "\n" + proc.stderr
        with open(self.out_log.get_path(), "w") as f:
            f.write(log)
        assert proc.returncode == 0, (
            f"turnbench.score exited {proc.returncode}; see {self.out_log.get_path()}\n" + log[-2000:]
        )

        scores = parse_aggregate(log)
        assert scores, (
            "could not parse an aggregate row out of turnbench.score's output. The numbers exist in "
            f"{self.out_log.get_path()} -- the PARSER is what broke, so fix it rather than trusting "
            "a silently empty result."
        )
        scores["tag"] = self.tag
        scores["predictions"] = self.predictions_rel or self.predictions_path.get_path()
        with open(self.out_json.get_path(), "w") as f:
            json.dump(scores, f, indent=2)
        print(f"[turnbench] {self.tag}: {json.dumps(scores)}", flush=True)


#: One aggregate row: task name, recall, fp_rate, the p10/50/90 latency cell, then tp/fn/fp/tn.
_ROW = re.compile(
    r"\b(EOT|INT)\b\D*?"
    r"(\d+\.\d+)\s*[│|]\s*"  # recall
    r"(\d+\.\d+)\s*[│|]\s*"  # fp_rate
    r"([^│|]*?)\s*[│|]\s*"  # latency cell (may be "-" when there are no hits)
    r"(\d+)\s*[│|]\s*(\d+)\s*[│|]\s*(\d+)\s*[│|]\s*(\d+)"
)


def parse_aggregate(text: str) -> dict:
    """Pull the EOT/INT aggregate rows out of the scorer's rich table.

    Tolerant on purpose: their table is a presentation format, not an interface, so this reads the
    numbers positionally and the raw log is always kept beside the parsed JSON. Returns {} when
    nothing matched, which the caller turns into a loud failure rather than an empty result.
    """
    out: dict = {}
    for m in _ROW.finditer(re.sub(r"\x1b\[[0-9;]*m", "", text)):
        task, recall, fp, lat, tp, fn, fp_n, tn = m.groups()
        out[task.lower()] = {
            "recall": float(recall),
            "fp_rate": float(fp),
            "latency_p10_p50_p90": lat.strip(),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp_n),
            "tn": int(tn),
        }
    return out


def turnbench_baselines_py(*, venv_python, baselines: tuple[str, ...] = ("moshi_vad",)):
    """Vendor TurnBench and score the named shipped baselines with our harness.

    This is the CALIBRATION step and it comes first: these are numbers Sesame published, produced by
    systems we did not run, so reproducing them tests our scorer wiring and nothing else. Only once
    that matches is a number we generate ourselves worth reporting.
    """
    vendored = VendorTurnBench()
    tk.register_output("turnbench/repo", vendored.out_repo)
    jobs = {}
    for name in baselines:
        job = TurnBenchScore(
            repo=vendored.out_repo,
            venv_python=venv_python,
            tag=name,
            predictions_rel=f"baselines/{name}/predictions-dev.json",
        )
        tk.register_output(f"turnbench/{name}/scores.json", job.out_json)
        tk.register_output(f"turnbench/{name}/score.log", job.out_log)
        jobs[name] = job
    return vendored, jobs
