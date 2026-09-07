"""'Never miss a result' sink.

The problem: we kick off experiments and forget to check them later. Fix: a downstream Sisyphus job that
DEPENDS on an experiment's output(s). Because Sisyphus only runs a job once its inputs exist, ``ResultNotify``
fires EXACTLY when the experiment finishes -- and when it fires it (a) writes a durable per-experiment digest
``output/RESULTS/<tag>.json`` (the parsed key metrics + source paths + a timestamp), and (b) appends one line
to a single central ``RESULTS.jsonl`` at the setup root. So a completed experiment ALWAYS leaves a fresh,
timestamped, easy-to-scan artifact -- nothing silently completes unseen.

Consumption: ``ls -lt output/RESULTS/`` or ``tail RESULTS.jsonl`` (newest last) at the start of any session.
It's a light ``mini_task`` (runs on the login node in the manager loop -- no GPU, negligible cost), so adding
one per experiment is free.

Usage (recipe):
    from i6_experiments.users.dorian_koch.speech_llm.result_notify import notify_result
    notify_result("audex_stage1_final", {"knowledge": grading.out_summary, "run_dir": ft.out_rundir})
"""

import json
import os
import time

from sisyphus import Job, Task, tk


class ResultNotify(Job):
    """Fires when its `results` inputs are all produced; writes a digest file + appends to RESULTS.jsonl."""

    # Re-fire (new digest) whenever the tag or the tracked result set changes; `note` is cosmetic.
    __sis_hash_exclude__ = {"note": ""}

    def __init__(self, *, tag: str, results: dict, note: str = ""):
        self.tag = tag
        self.results = results  # {label: tk.Path}  (a summary.json, a run_dir, any output to wait on)
        self.note = note
        self.out_digest = self.output_path("result.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        digest = {
            "tag": self.tag,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": self.note,
            "results": {},
        }
        for label, p in self.results.items():
            path = p.get()
            entry = {"path": path}
            try:
                base = os.path.basename(path)
                if os.path.isfile(path) and (path.endswith(".json") or "summary" in base):
                    entry["content"] = json.load(open(path))
                elif os.path.isfile(path) and path.endswith(".jsonl"):
                    entry["metrics"] = summarize_metrics_jsonl(path)
                elif os.path.isfile(path):
                    entry["preview"] = open(path, encoding="utf-8", errors="replace").read()[:2000]
                else:
                    entry["is_dir"] = os.path.isdir(path)
                    if os.path.isdir(path):
                        # A training arm's only declared output is its run_dir, so the numbers that
                        # say how the run WENT live in files inside it. Summarise them here or the
                        # digest records a directory path and nothing else -- which is how a 600-step
                        # arm could finish, fire its notify, and still tell you nothing.
                        found = sorted(f for f in os.listdir(path) if f.startswith("metrics") and f.endswith(".jsonl"))
                        if found:
                            entry["metrics"] = {f: summarize_metrics_jsonl(os.path.join(path, f)) for f in found}
            except Exception as e:  # a digest read must never fail the notify
                entry["read_error"] = str(e)
            digest["results"][label] = entry

        with open(self.out_digest.get(), "w") as f:
            json.dump(digest, f, indent=2, default=str)

        # Central append-only log at the setup root (cwd of the mini_task is the setup dir). One line per
        # firing -> a single time-ordered place to scan. Best-effort; never fail the job on a log hiccup.
        # Robust central-log path: derive the setup root from this job's own output path (mini_task
        # cwd is not guaranteed to be the setup root).
        _root = self.out_digest.get().split("/work/")[0]
        try:
            with open(os.path.join(_root, "RESULTS.jsonl"), "a") as f:
                f.write(
                    json.dumps(
                        {
                            "at": digest["finished_at"],
                            "tag": self.tag,
                            "note": self.note,
                            "digest": self.out_digest.get(),
                            # "metrics" before "preview"/"path": the central log is the thing that
                            # gets SCANNED at session start, so a line saying only where a run_dir
                            # lives reports that an arm finished and not how it went.
                            "summary": {
                                k: v.get(
                                    "content",
                                    v.get("metrics", v.get("preview", v.get("path"))),
                                )
                                for k, v in digest["results"].items()
                            },
                        },
                        default=str,
                    )
                    + "\n"
                )
        except Exception as e:
            print(f"[result-notify] central-log append failed: {e}", flush=True)

        print(f"[result-notify] {self.tag} -> {self.out_digest.get()} (+ RESULTS.jsonl)", flush=True)


#: Scalars worth carrying into a digest, in the order a reader wants them. Anything else in the file
#: is ignored rather than dumped -- the point of a digest is that it fits in a notification.
DIGEST_KEYS = (
    "loss",
    "text_loss",
    "audio_loss",
    "eval_loss",
    "knowledge_accuracy",
    "knowledge_avg_quality",
    "grad_norm",
    "lr",
)


def summarize_metrics_jsonl(path: str, keys=DIGEST_KEYS) -> dict:
    """first / last / min / max per tracked scalar in a metrics ``.jsonl``.

    A metrics file must NOT be digested with the generic head-preview branch: it is append-ordered,
    so the first 2000 characters are step 1 -- the state of the run before it trained, the least
    informative thing in the file. And it is append- not step-ordered (a preempted run replays steps
    on resume, see check_train_metrics_plot), so "last row" is not "highest step": rows are sorted by
    step here before first/last are taken.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written final line while the writer is still live
    if not rows:
        return {"rows": 0}
    out = {"rows": len(rows)}
    steps = [r["step"] for r in rows if isinstance(r.get("step"), (int, float))]
    if steps:
        out["last_step"] = max(steps)
    for key in keys:
        pairs = [
            (r.get("step", i), r[key])
            for i, r in enumerate(rows)
            if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
        ]
        if not pairs:
            continue
        pairs.sort(key=lambda sv: sv[0])
        vals = [v for _, v in pairs]
        out[key] = {"first": vals[0], "last": vals[-1], "min": min(vals), "max": max(vals), "n": len(vals)}
    return out


def notify_result(tag: str, results: dict, note: str = "") -> ResultNotify:
    """Attach a 'never miss it' sink to an experiment. `results` = {label: tk.Path} to wait on + digest.
    Registers the digest under ``RESULTS/<tag>`` and returns the job."""
    job = ResultNotify(tag=tag, results=results, note=note)
    tk.register_output(f"RESULTS/{tag}", job.out_digest)
    return job
