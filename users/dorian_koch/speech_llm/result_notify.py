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
                elif os.path.isfile(path):
                    entry["preview"] = open(path, encoding="utf-8", errors="replace").read()[:2000]
                else:
                    entry["is_dir"] = os.path.isdir(path)
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
                            "summary": {
                                k: v.get("content", v.get("preview", v.get("path")))
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


def notify_result(tag: str, results: dict, note: str = "") -> ResultNotify:
    """Attach a 'never miss it' sink to an experiment. `results` = {label: tk.Path} to wait on + digest.
    Registers the digest under ``RESULTS/<tag>`` and returns the job."""
    job = ResultNotify(tag=tag, results=results, note=note)
    tk.register_output(f"RESULTS/{tag}", job.out_digest)
    return job
