"""Guard: no job in the graph reaches into another job's work dir by a raw path.

A job's file must reach its reader as that job's OUTPUT (``job.out_x``, or ``tk.Path(rel, creator=job, ...)``), never
as ``tk.Path("/…/work/…/<Job>.<hash>/output/…")`` without a creator and never as a string inside an argument tuple.
A raw path records no dependency: Sisyphus cannot order the jobs, nothing tells the reader to wait, and every tool that
asks "who reads this job?" (cleanup, ``graph_waiting.py``) misses the reader. 2026-10-16: ``prune_checkpoints.py``
deleted ``RLFinetune.FESvleKrAwPk``'s checkpoint_1400 because ``ResolveOverlayCheckpoint.0uP0BA4mbf8b`` named the
run dir by raw path (``track_rl.py`` salvage); its 17 finished evals can no longer be re-run. User: "This should never
happen ... make sure this doesnt become a pattern".

Checks every graph job's inputs (creator-less path into a job dir under work/) and attributes (a string naming a job
dir under work/ is allowed ONLY if that job is also a recorded input of the reader, e.g. ``SpeechInference``'s
``offline_extra_args`` string plus the same output in ``extra_inputs``: the hash keeps the string, the dependency is
real). ``KNOWN`` lists deliberate exceptions; anything else fails. ``--strict`` ignores ``KNOWN``.
2026-10-16: the four sites found at creation (TEXT_PROBE_QUESTIONS, VOICE_WAVS_JOB['know'], two --voice_prompt_dir
dirs) were converted the same day, 0 hashes moved.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_no_raw_job_paths.py
"""

import logging
import os
import re
import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from sisyphus import tk  # noqa: E402
from sisyphus.loader import config_manager  # noqa: E402

JOB_DIR = re.compile(r"(?:^|/)work/(?:.+?/)?([A-Za-z0-9_]+\.[A-Za-z0-9_-]{12})(?:/|$)")
#: target job (Class.hash) -> where the raw path is built. Convert these; never add to this list.
KNOWN = {
    "CreateVenv.vllm_venv_v030": "common.py _VLLM_VENV_JOB (deliberate: hash_overwrite-fixed id, documented)",
}


def strings(v, depth=0):
    if depth > 6:
        return
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from strings(k, depth + 1)
            yield from strings(x, depth + 1)
    elif isinstance(v, (list, tuple, set, frozenset)):
        for x in v:
            yield from strings(x, depth + 1)


def main():
    strict = "--strict" in sys.argv
    config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
    found = {}
    for j in tk.sis_graph.jobs():
        reader = f"{type(j).__name__}.{os.path.basename(j._sis_path()).split('.')[-1]}"
        deps = {os.path.basename(p.creator._sis_path()) for p in j._sis_inputs if p.creator is not None}
        for p in j._sis_inputs:
            if p.creator is None:
                m = JOB_DIR.search(str(p))
                if m:
                    found.setdefault(m.group(1), set()).add(f"{reader} input")
        for k, v in vars(j).items():
            if k.startswith("_sis"):
                continue
            for s in strings(v):
                m = JOB_DIR.search(s)
                if m and m.group(1) not in deps:  # named as a string AND not a recorded dependency
                    found.setdefault(m.group(1), set()).add(f"{reader}.{k}")
    bad = {t: r for t, r in found.items() if strict or t not in KNOWN}
    for t, r in sorted(found.items()):
        tag = "FAIL" if t in bad else "known"
        print(f"{tag:5s} {t}  read by {len(r)}: {sorted(r)[:3]}  ({KNOWN.get(t, 'NEW raw path')})")
    if bad:
        raise SystemExit(f"check_no_raw_job_paths: {len(bad)} job dir(s) reached by raw path (pass the job's output)")
    print(f"check_no_raw_job_paths: OK ({len(found)} deliberate exception(s), no raw-path reader)")


if __name__ == "__main__":
    main()
