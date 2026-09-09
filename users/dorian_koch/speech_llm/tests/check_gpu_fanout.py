"""Guard: a job scales to however many GPUs the cluster hands it, without changing what it produces.

Backlog G1 (user directive 2026-09-08/09): the next cluster's smallest GPU node is 4 cards. Nothing in
a recipe says "4"; ``settings.MIN_GPUS_PER_JOB`` raises the request and the job reads what it got
(``common.visible_gpus``) and fans out. Three things have to hold for that to be safe, and each is
asserted here against the REAL helpers and the REAL source:

  * **The fan-out itself** -- ``run_worker_script_per_gpu`` runs one pinned worker per visible GPU,
    in its own ``gpu<k>/`` cwd, with the argv ``args_for_shard(k, n)`` names; a failing worker
    fails the job AND stops the others (a partial output must never be merged); one visible GPU is
    byte-for-byte the old single call.
  * **The arithmetic that keeps outputs identical** -- the contiguous cut equals
    ``datasets.Dataset.shard(contiguous=True)`` (global row numbering, ids, seeds unchanged); the
    nested stride partitions a Sisyphus shard exactly; the jsonl merge restores clip order and
    refuses overlaps; the progress sum reads every worker.
  * **The wiring** -- every converted job calls the helper, every converted worker takes the flag,
    vLLM asks for tensor-parallel, settings raises to the minimum, and training caps its ranks at
    the DECLARED count (an inference job given more GPUs is free speed; a training run given more
    would silently quadruple its effective batch).

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_gpu_fanout.py
"""

import os
import re
import sys
import tempfile
import textwrap
import time
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm import common  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.common import (  # noqa: E402
    contiguous_slice,
    job_progress_fraction,
    merge_jsonl_parts,
    nested_shard,
    run_worker_script_per_gpu,
    visible_gpus,
)

K = SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm"

WORKER = textwrap.dedent(
    """
    import json, os, sys, time
    # record what this worker saw, where the job can read it
    with open("seen.json", "w") as f:
        json.dump({"argv": sys.argv[1:], "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "cwd": os.getcwd()}, f)
    if "--fail" in sys.argv:
        sys.exit(3)
    if "--slow" in sys.argv:
        time.sleep(60)
    """
)


def _with_env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def check_visible_gpus():
    s = _with_env(CUDA_VISIBLE_DEVICES="2,5,7")
    try:
        assert visible_gpus() == ["2", "5", "7"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        assert visible_gpus() == [], "an explicitly empty CUDA_VISIBLE_DEVICES means no GPU"
    finally:
        _restore(s)
    print("PASS  visible_gpus reads the cgroup's CUDA_VISIBLE_DEVICES, empty means none")


def check_fan_out_runs_one_pinned_worker_per_gpu(tmp):
    script = Path(tmp, "worker.py")
    script.write_text(WORKER)
    cwd = Path(tmp, "job")
    cwd.mkdir()
    here = os.getcwd()
    s = _with_env(CUDA_VISIBLE_DEVICES="0,1,2,3")
    try:
        os.chdir(cwd)
        n = run_worker_script_per_gpu(
            sys.executable, script, lambda k, n: ["--part", k, "--of", n], log_label="t", with_hf_home=False
        )
        assert n == 4, n
        import json

        for k in range(4):
            seen = json.load(open(cwd / f"gpu{k}" / "seen.json"))
            assert seen["gpu"] == str(k), seen
            assert seen["argv"] == ["--part", str(k), "--of", "4"], seen
            assert Path(seen["cwd"]).resolve() == (cwd / f"gpu{k}").resolve(), seen
            assert (cwd / f"gpu{k}" / "worker.log").is_file()
    finally:
        os.chdir(here)
        _restore(s)
    print("PASS  4 visible GPUs -> 4 workers, each pinned to its card, own cwd, own argv")


def check_single_gpu_path_is_the_old_call(tmp):
    script = Path(tmp, "worker1.py")
    script.write_text(WORKER)
    cwd = Path(tmp, "job1")
    cwd.mkdir()
    here = os.getcwd()
    calls = []
    orig = common.run_worker_script
    common.run_worker_script = lambda *a, **kw: calls.append((a, kw)) or orig(*a, **kw)
    s = _with_env(CUDA_VISIBLE_DEVICES="0")
    try:
        os.chdir(cwd)
        n = run_worker_script_per_gpu(
            sys.executable, script, lambda k, n: ["--k", k], log_label="t", with_hf_home=False
        )
        assert n == 1 and len(calls) == 1, (n, calls)
        assert calls[0][0][2] == ["--k", 0], calls[0]
        assert (cwd / "seen.json").is_file() and not (cwd / "gpu0").exists(), "one GPU must not create gpu0/"
    finally:
        common.run_worker_script = orig
        os.chdir(here)
        _restore(s)
    print("PASS  one visible GPU is exactly the old run_worker_script call, in the job cwd")


def check_failure_stops_the_others(tmp):
    script = Path(tmp, "worker_fail.py")
    script.write_text(WORKER)
    cwd = Path(tmp, "jobf")
    cwd.mkdir()
    here = os.getcwd()
    s = _with_env(CUDA_VISIBLE_DEVICES="0,1,2")
    t0 = time.monotonic()
    try:
        os.chdir(cwd)
        try:
            run_worker_script_per_gpu(
                sys.executable,
                script,
                lambda k, n: ["--fail"] if k == 1 else ["--slow"],
                log_label="t",
                with_hf_home=False,
            )
        except RuntimeError as e:
            assert "worker 1/3" in str(e) and "exited with 3" in str(e), e
        else:
            raise SystemExit("FAIL: a failing worker did not fail the job")
    finally:
        os.chdir(here)
        _restore(s)
    elapsed = time.monotonic() - t0
    assert elapsed < 40, f"the slow workers were not terminated after the failure ({elapsed:.0f}s)"
    print("PASS  a failing worker fails the job and the other workers are stopped, not awaited")


def check_contiguous_slice_matches_datasets():
    from datasets import Dataset

    for total in (1, 4, 7, 100, 1001):
        ds = Dataset.from_dict({"i": list(range(total))})
        for n in (1, 2, 3, 4):
            if n > total:
                continue
            covered = []
            for k in range(n):
                a, b = contiguous_slice(total, k, n)
                ref = ds.shard(num_shards=n, index=k, contiguous=True)["i"]
                assert list(range(a, b)) == ref, (total, n, k, (a, b), ref[:3])
                covered += list(range(a, b))
            assert covered == list(range(total)), (total, n)
    print("PASS  contiguous_slice == datasets.shard(contiguous=True): row numbering survives the split")


def check_nested_shard_partitions_the_outer_shard():
    total = 97
    for S in (1, 3, 5):
        for s in range(S):
            outer = list(range(total))[s::S]
            for n in (1, 2, 4):
                parts = []
                for k in range(n):
                    sh, num = nested_shard(s if S > 1 else None, S if S > 1 else None, k, n)
                    parts.append(list(range(total))[sh::num])
                flat = sorted(x for p in parts for x in p)
                assert flat == sorted(outer), (S, s, n, flat[:5], outer[:5])
                assert sum(len(p) for p in parts) == len(outer)
    print("PASS  nested_shard(k, n) partitions exactly the job's own Sisyphus shard, driver unchanged")


def check_jsonl_merge_restores_clip_order(tmp):
    import json

    parts = []
    for k in range(3):
        p = Path(tmp, f"out.jsonl.part{k}")
        idx = [i for i in range(20) if i % 3 == k and i != 7]  # clip 7 missing: a silent reply dropped
        p.write_text("".join(json.dumps({"index": i, "v": i * i}) + "\n" for i in idx))
        Path(str(p) + ".idx").write_text("".join(f"{i}\n" for i in idx))
        parts.append(p)
    out = Path(tmp, "out.jsonl")
    n = merge_jsonl_parts(out, parts)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 19 and [r["index"] for r in rows] == [i for i in range(20) if i != 7], rows
    Path(tmp, "dup.part0").write_text(json.dumps({"index": 1}) + "\n")
    Path(tmp, "dup.part0.idx").write_text("1\n")
    try:
        merge_jsonl_parts(Path(tmp, "dup.jsonl"), [parts[1], Path(tmp, "dup.part0")])
    except AssertionError as e:
        assert "overlap" in str(e), e
    else:
        raise SystemExit("FAIL: overlapping parts were merged")
    print("PASS  merge_jsonl_parts restores clip order across parts and refuses an overlap")


def check_progress_sums_every_worker(tmp):
    import json

    work = Path(tmp, "work")
    for k, (d, t) in enumerate(((10, 100), (30, 100), (0, 100))):
        (work / f"gpu{k}").mkdir(parents=True)
        (work / f"gpu{k}" / "progress.json").write_text(json.dumps({"done": d, "total": t}))

    class _J:
        def _sis_path(self, *_a, **_k):
            return str(work)

    assert abs(job_progress_fraction(_J()) - 40 / 300) < 1e-9
    (work / "progress.json").write_text(json.dumps({"done": 5, "total": 10}))
    assert abs(job_progress_fraction(_J()) - 45 / 310) < 1e-9
    print("PASS  completed_fraction sums the per-GPU progress files (and a lone one still works)")


def check_wiring():
    src = {
        p: (K / p).read_text()
        for p in (
            "tts.py",
            "knowledge_benchmark.py",
            "moshi.py",
            "voicebench.py",
            "speech_inference.py",
            "common.py",
            "finetune.py",
            "inference_harness.py",
        )
    }
    # converted jobs
    assert src["tts.py"].count("run_worker_script_per_gpu(") == 1, "ChatterboxInference"
    assert src["knowledge_benchmark.py"].count("run_worker_script_per_gpu(") == 2, (
        "ChatterboxSingleSpeakerInference + WhisperTranscription"
    )
    assert src["moshi.py"].count("run_worker_script_per_gpu(") == 1, "MoshiAnnotate"
    assert src["voicebench.py"].count("run_worker_script_per_gpu(") == 1, "VoiceBenchResponses"
    assert "def _driver_shards" in src["speech_inference.py"] and "nested_shard(" in src["speech_inference.py"]
    assert 'name=f"offline_manifest.gpu{k}.json"' in src["speech_inference.py"], "FDB manifests per GPU"
    # every merge happens in index/row order, never by finish order
    assert "merge_hf_parts(out_hf" in src["tts.py"] and "merge_hf_parts(out_hf" in src["moshi.py"]
    assert "merge_clip_datasets(out_dir, parts)" in src["knowledge_benchmark.py"]
    assert (
        src["knowledge_benchmark.py"].count("merge_jsonl_parts(") == 1
        and src["voicebench.py"].count("merge_jsonl_parts(") == 1
    )
    # workers take the flags the jobs send
    for w, flag in (
        ("chatterbox_inference.py", "--sub_shard"),
        ("moshi_annotate_inference.py", "--sub_shard"),
        ("chatterbox_benchmark_inference.py", "--shard"),
        ("whisper_benchmark_inference.py", "--shard"),
        ("voicebench_responses.py", "--shard"),
    ):
        assert f'"{flag}"' in (K / w).read_text(), (w, flag)
    # the index sidecar both writers emit is what the merge reads
    for w in ("whisper_benchmark_inference.py", "voicebench_responses.py"):
        assert '+ ".idx", "w")' in (K / w).read_text(), w
    # vLLM: tensor-parallel over the visible GPUs
    assert '"--tensor-parallel-size", str(n_gpus)' in src["common.py"]
    # the retrieval seam's pin survives the fan-out's pin
    assert "{**pinned_env, **(extra_env or {})}" in src["inference_harness.py"]
    # settings: the cluster minimum, applied to GPU jobs only
    st = (SETUP / "settings.py").read_text()
    assert re.search(r"^\s*MIN_GPUS_PER_JOB = \d+", st, re.M), "settings.MIN_GPUS_PER_JOB"
    assert 'current_rqmt["gpu"] < MIN_GPUS_PER_JOB' in st and 'current_rqmt["gpu"] = MIN_GPUS_PER_JOB' in st
    # training never takes more ranks than it declared
    assert (
        'declared = int(job.hparams.get("gpu", 1))' in src["finetune.py"]
        and "n_train_gpus = declared" in src["finetune.py"]
    )
    print("PASS  every converted job/worker is wired, vLLM is TP, settings raises, training caps at declared")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        check_visible_gpus()
        check_fan_out_runs_one_pinned_worker_per_gpu(tmp)
        check_single_gpu_path_is_the_old_call(tmp)
        check_failure_stops_the_others(tmp)
        check_contiguous_slice_matches_datasets()
        check_nested_shard_partitions_the_outer_shard()
        check_jsonl_merge_restores_clip_order(tmp)
        check_progress_sums_every_worker(tmp)
    check_wiring()
    print("ALL PASS")
