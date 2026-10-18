"""Guard: multi-task GRPO with ONE judge for both rewards (``RLFinetune._run_shared_judge``; 2026-10-18).

User (2026-10-18): "having two cards for graders is a bit stupid, maybe we just consolidate to a single model", and to
move the reward Whisper off the trainer's card ("there really is no reason to sequentially wait here right?"). Through
the REAL ``RLFinetune.run`` with ``vllm_server`` and ``launch_training`` replaced by recorders:

  1. ``multitask["selfplay_judge"] == quality_judge`` on 2 GPUs: ONE server, on the first card, at
     ``SHARED_JUDGE_GPU_MEM``, ``max_model_len`` = the larger of the two judges'; training sees
     ``<trainer>,<judge>`` (its own card first = cuda:0) with ``RL_ASR_DEVICE=cuda:1``, both RL_JUDGE_* and
     RL_SP_JUDGE_* on that server; the launcher's ``quality.set_asr_device_from_env`` reads it as Whisper's own card;
     ``launch_training`` still runs ONE rank (RLFinetune declares 1 training GPU); the env is restored afterwards.
  2. Two different judges on 3 GPUs: the two-server path, unchanged (no RL_ASR_DEVICE: Whisper on the trainer's card).
  3. CAN FAIL: the judge card first in the trainer's CUDA_VISIBLE_DEVICES (training would land on the judge's card),
     the smaller max_model_len, or a leftover RL_ASR_DEVICE all fail [1] (each mutant re-runs [1] on a patched copy).

Setup ``.venv``, CPU, seconds (the full_duplex half: ``moshi_family/rl/tests/test_asr_device.py``):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_rl_shared_judge.py
"""

import contextlib
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import common, rl_finetune  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.finetune import declared_train_gpus  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.rl_finetune import RLFinetune  # noqa: E402

GEMMA, OSS = "google/gemma-4-31B-it", "openai/gpt-oss-120b"
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "RL_ASR_DEVICE", "RL_JUDGE_BASE_URL", "RL_SP_JUDGE_BASE_URL")


def drive(sp_judge: str, visible: str, run=None):
    """``run`` (default RLFinetune.run) on a job with the two judges; -> (servers started, envs seen by training,
    env afterwards)."""
    job = object.__new__(RLFinetune)
    job.quality_judge, job.judge_max_model_len, job.adapter = GEMMA, 8192, object()
    job.multitask = {"selfplay_judge": sp_judge, "selfplay_judge_max_model_len": 16384}
    servers, trainings = [], []

    @contextlib.contextmanager
    def fake_server(model, max_model_len=None, gpu_memory_utilization=0.9, **kw):
        servers.append({"model": model, "len": max_model_len, "mem": gpu_memory_utilization,
                        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})  # fmt: skip
        yield f"http://localhost:{1000 + len(servers)}/v1"

    def fake_launch(j, adapter):
        from moshi_family.rl import quality

        env = {k: os.environ.get(k) for k in ENV_KEYS}
        n = len(env["CUDA_VISIBLE_DEVICES"].split(","))
        with mock.patch("torch.cuda.device_count", return_value=n):
            env["asr_own_gpu"] = quality.set_asr_device_from_env()
        env["asr_device"] = quality.ASR_DEVICE
        quality.ASR_DEVICE = "cuda"
        trainings.append(env)

    saved = {k: os.environ.pop(k, None) for k in ENV_KEYS}
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    try:
        with (
            mock.patch.object(common, "vllm_server", fake_server),
            mock.patch.object(rl_finetune, "launch_training", fake_launch),
        ):
            (run or RLFinetune.run)(job)
        after = {k: os.environ.get(k) for k in ENV_KEYS}
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return servers, trainings, after


def shared_ok(servers, trainings, after) -> dict:
    t = trainings[0] if len(trainings) == 1 else {}
    return {
        "one server": len(servers) == 1,
        "server on the first card": servers[:1] and servers[0]["gpu"] == "0",
        "server memory share": servers[:1] and servers[0]["mem"] == RLFinetune.SHARED_JUDGE_GPU_MEM,
        "max_model_len = the larger": servers[:1] and servers[0]["len"] == 16384,
        "one training launch": len(trainings) == 1,
        "trainer sees <trainer>,<judge>": t.get("CUDA_VISIBLE_DEVICES") == "1,0",
        "RL_ASR_DEVICE = cuda:1 -> Whisper's own card": t.get("RL_ASR_DEVICE") == "cuda:1"
        and t.get("asr_own_gpu") is True,
        "both rewards on the one server": t.get("RL_JUDGE_BASE_URL") is not None
        and t.get("RL_JUDGE_BASE_URL") == t.get("RL_SP_JUDGE_BASE_URL"),
        "env restored": after
        == {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "RL_ASR_DEVICE": None,
            "RL_JUDGE_BASE_URL": t.get("RL_JUDGE_BASE_URL"),
            "RL_SP_JUDGE_BASE_URL": t.get("RL_SP_JUDGE_BASE_URL"),
        },  # fmt: skip
    }


def main():
    fails = []

    def report(prefix, checks, expect=True):
        for k, v in checks.items():
            ok = bool(v) == expect
            print(f"[{'ok' if ok else 'FAIL'}] {prefix}{k}")
            if not ok:
                fails.append(prefix + k)

    report("1 ", {"RLFinetune declares 1 training GPU (one rank)": declared_train_gpus(
        RLFinetune(venv_python_path=tk.Path("/nonexistent/python"))) == 1})  # fmt: skip
    report("1 ", shared_ok(*drive(GEMMA, "0,1")))

    servers, trainings, _ = drive(OSS, "0,1,2")
    t = trainings[0] if trainings else {}
    report("2 ", {
        "two servers on cards 0 and 1": [s["gpu"] for s in servers] == ["0", "1"] and [s["model"] for s in servers] == [GEMMA, OSS],
        "trainer on the last card only": t.get("CUDA_VISIBLE_DEVICES") == "2",
        "no RL_ASR_DEVICE: Whisper on the trainer's card": t.get("RL_ASR_DEVICE") is None and t.get("asr_own_gpu") is False,
        "separate judge urls": t.get("RL_JUDGE_BASE_URL") != t.get("RL_SP_JUDGE_BASE_URL"),
    })  # fmt: skip

    # 3. mutants of _run_shared_judge: each must fail at least one check of [1]
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(RLFinetune._run_shared_judge))
    mutants = {
        "judge card first": ('CUDA_VISIBLE_DEVICES=f"{visible[-1]},{visible[0]}"', 'CUDA_VISIBLE_DEVICES=f"{visible[0]},{visible[-1]}"'),
        "smaller max_model_len": ("max(lens) if lens", "min(lens) if lens"),
        "RL_ASR_DEVICE left set": ('os.environ.pop("RL_ASR_DEVICE", None)', "pass"),
    }  # fmt: skip
    for name, (old, new) in mutants.items():
        assert src.count(old) == 1, (name, old)
        ns = vars(rl_finetune)  # the module's own globals, so the mutant sees the patched launch_training
        exec(compile(src.replace(old, new), "<mutant>", "exec"), ns)
        mutant_cls = type("Mutant", (RLFinetune,), {"_run_shared_judge": ns.pop("_run_shared_judge")})

        def run_mutant(job, cls=mutant_cls):
            job.__class__ = cls
            return RLFinetune.run(job)

        caught = not all(shared_ok(*drive(GEMMA, "0,1", run=run_mutant)).values())
        os.environ.pop("RL_ASR_DEVICE", None)
        report("3 CAN FAIL: ", {name: caught})
    print("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
