"""Every job that imports torchcodec must CARRY its FFmpeg, not hope the node has one.

THE BUG THIS EXISTS FOR (2026-09-17, cost the n=1000 TTS job + a stalled graph):
``ChatterboxSingleSpeakerInference`` dropped ``requires: ["system_ffmpeg"]`` on 2026-09-16 on the
stated grounds that "FFmpeg, CUDA NPP and libpython all travel with the job via env_hook, proven on
c25g". The proof was real and the wiring was not::

    def env_hook(env):
        if self.ffmpeg_path is not None:          # <- never true for benchmarks
            InstallFFmpeg.add_to_env(self.ffmpeg_path, env)

``knowledge_benchmark_py`` declares ``ffmpeg_path: tk.Path | None = None`` and not one of the ~40
``benchmarks.py`` call sites passed it. So the job routed to c25g carrying nothing and died at
import on ``libavutil.so.60: cannot open shared object file`` -- our OWN build's libavutil, i.e. it
was never on the loader path at all -- ~4 min in, after the GPU was allocated.

Why no existing guard caught it: the c25g proof exercised the CALLEE with the argument supplied.
This is the caller/callee trap in CLAUDE.md, and it is the third time it has cost us a run, so the
check has to be on the real graph: the defective call site was the DEFAULT, and a fixture would
have been written by passing the argument -- i.e. written from the working case.

Deliberately NOT covered, and this is the honest gap rather than an oversight: ``MoshiAnnotate``
still declares ``requires: ["system_ffmpeg"]`` and relies on the node providing the libraries. It
cannot yet be converted, because its venv (``moshi_venv_v2``) ships no ``nvidia-npp-cu12`` and
``ldd libtorchcodec_core8.so`` there reports ``libnppicc.so.12 => not found`` even with our FFmpeg
on the path -- so making it self-contained needs a venv rebuild, not a wiring change. It is listed
below so the gap is recorded and greppable instead of silently absent.
"""

import os
import sys
import logging

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from sisyphus import tk
from sisyphus.loader import config_manager

CONFIG = "recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"

# Jobs that import torchcodec AND no longer declare `requires: ["system_ffmpeg"]`, so nothing but
# their own env_hook can supply the libraries. Each must have an FFmpeg build on one of its two
# channels: `ffmpeg_path` (hashed, used by the corpus pipeline) or `env_ffmpeg_path` (dropped in
# hash(), used by the benchmarks -- see ChatterboxSingleSpeakerInference.hash).
MUST_CARRY_FFMPEG = ("ChatterboxInference", "ChatterboxSingleSpeakerInference")

# Jobs that import torchcodec but still ride on the capability tag. Not a pass -- a recorded gap.
RELIES_ON_NODE = {"MoshiAnnotate": "moshi_venv_v2 ships no nvidia-npp-cu12; needs a venv rebuild"}


def _finished(job) -> bool:
    p = job._sis_path()
    return os.path.exists(p + "/finished") or os.path.exists(p + "/finished.tar.gz")


def _carries_ffmpeg(job) -> bool:
    return (getattr(job, "ffmpeg_path", None) or getattr(job, "env_ffmpeg_path", None)) is not None


def main() -> int:
    config_manager.load_configs([CONFIG])

    checked, broken, tagged = 0, [], []
    for job in tk.sis_graph.jobs():
        name = type(job).__name__
        if name in RELIES_ON_NODE:
            if not _finished(job):
                tagged.append(job._sis_id().split(".")[-1])
            continue
        if name not in MUST_CARRY_FFMPEG:
            continue
        # A finished job already proved itself on whatever node it got; re-flagging it would make
        # the guard un-greenable for history we cannot change.
        if _finished(job):
            continue
        checked += 1
        if not _carries_ffmpeg(job):
            broken.append((name, job._sis_id().split(".")[-1]))

    if broken:
        print(f"FAIL  {len(broken)} of {checked} pending torchcodec jobs carry no FFmpeg build:")
        for n, h in broken[:10]:
            print(f"        {n}.{h}")
        print()
        print("      torchcodec dlopens libavutil at IMPORT, and neither partition is guaranteed to")
        print("      provide one, so these die after the GPU is allocated. Supply the build in the")
        print("      builder rather than per call site -- InstallFFmpeg() is argument-free, so")
        print("      JobSingleton hands back the same job the corpus pipeline already builds:")
        print("          env_ffmpeg_path=InstallFFmpeg().out_path   # dropped in hash()")
        return 1

    print(f"PASS  all {checked} pending {'/'.join(MUST_CARRY_FFMPEG)} jobs carry an FFmpeg build")
    for n, why in RELIES_ON_NODE.items():
        print(f"NOTE  {n} still relies on `requires: [system_ffmpeg]` -- {why}")
    if tagged:
        print(f"      ({len(tagged)} pending, e.g. {', '.join(tagged[:3])})")

    # Non-vacuous: the check must be able to FAIL. Build the job the way the broken call site did
    # -- every argument except the FFmpeg one -- and assert it really carries nothing. Without this,
    # a refactor that gave every job a default path would leave the guard green for the wrong
    # reason, which is precisely how the original wiring looked correct.
    from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import (
        ChatterboxSingleSpeakerInference,
    )

    probe = None
    for job in tk.sis_graph.jobs():
        if type(job).__name__ == "ChatterboxSingleSpeakerInference":
            probe = job
            break
    assert probe is not None, "no ChatterboxSingleSpeakerInference in the graph to model the probe on"

    # ⚠ Each construction below happens in a CLEARED `created_jobs`. JobSingleton returns the same
    # OBJECT for two constructions that hash alike, so without this the id-equality assertion at the
    # end would pass because `a is b` -- with the second construction's env_ffmpeg_path silently
    # discarded. That is a vacuous pass in the exact place this guard is watching, and it is how the
    # first version of check_rqmt_not_hashed went green while testing nothing.
    import sisyphus.job as _sis_job

    def _fresh(**kw):
        _sis_job.created_jobs.clear()
        return ChatterboxSingleSpeakerInference(
            venv_python_path=probe.venv_python_path,
            in_hf=probe.in_hf,
            speaker_dir=probe.speaker_dir,
            speaker_name=probe.speaker_name,
            storage=probe.storage,
            **kw,
        )

    naked = _fresh()
    assert not _carries_ffmpeg(naked), (
        "the guard is vacuous: a job built with neither ffmpeg_path nor env_ffmpeg_path still "
        "reports as carrying one, so the real-graph loop above can never fail"
    )
    print("PASS  the guard can still fail (a job built with neither channel carries no FFmpeg)")

    # And the hash-excluded channel must genuinely not reach the hash -- if it ever did, supplying
    # it to the ~40 settled benchmark TTS jobs would re-run every one of them and cascade through
    # transcription -> grading -> the whole judged ledger.
    from i6_experiments.users.dorian_koch.speech_llm.tts import InstallFFmpeg

    with_env = _fresh(env_ffmpeg_path=InstallFFmpeg().out_path)
    assert with_env is not naked, "cleared created_jobs did not take effect; the check below is vacuous"
    assert _carries_ffmpeg(with_env), "the populated construction lost its env_ffmpeg_path"
    assert naked._sis_id() == with_env._sis_id(), (
        f"env_ffmpeg_path reached the hash ({naked._sis_id()} != {with_env._sis_id()}); supplying "
        "it would re-run every settled benchmark TTS and its whole downstream chain"
    )
    print("PASS  env_ffmpeg_path does not reach the hash (settled benchmark TTS jobs stay put)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
