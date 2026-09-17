"""Every job that imports torchcodec must CARRY its libraries, not hope the node has them.

THE BUGS THIS EXISTS FOR (both 2026-09-17, together they stalled the graph):

1. ``ChatterboxSingleSpeakerInference`` dropped ``requires: ["system_ffmpeg"]`` on 2026-09-16 on the
   stated grounds that "FFmpeg, CUDA NPP and libpython all travel with the job via env_hook, proven
   on c25g". The proof was real and the wiring was not::

       def env_hook(env):
           if self.ffmpeg_path is not None:          # <- never true for benchmarks
               InstallFFmpeg.add_to_env(self.ffmpeg_path, env)

   ``knowledge_benchmark_py`` declares ``ffmpeg_path: tk.Path | None = None`` and not one of the ~40
   ``benchmarks.py`` call sites passed it. So the job routed to c25g carrying nothing and died at
   import on ``libavutil.so.60: cannot open shared object file`` -- our OWN build's libavutil, i.e.
   never on the loader path -- ~4 min in, after the GPU was allocated. The c25g proof exercised the
   CALLEE with the argument supplied; this is the caller/callee trap in CLAUDE.md, third instance.

2. ``MoshiAnnotate`` kept the tag, was routed to c23g AS INTENDED, and died anyway. The alternative
   explanation was checked and refuted -- identical input schema (exactly one ``Audio`` feature),
   same venv, same code path as the five runs that succeeded on 09-12. The difference is the NODE:
   successes on ``n23g*``/``w23g*``, failures on ``r23g0004``. ``sinfo`` reports identical feature
   strings for all three prefixes, so SLURM cannot express it and the capability tag cannot be made
   to respect it. **A capability tag is not a guarantee; carrying the library is.**

Why the checks are on the REAL graph rather than a fixture: in (1) the defective call site was the
DEFAULT, and a fixture would have been written by passing the argument -- i.e. from the working case.

``system_ffmpeg`` is now declared by nothing, and the last assertion keeps it that way: re-adding it
would be a silent return to depending on node luck.
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

# Jobs that import torchcodec, mapped to the attributes that must actually carry the libraries.
# torchcodec needs THREE things and each partition supplied a different subset, which is why
# "it works on c23g" was never evidence a job was self-contained:
#   FFmpeg (ours)            -- libavutil/libavcodec
#   CUDA NPP                 -- libnppicc.so.12; torch bundles cublas/cudnn/cufft and NOT npp
#   base interpreter lib/    -- libpython3.12.so.1.0, resolved transitively
# libpython is added unconditionally in every env_hook, so only the first two are per-job data.
# Chatterbox gets NPP from its own venv (chatterbox_tts_venv_v5 ships nvidia-npp-cu12); annotate
# gets it from the dedicated npp_venv, because rebuilding moshi_venv to add one library would
# re-resolve `moshi` (a bare git+ requirement) and `whisper_timestamped` (which produces the word
# alignments arms train on) under hashes that do not cover venv contents.
NEEDS = {
    "ChatterboxInference": ("ffmpeg_path", "env_ffmpeg_path"),
    "ChatterboxSingleSpeakerInference": ("ffmpeg_path", "env_ffmpeg_path"),
    "MoshiAnnotate": ("env_ffmpeg_path",),
}
# Jobs that must additionally be handed NPP explicitly, because their own venv has none.
NEEDS_NPP = {"MoshiAnnotate": ("env_npp_venv_python",)}


def _finished(job) -> bool:
    p = job._sis_path()
    return os.path.exists(p + "/finished") or os.path.exists(p + "/finished.tar.gz")


def _has_any(job, attrs) -> bool:
    return any(getattr(job, a, None) is not None for a in attrs)


def main() -> int:
    config_manager.load_configs([CONFIG])

    checked, broken, tagged = 0, [], []
    for job in tk.sis_graph.jobs():
        name = type(job).__name__
        if name not in NEEDS:
            # Independently of the torchcodec jobs: nothing anywhere may go back to asking a
            # partition for these libraries.
            if "system_ffmpeg" in (getattr(job, "rqmt", None) or {}).get("requires", []):
                tagged.append(f"{name}.{job._sis_id().split('.')[-1]}")
            continue
        # A finished job already proved itself on whatever node it got; re-flagging it would make
        # the guard un-greenable for history we cannot change.
        if _finished(job):
            continue
        checked += 1
        missing = []
        if not _has_any(job, NEEDS[name]):
            missing.append("FFmpeg")
        if name in NEEDS_NPP and not _has_any(job, NEEDS_NPP[name]):
            missing.append("NPP")
        if "system_ffmpeg" in (job.rqmt or {}).get("requires", []):
            missing.append("still declares requires:[system_ffmpeg]")
        if missing:
            broken.append((name, job._sis_id().split(".")[-1], ", ".join(missing)))

    if broken:
        print(f"FAIL  {len(broken)} of {checked} pending torchcodec jobs are not self-contained:")
        for n, h, why in broken[:10]:
            print(f"        {n}.{h}  -- missing {why}")
        print()
        print("      torchcodec dlopens libavutil AND libnppicc.so.12 at IMPORT, and no partition")
        print("      reliably provides either, so these die after the GPU is allocated. Supply the")
        print("      libraries in the builder rather than per call site -- InstallFFmpeg() is")
        print("      argument-free, so JobSingleton hands back the job the corpus pipeline already")
        print("      builds, and npp_venv() is a single pinned package:")
        print("          env_ffmpeg_path=InstallFFmpeg().out_path   # both dropped in hash()")
        print("          env_npp_venv_python=npp_venv()")
        return 1

    print(f"PASS  all {checked} pending torchcodec jobs carry their own FFmpeg (+ NPP where needed)")

    if tagged:
        print(f"FAIL  {len(tagged)} job(s) still ask a PARTITION for the libraries:")
        for t in tagged[:10]:
            print(f"        {t}")
        print()
        print("      `requires: [system_ffmpeg]` routes to c23g, and c23g is heterogeneous: five")
        print("      MoshiAnnotate jobs succeeded on n23g*/w23g* and two died on r23g0004 with the")
        print("      same input, venv and code. sinfo gives all three prefixes identical features,")
        print("      so the tag cannot be made to mean what it says. Carry the libraries instead.")
        return 1

    print("PASS  no job anywhere declares requires:[system_ffmpeg] -- the capability is retired")

    # --- Non-vacuity. Each assertion below must be able to FAIL, or the loops above prove nothing.
    from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import (
        ChatterboxSingleSpeakerInference,
    )
    from i6_experiments.users.dorian_koch.speech_llm.tts import InstallFFmpeg

    probe = next(
        (j for j in tk.sis_graph.jobs() if type(j).__name__ == "ChatterboxSingleSpeakerInference"),
        None,
    )
    assert probe is not None, "no ChatterboxSingleSpeakerInference in the graph to model the probe on"

    # ⚠ Each construction happens in a CLEARED `created_jobs`. JobSingleton returns the same OBJECT
    # for two constructions that hash alike, so without this the id-equality assertion would pass
    # because `a is b` -- with the second construction's env_ffmpeg_path silently discarded. That is
    # a vacuous pass in the exact place this guard watches, and is how the first version of
    # check_rqmt_not_hashed went green while testing nothing.
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
    assert not _has_any(naked, NEEDS["ChatterboxSingleSpeakerInference"]), (
        "the guard is vacuous: a job built with neither channel still reports as carrying FFmpeg, "
        "so the real-graph loop above can never fail"
    )
    print("PASS  the guard can still fail (a job built with neither channel carries no FFmpeg)")

    with_env = _fresh(env_ffmpeg_path=InstallFFmpeg().out_path)
    assert with_env is not naked, "cleared created_jobs did not take effect; the check below is vacuous"
    assert _has_any(with_env, ("env_ffmpeg_path",)), "the populated construction lost its env_ffmpeg_path"
    assert naked._sis_id() == with_env._sis_id(), (
        f"env_ffmpeg_path reached the hash ({naked._sis_id()} != {with_env._sis_id()}); supplying "
        "it would re-run every settled benchmark TTS and its whole downstream chain"
    )
    print("PASS  env_ffmpeg_path does not reach the hash (settled benchmark TTS jobs stay put)")

    # Same for annotate's two channels -- these gate ~400 GB of corpora, so a hash move here is far
    # more expensive than the bug it fixes.
    from i6_experiments.users.dorian_koch.speech_llm.moshi import MoshiAnnotate

    ann = next((j for j in tk.sis_graph.jobs() if type(j).__name__ == "MoshiAnnotate"), None)
    assert ann is not None, "no MoshiAnnotate in the graph"

    def _fresh_ann(**kw):
        _sis_job.created_jobs.clear()
        return MoshiAnnotate(venv_python_path=ann.venv_python_path, in_hf=ann.in_hf, **kw)

    bare = _fresh_ann()
    loaded = _fresh_ann(
        env_ffmpeg_path=InstallFFmpeg().out_path,
        env_npp_venv_python=ann.env_npp_venv_python,
    )
    assert loaded is not bare, "cleared created_jobs did not take effect for MoshiAnnotate"
    assert loaded.env_npp_venv_python is not None, "the populated annotate construction lost its NPP venv"
    assert bare._sis_id() == loaded._sis_id(), (
        f"MoshiAnnotate's env channels reached the hash ({bare._sis_id()} != {loaded._sis_id()}); "
        "supplying them would re-run every annotate job and every corpus built from one"
    )
    print("PASS  MoshiAnnotate's env channels do not reach the hash (~400 GB of corpora stay put)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
