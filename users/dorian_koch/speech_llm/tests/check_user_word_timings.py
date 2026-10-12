"""Guard: every NEW synthetic / knowledge / Fisher corpus carries the USER side's word timings (2026-10-12).

User (2026-10-12): "make sure that for future synthetic and knowledge generations we always also generate the word
timings" -- the user-text head (``user_text_loss_weight``) reads its targets from ``alignments_other`` + ``other_ok``
(the JRE / Open Yap layout); the knowledge v6/v7 rows and the MoshiAnnotate corpora carried the assistant's words only.
The builders now write them by default (``user_words=True``; ``False`` = every corpus declared before, hash-excluded).
Real code throughout: the real graph, the real annotate worker (``annotate_hf_to_hf``, Whisper replaced by a fake that
labels a channel's words by its signal), the real ``knowledge_codes`` worker (CPU Mimi) and the real loader item builder
(``persona_item_builder(user_text_delay=)``).
  g  GRAPH (setup .venv): every MoshiAnnotate / KnowledgeDialogueCodes / FisherToMoshiTrainData in the graph has
     ``user_words`` on unless it is one of the FROZEN pre-2026-10-12 jobs (all of which are still in the graph with
     their old ids, i.e. ``False`` keeps the old hash); a fresh declaration defaults to on and hashes differently from
     ``False``; ``run()`` passes the flag to the worker; the Fisher converter's windows / ``alignments`` are unchanged
     and its user words land in ``alignments_other`` only.
  a  ANNOTATE (moshi_venv): ``alignments`` = the assistant channel's words only, ``alignments_other`` = the user
     channel's (speaker ``user``), ``other_ok`` false for a silent user channel (row kept); without the flag the
     schema and ``alignments`` are the pre-2026-10-12 ones.
  c  CODES + LOADER (moshi_family_venv; reads stage a's output): ``--user_words`` keeps ``alignments`` and the codes
     bit-identical to a build without it, carries the annotated user words (clamped to the user's speech onset) as
     ``alignments_other``; the loader's user-text row = those words delayed 6 frames (IGNORE when ``other_ok`` false),
     the assistant text row unchanged; a corpus WITHOUT user timings is refused by ``--user_words`` and by
     ``persona_item_builder(user_text_delay=)``.
``--selftest``: each stage's mutants (a new declaration pinned off, the default flipped off, ``run()`` not passing the
flag, the user pass reading the assistant channel, user words merged into ``alignments``, the codes worker dropping /
mixing up / not clamping them) must each fail.

Setup root (the graph stage runs here; the other two in their venvs, launched by this script; ~4 min):
    CUDA_HOME=/usr HF_HUB_OFFLINE=1 .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_user_word_timings.py [--selftest]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile

SETUP = os.getcwd()
HERE = os.path.abspath(__file__)
VENV = "work/i6_experiments/users/dorian_koch/jobs/venv/CreateVenv.{}/output/venv/bin/python"
STAGE_PY = {
    "graph": ".venv/bin/python",
    "annotate": None,  # the real annotate job's venv, recorded by the graph stage
    "codes": VENV.format("moshi_family_venv_v1"),
}
MUTANTS = {
    "graph": ["new_off", "default_off", "run_no_flag"],
    "annotate": ["user_from_assistant", "merge", "flag_off"],
    "codes": ["drop", "merge", "wrong_side", "no_clamp"],
}
BUILDERS = ("MoshiAnnotate", "KnowledgeDialogueCodes", "FisherToMoshiTrainData")
# Every builder job of the graph on 2026-10-12, before user_words existed (hash_snapshot): they keep user_words=False.
FROZEN = frozenset(
    """
    FisherToMoshiTrainData.98IFVP8879x4 FisherToMoshiTrainData.E6HnLS4wyaba FisherToMoshiTrainData.JJi8ACF3GodT
    KnowledgeDialogueCodes.DMgw9Srb4bmg KnowledgeDialogueCodes.uNaVPOSTXVly MoshiAnnotate.0KnpHqmljOCi
    MoshiAnnotate.0MZk9MmJqzvH MoshiAnnotate.0PPcZnXX1UHj MoshiAnnotate.0SfaiOeQVrFf MoshiAnnotate.0vryqk5M9WEY
    MoshiAnnotate.1DOiRVGjdQBA MoshiAnnotate.1dKGCorzRwUK MoshiAnnotate.2VFyLEfxkTgh MoshiAnnotate.2bgGqNOrtf5x
    MoshiAnnotate.2oZ7kqP65EsG MoshiAnnotate.3OXrmGCVy444 MoshiAnnotate.3gn5cbFnswK5 MoshiAnnotate.3poRtagKh2sh
    MoshiAnnotate.4EQBsbPFYsbb MoshiAnnotate.4mkACUUlRyCB MoshiAnnotate.5BVidV9iX0hY MoshiAnnotate.5Kb7JXxNBdyj
    MoshiAnnotate.5MOLOSQ1d8c7 MoshiAnnotate.5v6WA5peOnLV MoshiAnnotate.61s5Sfyj7OQ7 MoshiAnnotate.6GmyJYSRaldb
    MoshiAnnotate.6PCs3iMDg06s MoshiAnnotate.6YYq8askAAFN MoshiAnnotate.6xwIy0MTQCo4 MoshiAnnotate.76o345rNy9cO
    MoshiAnnotate.7B8N4EnBlNdv MoshiAnnotate.7psBhOWRvVe1 MoshiAnnotate.7yEMLtYm0axe MoshiAnnotate.8DFhYQuFD3Pv
    MoshiAnnotate.8rxjZdiRbZmp MoshiAnnotate.9cLapUfThI75 MoshiAnnotate.AF4RCfFgc2SL MoshiAnnotate.B6ca3ht79qM4
    MoshiAnnotate.CEUQFZZqFcRU MoshiAnnotate.Ckxhkx6gRO7q MoshiAnnotate.DlNVu9xMp1Kf MoshiAnnotate.Dxa77b1Oqa4j
    MoshiAnnotate.EGhZQo4wsp7g MoshiAnnotate.EdfrF37PHup5 MoshiAnnotate.ElEaln8J1ZoJ MoshiAnnotate.ErExgS4vSaVj
    MoshiAnnotate.F1IPmHRoLgsV MoshiAnnotate.F3qILN3PX2vc MoshiAnnotate.G7YJTJmkfDI4 MoshiAnnotate.GEcGMRPZ9rzx
    MoshiAnnotate.GtDjjSmd7rRj MoshiAnnotate.HRu1amRlQ6b7 MoshiAnnotate.Hgp5JVs9DcEN MoshiAnnotate.IQ8Wqr84PuND
    MoshiAnnotate.IevDqEc3bRK1 MoshiAnnotate.IfqsKgU0rpZE MoshiAnnotate.JPDGwTcDQX23 MoshiAnnotate.Jhaq59JBRwnh
    MoshiAnnotate.Jr7RQKE2wK0f MoshiAnnotate.KbAC1CjqF6nH MoshiAnnotate.Lk7c4Bx7KwFh MoshiAnnotate.M0rFg5l8UWci
    MoshiAnnotate.M8upI2KLIdeu MoshiAnnotate.Msl1lmsm7ppy MoshiAnnotate.N61vvXwi29Zt MoshiAnnotate.NrZ4re35QF7L
    MoshiAnnotate.OOZSRGc2s14u MoshiAnnotate.OxECX1FKWCoR MoshiAnnotate.P970hoRyEb7e MoshiAnnotate.Pc0SsnufXfqs
    MoshiAnnotate.PttoCc8n4a7a MoshiAnnotate.Q9dv1DzW5ZNe MoshiAnnotate.QIlVfH4N08i8 MoshiAnnotate.R4d7kdZmERew
    MoshiAnnotate.RTenbtSR1VMx MoshiAnnotate.S7tPIHNJ7IQ4 MoshiAnnotate.SJiyeAzP9N4l MoshiAnnotate.SVVBbOLx9rbc
    MoshiAnnotate.SWkUU03jIqWS MoshiAnnotate.Sd9O3XQByG20 MoshiAnnotate.T5l4jD6NFlKh MoshiAnnotate.T9iiPjEx5Bi4
    MoshiAnnotate.THfbOIjgUKpI MoshiAnnotate.TKy2ItWXPGYe MoshiAnnotate.TPpbh1tEj1YB MoshiAnnotate.TZYbpPxN6iys
    MoshiAnnotate.TlXOgEIf7zVt MoshiAnnotate.UNXKNYcgfqXo MoshiAnnotate.UP7PD7lqIQc6 MoshiAnnotate.UwC0GDym5Rfl
    MoshiAnnotate.XATgGedjkspH MoshiAnnotate.XIZ6fjaEGjhy MoshiAnnotate.XndQoLFCmszQ MoshiAnnotate.Xye5mE9L81HN
    MoshiAnnotate.Y6urTHFSgFbE MoshiAnnotate.Z5dWXkpn8SE1 MoshiAnnotate.ZHz0ftsXu28a MoshiAnnotate.ZUHn4qgWI1Rb
    MoshiAnnotate.at99NbqQtlAI MoshiAnnotate.cDIeCmDWCKQN MoshiAnnotate.cNQkIroLZ7cy MoshiAnnotate.ciiCuEIAnxuv
    MoshiAnnotate.d2AiFoQlPMfh MoshiAnnotate.dEyHp3bF4eXb MoshiAnnotate.e4ZiIzl4VUSv MoshiAnnotate.eCpyAaJkJkN5
    MoshiAnnotate.erUTe0M4w69o MoshiAnnotate.ezQLCqFGvXiS MoshiAnnotate.ftzUtj5EFeVI MoshiAnnotate.g2D6idcKiOoF
    MoshiAnnotate.gChSgTVApvrk MoshiAnnotate.gQiAhC7euDKU MoshiAnnotate.gRzdNOOn59uv MoshiAnnotate.h528yK1yrO1l
    MoshiAnnotate.hOMuutWAu0wq MoshiAnnotate.hQLZcvX8Z9sY MoshiAnnotate.hftLNEnHBwd6 MoshiAnnotate.iAEKcKVOyJUK
    MoshiAnnotate.iaHZNwj0C1a8 MoshiAnnotate.j6c4H139YDR2 MoshiAnnotate.jBef06SwZP7D MoshiAnnotate.jODed5tLeO1b
    MoshiAnnotate.kNsxHkvPOIqi MoshiAnnotate.kwrBwM7xLnoC MoshiAnnotate.lm7QSqrmyaIj MoshiAnnotate.mg03whAVcxs6
    MoshiAnnotate.n6eoHuxu5Vhd MoshiAnnotate.nflQMvdZHuAN MoshiAnnotate.o8S8IF24V2gR MoshiAnnotate.oCASmvN8YUWP
    MoshiAnnotate.oqLIRumV3iPh MoshiAnnotate.oub9NbzkaOM1 MoshiAnnotate.p6Ds6R5Tf26l MoshiAnnotate.qBRYBOJtkE6e
    MoshiAnnotate.qdOWU3gxFrps MoshiAnnotate.qhEsIt08A6jO MoshiAnnotate.r2e5ZWFsY8i2 MoshiAnnotate.r5bBufwPOjev
    MoshiAnnotate.r7QCzCelOnWu MoshiAnnotate.saZKSiCHtMLH MoshiAnnotate.scWcc2tG4a4U MoshiAnnotate.tK8lOpgv32a0
    MoshiAnnotate.tLaoavjR92uy MoshiAnnotate.uPChT5nyqrVO MoshiAnnotate.ue6MXyDadtkW MoshiAnnotate.wAdimQob6WKC
    MoshiAnnotate.wW93LzWrsgnD MoshiAnnotate.x4Cw9ItmIqq9 MoshiAnnotate.x7wdp7vNKBkN MoshiAnnotate.xd9UvHQITeIU
    MoshiAnnotate.yT91EF4D0mMS MoshiAnnotate.zBMCpJu0BgIR MoshiAnnotate.zSr2ESCcmzOS MoshiAnnotate.zhEDP3bhDMWO
    MoshiAnnotate.zpwUDi2Z9C0Y
""".split()
)
ASST, USER = ["paris", "is", "the", "answer"], ["what", "is", "the", "capital"]
SR = 24000
D = 6
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------ graph (.venv)


def stage_graph(mutant, tmp):
    sys.path.insert(0, os.path.join(SETUP, "recipe"))
    sys.path.insert(0, os.path.join(SETUP, "recipe", "sisyphus"))
    os.environ.setdefault("CUDA_HOME", "/usr")
    logging.disable(logging.WARNING)
    from sisyphus import tk
    from sisyphus.loader import config_manager

    from i6_experiments.users.dorian_koch.speech_llm import fisher_prep, moshi, persona_knowledge

    classes = (moshi.MoshiAnnotate, persona_knowledge.KnowledgeDialogueCodes, fisher_prep.FisherToMoshiTrainData)
    if mutant == "default_off":
        for cls in classes:
            cls.__init__.__kwdefaults__["user_words"] = False
    config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
    graph = [j for j in tk.sis_graph.jobs() if type(j).__name__ in BUILDERS]

    def key(j):
        return f"{type(j).__name__}.{j._sis_id().split('.')[-1]}"

    x = tk.Path("/nonexistent/check_user_word_timings")
    voice_dir = os.path.join(tmp, "voices")
    os.makedirs(voice_dir, exist_ok=True)
    open(os.path.join(voice_dir, "v.wav"), "wb").close()
    kw = {
        "MoshiAnnotate": dict(venv_python_path=x, in_hf=x),
        "KnowledgeDialogueCodes": dict(
            corpora=[("a", x)], voice_dir=tk.Path(voice_dir), voice_alias="v", venv_python_path=x, loudness_norm={}
        ),
        "FisherToMoshiTrainData": dict(cutset=x, wav_dir=x),
    }
    fresh = {cls.__name__: cls(**kw[cls.__name__]) for cls in classes}  # a new corpus, declared as it would be
    off = {cls.__name__: cls(**kw[cls.__name__], user_words=False) for cls in classes}
    extra = []
    if mutant == "new_off":
        extra.append(moshi.MoshiAnnotate(venv_python_path=x, in_hf=tk.Path("/nonexistent/new"), user_words=False))
    keys = {key(j) for j in graph}
    bad = [
        key(j)
        for j in graph + list(fresh.values()) + extra
        if not getattr(j, "user_words", False) and key(j) not in FROZEN
    ]
    check("g1 every builder outside the frozen pre-2026-10-12 list has user_words on", not bad, f"{bad[:4]}")
    check(
        "g2 the frozen jobs are all in the graph with their old ids (user_words=False keeps the hash)",
        FROZEN <= keys,
        f"({len(FROZEN & keys)}/{len(FROZEN)} found; {len(graph)} builder jobs)",
    )
    check(
        "g3 a fresh declaration defaults to on and hashes differently from user_words=False",
        all(fresh[n].user_words and fresh[n]._sis_id() != off[n]._sis_id() for n in fresh),
    )
    # run() hands the flag to the worker
    ma = fresh["MoshiAnnotate"]
    if mutant == "run_no_flag":
        ma.user_words = False
    got = {}

    def fake_per_gpu(_py, _script, args_for, **_kw):
        got["annotate"] = [str(a) for a in args_for(0, 1)]
        return 1

    real = moshi.run_worker_script_per_gpu
    moshi.run_worker_script_per_gpu = fake_per_gpu
    try:
        off["MoshiAnnotate"].run()
        got["annotate_off"] = got.pop("annotate")
        ma.run()
    finally:
        moshi.run_worker_script_per_gpu = real
    kd = fresh["KnowledgeDialogueCodes"]

    def fake_run(cmd, **_kw):
        got["codes"] = [str(c) for c in cmd]

    real_run = persona_knowledge.subprocess.run
    persona_knowledge.subprocess.run = fake_run
    try:
        try:
            kd.run()
        except RuntimeError:
            pass  # "worker wrote no summary.json": the fake worker wrote nothing
    finally:
        persona_knowledge.subprocess.run = real_run
    check(
        "g4 run() passes the flag (--annotate_user / --user_words), and not when off",
        "--annotate_user" in got["annotate"]
        and "--annotate_user" not in got["annotate_off"]
        and "--user_words" in got["codes"],
    )
    # Fisher: the converter's planner + row builder on a one-conversation cutset
    cut = {
        "id": "fe00001",
        "duration": 50.0,
        "supervisions": [
            {
                "id": "u1",
                "speaker": "user",
                "recording_id": "fe00001_user",
                "start": 0.5,
                "duration": 2.0,
                "text": "what is the capital",
            },
            {
                "id": "a1",
                "speaker": "agent",
                "recording_id": "fe00001_agent",
                "start": 3.0,
                "duration": 4.0,
                "text": "paris [laughter] is the answer",
            },
            {
                "id": "u2",
                "speaker": "user",
                "recording_id": "fe00001_user",
                "start": 8.0,
                "duration": 1.0,
                "text": "((what)) is",
            },
        ],
    }
    cutset = os.path.join(tmp, "cutset.jsonl")
    with open(cutset, "w") as f:
        f.write(json.dumps(cut) + "\n")
    import numpy as np
    import soundfile as sf

    wav_dir = os.path.join(tmp, "fisher_wav")
    os.makedirs(wav_dir, exist_ok=True)
    for ch in "AB":
        sf.write(os.path.join(wav_dir, f"fe_03_00001{ch}.wav"), np.zeros(8000 * 50, dtype=np.int16), 8000)
    fk = dict(
        cutset=tk.Path(cutset), wav_dir=tk.Path(wav_dir), duration_sec=40.0, min_agent_speech_sec=1.0, min_agent_words=2
    )
    f_on = fisher_prep.FisherToMoshiTrainData(**fk)
    f_off = fisher_prep.FisherToMoshiTrainData(**fk, user_words=False)
    w_on, _ = f_on._plan_windows()
    w_off, _ = f_off._plan_windows()
    row = f_on._build_row(w_on[0])
    texts = lambda al: [a["text"] for a in al]  # noqa: E731
    # the annotate stage's environment = MoshiAnnotate.run's env_hook for a real annotate job (FFmpeg + NPP + libpython
    # for torchcodec, which a datasets Audio() column needs) and that job's venv
    real_job = next(j for j in graph if type(j).__name__ == "MoshiAnnotate" and j.env_ffmpeg_path is not None)
    env = {k: os.environ.get(k, "") for k in ("PATH", "LD_LIBRARY_PATH", "PKG_CONFIG_PATH")}
    moshi.InstallFFmpeg.add_to_env(real_job.env_ffmpeg_path, env)
    moshi.add_cuda_npp_to_env(real_job.env_npp_venv_python.get(), env)
    moshi.add_venv_python_lib_to_env(real_job.venv_python_path.get(), env)
    with open(os.path.join(tmp, "annotate_env.json"), "w") as f:
        json.dump({"env": env, "python": real_job.venv_python_path.get()}, f)
    check(
        "g5 Fisher: windows + alignments unchanged; the user's words only in alignments_other",
        [w["alignments"] for w in w_on] == [w["alignments"] for w in w_off]
        and "alignments_other" not in w_off[0]
        and texts(row["alignments"]) == ASST
        and texts(row["alignments_other"]) == USER + ["what", "is"]
        and {a["speaker"] for a in row["alignments_other"]} == {"user"}
        and row["other_ok"] is True,
        f"({texts(row['alignments'])} / {texts(row.get('alignments_other', []))})",
    )


# ------------------------------------------------------------------------------------------------ annotate (moshi_venv)


def stage_annotate(mutant, tmp):
    sys.path.insert(0, os.path.join(SETUP, "recipe"))
    sys.path.insert(0, os.path.join(SETUP, "recipe", "i6_experiments", "users", "dorian_koch", "speech_llm"))
    import numpy as np
    from datasets import Dataset, Features, Sequence, Value, load_from_disk

    import moshi_annotate_inference as mai

    def words_at(ws, t0):
        return {
            "segments": [
                {"words": [{"text": w, "start": t0 + 0.3 * k, "end": t0 + 0.3 * k + 0.25} for k, w in enumerate(ws)]}
            ]
        }

    class FakeWhisper:
        """Labels a channel's words by its signal: silence -> none, loud (the user fixture) -> USER, else ASST."""

        @staticmethod
        def load_model(*_a, **_k):
            return "fake"

        @staticmethod
        def transcribe(_model, audio, **_kw):
            a = np.abs(np.asarray(audio))
            if a.max() < 1e-6:
                return {"segments": []}
            loud = float(a[a > 1e-6].mean()) > 0.1
            return words_at(USER, 0.2) if loud else words_at(ASST, 2.5)

    mai.whisper = FakeWhisper
    real_annotate = mai.annotate_array
    calls = []

    def annotate(x, sr, w, p, language="en"):
        calls.append(x)
        if mutant == "user_from_assistant" and len(calls) % 2 == 0:
            x = calls[-2]  # the user pass reads the assistant channel
        out = real_annotate(x, sr, w, p, language=language)
        if mutant == "merge" and len(calls) % 2 == 1:
            out = out + [[w, [0.2 + 0.3 * k, 0.45 + 0.3 * k], "SPEAKER_MAIN"] for k, w in enumerate(USER)]
        return out

    mai.annotate_array = annotate
    rng = np.random.default_rng(0)

    def chan(t0, t1, std, dur=8.0):
        x = np.zeros(int(dur * SR), dtype=np.float32)
        x[int(t0 * SR) : int(t1 * SR)] = rng.normal(0, std, int(t1 * SR) - int(t0 * SR))
        return x

    rows = []
    for i in range(3):
        user = np.zeros(int(8 * SR), dtype=np.float32) if i == 2 else chan(1.0, 4.0, 0.3)
        rows.append(
            {
                "id": f"row{i}",
                # columnar, as ChatterboxInference stores it (process_row zips the dict of lists back into tracks)
                "speaker_audio": {
                    "speaker": ["user", "assistant"],
                    "audio": [
                        {"array": user.tolist(), "sampling_rate": SR},
                        {"array": chan(2.0, 7.5, 0.05).tolist(), "sampling_rate": SR},
                    ],
                },
            }
        )
    feats = Features(
        {
            "id": Value("string"),
            "speaker_audio": Sequence(
                {
                    "speaker": Value("string"),
                    "audio": {"array": Sequence(Value("float32")), "sampling_rate": Value("int32")},
                }
            ),
        }
    )
    src = os.path.join(tmp, "tts")
    Dataset.from_list(rows, features=feats).save_to_disk(src)
    out_on, out_off = os.path.join(tmp, "ann_on"), os.path.join(tmp, "ann_off")
    cwd = os.getcwd()
    os.chdir(tmp)  # progress.json lands in the cwd
    try:
        mai.annotate_hf_to_hf(src, out_on, annotate_user=mutant != "flag_off")
        calls.clear()
        mai.annotate_hf_to_hf(src, out_off)
    finally:
        os.chdir(cwd)
    on, off = load_from_disk(out_on).data.table, load_from_disk(out_off).data.table
    texts = lambda al: [a["text"] for a in al]  # noqa: E731
    check(
        "a1 annotate_user: alignments_other + other_ok columns (the JRE layout)",
        {"alignments_other", "other_ok"} <= set(on.column_names)
        and str(load_from_disk(out_on).features.get("alignments_other")) == str(mai.ALIGNMENT_FEATURE),
    )
    if FAILS:
        return
    al, ao, ok = (on.column(c).to_pylist() for c in ("alignments", "alignments_other", "other_ok"))
    check(
        "a2 alignments = the assistant's words only; alignments_other = the user's (speaker 'user'); silent user -> other_ok false",
        all(texts(a) == ASST for a in al)
        and [texts(a) for a in ao] == [USER, USER, []]
        and all(a["speaker"] == mai.USER_SPEAKER for r in ao for a in r)
        and ok == [True, True, False],
        f"({[texts(a) for a in al][:1]} / {[texts(a) for a in ao]} / {ok})",
    )
    check(
        "a3 without the flag: the pre-2026-10-12 schema, the same rows and alignments",
        off.column_names == ["id", "audio_assistant", "audio_user", "alignments", "duration"]
        and off.column("alignments").to_pylist() == on.column("alignments").to_pylist()
        and off.column("id").to_pylist() == on.column("id").to_pylist() == ["row0", "row1", "row2"],
        f"({off.column_names})",
    )


# ------------------------------------------------------------------------------------------------ codes + loader (moshi_family_venv)


def stage_codes(mutant, tmp):
    sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import load_from_disk

    import moshi_family.knowledge_codes as kc
    import moshi_family.train_data_common as tdc
    from moshi_family.personaplex import train_data as td

    ann_on, ann_off = os.path.join(tmp, "ann_on"), os.path.join(tmp, "ann_off")
    if not os.path.isdir(ann_on):
        raise RuntimeError(f"{ann_on} missing: run the annotate stage first (the default run does)")
    real_prepare = kc.prepare_row

    def prepare(table, i, *a, **k):
        if mutant == "drop":
            k["user_words"] = False
        a_, u_, aligns, info = real_prepare(table, i, *a, **k)
        if mutant == "drop":
            info["user_words"], info["user_clamp_shift_sec"] = ([], False), 0.0
        elif mutant == "merge":
            aligns = aligns + info["user_words"][0]
        elif mutant == "wrong_side":
            info["user_words"] = (aligns, True)
        elif mutant == "no_clamp":
            raw = tdc.normalize_alignments(table.column("alignments_other")[i].as_py())
            info["user_words"] = (raw, bool(raw))
        return a_, u_, aligns, info

    kc.prepare_row = prepare
    voice = os.path.join(tmp, "voice.wav")
    sf.write(voice, np.random.default_rng(1).normal(0, 0.05, 5 * SR).astype(np.float32), SR)

    def build(out, ann, user_words):
        argv = ["knowledge_codes", "--out", out, "--voice_wav", voice, "--system_prompt", "You are a wise teacher."]
        argv += ["--loudness_norm", json.dumps({"target_lufs": -24.0, "max_gain_db": 10.0, "min_lufs": -45.0})]
        argv += ["--device", "cpu", "--progress", os.path.join(tmp, "progress.json"), "--workers", "1"]
        argv += (["--user_words"] if user_words else []) + [f"k={ann}"]
        sys.argv = argv
        kc.main()
        return tdc.read_arrow_table(out)

    on = build(os.path.join(tmp, "codes_on"), ann_on, True)
    plain = build(os.path.join(tmp, "codes_plain"), ann_on, False)
    texts = lambda al: [a["text"] for a in al]  # noqa: E731
    same = all(
        on.column(c).to_pylist() == plain.column(c).to_pylist()
        for c in ("id", "codes_assistant", "codes_user", "alignments")
    )
    check("c1 --user_words: ids / codes / alignments bit-identical to the build without it", same)
    src = load_from_disk(ann_on).data.table
    ao, ok = on.column("alignments_other").to_pylist(), on.column("other_ok").to_pylist()
    check(
        "c1b alignments_other = the annotated user words, other_ok as annotated; never in alignments",
        [texts(a) for a in ao] == [texts(a) for a in src.column("alignments_other").to_pylist()]
        and ok == src.column("other_ok").to_pylist()
        and not any(set(texts(a)) - set(ASST) for a in on.column("alignments").to_pylist())
        and "alignments_other" not in plain.column_names,
        f"({[texts(a) for a in ao]} / {ok})",
    )
    first = [a[0]["start"] for a in ao if a]
    check(
        "c4 user words clamped to the user channel's speech onset (word 0 at 0.2 s, speech from 1.0 s)",
        all(s >= 0.9 for s in first),
        f"({first})",
    )
    tok = td.PersonaPlexTokenizer(
        "nvidia/personaplex-7b-v1",
        device="cpu",
        cfg=td.PersonaPlexDataConfig(
            duration_sec=60.0, silence_frames=6, emit_epad=True, onset_floor=True, word_overlap="drop"
        ),
    )
    kw = dict(system_prompt_key="context", voice_codes_key="voice_codes")
    with_tgt = td.persona_item_builder(on, tok, user_text_delay=D, **kw)
    base = td.persona_item_builder(plain, tok, **kw)
    tgt_ok = row_same = has_words = leak_visible = True
    for r in range(on.num_rows):
        b = with_tgt(r, np.random.default_rng([3, r]))
        a = base(r, np.random.default_rng([3, r]))
        row_same &= all(torch.equal(x, y) for x, y in zip(a, b[: len(a)])) and len(b) == len(a) + 1
        codes, mask, tgt = b[0], b[1], b[-1]
        n = int(mask.sum())
        P = codes.shape[1] - n
        if not ok[r]:
            tgt_ok &= bool((tgt == tdc.USER_TEXT_IGNORE).all())
            continue
        row = tok.interleave_text(tdc.normalize_alignments(ao[r]), n, **tok._text_kwargs())[0].cpu()
        exp = torch.full((n,), tdc.TEXT_PADDING_ID, dtype=torch.long)
        exp[D:] = row[: n - D]
        tgt_ok &= bool((tgt[:P] == tdc.USER_TEXT_IGNORE).all()) and torch.equal(tgt[P:].cpu(), exp)
        has_words &= int((tgt[P:] >= 0).sum() - (tgt[P:] == tdc.TEXT_PADDING_ID).sum()) > 0
        leak_visible &= not torch.equal(codes[tdc.TEXT_ROW, P:].cpu(), row)
    check(
        "c2 loader: items = the plain build's + one tensor; the assistant text row is not the user's words",
        row_same and leak_visible,
    )
    check(
        "c3 loader: user-text targets = the stored user words delayed 6 frames (word tokens present); other_ok false -> IGNORE",
        tgt_ok and has_words,
    )
    refused = []
    try:
        td.persona_item_builder(plain, tok, user_text_delay=D, **kw)
    except ValueError:
        refused.append("loader")
    try:
        build(os.path.join(tmp, "codes_refused"), ann_off, True)
    except RuntimeError:
        refused.append("worker")
    check(
        "c5 a corpus without user timings is refused (loader + --user_words worker)",
        refused == ["loader", "worker"],
        f"({refused})",
    )


# ------------------------------------------------------------------------------------------------ driver


def run_stage(stage, mutant, tmp, base):
    py = os.path.join(SETUP, STAGE_PY[stage]) if STAGE_PY[stage] else None
    env = dict(os.environ, CUDA_HOME=os.environ.get("CUDA_HOME", "/usr"), HF_HUB_OFFLINE="1")
    if stage == "annotate":  # written by the graph stage: the real annotate job's venv and env_hook
        with open(os.path.join(base, "annotate_env.json")) as f:
            spec = json.load(f)
        py = spec["python"]
        env.update(spec["env"])
    cmd = [py, HERE, "--stage", stage, "--tmp", tmp] + (["--mutant", mutant] if mutant else [])
    r = subprocess.run(cmd, env=env, capture_output=mutant is not None, text=True, cwd=SETUP)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGE_PY), default=None)
    ap.add_argument("--mutant", default=None)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.stage is not None:
        if args.mutant is not None:
            assert args.mutant in MUTANTS[args.stage], args.mutant
        {"graph": stage_graph, "annotate": stage_annotate, "codes": stage_codes}[args.stage](args.mutant, args.tmp)
        print(f"{args.stage}{'/' + args.mutant if args.mutant else ''}: {len(FAILS)} failed {FAILS}", flush=True)
        sys.exit(1 if FAILS else 0)
    with tempfile.TemporaryDirectory(prefix="check_user_word_timings_") as tmp:
        rc = [run_stage(s, None, tmp, tmp) for s in ("graph", "annotate", "codes")]
        ok = all(c == 0 for c in rc)
        print(f"{'ALL PASS' if ok else 'FAILED'} (stages graph/annotate/codes exit {rc})", flush=True)
        if not args.selftest:
            sys.exit(0 if ok else 1)
        caught = []
        for stage, ms in MUTANTS.items():
            for m in ms:
                mtmp = tempfile.mkdtemp(prefix=f"m_{m}_", dir=tmp)
                if stage == "codes":  # the codes stage reads the (unmutated) annotate stage's output
                    for d in ("ann_on", "ann_off"):
                        os.symlink(os.path.join(tmp, d), os.path.join(mtmp, d))
                c = run_stage(stage, m, mtmp, tmp)
                caught.append(c != 0)
                print(f"{'CAUGHT' if c != 0 else 'MISSED'} mutant {stage}/{m}", flush=True)
        print(f"selftest: {sum(caught)}/{len(caught)} mutants caught", flush=True)
        sys.exit(0 if ok and all(caught) else 1)


if __name__ == "__main__":
    main()
