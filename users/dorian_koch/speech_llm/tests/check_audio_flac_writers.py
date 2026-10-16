"""Guard: the ``Audio()`` writers (MoshiAnnotate, Chatterbox TTS) store exact FLAC that decodes to what their WAV did.

FLAC step 2c (2026-10-16; user: "new behaviour exactly matches old behaviour"): ``moshi_annotate_inference._audio_cell``
and ``chatterbox_inference._audio_value`` keep each writer's own WAV encode and store its PCM_16 samples as FLAC
(``clip_codec.wav_to_flac``) under the same stored ``path``. This drives both helpers from the worker sources (the
workers import Whisper / Chatterbox at module level, so the helper functions and ``dialogue_features`` are executed
from their source), checks the rows are wired through them, writes each corpus the old way and the new way through
``datasets`` and compares what ``datasets``' own decode (torchcodec) returns: identical arrays and rates, identical
stored paths, FLAC bytes actually stored. Plus ``wav_to_flac`` unit cases (float WAV, FLAC, garbage, empty -> None).

Needs a working torchcodec, which the setup .venv lacks, so it re-runs itself under ``moshi_venv_v2`` with exactly
MoshiAnnotate's env (our FFmpeg, CUDA NPP, the interpreter's lib). From the setup root, no GPU:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_audio_flac_writers.py
Shown able to fail: before 2c neither helper exists (and the rows are not built through one).
"""

import ast
import io
import os
import subprocess
import sys
import tempfile

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _reexec():
    sys.path.insert(0, "recipe")
    sys.path.insert(0, "recipe/sisyphus")
    from sisyphus import tk

    from i6_experiments.users.dorian_koch.speech_llm.common import add_cuda_npp_to_env, add_venv_python_lib_to_env
    from i6_experiments.users.dorian_koch.speech_llm.tts import InstallFFmpeg

    vp = "work/i6_experiments/users/dorian_koch/jobs/venv/CreateVenv.moshi_venv_v2/output/venv_python_path"
    py = open(vp).read().strip().strip("'\"")
    env = dict(os.environ, CHECK_AUDIO_FLAC_INNER="1", HF_DATASETS_OFFLINE="1")
    InstallFFmpeg.add_to_env(tk.Path(os.path.realpath("output/ffmpeg")), env)
    add_cuda_npp_to_env(os.path.realpath("output/venv/cuda_npp") + "/bin/python", env)
    add_venv_python_lib_to_env(py, env)
    sys.exit(subprocess.run([py, os.path.abspath(__file__)], env=env).returncode)


def _from_source(path, names, ns):
    tree = ast.parse(open(path).read())
    body = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            body.append(n)
        elif isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets):
            body.append(n)
    found = {getattr(n, "name", None) or n.targets[0].id for n in body}
    missing = set(names) - found
    assert not missing, f"{os.path.basename(path)} has no {sorted(missing)} (the FLAC writer helper of step 2c)"
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


def main():
    import numpy as np
    import soundfile as sf
    from datasets import Audio, Dataset, Features, List, Sequence, Value, load_from_disk

    sys.path.insert(0, D)
    import clip_codec

    def wav(x, sr, subtype="PCM_16"):
        b = io.BytesIO()
        sf.write(b, x, sr, format="WAV", subtype=subtype)
        return b.getvalue()

    rng = np.random.default_rng(0)
    sr = 24000
    mono = rng.integers(-20000, 20000, sr // 3).astype(np.int16)
    stereo = rng.integers(-20000, 20000, (sr // 4, 2)).astype(np.int16)
    # wav_to_flac: exact on PCM_16 WAV, None for everything else
    for x in (mono, stereo):
        f = clip_codec.wav_to_flac(wav(x, sr))
        assert f is not None and sf.info(io.BytesIO(f)).format == "FLAC"
        back, r = sf.read(io.BytesIO(f), dtype="int16", always_2d=True)
        assert r == sr and np.array_equal(back, x.reshape(len(x), -1))
    fl = io.BytesIO()
    sf.write(fl, mono, sr, format="FLAC", subtype="PCM_16")
    for bad in (wav(mono / 32768.0, sr, "FLOAT"), fl.getvalue(), b"not audio", wav(np.zeros(0, np.int16), sr)):
        assert clip_codec.wav_to_flac(bad) is None
    print("[ok] wav_to_flac: exact on PCM_16 WAV (mono/stereo); float WAV, FLAC, garbage, empty -> None")

    ns = {"Audio": Audio, "wav_to_flac": clip_codec.wav_to_flac, "os": os, "np": np}
    ann = os.path.join(D, "moshi_annotate_inference.py")
    _from_source(ann, ["_audio_cell"], ns)
    src = open(ann).read()
    assert src.count('"audio_assistant": _audio_cell(') == 1 and src.count('"audio_user": _audio_cell(') == 1, (
        "MoshiAnnotate rows are not built through _audio_cell"
    )
    tts = os.path.join(D, "chatterbox_inference.py")
    ns2 = _from_source(
        tts,
        ["_audio_value", "dialogue_features"],
        {**ns, "Features": Features, "Value": Value, "Sequence": Sequence, "List": List},
    )
    assert open(tts).read().count("speaker_audio[speaker] = _audio_value(audio_path)") == 1, (
        "Chatterbox rows are not built through _audio_value"
    )

    def decoded(ds, get):
        return [(np.asarray(a["array"]), a["sampling_rate"]) for r in ds for a in get(r)]

    def stored(ds, get_raw):
        return [get_raw(r) for r in ds.data.table.to_pylist()]

    with tempfile.TemporaryDirectory() as tmp:
        # MoshiAnnotate: the same float arrays (int16-valued and arbitrary) through dataset.map(features=Audio())
        arrays = [mono.astype(np.float32) / 32768.0, rng.uniform(-0.7, 0.7, sr // 5).astype(np.float32)]
        feats = Features({"i": Value("int64"), "audio_assistant": Audio()})
        base = Dataset.from_dict({"i": list(range(len(arrays)))})
        old = base.map(lambda r: {"audio_assistant": {"array": arrays[r["i"]], "sampling_rate": sr}}, features=feats)
        new = base.map(lambda r: {"audio_assistant": ns["_audio_cell"](arrays[r["i"]], sr)}, features=feats)
        old.save_to_disk(f"{tmp}/a_old")
        new.save_to_disk(f"{tmp}/a_new")
        old, new = load_from_disk(f"{tmp}/a_old"), load_from_disk(f"{tmp}/a_new")
        g = lambda r: [r["audio_assistant"]]  # noqa: E731
        for (x, rx), (y, ry) in zip(decoded(old, g), decoded(new, g)):
            assert rx == ry and x.shape == y.shape and np.array_equal(x, y), "MoshiAnnotate: FLAC decodes differently"
        so = stored(old, lambda r: r["audio_assistant"])
        sn = stored(new, lambda r: r["audio_assistant"])
        assert [a["path"] for a in so] == [a["path"] for a in sn]
        assert all(sf.info(io.BytesIO(a["bytes"])).format == "FLAC" for a in sn), "MoshiAnnotate stored no FLAC"
        print(
            f"[ok] MoshiAnnotate cells: identical decode, FLAC stored ({sum(len(a['bytes']) for a in sn)} vs "
            f"{sum(len(a['bytes']) for a in so)} bytes)"
        )

        # Chatterbox: per-speaker .wav files -> dialogue_features rows, old (paths) vs new (_audio_value)
        paths = []
        for k, spk in enumerate(("user", "assistant")):
            d = f"{tmp}/dialogue_{k}"
            os.makedirs(d)
            p = f"{d}/{spk}.wav"
            sf.write(p, (mono if k == 0 else stereo[:, 0]).astype(np.float32) / 32768.0, sr, subtype="PCM_16")
            paths.append((spk, p))

        def rows(value):
            def gen():
                yield {
                    "id": "dialogue_0",
                    "speaker_audio": {"speaker": [s for s, _ in paths], "audio": [value(p) for _, p in paths]},
                    "turns": {"speaker": ["user"], "start_time": [0.0], "end_time": [1.0], "text": ["hi"]},
                }

            return gen

        F = ns2["dialogue_features"]
        Dataset.from_generator(rows(lambda p: p), features=F).save_to_disk(f"{tmp}/t_old")
        Dataset.from_generator(rows(ns2["_audio_value"]), features=F).save_to_disk(f"{tmp}/t_new")
        old, new = load_from_disk(f"{tmp}/t_old"), load_from_disk(f"{tmp}/t_new")
        g = lambda r: r["speaker_audio"]["audio"]  # noqa: E731
        do, dn = decoded(old, g), decoded(new, g)
        assert len(do) == len(dn) == 2
        for (x, rx), (y, ry) in zip(do, dn):
            assert rx == ry and x.shape == y.shape and np.array_equal(x, y), "Chatterbox: FLAC decodes differently"
        so = stored(old, lambda r: r["speaker_audio"]["audio"])[0]
        sn = stored(new, lambda r: r["speaker_audio"]["audio"])[0]
        assert [a["path"] for a in so] == [a["path"] for a in sn] == ["user.wav", "assistant.wav"], (so, sn)
        assert all(sf.info(io.BytesIO(a["bytes"])).format == "FLAC" for a in sn), "Chatterbox stored no FLAC"
        print("[ok] Chatterbox rows: identical decode, same stored names, FLAC stored")
    print("check_audio_flac_writers: OK")


if __name__ == "__main__":
    if os.environ.get("CHECK_AUDIO_FLAC_INNER"):
        main()
    else:
        _reexec()
