"""Guard: every clip-store reader decodes the FLAC layout to exactly what it decodes from the float layout.

FLAC step 1 (2026-10-16; user: "make sure old jobs keep working and new behaviour exactly matches old behaviour"):
clip stores may hold their samples as ``audio_flac`` (exact FLAC PCM_16, ``clip_codec.encode_exact``) instead of
``audio`` (list<float32>). This writes the same int16-valued clips in both layouts and drives the real readers:
``clip_store.open_clips`` / ``materialise_clips`` (bit-identical arrays, byte-identical wavs),
``moshi_family.offline_cli._materialise_arrow_clips`` (byte-identical wavs: all offline drivers read through it),
``rl.knowledge_segments.KnowledgeQuestionSource.segment`` (identical ``user_pcm``) and the whisper worker's
``_LazyArrowClips`` (its class source, executed: the worker module itself imports faster_whisper). Also: the two
``clip_codec.py`` copies are byte-identical, and ``encode_exact`` refuses anything it could not store losslessly.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_clip_codec.py
"""

import ast
import filecmp
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import datasets  # noqa: E402
import numpy as np  # noqa: E402
from datasets import Dataset  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import clip_codec  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.clip_store import materialise_clips, open_clips, write_clips  # noqa: E402

datasets.disable_progress_bars()
D = os.path.join(os.getcwd(), "recipe/i6_experiments/users/dorian_koch/speech_llm")
F = os.path.join(os.getcwd(), "recipe/speech_llm/full_duplex/moshi_family")


def main():
    assert filecmp.cmp(os.path.join(D, "clip_codec.py"), os.path.join(F, "clip_codec.py"), shallow=False), (
        "the two clip_codec.py copies differ"
    )
    rng = np.random.default_rng(0)
    sr = 24000
    clips = {}
    for i in (0, 3, 7):
        q = rng.integers(-32768, 32768, size=sr // 2 + 37 * i).astype(np.float64)
        q[:200] = 0
        clips[i] = (q / 32768.0).astype(np.float32)
    # encode_exact: exact on int16 values, refuses the rest
    for x in clips.values():
        raw = clip_codec.encode_exact(x, sr)
        assert raw is not None and np.array_equal(clip_codec.decode_row({clip_codec.COL_AUDIO_FLAC: raw}), x)
    for bad in (
        np.array([0.1, 0.2], np.float32),
        np.array([1.0], np.float32),
        np.array([], np.float32),
        np.array([np.nan], np.float32),
    ):
        assert clip_codec.encode_exact(bad, sr) is None, bad
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
        old = os.path.join(tmp, "old")
        write_clips(old, [(i, x, sr) for i, x in clips.items()])
        new = os.path.join(tmp, "new")
        Dataset.from_dict(
            {
                "index": list(clips),
                clip_codec.COL_AUDIO_FLAC: [clip_codec.encode_exact(x, sr) for x in clips.values()],
                "sampling_rate": [sr] * len(clips),
                "monologue": [None] * len(clips),
                "trace_json": [None] * len(clips),
            }
        ).save_to_disk(new)
        # clip_store readers
        a, b = open_clips(old), open_clips(new)
        assert sorted(a) == sorted(b) == sorted(clips)
        for i, x in clips.items():
            (xa, ra), (xb, rb) = a[i], b[i]
            assert ra == rb == sr and xa.dtype == xb.dtype == np.float32
            assert np.array_equal(xa, x) and np.array_equal(xb, x), f"open_clips clip {i}"
        ma, mb = os.path.join(tmp, "ma"), os.path.join(tmp, "mb")
        materialise_clips(old, ma)
        materialise_clips(new, mb)
        for i in clips:
            assert filecmp.cmp(f"{ma}/{i}.wav", f"{mb}/{i}.wav", shallow=False), f"materialise_clips clip {i}"
        # moshi_family offline drivers
        from moshi_family.offline_cli import _materialise_arrow_clips
        from pathlib import Path

        oa, ob = (
            _materialise_arrow_clips(Path(old), Path(tmp, "oa")),
            _materialise_arrow_clips(Path(new), Path(tmp, "ob")),
        )
        for i in clips:
            assert filecmp.cmp(oa / f"{i}.wav", ob / f"{i}.wav", shallow=False), f"offline_cli clip {i}"
        # RL question clips
        from moshi_family.rl.knowledge_segments import KnowledgeQuestionSource

        qd = os.path.join(tmp, "q")
        Dataset.from_dict(
            {"question": [f"q{i}" for i in range(8)], "answer": ["a"] * 8, "aliases": [["a"]] * 8}
        ).save_to_disk(qd)
        segs = []
        for audio in (old, new):
            man = os.path.join(tmp, f"man_{os.path.basename(audio)}.json")
            open(man, "w").write(json.dumps({"questions": qd, "audio": audio}))
            src = KnowledgeQuestionSource(man, sample_rate=sr, log=lambda *a: None)
            segs.append({i: src.segment(i).user_pcm for i in clips})
        for i in clips:
            assert np.array_equal(segs[0][i], segs[1][i]), f"knowledge_segments clip {i}"
        # whisper worker's reader, executed from its own source
        tree = ast.parse(open(os.path.join(D, "whisper_benchmark_inference.py")).read())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_LazyArrowClips")
        ns = {"np": np, "decode_row": clip_codec.decode_row}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "whisper_benchmark_inference.py", "exec"), ns)
        from datasets import load_from_disk

        wa, wb = ns["_LazyArrowClips"](load_from_disk(old)), ns["_LazyArrowClips"](load_from_disk(new))
        for i, x in clips.items():
            assert np.array_equal(wa[i][0], x) and np.array_equal(wb[i][0], x) and wa[i][1] == wb[i][1] == sr, (
                f"whisper clip {i}"
            )
    print("check_clip_codec: OK (copies identical; float and FLAC layouts decode identically in all 5 readers)")


if __name__ == "__main__":
    main()
