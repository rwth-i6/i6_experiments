"""Guard: every clip-store reader decodes the FLAC layouts to exactly what it decodes from the float layout.

FLAC (2026-10-16; old jobs must keep working and new behaviour must exactly match old behaviour): clip
stores may hold a row's samples as ``audio_flac`` (exact FLAC PCM_16, ``clip_codec.encode_exact``) instead of ``audio``
(list<float32>), per row: the writer streams, so a store can mix FLAC rows and float rows (the other column null).
This writes the same clips in three layouts -- old float, FLAC-only, hybrid (int16-valued rows FLAC, one non-int16
row kept as floats) -- and drives the real readers on each: ``clip_store.open_clips`` / ``materialise_clips``
(bit-identical arrays, byte-identical wavs), ``moshi_family.offline_cli._materialise_arrow_clips`` (byte-identical
wavs: all offline drivers read through it), ``rl.knowledge_segments.KnowledgeQuestionSource.segment`` (identical
``user_pcm``) and the whisper worker's ``_LazyArrowClips`` (its class source, executed: the worker module itself
imports faster_whisper). Also: the two ``clip_codec.py`` copies are byte-identical, and ``encode_exact`` refuses
anything it could not store losslessly. Shown able to fail: the step-1 readers raise ``KeyError: 'audio'`` on the
FLAC-only store, and the step-1 ``decode_row`` (``"audio_flac" in row``) hands a null to soundfile on the hybrid one.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_clip_codec.py
"""

import ast
import filecmp
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))  # merge_clip_datasets(link=True) -> jobs.hf
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import datasets  # noqa: E402
import numpy as np  # noqa: E402
from datasets import Dataset, Features, Sequence, Value, load_from_disk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import clip_codec  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.clip_store import (  # noqa: E402
    materialise_clips,
    merge_clip_datasets,
    open_clips,
    write_clips,
)

datasets.disable_progress_bars()
D = os.path.join(os.getcwd(), "recipe/i6_experiments/users/dorian_koch/speech_llm")
F = os.path.join(os.getcwd(), "recipe/speech_llm/full_duplex/moshi_family")
SR = 24000
HYBRID = Features(
    {
        "index": Value("int64"),
        "audio": Sequence(Value("float32")),
        "audio_flac": Value("binary"),
        "sampling_rate": Value("int64"),
        "monologue": Value("string"),
        "trace_json": Value("string"),
    }
)


def _clips():
    rng = np.random.default_rng(0)
    clips = {}
    for i in (0, 3, 7):
        q = rng.integers(-32768, 32768, size=SR // 2 + 37 * i).astype(np.float64)
        q[:200] = 0
        clips[i] = (q / 32768.0).astype(np.float32)
    clips[9] = rng.uniform(-0.5, 0.5, SR // 3).astype(np.float32)  # not int16-valued: must stay float
    # a clip whose FLAC bytes END IN NUL (~1 in 256 do: the stream ends in a CRC-16): datasets' numpy format strips
    # trailing NULs from binary values (2026-10-16, "flac decoder lost sync") -- every reader must survive it
    for seed in range(5000):
        q = np.random.default_rng(1000 + seed).integers(-30000, 30000, SR // 10)
        x = (q / 32768.0).astype(np.float32)
        raw = clip_codec.encode_exact(x, SR)
        if raw is not None and raw.endswith(b"\x00"):
            clips[11] = x
            break
    assert 11 in clips, "no FLAC stream ending in a NUL byte found"
    return clips


def _old_store(path, clips):
    """What ``write_clips`` wrote before 2026-10-16: float rows only, column types inferred."""
    Dataset.from_dict(
        {
            "index": list(clips),
            "audio": list(clips.values()),
            "sampling_rate": [SR] * len(clips),
            "monologue": [None] * len(clips),
            "trace_json": [None] * len(clips),
        }
    ).save_to_disk(path)


def _store(path, clips, *, flac_only):
    rows = {k: [] for k in HYBRID}
    for i, x in clips.items():
        raw = clip_codec.encode_exact(x, SR)
        if flac_only and raw is None:
            continue
        rows["index"].append(i)
        rows["audio"].append(None if raw is not None else x)
        rows["audio_flac"].append(raw)
        rows["sampling_rate"].append(SR)
        rows["monologue"].append(None)
        rows["trace_json"].append(None)
    ds = Dataset.from_dict(rows, features=HYBRID)
    if flac_only:
        ds = ds.remove_columns("audio")
    ds.save_to_disk(path)
    return sorted(rows["index"])


def _check_readers(tmp, name, path, want):
    a = open_clips(path)
    assert sorted(a) == sorted(want), (name, sorted(a))
    for i, x in want.items():
        got, sr = a[i]
        assert sr == SR and got.dtype == np.float32 and np.array_equal(got, x), f"{name}: open_clips clip {i}"
    m = os.path.join(tmp, f"m_{name}")
    materialise_clips(path, m)
    ref = os.path.join(tmp, "ref_wav")
    from moshi_family.offline_cli import _materialise_arrow_clips

    o = _materialise_arrow_clips(Path(path), Path(tmp, f"o_{name}"))
    ro = Path(tmp, "ref_offline")
    for i in want:
        assert filecmp.cmp(f"{ref}/{i}.wav", f"{m}/{i}.wav", shallow=False), f"{name}: materialise_clips clip {i}"
        assert filecmp.cmp(ro / f"{i}.wav", o / f"{i}.wav", shallow=False), f"{name}: offline_cli clip {i}"
    from moshi_family.rl.knowledge_segments import LEAD_IN_S, KnowledgeQuestionSource

    lead = np.zeros(int(round(LEAD_IN_S * SR)), dtype=np.float32)

    qd = os.path.join(tmp, "q")
    if not os.path.exists(qd):
        Dataset.from_dict(
            {"question": [f"q{i}" for i in range(12)], "answer": ["a"] * 12, "aliases": [["a"]] * 12}
        ).save_to_disk(qd)
    man = os.path.join(tmp, f"man_{name}.json")
    Path(man).write_text(json.dumps({"questions": qd, "audio": path}))
    src = KnowledgeQuestionSource(man, sample_rate=SR, log=lambda *a: None)
    for i, x in want.items():
        pcm = src.segment(i).user_pcm
        assert np.array_equal(pcm, np.concatenate([lead, x])), f"{name}: knowledge_segments clip {i}"
    tree = ast.parse(open(os.path.join(D, "whisper_benchmark_inference.py")).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_LazyArrowClips")
    ns = {"np": np, "decode_row": clip_codec.decode_row}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "whisper_benchmark_inference.py", "exec"), ns)
    w = ns["_LazyArrowClips"](load_from_disk(path))
    for i, x in want.items():
        assert np.array_equal(w[i][0], x) and w[i][1] == SR, f"{name}: whisper clip {i}"


def main():
    assert filecmp.cmp(os.path.join(D, "clip_codec.py"), os.path.join(F, "clip_codec.py"), shallow=False), (
        "the two clip_codec.py copies differ"
    )
    clips = _clips()
    for i, x in clips.items():
        raw = clip_codec.encode_exact(x, SR)
        if i == 11:
            assert raw.endswith(b"\x00")
        if i == 9:
            assert raw is None, "a non-int16 clip must not be FLAC-encoded"
        else:
            assert raw is not None and np.array_equal(clip_codec.decode_row({clip_codec.COL_AUDIO_FLAC: raw}), x)
    from datasets import Features as _F, Value as _V

    nul = clip_codec.encode_exact(clips[11], SR)
    stripped = Dataset.from_dict({"audio_flac": [nul]}, features=_F({"audio_flac": _V("binary")})).with_format("numpy")[
        0
    ]
    try:
        clip_codec.decode_row(stripped)
        raise AssertionError("decode_row decoded a numpy-format (NUL-stripped) audio_flac")
    except TypeError:
        pass
    assert (
        clip_codec.decode_row(
            clip_codec.numpy_rows(Dataset.from_dict({"audio_flac": [nul]}, features=_F({"audio_flac": _V("binary")})))[
                0
            ]
        ).shape
        == clips[11].shape
    )
    for bad in (
        np.array([0.1, 0.2], np.float32),
        np.array([1.0], np.float32),
        np.array([], np.float32),
        np.array([np.nan], np.float32),
    ):
        assert clip_codec.encode_exact(bad, SR) is None, bad
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
        old = os.path.join(tmp, "old")
        _old_store(old, clips)
        # references: what the readers produce from the old float layout
        materialise_clips(old, os.path.join(tmp, "ref_wav"))
        from moshi_family.offline_cli import _materialise_arrow_clips

        _materialise_arrow_clips(Path(old), Path(tmp, "ref_offline"))
        flac_only, hybrid = os.path.join(tmp, "flac_only"), os.path.join(tmp, "hybrid")
        kept = _store(flac_only, clips, flac_only=True)
        _store(hybrid, clips, flac_only=False)
        _check_readers(tmp, "old", old, clips)
        _check_readers(tmp, "flac_only", flac_only, {i: clips[i] for i in kept})
        _check_readers(tmp, "hybrid", hybrid, clips)
        # the writer (FLAC step 2): int16-valued clips as FLAC, the rest as floats, one declared schema
        new = os.path.join(tmp, "write_clips")
        write_clips(new, [(i, x, SR) for i, x in clips.items()])
        ds = load_from_disk(new)
        flac_rows = {int(r["index"]) for r in ds if r["audio_flac"] is not None}
        assert flac_rows == {0, 3, 7, 11}, f"write_clips stored {sorted(flac_rows)} as FLAC, want the int16 clips"
        _check_readers(tmp, "write_clips", new, clips)
        # shards with different content share one schema, so their merge links (no second copy) ...
        sa, sb = os.path.join(tmp, "shard_a"), os.path.join(tmp, "shard_b")
        write_clips(sa, [(0, clips[0], SR), (11, clips[11], SR)])
        write_clips(sb, [(9, clips[9], SR)], monologues={9: "hm"})
        assert load_from_disk(sa).features == load_from_disk(sb).features, "shard schemas differ: the merge would copy"
        lm = os.path.join(tmp, "link_merge")
        merge_clip_datasets(lm, [sa, sb], link=True)
        assert all(os.path.islink(os.path.join(lm, f)) for f in os.listdir(lm) if f.endswith(".arrow")), "merge copied"
        _check_readers(tmp, "link_merge", lm, {i: clips[i] for i in (0, 9, 11)})
        # ... and an old float shard still merges with a new one (copied: the schemas differ)
        so = os.path.join(tmp, "shard_old")
        _old_store(so, {3: clips[3], 7: clips[7]})
        mm = os.path.join(tmp, "mixed_merge")
        before = sorted(os.listdir(so))
        merge_clip_datasets(mm, [sa, so, sb], link=True)
        # the old shard is another job's output: the cast must not leave a cache-*.arrow copy in it
        assert sorted(os.listdir(so)) == before, f"merge wrote into the old shard: {set(os.listdir(so)) - set(before)}"
        _check_readers(tmp, "mixed_merge", mm, {i: clips[i] for i in (0, 3, 7, 9, 11)})
    print(
        "check_clip_codec: OK (copies identical; float, FLAC-only, hybrid, write_clips and merged stores decode identically in 5 readers)"
    )


if __name__ == "__main__":
    main()
