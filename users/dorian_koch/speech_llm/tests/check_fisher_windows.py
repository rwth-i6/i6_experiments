"""Login-node guard for FisherToMoshiTrainData, driven on the corpus it actually built (no GPU, no Sisyphus graph).

The per-channel wavs this used to read (``FisherSphToWav.bb5Gjm6qeOg8``) were deleted on 2026-09-28 (211 GB, 23k
inodes). The 40 s training windows (``FisherToMoshiTrainData.E6HnLS4wyaba`` = ``output/fisher/train_windows``) embed
both channels' raw wav bytes, so the guard reads that arrow output and re-runs the REAL job methods against it (a shim
carrying the class's functions, since sisyphus overrides ``Job.__new__``):

[1] markup cleaning (fixed cases).
[2] planning: ``_plan_windows`` with the built job's own parameters (read from its ``info``) reproduces every stored
    row id in order (20,000) and the stored rejection counts; on a seeded sample of rows the stored ``alignments``
    word for word.
[3] alignment invariants on the sample (agent-only label, in-window, sorted, no markup).
[4] channel mapping is role-keyed (user -> A, agent -> B).
[5] slicing: per sampled row, the stored channels are written as that conversation's A / B wavs at the window's
    offset (seeded noise elsewhere), and ``_build_row`` must return the stored bytes exactly: channel mapping, seek
    offset, length and the PCM_16 re-encode all agree with the corpus training read.
[6] the independent signal, on the stored audio: the agent's word spans carry clearly more energy on the assistant
    channel than outside them (median in/out RMS ratio > 2) and more than the same spans on the user channel.

``--selftest``: source mutants of ``fisher_prep.py`` (channel swap in the path map or in the row, seek offset, user
words in the monologue, markup, shuffle seed, word timing) run through the same checks; each must fail.

Never registers or constructs a Sisyphus job, and never reads ``FisherSphToWav`` (its ``wav_dir`` parameter is
replaced by a temp dir). Setup ``.venv``, ~2 min (~10 min with ``--selftest``):
``CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_fisher_windows.py``
"""

import inspect
import io
import os
import random
import sys
import tempfile
import types

import numpy as np
import soundfile as sf

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")

import i6_experiments.users.dorian_koch.speech_llm.fisher_prep as fisher_prep  # noqa: E402

JOB_DIR = "work/i6_experiments/users/dorian_koch/speech_llm/fisher_prep/FisherToMoshiTrainData.E6HnLS4wyaba"
N_SAMPLE = 12
SAMPLE_SEED = 0


class FakeTkPath:
    """Stand-in for `tk.Path`, which needs a Sisyphus job context to construct."""

    def __init__(self, path):
        self.path = path

    def get_path(self):
        return self.path


def job_params(job_dir: str) -> dict:
    """The built job's parameters, from the ``PARAMETER:`` lines Sisyphus wrote into its ``info``."""
    params = {}
    with open(os.path.join(job_dir, "info")) as f:
        for line in f:
            if not line.startswith("PARAMETER: "):
                continue
            key, value = line[len("PARAMETER: ") :].rstrip("\n").split(": ", 1)
            if value == "None":
                params[key] = None
            elif key in ("cutset", "wav_dir"):
                params[key] = value
            elif key == "user_words":
                params[key] = value == "True"
            else:
                params[key] = float(value) if "." in value else int(value)
    return params


def make_job(mod, params: dict, wav_dir: str):
    shim_cls = type(
        "JobShim", (), {k: v for k, v in mod.FisherToMoshiTrainData.__dict__.items() if not k.startswith("__")}
    )
    job = shim_cls()
    job.cutset = FakeTkPath(params["cutset"])
    job.wav_dir = FakeTkPath(wav_dir)  # never the deleted FisherSphToWav output
    job.duration_sec = params["duration_sec"]
    job.stride_sec = params["stride_sec"] if params["stride_sec"] is not None else params["duration_sec"]
    job.min_agent_speech_sec = params["min_agent_speech_sec"]
    job.min_agent_words = params["min_agent_words"]
    job.windows_per_conv = params["windows_per_conv"]
    job.max_windows = params["max_windows"]
    job.seed = params["seed"]
    job.user_words = params.get("user_words", False)  # absent from info = False (declared before 2026-10-12)
    job.rqmt = {"cpu": 4}
    return job


def decode(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    samples, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    return samples, sr


def run_checks(mod, corpus: dict) -> list[str]:
    """All checks against module ``mod`` (the real fisher_prep or a mutant); returns the failures."""
    failures = []

    def check(cond, msg):
        print(f"  {'[ok]' if cond else '[FAIL]'} {msg}")
        if not cond:
            failures.append(msg)

    # [1] markup cleaning
    markup_cases = [
        ("and i generally prefer", "and i generally prefer"),
        ("[laughter] yeah [noise] okay", "yeah okay"),
        ("(( i think )) so", "i think so"),
        ("(( ))", ""),
        ("we watched t._v.", "we watched t.v."),
        ("[mn] uh huh [sigh]", "uh huh"),
    ]
    bad = [(r, mod._clean_fisher_text(r), e) for r, e in markup_cases if mod._clean_fisher_text(r) != e]
    check(not bad, f"[1] markup cleaning {len(markup_cases) - len(bad)}/{len(markup_cases)} {bad[:2]}")

    # [2] planning reproduces the stored corpus
    params = corpus["params"]
    job = make_job(mod, params, wav_dir="/nonexistent")
    windows, stats = job._plan_windows()
    planned_ids = [f"{w['conv']}_{w['start']:.1f}" for w in windows]
    stored_ids = corpus["ids"]
    first_diff = next((i for i, (a, b) in enumerate(zip(planned_ids, stored_ids)) if a != b), None)
    check(
        planned_ids == stored_ids,
        f"[2] planned ids == stored ids ({len(planned_ids)} vs {len(stored_ids)}; first diff at {first_diff})",
    )
    shared = {k: (stats[k], corpus["stats"][k]) for k in ("rejected_quiet", "rejected_short_text", "windows")}
    check(all(a == b for a, b in shared.values()), f"[2] planned stats == stored stats {shared}")
    by_id = dict(zip(planned_ids, windows))
    align_bad = []
    for row in corpus["rows"]:
        w = by_id.get(row["id"])
        if w is None:
            align_bad.append((row["id"], "not planned"))
            continue
        a, b = w["alignments"], row["alignments"]
        same = len(a) == len(b) and all(
            x["text"] == y["text"]
            and x["speaker"] == y["speaker"]
            and abs(x["start"] - y["start"]) < 1e-3
            and abs(x["end"] - y["end"]) < 1e-3
            for x, y in zip(a, b)
        )
        if not same:
            align_bad.append((row["id"], len(a), len(b)))
    check(not align_bad, f"[2] planned alignments == stored, {len(corpus['rows'])} sampled rows {align_bad[:3]}")

    # [3] alignment invariants (stored rows, which the planner just reproduced)
    inv_bad = []
    for row in corpus["rows"]:
        al = row["alignments"]
        ok = bool(al)
        ok &= all(0.0 <= a["start"] < job.duration_sec and a["end"] > a["start"] for a in al)
        ok &= all(a["start"] <= b["start"] for a, b in zip(al, al[1:]))
        ok &= all(a["speaker"] == mod.MONOLOGUE_SPEAKER for a in al)
        ok &= all(
            not set("[()_") & set(a["text"]) and a["text"] == a["text"].strip() and " " not in a["text"] for a in al
        )
        if not ok:
            inv_bad.append(row["id"])
    check(not inv_bad, f"[3] alignments agent-only, in-window, sorted, markup-free {inv_bad[:3]}")

    # [4] channel mapping is role-keyed
    paths = job._channel_wav_paths("fe00001")
    check(
        paths["user"].endswith("fe_03_00001A.wav") and paths["agent"].endswith("fe_03_00001B.wav"),
        f"[4] channel mapping user->A, agent->B {paths}",
    )

    # [5] _build_row on the stored audio returns the stored bytes
    slice_bad = []
    for k, row in enumerate(corpus["rows"]):
        w = by_id.get(row["id"])
        if w is None:
            slice_bad.append((row["id"], "not planned"))
            continue
        with tempfile.TemporaryDirectory(prefix="fisher_guard_") as tmp:
            num = w["conv"][2:]
            rng = np.random.default_rng(k)
            for ch, col in (("A", "audio_user"), ("B", "audio_assistant")):
                samples, sr = decode(row[col]["bytes"])
                off = int(round(w["start"] * sr))
                conv = rng.integers(-3000, 3000, size=off + samples.shape[0] + 5 * sr, dtype=np.int16)
                conv[off : off + samples.shape[0]] = samples
                sf.write(os.path.join(tmp, f"fe_03_{num}{ch}.wav"), conv, sr, format="WAV", subtype="PCM_16")
            job.wav_dir = FakeTkPath(tmp)
            try:
                built = job._build_row(w)
            except Exception as e:  # noqa: BLE001 -- a mutant may raise; that is a failure, not a crash
                slice_bad.append((row["id"], repr(e)[:80]))
                continue
        for col in ("audio_assistant", "audio_user"):
            if built[col]["bytes"] != row[col]["bytes"]:
                slice_bad.append((row["id"], col))
        if built["id"] != row["id"]:
            slice_bad.append((row["id"], built["id"]))
    check(not slice_bad, f"[5] _build_row reproduces the stored bytes on {len(corpus['rows'])} rows {slice_bad[:3]}")

    # [6] independent signal: the agent's words sit on the assistant channel's energy
    ratios_a, ratios_u = [], []
    for row in corpus["rows"]:
        a, sr = decode(row["audio_assistant"]["bytes"])
        u, _ = decode(row["audio_user"]["bytes"])
        mask = np.zeros(a.shape[0], dtype=bool)
        for al in row["alignments"]:
            mask[int(al["start"] * sr) : int(al["end"] * sr)] = True
        for x, out in ((a.astype(np.float64), ratios_a), (u.astype(np.float64), ratios_u)):
            rin = float(np.sqrt((x[mask] ** 2).mean())) if mask.any() else 0.0
            rout = float(np.sqrt((x[~mask] ** 2).mean())) if (~mask).any() else 1e-9
            out.append(rin / max(rout, 1e-9))
    med_a, med_u = float(np.median(ratios_a)), float(np.median(ratios_u))
    wins = sum(ra > ru for ra, ru in zip(ratios_a, ratios_u))
    check(
        med_a > 2.0 and med_a > med_u and wins >= 0.8 * len(ratios_a),
        f"[6] in/out-of-word RMS: assistant median {med_a:.2f}x, user median {med_u:.2f}x, "
        f"assistant > user on {wins}/{len(ratios_a)} rows",
    )
    return failures


def load_corpus() -> dict:
    import json

    from datasets import load_from_disk

    params = job_params(JOB_DIR)
    ds = load_from_disk(os.path.join(JOB_DIR, "output", "dataset"))
    ids = ds.data.column("id").to_pylist()
    with open(os.path.join(JOB_DIR, "output", "stats.json")) as f:
        stats = json.load(f)
    idx = sorted(random.Random(SAMPLE_SEED).sample(range(len(ids)), N_SAMPLE))
    rows = [ds[i] for i in idx]
    n_offset = sum(1 for r in rows if not r["id"].endswith("_0.0"))
    print(f"corpus {JOB_DIR}: {len(ids)} rows; params {params}")
    print(f"sample (seed {SAMPLE_SEED}): rows {idx}, {n_offset}/{N_SAMPLE} with a window start > 0")
    return {"params": params, "ids": ids, "stats": stats, "rows": rows}


#: (name, [(old, new), ...]) source mutants of fisher_prep.py; each must make run_checks fail.
MUTANTS = [
    (
        "channel swap in _channel_wav_paths",
        [
            (
                '"user": os.path.join(wav_root, f"fe_03_{conv_number}A.wav")',
                '"user": os.path.join(wav_root, f"fe_03_{conv_number}B.wav")',
            ),
            (
                '"agent": os.path.join(wav_root, f"fe_03_{conv_number}B.wav")',
                '"agent": os.path.join(wav_root, f"fe_03_{conv_number}A.wav")',
            ),
        ],
    ),
    (
        "channel swap in _build_row",
        [
            ('"audio_assistant": {"bytes": agent_wav', '"audio_assistant": {"bytes": user_wav'),
            ('"audio_user": {"bytes": user_wav', '"audio_user": {"bytes": agent_wav'),
        ],
    ),
    (
        "seek 0.5 s late",
        [
            (
                "snd.seek(int(round(window_start_sec * sample_rate)))",
                "snd.seek(int(round((window_start_sec + 0.5) * sample_rate)))",
            )
        ],
    ),
    (
        "user words in the monologue",
        [
            (
                "return self._word_alignments(agent_supervisions, win_start, win_end, MONOLOGUE_SPEAKER)",
                "return self._word_alignments([s for s in self._all_sups if s.get('speaker') != CUTSET_AGENT_SPEAKER]"
                " + agent_supervisions, win_start, win_end, MONOLOGUE_SPEAKER)",
            ),
            (
                "            if not agent_supervisions:\n                continue\n",
                "            if not agent_supervisions:\n                continue\n            self._all_sups = cut.get('supervisions', [])\n",
            ),
        ],
    ),
    ("markup: underscores kept", [('t = t.replace("_", "")', "t = t")]),
    ("shuffle seed drift", [("rng = random.Random(self.seed)", "rng = random.Random(self.seed + 1)")]),
    (
        "word onsets shifted half a word",
        [
            (
                "word_start = sup_start + word_index * sec_per_word",
                "word_start = sup_start + (word_index + 0.5) * sec_per_word",
            )
        ],
    ),
]


def mutant_module(name: str, edits: list) -> types.ModuleType:
    src = inspect.getsource(fisher_prep)
    for old, new in edits:
        assert src.count(old) == 1, (
            f"mutant {name!r}: {old!r} occurs {src.count(old)}x in fisher_prep.py (stale mutant)"
        )
        src = src.replace(old, new)
    mod = types.ModuleType(f"fisher_prep_mutant_{abs(hash(name))}")
    mod.__file__ = fisher_prep.__file__
    exec(compile(src, fisher_prep.__file__, "exec"), mod.__dict__)
    return mod


def main():
    corpus = load_corpus()
    print("\n== real fisher_prep ==")
    failures = run_checks(fisher_prep, corpus)
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")
    if "--selftest" not in sys.argv:
        return
    caught = []
    for name, edits in MUTANTS:
        print(f"\n== mutant: {name} ==")
        fails = run_checks(mutant_module(name, edits), corpus)
        caught.append(bool(fails))
        print(f"  -> {'CAUGHT' if fails else 'MISSED'} ({len(fails)} failing checks)")
    print(f"\nselftest: {sum(caught)}/{len(caught)} mutants caught")
    if not all(caught):
        sys.exit(1)


if __name__ == "__main__":
    main()
