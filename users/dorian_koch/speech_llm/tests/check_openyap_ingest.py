"""Guard: the Open Yap ingest (``moshi_family/openyap_common.py`` + ``openyap_ingest_main.py``) produces EPISODE rows
the JRE pipeline's downstream stages read correctly.

WHY. Open Yap skips diarization / separation / ASR, so four things the JRE path gets from its own stages are
built here instead, each a silent-corruption risk: (a) the archive is streamed, never extracted -- a conversation
split across the stream, or the wrong conversation's bytes, would look like valid audio; (b) speaker segments come
from the vendor words (``ipu_turns``) and feed DuplexChat's slicer and ``select_windows`` as if they were a
diarization; (c) track a MUST be channel 0 in the codes, in ``asr_json`` and in ``channel_speakers_json`` -- the
slicer cuts ``words_a`` from asr channel 0, and a swap would train the assistant on its partner's words, audibly
fine and wrong; (d) the tracks are ~-45 LUFS and need ~+20 dB: ``normalize_per_track`` must land them at -24 LUFS.
Checked:
  1. ``iter_conversations`` on a synthetic tar.gz of the vendor layout: every conversation once, in archive order,
     with its own bytes; ``want`` / ``want_audio`` / ``stop_after`` select; a non-contiguous conversation raises.
  2. ``clean_words`` keeps only ``type == "word"`` with text and valid times, sorted; ``ipu_turns`` joins a pause
     < gap and splits at >= gap; ``conversation_turns`` merges both speakers sorted; ``bin_pack`` / ``seeded_unit``
     are deterministic and balanced.
  3. ``normalize_per_track`` brings a -45 LUFS track (true peak -16 dBFS, like the first archive conversation) to
     -24 +-0.5 LU with ONE gain (>= 99% of samples at one output/input ratio; the rest is the limiter) and no boosted sample above -1 dBFS; a
     track below ``min_lufs`` is untouched. Can fail: the one-gain ``per_channel`` mode (peak cap) misses the target
     by > 3 LU on the same track -- the reason per_track exists.
  4. The real worker (subprocess, CPU mimi, our ffmpeg) on a 2-conversation synthetic archive: the part's columns are
     exactly ``part_features("episode")``; frames = 12.5 Hz x duration; labels = vendor speaker ids in track order;
     asr channel 0 = track a's words; the recorded per-track output loudness is -24 +-1 LU; one reference clip for
     the speaker with >= 60 s of exclusive speech only; one stored user variant.
  4b. The worker writes ``progress.json`` into its cwd (the job work dir the manager reads), done == total. A resumed
     shard whose one remaining conversation fails every time passes the failure gate when that is within
     ``max_failure_frac`` of the SHARD (a gate on the resumed remainder could never pass again).
  5. DuplexChat's slicer (``podcast_slice_main``, pinned clone) on that part: the dialogue rows' ``words_a`` are track
     a's words (re-based), ``turns_json`` carries the two speaker ids. Can fail: the same check on a copy of the part
     whose ``asr_json`` channels are swapped FAILS.

moshi_family venv, CPU, ~2 min (setup root):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_openyap_ingest.py'
"""

import glob
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.getcwd()
LIB = os.path.join(ROOT, "recipe", "speech_llm", "full_duplex")
sys.path.insert(0, LIB)

import numpy as np  # noqa: E402
import pyloudnorm  # noqa: E402
import soundfile as sf  # noqa: E402

spec = importlib.util.spec_from_file_location("openyap_common", os.path.join(LIB, "moshi_family", "openyap_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
spec = importlib.util.spec_from_file_location("oyi", os.path.join(LIB, "moshi_family", "openyap_ingest_main.py"))
OYI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OYI)

FFMPEG = sorted(
    glob.glob(
        os.path.join(ROOT, "work/i6_experiments/users/dorian_koch/speech_llm/tts/InstallFFmpeg.51Cp89ANVlZ9/output/out")
    )
)
DUPLEXCHAT_SRC = sorted(
    glob.glob(os.path.join(ROOT, "work/i6_core/tools/git/CloneGitRepositoryJob.al9vaaHw0SWr/output/DuplexChat/src"))
)
SR48 = 48000
rng = np.random.default_rng(0)
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def speechlike(sec, sr):
    n = int(sec * sr)
    env = np.repeat(rng.random(int(sec * 4) + 1) > 0.3, sr // 4 + 1)[:n]
    t = np.arange(n) / sr
    x = (0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * rng.normal(0, 1, n)) * env
    return x.astype(np.float32)


def at_lufs(x, lufs, sr):
    m = pyloudnorm.Meter(sr).integrated_loudness(x.astype(np.float64))
    return (x * 10 ** ((lufs - m) / 20)).astype(np.float32)


def track(spans, dur, sr=SR48, lufs=-45.0):
    """Speech-like signal inside ``spans`` (s), silence elsewhere, at ``lufs``."""
    x = np.zeros(int(dur * sr), dtype=np.float32)
    for a, b in spans:
        x[int(a * sr) : int(a * sr) + int((b - a) * sr)] = speechlike(b - a, sr)[: int((b - a) * sr)]
    return at_lufs(x, lufs, sr)


def words_for(spans, tag):
    """One word every 0.4 s inside each span (text = tag + index): what the vendor table would hold."""
    out = []
    for a, b in spans:
        t = a
        while t + 0.3 <= b:
            out.append({"word": f"{tag}{len(out)}", "start": round(t, 3), "end": round(t + 0.3, 3), "type": "word"})
            t += 0.4
    return out


def flac(x, sr=SR48):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def conversation(cid, dur, spans_a, spans_b, spk=("spk_a1", "spk_b1")):
    xa, xb = track(spans_a, dur), track(spans_b, dur)
    wa, wb = words_for(spans_a, "alpha"), words_for(spans_b, "beta")
    # a non-word event and an empty word: must never reach the text stream
    wa.append({"word": "[laugh]", "start": 1.0, "end": 1.2, "type": "laugh"})
    wb.append({"word": "  ", "start": 2.0, "end": 2.1, "type": "word"})
    meta = {
        "conversation_id": cid,
        "language": "en",
        "duration_seconds": dur,
        "speaker_a_id": spk[0],
        "speaker_b_id": spk[1],
    }
    files = {
        "speaker_a.flac": flac(xa),
        "speaker_b.flac": flac(xb),
        "meta.json": json.dumps(meta).encode(),
    }
    for t, s, w in (("a", spk[0], wa), ("b", spk[1], wb)):
        files[f"speaker_{t}_meta.json"] = json.dumps(
            {"speaker_id": s, "recording": {"integrated_lufs": -45.0}}
        ).encode()
        files[f"speaker_{t}_transcript.json"] = json.dumps(
            {"speaker_index": t, "corrections_applied": False, "words": w}
        ).encode()
        files[f"speaker_{t}_dnsmos.json"] = json.dumps({"ovr": "[3.1, None]"}).encode()
    return files, wa, wb


def write_archive(path, convs, order=None):
    with tarfile.open(path, "w:gz") as tf:
        members = [(cid, name, data) for cid, files in convs for name, data in files.items()]
        for cid, name, data in order(members) if order else members:
            ti = tarfile.TarInfo(f"conversations/{cid}/{name}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def main():
    tmp = tempfile.mkdtemp(prefix="oy_guard_", dir="/tmp")
    # conv 1: 120 s, A has ~62 s of exclusive speech (gets a reference clip), B ~53 s; conv 2: 40 s, nobody >= 60 s
    c1, wa1, wb1 = conversation(
        "conv_one", 120.0, [(0.5, 35.0), (45.0, 75.0)], [(34.0, 46.0), (74.5, 118.0)], spk=("spk_a1", "spk_b1")
    )
    c2, wa2, wb2 = conversation("conv_two", 40.0, [(0.5, 18.0)], [(19.0, 39.0)], spk=("spk_x", "spk_x"))
    arch = os.path.join(tmp, "oy.tar.gz")
    write_archive(arch, [("conv_one", c1), ("conv_two", c2)])

    # ---- [1] streaming -------------------------------------------------------------------------------------------
    got = list(C.iter_conversations(arch))
    check("[1] all conversations, archive order", [g[0] for g in got] == ["conv_one", "conv_two"])
    check(
        "[1] own bytes",
        got[0][1]["speaker_a.flac"] == c1["speaker_a.flac"] and got[1][1]["meta.json"] == c2["meta.json"],
    )
    got = list(C.iter_conversations(arch, want={"conv_two"}, want_audio=False))
    check(
        "[1] want + want_audio=False",
        [g[0] for g in got] == ["conv_two"] and "speaker_a.flac" not in got[0][1] and "meta.json" in got[0][1],
    )
    got = list(C.iter_conversations(arch, stop_after=1))
    check("[1] stop_after", [g[0] for g in got] == ["conv_one"])
    bad = os.path.join(tmp, "bad.tar.gz")
    write_archive(bad, [("conv_one", c1), ("conv_two", c2)], order=lambda m: m[:3] + m[9:] + m[3:9])
    try:
        list(C.iter_conversations(bad))
        check("[1] non-contiguous conversation raises", False)
    except RuntimeError:
        check("[1] non-contiguous conversation raises", True)

    # ---- [2] words / segments / shards -------------------------------------------------------------------------
    ws, tally = C.clean_words(
        [
            {"word": "b", "start": 1.0, "end": 1.2, "type": "word"},
            {"word": "a", "start": 0.5, "end": 0.7, "type": "word"},
            {"word": "[cough]", "start": 0.8, "end": 0.9, "type": "cough"},
            {"word": " ", "start": 0.9, "end": 1.0, "type": "word"},
            {"word": "x", "start": 2.0, "end": 1.0, "type": "word"},
        ]
    )
    check(
        "[2] clean_words",
        [w["text"] for w in ws] == ["a", "b"]
        and tally["dropped_type"] == 1
        and tally["dropped_empty"] == 1
        and tally["dropped_time"] == 1,
        str(tally),
    )
    w3 = [
        {"text": "p", "start": 0.0, "end": 0.5},
        {"text": "q", "start": 0.79, "end": 1.0},
        {"text": "r", "start": 1.3, "end": 1.5},
    ]
    segs = C.ipu_turns(w3, "S", 0.3)
    check(
        "[2] ipu_turns joins < gap, splits >= gap",
        [(s["start"], s["end"]) for s in segs] == [(0.0, 1.0), (1.3, 1.5)],
        str(segs),
    )
    t2 = C.conversation_turns({"A": w3, "B": [{"text": "z", "start": 0.6, "end": 0.7}]}, 0.3)
    check("[2] conversation_turns sorted, both speakers", [t["speaker"] for t in t2] == ["A", "B", "A"])
    items = [(f"c{i}", float(s)) for i, s in enumerate(rng.integers(100, 7000, 200))]
    b1, b2 = C.bin_pack(items, 4), C.bin_pack(list(reversed(items)), 4)
    loads = [sum(dict(items)[c] for c in b) for b in b1]
    check(
        "[2] bin_pack deterministic + balanced",
        b1 == b2 and max(loads) - min(loads) <= 7000,
        str([round(x) for x in loads]),
    )
    check(
        "[2] seeded_unit is a pure function",
        C.seeded_unit(0, "conv_one") == C.seeded_unit(0, "conv_one") != C.seeded_unit(1, "conv_one"),
    )

    # ---- [3] loudness ------------------------------------------------------------------------------------------
    from moshi_family.podcast_mimi_encode import MIMI_SR, apply_loudness_norm

    meter = pyloudnorm.Meter(MIMI_SR)
    x = track([(0.5, 25.0)], 30.0, sr=MIMI_SR, lufs=-45.0)
    x[int(5 * MIMI_SR)] = 10 ** (-16 / 20)  # a -16 dBFS click, like the true peak of the first archive track
    silent = np.zeros_like(x)
    cfg = {
        "target_lufs": -24.0,
        "max_gain_db": 30.0,
        "min_lufs": -70.0,
        "mode": "per_track",
        "limiter_ceiling_dbfs": -1.0,
    }
    ya, yb, rec = OYI.normalize_per_track(x, silent, cfg)
    out = meter.integrated_loudness(ya.astype(np.float64))
    nz = x != 0
    ratio = ya[nz] / x[nz]
    same = float(np.mean(np.abs(ratio / np.median(ratio) - 1.0) < 1e-4))
    check("[3] per_track: -45 -> -24 LUFS", abs(out + 24.0) <= 0.5, f"out {out:.2f}, gain {rec['a']['gain_db_median']}")
    # one gain: every sample the limiter did not touch has the same output/input ratio
    check("[3] per_track: one gain (>= 99% of samples at the median ratio)", same >= 0.99, f"{same:.4f}")
    check(
        "[3] per_track: no sample above -1 dBFS",
        float(np.abs(ya).max()) <= 10 ** (-1 / 20) + 1e-4,
        f"peak {np.abs(ya).max():.4f}",
    )
    check("[3] silent track untouched", np.array_equal(yb, silent))
    pa, _, prec = apply_loudness_norm(x, silent, {"target_lufs": -24.0, "max_gain_db": 30.0, "min_lufs": -70.0})
    miss = abs(meter.integrated_loudness(pa.astype(np.float64)) + 24.0)
    check(
        "[3] can fail: per_channel (peak cap) misses the target by > 3 LU",
        miss > 3.0,
        f"miss {miss:.1f} LU ({prec['a']['reason']})",
    )

    # ---- [4] the worker -----------------------------------------------------------------------------------------
    if not FFMPEG or not DUPLEXCHAT_SRC:
        check("[4] ffmpeg + DuplexChat clone present", False, f"{FFMPEG} {DUPLEXCHAT_SRC}")
        return
    shard = os.path.join(tmp, "shard.jsonl")
    with open(shard, "w") as f:
        for cid, d in (("conv_one", 120.0), ("conv_two", 40.0)):
            f.write(json.dumps({"id": cid, "duration_sec": d}) + "\n")
    out_dir = os.path.join(tmp, "codes")
    env = dict(os.environ)
    env["PYTHONPATH"] = LIB + os.pathsep + os.path.join(ROOT, "recipe")
    env["PATH"] = f"{FFMPEG[0]}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{FFMPEG[0]}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("HF_HOME", "/hpcwork/p0023999/common_hf_home")
    env.setdefault("HF_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))
    env["HF_HUB_OFFLINE"] = "1"
    aug = {
        "variants": 1,
        "seed": 0,
        "augment": {"gain_prob": 1.0, "noise_prob": 0.0, "echo_reverb_prob": 0.0, "clip_peak": 1.0},
    }
    subprocess.run(
        [sys.executable, os.path.join(LIB, "moshi_family", "openyap_ingest_main.py"), "--archive", arch,
         "--common", os.path.join(LIB, "moshi_family", "openyap_common.py"), "--shard_jsonl", shard,
         "--out_dir", out_dir, "--ffmpeg_dir", FFMPEG[0], "--gap", "0.5", "--loudness_norm", json.dumps(cfg),
         "--user_augment", json.dumps(aug), "--device", "cpu", "--convs_per_part", "8"],
        env=env, check=True, cwd=tmp,
    )  # fmt: skip
    prog = json.load(open(os.path.join(tmp, "progress.json")))
    check("[4b] progress.json in the worker cwd, done == total", prog["done"] == prog["total"] == 2, str(prog))
    from datasets import load_from_disk

    from moshi_family.podcast_mimi_encode import part_features

    parts = sorted(glob.glob(os.path.join(out_dir, "parts", "part_*")))
    ds = load_from_disk(parts[0])
    check("[4] one part, two rows", len(parts) == 1 and ds.num_rows == 2)
    check("[4] columns == part_features('episode')", set(ds.column_names) == set(part_features("episode")))
    rows = {r["item_id"]: r for r in ds}
    for cid, dur, wa, spk in (
        ("conv_one", 120.0, wa1, ["spk_a1", "spk_b1"]),
        ("conv_two", 40.0, wa2, ["spk_x_a", "spk_x_b"]),
    ):
        r = rows[cid]
        check(f"[4] {cid} frames = 12.5 Hz x duration", abs(r["n_frames"] - dur * 12.5) <= 1, f"{r['n_frames']}")
        check(
            f"[4] {cid} labels in track order",
            json.loads(r["channel_speakers_json"]) == spk and json.loads(r["speaker_labels_json"]) == spk,
        )
        asr = json.loads(r["asr_json"])
        want_a = [w["word"] for w in sorted(wa, key=lambda w: w["start"]) if w["type"] == "word"]
        check(f"[4] {cid} asr channel 0 = track a words", [w["text"] for w in asr["channels"][0]["words"]] == want_a)
        loud = json.loads(r["channel_info"])["loudness"]
        lo = [loud[c]["chunk_lufs_out"][0] for c in ("a", "b")]
        check(
            f"[4] {cid} recorded output loudness -24 +-1", all(v is not None and abs(v + 24) <= 1 for v in lo), str(lo)
        )
        check(f"[4] {cid} one user variant", r["n_user_aug"] == 1 and len(r["codes_a_aug"]) == len(r["codes_a"]))
        codes = np.asarray(r["codes_a"])
        check(f"[4] {cid} codes in range", codes.min() >= 0 and codes.max() < 2048)
    refs = json.loads(rows["conv_one"]["speaker_ref_meta_json"])
    check(
        "[4] reference clip only for >= 60 s exclusive",
        [m["speaker"] for m in refs] == ["spk_a1"] and not json.loads(rows["conv_two"]["speaker_ref_meta_json"]),
        str(refs),
    )

    # resume: conv_one is done (its part exists); conv_two now fails on every run (corrupt audio). 1 of 2 = 50%.
    c2bad = {**c2, "speaker_a.flac": b"not a flac"}
    bad_arch = os.path.join(tmp, "oy_bad.tar.gz")
    write_archive(bad_arch, [("conv_one", c1), ("conv_two", c2bad)])
    resume_out = os.path.join(tmp, "codes_resume")
    os.makedirs(os.path.join(resume_out, "parts"))
    one = ds.filter(lambda r: r["item_id"] == "conv_one")
    one.save_to_disk(os.path.join(resume_out, "parts", "part_00000"))
    with open(os.path.join(resume_out, "parts", "part_00000", "episodes.json"), "w") as f:
        json.dump(["conv_one"], f)
    rc = subprocess.run(
        [sys.executable, os.path.join(LIB, "moshi_family", "openyap_ingest_main.py"), "--archive", bad_arch,
         "--common", os.path.join(LIB, "moshi_family", "openyap_common.py"), "--shard_jsonl", shard,
         "--out_dir", resume_out, "--ffmpeg_dir", FFMPEG[0], "--gap", "0.5", "--device", "cpu",
         "--convs_per_part", "8", "--max_failure_frac", "0.5"],
        env=env, cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode  # fmt: skip
    check("[4b] resumed shard: 1 failure of 2 passes a 50% gate (was 1 of the 1 remaining)", rc == 0, f"rc {rc}")

    # ---- [5] DuplexChat's slicer on the part --------------------------------------------------------------------
    def sliced_words_ok(in_dir):
        sl = os.path.join(tmp, "slice_" + os.path.basename(in_dir))
        subprocess.run(
            [sys.executable, os.path.join(LIB, "moshi_family", "podcast_slice_main.py"), "--in_dir", in_dir,
             "--out_dir", sl, "--repo_src", DUPLEXCHAT_SRC[0], "--gap_seconds", "5", "--max_single_speaker_ratio", "0.8",
             "--min_duration_seconds", "10", "--max_duration_seconds", "600", "--min_dialogues_per_episode", "1",
             "--rows_per_part", "100"],
            env=env, check=True, stdout=subprocess.DEVNULL,
        )  # fmt: skip
        rows_s = [r for p in sorted(glob.glob(os.path.join(sl, "parts", "part_*"))) for r in load_from_disk(p)]
        ok = bool(rows_s)
        for r in rows_s:
            wa_ = json.loads(r["words_a"] or "[]")
            ok &= bool(wa_) and all(w["text"].startswith("alpha") for w in wa_)
            ok &= {t["speaker"] for t in json.loads(r["turns_json"])} <= set(
                json.loads(rows[r["episode_id"]]["channel_speakers_json"])
            )
        return ok, len(rows_s)

    ok, n = sliced_words_ok(out_dir)
    check("[5] slicer: words_a = track a, turns = speaker ids", ok, f"{n} dialogue rows")
    # can fail: the same part with the asr channels swapped
    swapped = os.path.join(tmp, "swapped")
    os.makedirs(os.path.join(swapped, "parts"))

    def swap(r):
        a = json.loads(r["asr_json"])
        for ch in a["channels"]:
            ch["channel"] = 1 - ch["channel"]
        r["asr_json"] = json.dumps(a)
        return r

    ds.map(swap).save_to_disk(os.path.join(swapped, "parts", "part_00000"))
    with open(os.path.join(swapped, "parts", "part_00000", "episodes.json"), "w") as f:
        json.dump(["conv_one", "conv_two"], f)
    ok, _ = sliced_words_ok(swapped)
    check("[5] can fail: swapped asr channels are caught", not ok)


if __name__ == "__main__":
    main()
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
