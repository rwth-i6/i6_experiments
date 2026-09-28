"""MFDB ``candor_turn_taking`` scorer, metric version 2 (2026-09-28).

Version 1 (the fork's ``evaluation/eval_smooth_turn_taking.py``, kept unchanged so old numbers reproduce) takes the
FIRST non-greeting VAD segment anywhere in the clip as the model's response onset. A model that says "Let me know." at
0.5 s, backchannels "Yeah." in the middle of the user's turn, or makes a remark during a user pause therefore gets a
latency of several seconds to −84 s, although its reply to the turn end starts near 0 s. Full-Duplex-Bench defines the
latency as "the time between the end of the user's speech and the start of the model's response"; upstream's code clamps
every negative latency to 0, which hides both the artefact and genuine talk-over.

Version 2 changes only which speech counts as the response:
- The response is the model speech still going at or after the turn end T: non-greeting VAD segments (same VAD as v1:
  silero, min silence 800 ms, pad 200 ms; same greeting filter; same ASR-word assignment) whose END is after T.
- Onset = start of the first such segment that holds at least one ASR word (a word-less VAD blip is not a response).
  A segment that started before T and runs past it is a real overlap at the turn end, so its negative latency stays.
- Take-turn (TOR) = the v1 rule (≥ 1 s of speech or > 3 words), applied to the response segments only.
- Speech that ENDED before T is not the response to this turn end. It is reported separately per clip:
  ``pre_turn`` (any worded pre-turn segment) and ``pre_turn_long`` (a pre-turn segment of > 3 words), so talking over
  mid-turn stays visible instead of being folded into the latency.

Run in the fdb_eval venv: ``python fdb_turn_taking_v2.py --root_dir <SpeechInference out_dir> --fork_eval_dir <fork
evaluation/> --out-json <eval.json>``. Per clip it prints the v1 and v2 values side by side.
"""

import argparse
import json
import os
import sys

METRIC_VERSION = 2
TURN_DURATION_THRESHOLD = 1.0  # same as v1
TURN_NUM_WORDS_THRESHOLD = 3  # same as v1


def response_segments(turn_end, segments):
    """The segments that make up the model's response to the turn end: those still going after it."""
    return [s for s in segments if s["end"] > turn_end]


def score_clip(turn_end, segments):
    """segments: non-greeting VAD segments in time order, each {"start", "end", "words": [str]}.

    Returns v2 ``tor``/``latency`` plus the pre-turn flags, and the v1 values for the same segments."""
    resp = response_segments(turn_end, segments)
    resp_ids = {id(s) for s in resp}
    pre = [s for s in segments if id(s) not in resp_ids and s["words"]]
    worded = [s for s in resp if s["words"]]
    cum_dur = sum(s["end"] - s["start"] for s in resp)
    cum_words = sum(len(s["words"]) for s in resp)
    take = bool(worded) and not (cum_dur < TURN_DURATION_THRESHOLD and cum_words <= TURN_NUM_WORDS_THRESHOLD)
    # v1 on the same segments (eval_smooth_turn_taking.py: first segment anywhere, cumulative over all)
    v1_dur = sum(s["end"] - s["start"] for s in segments)
    v1_words = sum(len(s["words"]) for s in segments)
    v1_take = bool(segments) and not (v1_dur < TURN_DURATION_THRESHOLD and v1_words <= TURN_NUM_WORDS_THRESHOLD)
    return {
        "tor": int(take),
        "latency": (worded[0]["start"] - turn_end) if take else None,
        "pre_turn": bool(pre),
        "pre_turn_long": any(len(s["words"]) > TURN_NUM_WORDS_THRESHOLD for s in pre),
        "pre_turn_words": sum(len(s["words"]) for s in pre),
        "tor_v1": int(v1_take),
        "latency_v1": (segments[0]["start"] - turn_end) if v1_take else None,
    }


def clip_segments(clip_dir, fork_eval_dir):
    """Non-greeting VAD segments of one clip's output.wav with their ASR words, exactly as v1 builds them."""
    if fork_eval_dir not in sys.path:
        sys.path.insert(0, fork_eval_dir)
    import soundfile as sf
    import torch
    from moshi_fixes import get_speech_timestamps, non_greeting_speech_segments

    with open(os.path.join(clip_dir, "output.json")) as f:
        chunks = json.load(f)["chunks"]
    if not chunks:  # v1: no transcription = no turn
        return []
    data, sr = sf.read(os.path.join(clip_dir, "output.wav"), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).t().contiguous()
    vad = get_speech_timestamps(wav, sr=sr, min_silence_duration_ms=800, resample=True)
    return [
        {"start": s["start"], "end": s["end"], "words": list(w)}
        for s, w in non_greeting_speech_segments(vad, chunks, verbose=False)
    ]


def _median(xs):
    return sorted(xs)[len(xs) // 2] if xs else None  # same (upper) median as v1


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def eval_smooth_turn_taking_v2(root_dir, fork_eval_dir):
    clips = sorted(d for d in os.listdir(root_dir) if os.path.exists(os.path.join(root_dir, d, "turn_taking.json")))
    if not clips:
        raise RuntimeError(f"no clips with turn_taking.json under {root_dir}")
    rows = []
    for c in clips:
        d = os.path.join(root_dir, c)
        with open(os.path.join(d, "turn_taking.json")) as f:
            turn_end = json.load(f)[0]["timestamp"][0]
        segs = clip_segments(d, fork_eval_dir)
        r = score_clip(turn_end, segs)
        r["clip"] = c
        rows.append(r)
        print(f"{d}/output.json")
        print(
            f"turn_end {turn_end:.2f} segments "
            + " | ".join(f"{s['start']:.2f}-{s['end']:.2f} {' '.join(s['words'])}" for s in segs)
        )
        print(f"v1: TOR {r['tor_v1']} latency {r['latency_v1']}")
        print(f"pre-turn speech: {int(r['pre_turn'])} long {int(r['pre_turn_long'])} words {r['pre_turn_words']}")
        print(f"the TOR is {r['tor']}")
        print(f"the latency is {r['latency']}")
    lat = [r["latency"] for r in rows if r["tor"]]
    lat1 = [r["latency_v1"] for r in rows if r["tor_v1"]]
    n = len(rows)
    res = {
        "metric_version": METRIC_VERSION,
        "n": n,
        "average_take_turn": sum(r["tor"] for r in rows) / n,
        "average_latency": _mean(lat),
        "median_latency": _median(lat),
        "share_latency_below_minus1s": (sum(x < -1 for x in lat) / len(lat)) if lat else None,
        "pre_turn_rate": sum(r["pre_turn"] for r in rows) / n,
        "pre_turn_long_rate": sum(r["pre_turn_long"] for r in rows) / n,
        # upstream Full-Duplex-Bench reports mean(max(0, latency)) over its own onset; given here on the v2 onset
        "average_latency_clamped0": _mean([max(0.0, x) for x in lat]),
        "v1_average_take_turn": sum(r["tor_v1"] for r in rows) / n,
        "v1_average_latency": _mean(lat1),
        "v1_median_latency": _median(lat1),
    }
    print("---------------------------------------------------")
    print("[Result v2]")
    for k, v in res.items():
        print(f"{k}: {v}")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True)
    p.add_argument("--fork_eval_dir", required=True)
    p.add_argument("--out-json", required=True)
    a = p.parse_args()
    out = eval_smooth_turn_taking_v2(a.root_dir, a.fork_eval_dir)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=4)
