"""A20: verify the SYNTHETIC corpus's word alignments against an independent signal.

Fisher has had this since the start (`check_fisher_windows.py`) and the synthetic corpus has never
had it -- "the corpus that collapses the model is the one with no alignment guard" (finetuning.md).
A19 is what that costs: 59.2% of QA rows planted the answer's first word a median 62 frames before
the speaker made any sound, and it survived for months because nothing looked.

The independent signal is the waveform itself: if a word's timing is right, the assistant channel
must be LOUDER inside word spans than between them. That is a property of the audio, not of the
alignment, so it cannot be satisfied by a self-consistent but wrong annotation.

Random sample with a fixed seed -- these corpora are shard merges in sorted key order, so reading
the head measures one template (see the sampling rule in CLAUDE.md).
"""
import os, sys, random
sys.path.insert(0, "recipe"); sys.path.insert(0, "recipe/speech_llm/full_duplex")
os.environ.setdefault("CUDA_HOME", "/usr")
import numpy as np
from moshi_family.train_data_common import (
    decode_duplex_row, normalize_alignments, read_arrow_table, speech_onset_sec,
)

CORPUS = sys.argv[1] if len(sys.argv) > 1 else (
    "/rwthfs/rz/cluster/home/tt201262/setups/2026-01-speech-llm/work/i6_experiments/users/"
    "dorian_koch/jobs/hf/HfMergeShards.UPwNJSN79ag0/output/merged_dataset")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
#: in-word RMS must exceed out-of-word RMS by at least this much. Fisher runs far above it; this is
#: a floor that catches a systematically shifted or scrambled annotation, not a quality target.
MIN_RATIO = 2.0
#: at most this fraction of rows may individually fall below the floor
MAX_BAD_FRAC = 0.10

table = read_arrow_table(CORPUS)
rng = random.Random(0)
n = min(N, table.num_rows)
idxs = sorted(rng.sample(range(table.num_rows), n))
print(f"corpus: {CORPUS.split('/')[-2]}")
print(f"sampling {n} of {table.num_rows} rows at random (seed 0)\n")

ratios, early_frames, used, bad = [], [], 0, 0
for i in idxs:
    assistant, _user, sr, raw = decode_duplex_row(table, i)
    al = normalize_alignments(raw)
    if not al or assistant.size == 0:
        continue
    mask = np.zeros(assistant.shape[0], dtype=bool)
    for _w, (s, e), _spk in al:
        a, b = int(max(0, s) * sr), int(min(e, assistant.shape[0] / sr) * sr)
        if b > a:
            mask[a:b] = True
    if mask.sum() == 0 or (~mask).sum() == 0:
        continue
    used += 1
    rin = float(np.sqrt((assistant[mask].astype(np.float64) ** 2).mean()))
    rout = float(np.sqrt((assistant[~mask].astype(np.float64) ** 2).mean()))
    r = rin / max(rout, 1e-9)
    ratios.append(r)
    if r < MIN_RATIO:
        bad += 1
    # A19's defect, measured directly: how far before the first sound is word 0 placed?
    onset = speech_onset_sec(assistant, sr)
    if onset is not None:
        early_frames.append((onset - al[0][1][0]) * 12.5)

ra = np.array(ratios); ef = np.array(early_frames)
print(f"rows measured: {used}")
print(f"in-word / out-of-word RMS ratio:")
print(f"   median {np.median(ra):6.1f}x   mean {ra.mean():6.1f}x   p10 {np.percentile(ra,10):5.1f}x   min {ra.min():.2f}x")
print(f"   rows below the {MIN_RATIO}x floor: {bad}/{used} ({100*bad/used:.1f}%)")
print(f"\nword 0 placed BEFORE the assistant makes a sound (frames, 12.5 Hz):")
print(f"   median {np.median(ef):+6.1f}   mean {ef.mean():+6.1f}   p90 {np.percentile(ef,90):+6.1f}")
print(f"   rows where word 0 precedes any sound: {100*(ef>0).mean():.1f}%")

fails = []
if bad / used > MAX_BAD_FRAC:
    fails.append(f"{100*bad/used:.1f}% of rows are below the {MIN_RATIO}x in/out RMS floor")
if np.median(ra) < MIN_RATIO:
    fails.append(f"median in/out RMS ratio {np.median(ra):.2f}x is below {MIN_RATIO}x")
print()
if fails:
    print("FAIL:"); [print("   -", f) for f in fails]
else:
    print("[ok] alignments agree with the waveform: words are where the sound is")
print("\nNOTE: the 'before any sound' number is the A19 defect and is EXPECTED to be large here --")
print("it is what clamp_text_to_speech exists to repair at training time. It is reported, not")
print("asserted, because the corpus on disk is not changed by that knob.")
sys.exit(1 if fails else 0)
