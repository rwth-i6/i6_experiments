"""Build the German per-phone mean-log-mel + duration tables. RUNS ON FZJ.

Split of labour: MFA aligns on RZ (x86-only, no aarch64 kalpy build exists); the TABLE is computed
here on FZJ because it must use RETURNN's *exact* `log_mel_filterbank_from_raw`. RZ has no
tools/returnn checkout and lacks dm-tree, and reimplementing the filterbank would risk the German
table not living in the same feature space as the English one -- which would silently invalidate
every comparison. TextGrids are a few KB each, so shipping them is trivial.

Mirrors ComputeMfaPhoneMeanLogMelJob.run() on every detail that affects the numbers:
  peak-normalise -> log_mel_filterbank_from_raw(16 kHz, 25 ms, 10 ms, 80 mel)
  frame centres at t*step + window/2, phone by [start, end)
  uncovered frames -> [space], spn -> [UNKNOWN], labels with no frames -> global mean

Audio comes from the MLS parquet already staged on FZJ (soundfile, not the datasets Audio feature --
datasets>=4 decodes via a torchcodec that is broken here).

Output contract (aed_glowtts_model_def :4947, :4950): means float32 [V,80], labels [V], `[space]` present.

Usage: build_de_table_fzj.py <textgrid_dir> <parquet> <dict> <out_prefix>
"""
import glob
import io
import json
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

# RETURNN is a CHECKOUT in the FZJ setup, not an installed package, so insert it explicitly.
# table_vs_hours.py does the same; without this the import fails.
sys.path.insert(0, "/e/home/jusers/koch13/jupiter/setups/2026-09-16-fzj-dlm/tools/returnn")
import returnn.frontend as rf
from returnn.tensor import Dim, Tensor, batch_dim

rf.select_backend_torch()
batch_dim.dyn_size_ext = rf.convert_to_tensor(torch.tensor(1, dtype=torch.int32), dims=[])

TG_DIR = sys.argv[1]
PARQ = sys.argv[2]
DICT = sys.argv[3]
OUT = sys.argv[4]

SR, WIN, STEP, NMEL = 16000, 0.025, 0.010, 80
SIL_LAB, UNK_LAB = "[space]", "[UNKNOWN]"

phones = set()
with open(DICT, encoding="utf8") as fh:
    for ln in fh:
        parts = ln.rstrip("\n").split("\t")
        if len(parts) >= 2:
            phones.update(p for p in parts[-1].split() if p)
phones.discard("spn")
# Match the English vocab convention exactly (ReturnnVocabFromPhonemeInventory.z2RlZd9Y0jWQ):
# sorted(phones + [UNKNOWN],[end],[space],[start]) then [blank] appended LAST.
# Diverging here would not break the model (it finds [space] by name) but is how a silent
# index mismatch between table rows and phonemizer output would later creep in.
labels = sorted(list(phones) + [UNK_LAB, "[end]", SIL_LAB, "[start]"]) + ["[blank]"]
idx = {l: i for i, l in enumerate(labels)}
SIL, UNK = idx[SIL_LAB], idx[UNK_LAB]
NL = len(labels)
print(f"vocab: {NL} labels ({len(phones)} phones + 5 specials)", flush=True)

out_dim = Dim(NMEL, name="mel")


def log_mel(a):
    peak = np.max(np.abs(a))
    if peak != 0.0:
        a = a / peak
    raw = torch.tensor(a[None, :], dtype=torch.float32)
    td = Dim(int(raw.shape[1]), name="time")
    src = Tensor("audio", dims=[batch_dim, td], dtype="float32", raw_tensor=raw)
    f, fd = rf.audio.log_mel_filterbank_from_raw(
        src, in_spatial_dim=td, out_dim=out_dim, sampling_rate=SR, window_len=WIN, step_len=STEP
    )
    return f.copy_compatible_to_dims_raw([batch_dim, fd, out_dim])[0].numpy()


_IV = re.compile(
    r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"', re.S)


def phones_of(path):
    s = open(path, encoding="utf8").read()
    tier = None
    for t in re.split(r"item\s*\[\d+\]:", s):
        m = re.search(r'name\s*=\s*"([^"]*)"', t)
        if m and "phone" in m.group(1).lower():
            tier = t
    return [] if tier is None else [(float(a), float(b), c.strip()) for a, b, c in _IV.findall(tier)]


tbl = pq.read_table(PARQ)
cols = {c.lower(): c for c in tbl.column_names}
acol = next(c for lc, c in cols.items() if "audio" in lc)
icol = next(c for lc, c in cols.items() if lc == "id")
audio_by_id = dict(zip(tbl.column(icol).to_pylist(), tbl.column(acol).to_pylist()))
print(f"parquet: {tbl.num_rows} rows", flush=True)

tgs = sorted(glob.glob(os.path.join(TG_DIR, "**", "*.TextGrid"), recursive=True))
print(f"textgrids: {len(tgs)}", flush=True)

sums = np.zeros((NL, NMEL), dtype=np.float64)
counts = np.zeros((NL,), dtype=np.int64)
durs = {l: [] for l in labels}
n_seq = n_miss = 0
unmapped = {}

for tg in tgs:
    uid = os.path.basename(tg)[: -len(".TextGrid")]
    a = audio_by_id.get(uid)
    if a is None:
        n_miss += 1
        continue
    ivs = phones_of(tg)
    if not ivs:
        n_miss += 1
        continue
    raw = a["bytes"] if isinstance(a, dict) else a
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    assert sr == SR, (uid, sr)
    feats = log_mel(np.asarray(arr, dtype=np.float32))
    tc = np.arange(feats.shape[0]) * STEP + WIN / 2
    fl = np.full((feats.shape[0],), SIL, dtype=np.int64)
    for s, e, lab in ivs:
        if lab in ("", "sil", "sp"):
            i = SIL
        elif lab == "spn":
            i = UNK
        elif lab in idx:
            i = idx[lab]
        else:
            unmapped[lab] = unmapped.get(lab, 0) + 1
            continue
        fl[(tc >= s) & (tc < e)] = i
        durs[labels[i]].append(e - s)
    np.add.at(sums, fl, feats.astype(np.float64))
    np.add.at(counts, fl, 1)
    n_seq += 1
    if n_seq % 200 == 0:
        print(f"  {n_seq}/{len(tgs)}", flush=True)

print(f"\nused {n_seq}, skipped {n_miss}")
if unmapped:
    print(f"unmapped labels: {unmapped}")

gm = sums.sum(0) / max(int(counts.sum()), 1)
means = np.zeros((NL, NMEL), dtype=np.float32)
for i in range(NL):
    means[i] = (sums[i] / counts[i]) if counts[i] > 0 else gm
np.savez(OUT + ".npz", means=means, labels=np.array(labels, dtype=object))

med = {l: (float(np.median(durs[l]) / STEP) if durs[l] else 1.0) for l in labels}
np.savez(OUT + "_durations.npz",
         medians=np.array([med[l] for l in labels], dtype="float32"),
         labels=np.array(labels, dtype=object))

nz = counts > 0
zero = [labels[i] for i in range(NL) if counts[i] == 0]
print(f"labels with NO frames (-> global mean): {len(zero)} {zero}")
print(f"frames/label: min {counts[nz].min()} median {int(np.median(counts[nz]))} max {counts.max()}")
print("rarest:", sorted(((int(counts[i]), labels[i]) for i in range(NL) if counts[i] > 0))[:6])
json.dump({"n_seq": n_seq, "n_missing": n_miss, "labels": labels,
           "frame_counts": {labels[i]: int(counts[i]) for i in range(NL)},
           "duration_medians_frames": med, "unmapped": unmapped},
          open(OUT + "_stats.json", "w"), indent=1, ensure_ascii=False)
print(f"wrote {OUT}.npz / _durations.npz / _stats.json")
