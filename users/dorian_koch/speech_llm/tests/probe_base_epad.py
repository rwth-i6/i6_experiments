"""What does BASE MOSHI itself emit for PAD / EPAD / word tokens?

Our training targets are built by `interleave_text`, which we have checked against the
moshi-finetune reference frame for frame. That proves we match the reference's *code*. It
does NOT prove the reference matches the released model's own convention -- and the base
model is the thing whose pretrained behaviour we are trying not to contradict.

So: generate from the released checkpoint and read its raw text stream.

Reuses the production load + generate path (MoshiModel + InferenceState.run), so what this
observes is what the benchmark jobs observe.
"""
import os, sys, json, collections
import numpy as np
import torch

sys.path.insert(0, "/home/tt201262/setups/2026-01-speech-llm/recipe/speech_llm/full_duplex")
sys.path.insert(0, "/home/tt201262/setups/2026-01-speech-llm/recipe")

from moshi_family.moshi_engine import MoshiModel

PAD_ID, EPAD_ID = 3, 0          # run_inference.py skips exactly {0, 3} when printing
CLIPS = "/home/tt201262/setups/2026-01-speech-llm/output/rehearsal_v1/user_audio"
N_CLIPS = 4
LEAD_S, CAPTURE_S = 2.0, 25.0

print("[probe] loading base moshiko", flush=True)
model = MoshiModel("kyutai/moshiko-pytorch-bf16", device="cuda", batch_size=N_CLIPS)
sr = model.sample_rate
tok = model.state.text_tokenizer

from i6_experiments.users.dorian_koch.speech_llm.clip_store import open_clips
clips = open_clips(CLIPS)
idxs = sorted(clips.keys())[:N_CLIPS]
print(f"[probe] {len(idxs)} prompts from {CLIPS}", flush=True)

lead = np.zeros(int(LEAD_S * sr), dtype=np.float32)
tail = np.zeros(int(CAPTURE_S * sr), dtype=np.float32)
ins = []
for i in idxs:
    samples, csr = clips[i]
    a = np.asarray(samples, dtype=np.float32).reshape(-1)
    assert int(csr) == sr, f"clip {i} sr {csr} != model sr {sr}"
    ins.append(np.concatenate([lead, a, tail]))

t_max = max(len(x) for x in ins)
arr = np.zeros((N_CLIPS, 1, t_max), dtype=np.float32)
for b, x in enumerate(ins):
    arr[b, 0, : len(x)] = x

model.state.mimi.reset_streaming()
model.state.lm_gen.reset_streaming()
print("[probe] generating", flush=True)
with torch.no_grad():
    out = model.state.run(torch.from_numpy(arr).to(model.device))

def kind(t):
    return "PAD" if t == PAD_ID else ("EPAD" if t == EPAD_ID else "WORD")

agg = collections.Counter()
epad_followed_by_word = epad_total = 0
runs_started_with_epad = runs_total = 0
other_pads = collections.Counter()

for b, i in enumerate(idxs):
    ids = [int(t) for t in out[b][0].detach().cpu().tolist()]
    for t in ids:
        agg[kind(t)] += 1
        if kind(t) == "PAD" or kind(t) == "EPAD":
            other_pads[t] += 1
    # EPAD -> is the very next frame a word?
    for k, t in enumerate(ids):
        if t == EPAD_ID:
            epad_total += 1
            if k + 1 < len(ids) and kind(ids[k + 1]) == "WORD":
                epad_followed_by_word += 1
    # every word-run emerging from padding: was the frame before it EPAD?
    for k, t in enumerate(ids):
        if kind(t) == "WORD" and (k == 0 or kind(ids[k - 1]) != "WORD"):
            runs_total += 1
            if k > 0 and ids[k - 1] == EPAD_ID:
                runs_started_with_epad += 1

    if b == 0:
        print("\n--- clip %d, first 160 frames (. PAD, E EPAD, W word) ---" % i)
        print("".join("." if kind(t) == "PAD" else ("E" if kind(t) == "EPAD" else "W")
                      for t in ids[:160]))
        txt = "".join(tok.id_to_piece(t) for t in ids if t not in (PAD_ID, EPAD_ID)).replace("▁", " ")
        print("decoded:", txt.strip()[:300])
        print("\nfirst 40 frames, verbatim:")
        for k, t in enumerate(ids[:40]):
            pc = "" if t in (PAD_ID, EPAD_ID) else repr(tok.id_to_piece(t))
            print(f"   {k:3d}  id={t:<6d} {kind(t):<5s} {pc}")

print("\n================ AGGREGATE over %d clips ================" % len(idxs))
print("frame token kinds:", dict(agg))
print("distinct pad-ish ids seen:", dict(other_pads))
print()
print(f"EPAD frames                      : {epad_total}")
print(f"  ...immediately before a WORD   : {epad_followed_by_word}"
      + (f"  ({100*epad_followed_by_word/epad_total:.1f}%)" if epad_total else ""))
print(f"word-runs emerging from padding  : {runs_total}")
print(f"  ...preceded by EPAD            : {runs_started_with_epad}"
      + (f"  ({100*runs_started_with_epad/runs_total:.1f}%)" if runs_total else ""))
print()
print("INTERPRETATION")
print(" - if EPAD->WORD is ~100%, EPAD is strictly an 'about to speak' marker")
print(" - if word-runs-preceded-by-EPAD is ~100%, the model emits it before EVERY word")
print("   emerging from padding -- i.e. NOT only sentence-initially, matching our targets")
