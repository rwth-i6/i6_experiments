# Long retained SIL runs after rVAD: analysis script (2026-09-24)

Status: DONE_WITH_CONCERNS. The script is written and tested. It cannot yet run on its real inputs:
- The VAD job `alias/sae/4a/data/vad` (BlankfreeVadHdfJob.RLrgIh6lFv9m) has not been created on disk.
  Its input `AssignUnitsJob.OqzzdiNBxGiO` was still unfinished when checked.
- The train-clean-100 MFA parquets (`get_mfa_alignments("train-clean-100")`) are in no graph and are
  not downloaded, so the train split cannot run.

Preliminary dev numbers exist. They use keep masks recomputed with the package's own VAD code, not the
job's HDFs (see below).

## Script
`analysis/long_sil_after_vad.py` (setup-local; read-only; the package is imported, not edited).

Constants are read from the package:
- stride 3: `training/config.py:48-50` (NET_ARGS) and `model/blankfree_model.py:465-468,499`
  (`recognizer_stride=3`);
- recognizer frame t sits at unit 3t: `model/lattice.py:916`; T = ceil(S/3): `model/recognizer.py:39-40,86-88`;
- W = 25: `EMC_BAND`, `training/config.py:61,310`, then `model/emc_model.py:358-359`;
- cap = floor(2W/3)+1 = 17: `model/blankfree_model.py:405-407`.

Mapping:
- The package keeps `raw_index = flatnonzero(~silence[:original])` and concatenates the kept frames
  (`data/vad.py:144-147,205`).
- 50 Hz frame i is gold silence iff its centre 0.02 i + 0.01 s lies in no MFA phone whose
  `canonical_phone` is not SIL. So gaps and `spn` count as silence (`phones.py:48-54`).
- Runs are maximal silent stretches of `silent[raw_index]`. Kept pieces that become adjacent merge.
- A run [a, b) of kept units is ceil(b/3) - ceil(a/3) recognizer frames: the frames t with 3t in [a, b).
- Before VAD, the same rules apply on 0..original-1.
- Each long run is labelled leading, internal, trailing, or "all" (the whole utterance is silent).
- The train split is restricted to `cv_split/output/train.segments` (28,254 utterances, the recognizer's
  training set, `inputs.py:164-178`).
- `pre_vad_implementer_replication` repeats the implementer's count: internal inter-phone gaps longer
  than 1.0 s.
- `--recompute-vad OGG_ZIP FEATURE_HDF` is a test-only path. It uses the logic of `data/vad.py:128-147`:
  rVADfast threshold 0.4, subframes 2, padded or cut to the feature-HDF length.

## Run command (once the VAD job is finished), from the setup dir
    /work/asr4/hwu/conda/envs/sae/bin/python analysis/long_sil_after_vad.py --out analysis/out/long_sil_after_vad \
        --splits dev-clean dev-other train --mfa-train-dir <snapshot dir of get_mfa_alignments("train-clean-100")>
- Without `--mfa-train-dir`, the train split is skipped and a note says why.
- The script refuses output paths under work/, /work/asr4/hwu/setups/ or recipe/.
- The expected runtime is a few minutes: it reads HDFs only, with no rVAD. This is not measured.

## Checks
- 20 dev-other utterances, recomputed masks, 8 s (`analysis/out/test20.*`): no run > 17. The maximum
  is 6 recognizer frames after VAD and 15 before.
- The implementer's pre-VAD figures are reproduced exactly:
  - dev-other: 76 gaps > 1 s in 62 of 2864 utterances (2.16%), maximum 2.11 s (105.5 units);
  - dev-clean: 39 gaps in 35 of 2703 utterances (1.29%), maximum 1.59 s (79.5 units).
- Hand check against the per-utterance dump (`analysis/out/test300d.json`, `dumps`):
  - 116-288045-0000 (532 frames, 469 kept; kept pieces [19,146) and [179,521)):
    - The leading silence [0,25) becomes kept [0,6).
    - The gap 1.74-1.87 s becomes frames [87,93), then kept [68,74).
    - The gap 3.07-3.55 s ([153,177)) falls entirely in the dropped stretch 146-178, so no run remains.
    - The gap at [323,325) becomes kept 127+144 = [271,273).
    - The trailing silence [519,532) becomes kept [467,469).
    - The recognizer-frame counts match by hand: [0,25) gives 9 and [427,429) gives 0.
  - 1255-138279-0000: the `spn` at 10.51-11.69 s plus the gap to 12.18 s is silence [525,609).
    - Kept pieces [474,590) and [604,633) map orig 525..589 to kept 429..493 and orig 604..608 to
      kept 494..498.
    - These merge into one run [429,499): 70 units, ceil(499/3) - ceil(429/3) = 24 recognizer frames,
      which is > 17. The script reports the same.

## Preliminary dev numbers (recomputed masks, NOT the VAD job output)
Files: `analysis/out/full_dev{other,clean}_recomp.{json,txt}`. Wall time 217 s and 231 s, dominated by rVAD.

| split | stage | utts | % utts with run > 17 | runs (lead/int/trail) | excess share of rec frames | max run |
|---|---|---|---|---|---|---|
| dev-other | post-VAD | 2864 | 1.22 | 38 (11/23/4) | 0.123 % | 32 |
| dev-other | pre-VAD | 2864 | 3.98 | 137 (18/103/15, +1 all-silent) | 0.279 % | 54 |
| dev-clean | post-VAD | 2703 | 0.33 | 10 (1/7/2) | 0.017 % | 28 |
| dev-clean | pre-VAD | 2703 | 2.26 | 67 (4/57/6) | 0.091 % | 41 |

The pre-VAD shares are larger than the implementer's 2.2 % / 1.3 %. Here edges and `spn` count, and
the threshold is 17 recognizer frames rather than a gap longer than 1.0 s.

## Undetermined / concerns
- The mask source is unconfirmed. The recomputed masks follow the job's code on the same zips and
  feature HDFs, but I did not compare them with the job output (it does not exist yet). The final
  numbers must come from the run command above.
- The train split needs the train-clean-100 MFA download, a sisyphus job that is not yet in any graph.
- `spn` counts as silence (the package's gold convention). The dev-other maximum post-VAD case
  (1255-138279-0000) is mostly `spn`.
- Assumption: the dispatch's cap of 17 is applied to every run. `model/blankfree_model.py:405-407` states
  a trailing run is capped at 10 frames. Trailing runs of 11-17 frames are therefore also forced under
  rc and are NOT counted. The first token's 28-unit limit is also not modelled.
- The frame rasterisation (frame centre inside a phone interval) is my choice. The package has no
  MFA-to-frame convention.
