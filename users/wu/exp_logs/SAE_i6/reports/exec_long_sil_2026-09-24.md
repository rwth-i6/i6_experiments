# Long retained SIL runs on the real VAD output (2026-09-24)

Status: DONE. I ran analysis/long_sil_after_vad.py on the finished VAD job
BlankfreeVadHdfJob.RLrgIh6lFv9m (alias/sae/4a/data/vad). It read the job's raw_index and orig_length HDFs
and finished in 3.6 s on the desktop CPU.
Cap: 17 recognizer frames (W=25, stride 3). A long run is a SIL run longer than 17 frames.

| split | stage | utts | % utts with run > 17 | runs (lead/int/trail) | excess frames, % of retained | max run |
|---|---|---|---|---|---|---|
| dev-clean | post-VAD | 2703 | 0.33 | 10 (1/7/2) | 0.0169 | 28 |
| dev-other | post-VAD | 2864 | 1.22 | 38 (11/23/4) | 0.1229 | 32 |
| dev-clean | pre-VAD | 2703 | 2.26 | 67 | 0.0905 | 41 |
| dev-other | pre-VAD | 2864 | 3.98 | 137 | 0.2792 | 54 |

## Mask check
I compared the job's masks with the script's recomputed masks (reports/exec_long_sil_2026-09-24.cmp_masks.py).
They cover the same utterances: 2864 in dev-other and 2703 in dev-clean. 0 utterances differ. The symmetric
difference of the kept frames is 0 in both splits. So the provisional numbers (1.22 % and 0.33 %) stand unchanged.

## Train split: skipped
get_mfa_alignments("train-clean-100") is a DownloadHuggingFaceSnapshotJob, a Sisyphus job. The 14 pinned
parquets of gilkeyio/librispeech-alignments@0daa1eb4 hold 6.46 GB (HF API), which is over the 5 GB limit.
Nothing was downloaded.

## Caveats (from the implementer's report)
- The cap of 17 is applied to every run. The trailing cap of 10 is not modelled, so trailing runs of
  11-17 frames are undercounted.
- spn counts as silence.
- Operational note: the script refuses to write to reports/, because it resolves into recipe/. The raw
  outputs went to scratch and were copied here as exec_long_sil_2026-09-24.dev_job.json.
