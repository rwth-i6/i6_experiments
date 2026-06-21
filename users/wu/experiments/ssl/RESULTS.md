# SSL results — LibriSpeech WER (greedy CTC/NAR decode + sclite)

_Generated 2026-06-21. One line per experiment; the checkpoint shown is the one with the **best dev-other**
WER for that experiment (LS100 overfits, so per-epoch selection is read off the recog tables, not auto-picked).
Experiment names are the config/alias prefixes verbatim. Collapsed runs are excluded (listed below)._

| experiment (config prefix)                                     | best ep | dev-clean | dev-other | test-clean | test-other |
| :------------------------------------------------------------- | ------: | --------: | --------: | ---------: | ---------: |
| `ssl/ctc_ls960/scratch_12x512_spm5k`                           |      60 |      2.98 |      7.15 |       3.07 |       7.13 |
| `ssl/ctc_ls100/finetune_bestrq_base_12x512_spm5k`              |     100 |      6.46 |     14.33 |       6.67 |      14.15 |
| `ssl/finetune_two_level/ls960_cif120ms_k128/ft1_ctc_frozenseg` |      90 |      8.50 |     15.91 |       8.57 |      15.86 |
| `ssl/finetune_two_level/ls960_meanpool80ms_k128/ft1_ctc`       |     100 |      8.50 |     16.01 |       8.41 |      16.03 |
| `ssl/finetune_two_level/ls960_cif80ms_k128/ft1_ctc_frozenseg`  |      90 |      8.41 |     16.03 |       8.24 |      16.18 |
| `ssl/finetune_two_level/ls960_cif120ms_k128/ft1_ctc_trainseg`  |      70 |      9.16 |     16.24 |       9.16 |      16.29 |
| `ssl/finetune_two_level/ls960_cif80ms_k128/ft1_ctc_trainseg`   |     100 |      9.37 |     16.67 |       9.24 |      16.84 |
| `ssl/ctc_ls100/scratch_12x512_spm5k`                           |     100 |     10.63 |     22.44 |      10.63 |      23.24 |
| `ssl/finetune_two_level/ls960_cif120ms_k128/ft2_ce_trainseg`   |      90 |     31.72 |     36.72 |      33.29 |      37.44 |
| `ssl/finetune_two_level/ls960_cif80ms_k128/ft2_ce_trainseg`    |      90 |     32.81 |     36.95 |      33.78 |      37.87 |

**Excluded — collapsed (diverged):**
- `ssl/ctc_ls100/finetune_bestrq_mask49_12x512_spm5k` — diverged to 100.0% dev-other (best pre-collapse 17.87); not a valid converged result.

**No completed recog yet (omitted):**
- `ssl/ctc_ls100/finetune_bestrq_18x512_spm5k` (depth control) — its BEST-RQ pretrain `ssl/pretrain_bestrq/ls960_18x512_n4` is still training, so no finetune WER yet.

**Notes:**
- `ctc_ls960/scratch` is the **supervised 960h topline**; `ctc_ls100/scratch` is the from-scratch 100h floor; everything else finetunes a frozen SSL encoder on LS100.
- Two-level segmenter A/B (80 ms): frozen CIF / fixed mean-pool / trainable CIF all land ~16.0–16.7 dev-other; **frozen CIF and mean-pool beat trainable CIF** — CTC co-adaptation of the segmenter does not help even after the rate-anchor fix.
- `ft2_ce_*` (Paraformer scaled-CIF + NAR per-token CE) underperform badly (~37 dev-other): not collapsed, but the CIF rate has not retuned to the label rate — open tunable.
