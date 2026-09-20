# Prior Probe Extraction Report

## NeighbourhoodProbeJob.ZhYOeCltF4wW (ctrl_50 ep4 decode)

**Source:** /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/prior_probe/NeighbourhoodProbeJob.ZhYOeCltF4wW/output/prior_probe.md

### Basic Information
- n utterances: 2864
- checkpoint/decode paths:
  - posteriors: `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_core/returnn/forward/ReturnnForwardJobV2.AY6gwxHxHcUR/output/posteriors.hdf`
  - neural_lm: `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/neural_phone_lm/NeuralPhoneLmTrainJob.Iv6P6YVPNWmB/output/model.pt`
  - raw_hyps: `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_eval_jobs/BlankfreeGreedyPerJob.Vz6QOYliPU40/output/greedy_raw.json`

### Rewards, nats per token (for reading against the banked prior_gap table)

| reward | decode | edits | gold |
| --- | --- | --- | --- |
| r_nn (neural LM - unigram) | -2.6561 | -2.6551 | 0.9484 |
| r_lex (lexicon ESCAPE - unigram) | -1.1890 | -1.1962 | 1.3089 |
| r_nn_orig (neural LM - trigram) | 2.3994 | 2.4011 | 0.6441 |
| r_lex_orig (lexicon ESCAPE - trigram) | 3.8665 | 3.8599 | 1.0046 |

### The probe, per reward (nats per utterance; deltas are totals)

| reward | definition | family | std neighbourhood median | std neighbourhood mean | std edits median | std edits mean | delta pearson | delta spearman | delta n | gold above max fraction | gold above max count | utterances | decode nats per utterance | gold nats per utterance | edit delta mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r_nn | neural LM - unigram | A1 (amended) | 3.9983 | 4.1778 | 4.1804 | 4.3561 | 0.6486 | 0.6144 | 22892 | 0.9993 | 2862 | 2864 | -95.1186 | 58.7026 | 1.6367 |
| r_lex | lexicon ESCAPE - unigram | A1 (amended) | 0.5036 | 0.7886 | 0.5240 | 0.8040 | 0.4223 | 0.5548 | 22892 | 0.9997 | 2863 | 2864 | -42.5810 | 81.0195 | 0.4575 |
| r_nn_orig | neural LM - trigram | original (Design line 160-163) | 5.0300 | 5.2592 | 5.2740 | 5.4925 | -0.7283 | -0.7060 | 22892 | 0.0552 | 158 | 2864 | 85.9277 | 39.8682 | -1.3630 |
| r_lex_orig | lexicon ESCAPE - trigram | original (Design line 160-163) | 6.5170 | 6.6868 | 6.7629 | 6.9724 | -0.9267 | -0.9211 | 22892 | 0.1068 | 306 | 2864 | 138.4653 | 62.1852 | -2.5421 |

---

## NeighbourhoodProbeJob.6JZvwE7T1UmA (ctrl_50 ep10 decode)

**Source:** /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/prior_probe/NeighbourhoodProbeJob.6JZvwE7T1UmA/output/prior_probe.md

### Basic Information
- n utterances: 2864
- checkpoint/decode paths:
  - posteriors: `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_core/returnn/forward/ReturnnForwardJobV2.QMYeLWX1G7Na/output/posteriors.hdf`
  - neural_lm: `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/neural_phone_lm/NeuralPhoneLmTrainJob.Iv6P6YVPNWmB/output/model.pt`
  - raw_hyps: `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_eval_jobs/BlankfreeGreedyPerJob.dG4n46xTRSl0/output/greedy_raw.json`

### Rewards, nats per token (for reading against the banked prior_gap table)

| reward | decode | edits | gold |
| --- | --- | --- | --- |
| r_nn (neural LM - unigram) | -0.6663 | -0.6969 | 0.9484 |
| r_lex (lexicon ESCAPE - unigram) | -0.7925 | -0.8075 | 1.3089 |
| r_nn_orig (neural LM - trigram) | 0.2937 | 0.3285 | 0.6441 |
| r_lex_orig (lexicon ESCAPE - trigram) | 0.1675 | 0.2180 | 1.0046 |

### The probe, per reward (nats per utterance; deltas are totals)

| reward | definition | family | std neighbourhood median | std neighbourhood mean | std edits median | std edits mean | delta pearson | delta spearman | delta n | gold above max fraction | gold above max count | utterances | decode nats per utterance | gold nats per utterance | edit delta mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r_nn | neural LM - unigram | A1 (amended) | 4.2332 | 4.3169 | 4.4275 | 4.5237 | 0.6781 | 0.6753 | 22912 | 0.9920 | 2841 | 2864 | -38.9680 | 58.7026 | -1.3202 |
| r_lex | lexicon ESCAPE - unigram | A1 (amended) | 2.4122 | 2.8292 | 2.5579 | 2.9672 | 0.3582 | 0.4126 | 22912 | 0.9972 | 2856 | 2864 | -46.3519 | 81.0195 | -0.3280 |
| r_nn_orig | neural LM - trigram | original (Design line 160-163) | 5.0147 | 5.1102 | 5.2393 | 5.3556 | -0.7531 | -0.6456 | 22912 | 0.6390 | 1830 | 2864 | 17.1801 | 39.8682 | 1.8117 |
| r_lex_orig | lexicon ESCAPE - trigram | original (Design line 160-163) | 6.6758 | 6.7396 | 6.9595 | 7.0560 | -0.8298 | -0.7519 | 22912 | 0.8492 | 2432 | 2864 | 9.7962 | 62.1852 | 2.8038 |

---

## PriorGapAnalysisJob.m6lUxAO65f6A (word-BIGRAM ESCAPE row)

**Source:** /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.m6lUxAO65f6A/output/prior_gap.md

### word_lm_order field
2

### Like-for-like decision table rows

| prior | gold /token | private /token | gold null /token | private null /token | gap /token (paired) | spread | n paired | IS sd gold | IS sd private | discriminates more than the trigram |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unigram (Witten-Bell, live) | -3.5035 | -3.6688 | -3.5035 | -3.6688 | 0.1556 | 0.0023 | 2863 | - | - | no |
| bigram (Witten-Bell, live) | -3.2502 | -3.7821 | -5.2160 | -5.5870 | 0.5205 | 0.0048 | 2863 | - | - | no |
| trigram (Witten-Bell, live) | -3.1992 | -4.6287 | -7.1911 | -7.5913 | 1.3928 | 0.0172 | 2863 | - | - | no |
| 4-gram (KenLM, modified Kneser-Ney) | -2.7862 | -4.5040 | -5.6404 | -5.6849 | 1.6662 | 0.0078 | 2863 | 12.7942 | 17.9207 | yes |
| 6-gram (KenLM, modified Kneser-Ney) | -2.5742 | -4.2614 | -5.0665 | -5.1232 | 1.6225 | 0.0138 | 2863 | 22.8347 | 19.9793 | yes |
| lexicon (word-LM vocabulary) + word 2-gram, STRICT (subset) | -2.1849 | -4.7151 | -6.4520 | -6.6241 | 2.5116 | 0.0627 | 609 | 29.6139 | 22.1416 | yes (subset) |
| lexicon (word-LM vocabulary) + word 2-gram + escape word | -2.2324 | -4.4709 | -4.4778 | -4.6603 | 2.2873 | 0.0144 | 2863 | 30.3377 | 24.6401 | yes |
| neural phone LM (transformer) | -2.5551 | -4.3350 | -5.6543 | -5.8763 | 1.7134 | 0.0094 | 2863 | 19.7760 | 17.1682 | yes |

### Step 0b neural row verdict line

Verdict: **partial_proxy** -- it is a partial proxy and the arm's design must say what it loses.  Its gap 1.7134 /token against the bar 1.9891 recomputed from this run's own rows (trigram 1.3928 + 0.667 x the lexicon ESCAPE excess 0.8945; the pre-registered arithmetic on the Step 0 numbers is 2.01), the 4-gram gap 1.6662, and on the `lexicon_strict` subset (609 utterances) 1.5202 against the strict lexicon's 2.5116 (distance 0.9914, tolerance 0.30).  The Step 0b row is read by THIS rule only; it fires no row of Step 0's decision table.

---

## PriorGapAnalysisJob.Gct95xZHe0zt (word-TRIGRAM comparison)

**Source:** /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.Gct95xZHe0zt/output/prior_gap.md

### word_lm_order field
3

### Like-for-like decision table rows

| prior | gold /token | private /token | gold null /token | private null /token | gap /token (paired) | spread | n paired | IS sd gold | IS sd private | discriminates more than the trigram |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unigram (Witten-Bell, live) | -3.5035 | -3.6688 | -3.5035 | -3.6688 | 0.1556 | 0.0023 | 2863 | - | - | no |
| bigram (Witten-Bell, live) | -3.2502 | -3.7821 | -5.2160 | -5.5870 | 0.5205 | 0.0048 | 2863 | - | - | no |
| trigram (Witten-Bell, live) | -3.1992 | -4.6287 | -7.1911 | -7.5913 | 1.3928 | 0.0172 | 2863 | - | - | no |
| 4-gram (KenLM, modified Kneser-Ney) | -2.7862 | -4.5040 | -5.6404 | -5.6849 | 1.6662 | 0.0078 | 2863 | 12.7942 | 17.9207 | yes |
| 6-gram (KenLM, modified Kneser-Ney) | -2.5742 | -4.2614 | -5.0665 | -5.1232 | 1.6225 | 0.0138 | 2863 | 22.8347 | 19.9793 | yes |
| lexicon (word-LM vocabulary) + word 3-gram, STRICT (subset) | -2.1450 | -4.6567 | -6.3860 | -6.5503 | 2.5057 | 0.0595 | 609 | 31.0117 | 20.9268 | yes (subset) |
| lexicon (word-LM vocabulary) + word 3-gram + escape word | -2.1945 | -4.4613 | -4.4791 | -4.6619 | 2.3213 | 0.0159 | 2863 | 31.6504 | 24.2607 | yes |
| neural phone LM (transformer) | -2.5551 | -4.3350 | -5.6543 | -5.8763 | 1.7134 | 0.0094 | 2863 | 19.7760 | 17.1682 | yes |

### Step 0b neural row verdict line

Verdict: **partial_proxy** -- it is a partial proxy and the arm's design must say what it loses.  Its gap 1.7134 /token against the bar 2.0118 recomputed from this run's own rows (trigram 1.3928 + 0.667 x the lexicon ESCAPE excess 0.9286; the pre-registered arithmetic on the Step 0 numbers is 2.01), the 4-gram gap 1.6662, and on the `lexicon_strict` subset (609 utterances) 1.5202 against the strict lexicon's 2.5057 (distance 0.9855, tolerance 0.30).  The Step 0b row is read by THIS rule only; it fires no row of Step 0's decision table.

