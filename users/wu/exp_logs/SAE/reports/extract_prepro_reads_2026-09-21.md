# SAE 4A Preprocessing Experiment: Numeric Extractions (2026-09-21)

## Setup
- Config: `config/sae_4a_prepro_pack.py`
- Pack job: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.5EIGJJ1MkcO9` (resolved from symlink)
- Arms: ctrl_20, ctrl_20_s1, prepro_20
- Kept epochs: 1, 4, 10, 20
- Primary dataset: dev-other (dev-clean registered but not extracted here)

## 1. PER Data (dev-other, ep20) with S/D/I counts and job hashes

| Arm | PER | S | D | I | Job Hash |
|-----|-----|-------|-------|-------|---------|
| ctrl_20 | 0.8745677619517699 | 126601 | 18578 | 9860 | 9GnrDF23mUuG |
| ctrl_20_s1 | 0.8758482583556622 | 127299 | 19515 | 8452 | MtzQhyVFh6xx |
| prepro_20 | 0.8888957833873925 | 129982 | 18000 | 9597 | bExeV4r9gCAk |

Job paths: `/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_eval_jobs/BlankfreeGreedyPerJob.<hash>/output/per.json`

## 2. Derangement Gap (dev-other, ep20)

| Arm | Gap |
|-----|------|
| ctrl_20 | 4.2659388706697525 |
| ctrl_20_s1 | 4.5992389181995055 |
| prepro_20 | 4.234906094123154 |

Source: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/{arm}/ep20/dev-other/derangement_gap.json`

## 3. Greedy Emitted Rate (decode_stats, dev-other, ep20)

Note: Field is `phone_rate_original_hz` from decode_stats.json (phones per second on original frame rate, not the computed emitted rate per second).

| Arm | phone_rate_original_hz | phones_emitted |
|-----|---------|---------|
| ctrl_20 | 9.160905671862432 | 168557 |
| ctrl_20_s1 | 9.033457249070633 | 166212 |
| prepro_20 | 9.178025609252375 | 168872 |

Source: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/{arm}/ep20/dev-other/decode_stats.json`

## 4. Paired Per Delta (ep20, dev-other)

### prepro_20 vs ctrl_20
- delta_per: 0.014328021435622618
- CI95: [0.011112504478357542, 0.017133126519636895]
- n: 2864 utterances
- Source: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/paired/prepro_20_vs_ctrl_20/ep20/dev-other/paired_per.json`

### ctrl_20 vs ctrl_20_s1
- delta_per: -0.0012804964038922728
- CI95: [-0.00385803321975395, 0.001493114593455582]
- n: 2864 utterances
- Source: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/paired/ctrl_20_vs_ctrl_20_s1/ep20/dev-other/paired_per.json`

### prepro_20 vs ctrl_20_s1
- delta_per: 0.013047525031730345
- CI95: [0.009863039709302012, 0.016126387436263344]
- n: 2864 utterances
- Source: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/paired/prepro_20_vs_ctrl_20_s1/ep20/dev-other/paired_per.json`

## 5. Wall Time Per Sub-Epoch and Training Loss (ep1 and ep20)

Source: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.5EIGJJ1MkcO9/output/{arm}/learning_rates`

### Epoch 1
| Arm | Total Time (sec) | Steps | Wall Time Per Step (sec/step) | train_loss_agg |
|-----|--------|-------|-----------|---------|
| ctrl_20 | 590 | 57 | 10.350877 | 1.657488321003161 |
| ctrl_20_s1 | 597 | 57 | 10.473684 | 1.6683929510283888 |
| prepro_20 | 595 | 57 | 10.438596 | 1.6738591967967518 |

### Epoch 20
| Arm | Total Time (sec) | Steps | Wall Time Per Step (sec/step) | train_loss_agg |
|-----|--------|-------|-----------|---------|
| ctrl_20 | 580 | 57 | 10.175439 | 1.5229638229336655 |
| ctrl_20_s1 | 584 | 57 | 10.245614 | 1.5216154048317356 |
| prepro_20 | 582 | 57 | 10.210526 | 1.4260900041513276 |

## 6. WAV2VEC-U Selection Statistic

MISSING: 4-gram phone-LM perplexity / vocabulary-seen fraction squared, SIL stripped

Searched at:
- `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/{arm}/{epoch}/dev-other/` (per.json, decode_stats.json, greedy_phones.json, greedy_raw.json)
- `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/` (all subdirectories)

The metric is mentioned in the design as to be "computed for both arms at every kept epoch from the existing greedy decodes" but the computation output is not present in the current outputs.

## Notes

- All paths are absolute starting from `/e/project1/spell/wu24/2026-07-13_unsupervised/`
- PER values are fractions (not percentages)
- Paired deltas are negative-is-better convention (BASELINE in delta_per name is ctrl_20 or ctrl_20_s1)
- Wall times derived from `:meta:epoch_train_time_secs` / `:meta:epoch_num_train_steps`
- Training loss from `train_loss_agg` field in learning_rates epochs
