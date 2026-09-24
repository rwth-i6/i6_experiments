# SAE 4A Preprocessing Pack Experiment: Numeric Extractions (2026-09-21)

## Status

DONE

## Setup
- Config: `config/sae_4a_prepro_pack.py`
- Pack job: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.5EIGJJ1MkcO9` (resolved from /e/scratch/)
- Arms: ctrl_20, ctrl_20_s1, prepro_20
- Kept epochs: 1, 4, 10, 20
- Dataset: dev-other

## Greedy PER (dev-other) by Arm and Epoch

Source: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/{arm}/ep{epoch}/dev-other/per.json`

| Arm | ep1 | ep4 | ep10 | ep20 |
|-----|-----|-----|------|------|
| ctrl_20 | 0.8554392892398816 | 0.8750585248907066 | 0.8688760400507686 | 0.8745677619517699 |
| ctrl_20_s1 | 0.8516711324213793 | 0.8729431673952898 | 0.88020307431956 | 0.8758482583556622 |
| prepro_20 | 0.8512875475955436 | 0.8808687068114511 | 0.8906219151036525 | 0.8888957833873925 |

## Greedy Emitted Rate (phone_rate_original_hz, dev-other) by Arm and Epoch

Source: `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/{arm}/ep{epoch}/dev-other/decode_stats.json`

| Arm | ep1 | ep4 | ep10 | ep20 |
|-----|-----|-----|------|------|
| ctrl_20 | 3.3047457553425073 | 9.174819017804735 | 9.159546946672753 | 9.160905671862432 |
| ctrl_20_s1 | 3.647253201156547 | 9.15394899889128 | 9.151014152481576 | 9.033457249070633 |
| prepro_20 | 3.4497489075849477 | 9.137046457531685 | 9.259549120633057 | 9.178025609252375 |

## WAV2VEC-U Selection Statistic (4-gram phone-LM perplexity / vocabulary-seen fraction squared, SIL stripped)

MISSING — searched:
- `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/{arm}/ep{epoch}/dev-other/` (all epochs, all arms)
- All subdirectories under `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/`

Not present in outputs. Per design doc (SAE_4A_prepro.md), the statistic is to be "computed for both arms at every kept epoch from the existing greedy decodes" but computation output not found.

## Notes

- All paths relative to `/e/project1/spell/wu24/2026-07-13_unsupervised/`
- PER values are fractions (not percentages)
- Rate is `phone_rate_original_hz` from decode_stats (phones per second at original 50 Hz frame rate)
- All 12 cells (3 arms × 4 epochs) successfully read for PER and rate
- Selection statistic: MISSING for all arms and epochs
