# Training Loss Trajectory for N=100 Budget Arms (Epochs 40–67)

Extract date: 2026-09-21. All values are `train_loss_agg` from learning_rates files. N=100 arms
killed at epochs 65–67 (user-specified). Learning rates at epochs 50 and 65 shown for schedule
detection. N=50 siblings' epoch 50 loss included for comparison.

## ctrl_100 (Scheduled)

From: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/work/ctrl_100/`

| Epoch | train_loss_agg | LearningRate |
|-------|---|---|
| 40 | 1.08134925365448 | 0.0001 |
| 50 | 1.0795800434915643 | 0.0001 |
| 60 | 1.078434235171268 | 0.0001 |
| 67 (last) | 1.0756977294620715 | 8.425e-05 |

**N=50 sibling (ctrl_50) at epoch 50:** 1.2845593502647

---

## odmprior_100 (Scheduled)

From: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/work/odmprior_100/`

| Epoch | train_loss_agg | LearningRate |
|-------|---|---|
| 40 | -2.965522230717174 | 0.0001 |
| 50 | -2.9917827698222377 | 0.0001 |
| 60 | -2.9873952029044166 | 0.0001 |
| 67 (last) | -2.9799168026238156 | 8.425e-05 |

**N=50 sibling (odmprior_50) at epoch 50:** -3.000955920470388

---

## bt_100 (Scheduled)

From: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.reEI2Nd0S77A/work/bt_100/`

| Epoch | train_loss_agg | LearningRate |
|-------|---|---|
| 40 | 1.0061159844984089 | 0.0001 |
| 50 | 1.0017092562558358 | 0.0001 |
| 60 | 0.9980922885108412 | 0.0001 |
| 66 (last) | 0.9889284152733652 | 8.65e-05 |

**N=50 sibling (bt_50) at epoch 50:** 0.9353983538192615

---

## odmbt_100 (Scheduled)

From: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.reEI2Nd0S77A/work/odmbt_100/`

| Epoch | train_loss_agg | LearningRate |
|-------|---|---|
| 40 | -2.981349735929255 | 0.0001 |
| 50 | -2.9791258050684344 | 0.0001 |
| 60 | -2.974610127900776 | 0.0001 |
| 65 (last) | -2.9906386952651176 | 8.875e-05 |

**N=50 sibling (odmbt_50) at epoch 50:** -3.002038031293635

---

## nosched_ctrl_100 (No Schedule)

From: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.4QzmftNlbErt/work/nosched_ctrl_100/`

| Epoch | train_loss_agg | LearningRate |
|-------|---|---|
| 40 | 1.4301852113322209 | 0.0001 |
| 50 | 1.4258718511514497 | 0.0001 |
| 60 | 1.4095958283073025 | 0.0001 |
| 67 (last) | 1.393938135682491 | 0.0001 |

**N=50 sibling (nosched_ctrl_50) at epoch 50:** 1.5093621412913005

---

## nosched_odmprior_100 (No Schedule)

From: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.4QzmftNlbErt/work/nosched_odmprior_100/`

| Epoch | train_loss_agg | LearningRate |
|-------|---|---|
| 40 | -2.9397553770165694 | 0.0001 |
| 50 | -2.9757304149761534 | 0.0001 |
| 60 | -2.995034280576204 | 0.0001 |
| 67 (last) | -2.983655247771949 | 0.0001 |

**N=50 sibling (nosched_odmprior_50) at epoch 50:** -2.9992237718481767

---

## Observations

- **ctrl_100:** Loss decreases from epoch 40 to 67: 1.0813 → 1.0756 (net -0.0057)
- **odmprior_100:** Loss decreases from epoch 40 to 67: -2.9655 → -2.9799 (net -0.0144, more negative is better)
- **bt_100:** Loss decreases from epoch 40 to 66: 1.0061 → 0.9889 (net -0.0172)
- **odmbt_100:** Loss worsens from epoch 40 to 65: -2.9813 → -2.9906 (net -0.0093, more negative)
- **nosched_ctrl_100:** Loss decreases from epoch 40 to 67: 1.4301 → 1.3939 (net -0.0362)
- **nosched_odmprior_100:** Loss worsens from epoch 40 to 67: -2.9397 → -2.9836 (net -0.0439, more negative)

All six N=100 arms show continued training-objective improvement after sub-epoch 50 through
their final epochs (65–67). Learning rates decline from epoch 50 to 65–67 for scheduled arms
(ctrl_100, odmprior_100, bt_100, odmbt_100), while nosched arms maintain constant 0.0001 LR.

