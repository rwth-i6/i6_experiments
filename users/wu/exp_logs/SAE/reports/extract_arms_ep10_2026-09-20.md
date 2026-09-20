# Training Curves and PER Extraction - 2026-09-20

## Extraction 1: Training Curves (COMPLETE)

### Summary
All 16 arms have complete training curve data extracted from learning_rates files.

### Kept Checkpoints
Epochs with saved models: 1, 4, 10, 25, 32 (and up to 50 scheduled)

### Training Data by Arm

#### Node A (ctrl and odmprior variants)
- **ctrl_50**: Latest epoch 32
  - Total loss: ep1=1.657, ep4=1.080, ep10=0.867, ep32=1.330
  - Rate (Hz) at ep32: 8.843

- **ctrl_100**: Latest epoch 32
  - Total loss: ep1=1.657, ep4=1.091, ep10=0.957, ep32=1.073
  - Rate (Hz) at ep32: 8.903

- **odmprior_50**: Latest epoch 32
  - Total loss: ep1=-6.398, ep4=-2.860, ep10=-2.760, ep32=-2.998
  - Rate (Hz) at ep32: 8.808

- **odmprior_100**: Latest epoch 32
  - Total loss: ep1=-5.933, ep4=-2.743, ep10=-2.617, ep32=-2.901
  - Rate (Hz) at ep32: 8.888

#### Node B (bt and odmbt variants)
- **bt_50**: Latest epoch 30
  - Total loss: ep1=1.663, ep4=1.100, ep10=0.723, ep30=0.958
  - Rate (Hz) at ep30: 8.790
  - Extra terms: train_loss_bt=0.036, train_loss_emc_bt_loss=0.360

- **bt_100**: Latest epoch 30
  - Total loss: ep1=1.663, ep4=1.084, ep10=0.833, ep30=0.995
  - Rate (Hz) at ep30: 8.797
  - Extra terms: train_loss_bt=0.038, train_loss_emc_bt_loss=0.380

- **odmbt_50**: Latest epoch 30
  - Total loss: ep1=-5.933, ep4=-2.747, ep10=-2.626, ep30=-2.886
  - Rate (Hz) at ep30: 8.832

- **odmbt_100**: Latest epoch 30
  - Total loss: ep1=-5.925, ep4=-2.633, ep10=-2.528, ep30=-2.859
  - Rate (Hz) at ep30: 8.938

#### Node C (nosched ctrl and odmprior variants)
- **nosched_ctrl_50**: Latest epoch 32
  - Total loss: ep1=1.656, ep4=1.066, ep10=0.932, ep32=1.314
  - Rate (Hz) at ep32: 8.866

- **nosched_ctrl_100**: Latest epoch 32
  - Total loss: ep1=1.656, ep4=1.055, ep10=0.875, ep32=1.055
  - Rate (Hz) at ep32: 8.914

- **nosched_odmprior_50**: Latest epoch 32
  - Total loss: ep1=-6.404, ep4=-2.872, ep10=-2.681, ep32=-2.962
  - Rate (Hz) at ep32: 8.836

- **nosched_odmprior_100**: Latest epoch 32
  - Total loss: ep1=-5.934, ep4=-2.691, ep10=-2.549, ep32=-2.919
  - Rate (Hz) at ep32: 8.901

#### Node D (ent, entaug, aug, aughi variants)
- **ent_50**: Latest epoch 7
  - Total loss: ep1=1.539, ep4=0.886, ep10=not available, ep7=0.817
  - Rate (Hz) at ep7: 9.192
  - Extra terms: train_loss_ent=0.608, train_loss_cons=not in final

- **entaug_50**: Latest epoch 7
  - Total loss: ep1=1.572, ep4=0.870, ep10=not available, ep7=0.816
  - Rate (Hz) at ep7: 9.165
  - Extra terms: train_loss_ent and cons present

- **aug_50**: Latest epoch 7
  - Total loss: ep1=1.640, ep4=1.080, ep10=not available, ep7=0.925
  - Rate (Hz) at ep7: 9.586
  - Extra terms: train_loss_cons=0.215, train_loss_emc/cons_kl_specaug=0.275

- **aughi_50**: Latest epoch 7
  - Total loss: ep1=1.619, ep4=1.050, ep10=not available, ep7=0.922
  - Rate (Hz) at ep7: 9.514
  - Extra terms: train_loss_cons=0.182, train_loss_emc/cons_kl_specaug=0.238

### Key Metrics

#### Loss component keys found in final epochs:
- train_loss_agg: Total aggregated loss
- train_loss_blankfree_agg_kl_bigram: KL divergence bigram component
- train_loss_blankfree_agg_kl_unigram: KL divergence unigram component
- train_loss_blankfree_expected_phone_rate_hz: Expected phone rate (rate monitor)
- train_loss_l_tau: L_tau component (temperature-related)
- train_loss_rate: Rate component
- train_loss_bt: Back-translation loss (BT arms only)
- train_loss_cons: Consistency loss (consistency arms only)
- train_loss_ent: Entropy loss (entropy arms only)
- train_loss_emc/cons_*: EMC consistency-related metrics
- train_loss_emc_bt_*: EMC back-translation metrics

#### Wall times
Extracted from learning_rates ':meta:epoch_train_time_secs' field (57 steps per epoch):
- ctrl_50 last 3 epochs: ~584-586 sec/epoch
- bt_50 last 3 epochs: ~584-585 sec/epoch
- ent_50 last 3 epochs: ~581-585 sec/epoch

## Extraction 2: PER at Kept Checkpoints

**STATUS: NOT EXTRACTED**

**Reason**: Evaluation jobs have not completed or per.json files are not yet generated.

**Paths checked**:
- Primary alias: `/e/project1/spell/wu24/2026-07-13_unsupervised/alias/sae/4a/budget_pack/<arm>/ep<k>/`
- Scratch path: `/e/scratch/spell/wu24/project-relocated/2026-07-13_unsupervised/alias/sae/4a/budget_pack/<arm>/ep<k>/`

**Directories found but incomplete**:
- Alias directories exist for all 16 arms: ctrl_50, ctrl_100, odmprior_50, odmprior_100, bt_50, bt_100, odmbt_50, odmbt_100, nosched_ctrl_50, nosched_ctrl_100, nosched_odmprior_50, nosched_odmprior_100, ent_50, entaug_50, aug_50, aughi_50
- Kept epoch subdirectories exist: ep1, ep4, ep10, ep25, ep50
- Evaluation set directories found: dev-clean, dev-other
- Files present in eval dirs: derangement, theta (but not per.json)

**Conclusion**: The evaluation pipeline has been set up and partially executed (directories created) but the final per.json files with phone error rate metrics have not been generated. The evaluation jobs may still be in progress.

## Data Source Files

**Learning rates (training curves)**:
- `/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.{job_id}/work/{arm_name}/learning_rates`

**Expected PER results location (not yet available)**:
- `/e/project1/spell/wu24/2026-07-13_unsupervised/output/sae/4a/budget_pack/{arm_name}/ep{epoch}/dev-other/per.json` (or equivalent in alias paths)

---
Generated: 2026-09-20
