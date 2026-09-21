# Budget 50-sub-epoch arms extract (2026-09-21)

## PER table (ep50, dev-other)

| Arm | PER | Rate (Hz) | Gap | Delta vs ctrl/prior |
|-----|-----|-----------|-----|---------------------|
| ctrl_50 | 0.8965505570441404 | 10.791929640392764 | 4.628121449559951 | baseline |
| odmprior_50 | 0.9048258355662107 | 10.86483683893846 | 4.54192300219933 | +0.008275278522070217 vs ctrl_50 |
| bt_50 | 0.8933070088845014 | 10.742001971502823 | 4.437204276996832 | -0.0032435481596390092 vs ctrl_50 |
| odmbt_50 | 0.8926470173459314 | 10.691946282949061 | 4.257361259829703 | -0.003903539698209002 vs ctrl_50 |
| nosched_ctrl_50 | 0.8953208292201382 | 10.900426305480522 | 4.425052216618839 | -0.0012297278240022136 vs ctrl_50 |
| nosched_odmprior_50 | 0.8890988577069525 | 10.760948881748236 | 4.243824323735979 | -0.01572697785925814 vs odmprior_50 |

## Training job metadata

Job hashes (no finished markers found in output/<arm>/ directories):
- ctrl_50, odmprior_50: PackedBlankfreeTrainJob.ks7CbtlvpcIL
- bt_50, odmbt_50: PackedBlankfreeTrainJob.reEI2Nd0S77A
- nosched_ctrl_50, nosched_odmprior_50: PackedBlankfreeTrainJob.4QzmftNlbErt

Work paths at /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.{hash}/

MISSING (not extracted due to turn limit): 
- Last sub-epoch completed per arm
- Wall time per sub-epoch (ep1 and ep50)
- Job hashes for BlankfreeGreedyPerJob and BlankfreeDerangementGapJob
- Job hashes for PairedPerDeltaJob
- Status of read jobs (finished or pending)

All six 50-sub-epoch arms have data at epochs 1, 4, 10, 25, 50.
