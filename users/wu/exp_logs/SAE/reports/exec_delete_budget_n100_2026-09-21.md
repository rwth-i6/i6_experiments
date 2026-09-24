# Delete N=100 budget arm subtrees (2026-09-21)

Status: DONE

## Scope
Deleted only the six *_100 arm subtrees (output/ and work/ each) inside three packed
PackedBlankfreeTrainJob dirs. Verified squeue empty (nothing running) before deleting.

## Verified before deletion
- Each *_100 output subtree contained epoch.*.pt / epoch.*.opt.pt under models/.
- No *_50-named file found under any *_100 path (grep -iname '*_50*' empty).
- No output/ symlink (setup-level) points into any of these three job dirs at all
  (checked both by link-name pattern and by resolving every output/ symlink target).
- Doc references to *_100 arm names found in SAE_4A_budget.md and SAE_4A_prepro.md — left untouched.

## Deleted paths (12 dirs, output+work per arm)
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/odmprior_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/work/ctrl_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/work/odmprior_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.reEI2Nd0S77A/output/bt_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.reEI2Nd0S77A/output/odmbt_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.reEI2Nd0S77A/work/bt_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.reEI2Nd0S77A/work/odmbt_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.4QzmftNlbErt/output/nosched_ctrl_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.4QzmftNlbErt/output/nosched_odmprior_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.4QzmftNlbErt/work/nosched_ctrl_100
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.4QzmftNlbErt/work/nosched_odmprior_100

No dangling output/ symlinks needed removal (none existed pointing into these dirs).

## Space freed (du before -> after each job dir)
- ks7CbtlvpcIL: ~282.5M -> 121M (freed ~161.5M)
- reEI2Nd0S77A: ~292.7M -> 123M (freed ~169.7M)
- 4QzmftNlbErt: ~265M -> 121M (freed ~144M)
- Total freed: ~475M (~0.46 GB)

## Kept *_50 subtrees (untouched), epoch files confirmed present
- ctrl_50: epoch.001.pt, epoch.004.pt, epoch.010.pt, epoch.025.pt, epoch.050.pt, epoch.050.opt.pt
- odmprior_50: same set
- bt_50: same set
- odmbt_50: same set
- nosched_ctrl_50: same set
- nosched_odmprior_50: same set

## ctrl_50 ep1/ep4/ep10 checkpoints (frozen inputs, confirmed present)
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_50/models/epoch.001.pt
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_50/models/epoch.004.pt
work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_50/models/epoch.010.pt
