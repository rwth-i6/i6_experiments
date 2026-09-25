# A15-F implementation: content resolution and a sharp null (2026-09-24)

Status: DONE_WITH_CONCERNS. Commit 8972c436 in recipe/2025-10-speech-llm, branch
haotian_modality_matching_jupiter (not pushed). No jobs launched.

## Files
- src/speech_llm/sae/emc/phi_content_controls.py (new, 854 lines): PhiContentJob, TableContentJob,
  PartitionBoundsJob, PhiContentReadJob. The reading rules and the class mappings are in the docstrings.
- src/speech_llm/sae/emc/test_phi_content_controls.py (new, 496 lines). Parts: unit, bounds, measure,
  control, table, graph, unfinished.
- .../librispeech/configs/config_sae_4a_lexlat_v2_phicontent_v1.py (new, 125 lines).
- config/sae_4a_lexlat_v2_phicontent.py (the shim, setup dir, not in git).
No existing module was edited and no file with "relabel" in its name was touched. settings.py is unchanged.

## Items
- (i) m_phi.npz per phi (40x500): built.
- (ii) 7-class split (class share, within-class accuracy) plus per-phone R4 emis: built. It is asserted
  equal to E.r4_frame_accuracy.
- (iii) PartitionBoundsJob reproduces the audit: random mean 0.1179, max 0.1290 (200 draws, seed 0);
  7-class 0.2872; 13-class 0.3767. The 40-way majority equals the oracle (asserted).
- (iv) Permuted-unit null: lin2 rows and bias permuted by default_rng(s).permutation(500), s = 1..5,
  on the six A10 ep48 phis. The job asserts m' = m[:, sigma]. It runs the same map, T3 primary and
  relabel, R4 emis and the relabelled gold fit.
- (v) Per-gold-phone relabelled fit, split claimed/unclaimed: built without module edits. Frame
  attribution is the segment posterior: the gradient of forward_logsum with respect to a detached
  float64 seg leaf, served by an instance-attribute override of segment_scores. The entropy is
  reported pooled as a residual. log p is asserted against the float32 T3 score to a relative
  tolerance of 1e-5, and the posterior frame sum to 1e-6.
- (vi) real_c_s16 table phi: built. m comes from blankfree_emtable.TableReverseModel (float64,
  SMOOTHING=0.1, the emission A11 scores with). The job asserts the selection names the table.

## Checks (all PASS, real objects, login node)
- unit: the segment posterior matches brute force; permute_units is exact.
- bounds: numbers as in (iii); 0.1-0.6 s, 0.12 GB.
- control (full 500 selected):
  - gold maps to the identity (40/40). R4 is 0.5895, equal to banked A15-E. Class share 0.728,
    within-class accuracy 0.8095. Tap relative diff 6.1e-7. 49 s wall, 800 CPU-s, 6.2 GB.
  - permphi: 40/40; T3 3.9236 (equal to A15-E).
  - permphi nulls s1/s2: 1/40 labels correct; R4 emis 0.098/0.083; T3 primary 0.125/0.167. The
    structure is destroyed. 107 s, 1917 CPU-s, 6.4 GB.
- measure (a10_durinit_s01_ep48 + 5 nulls, 35 items): 16.7 s, 296 CPU-s, 4.0 GB.
  - Phi: R4 emis 0.311, within-class 0.574.
  - Nulls: R4 emis 0.09-0.12, within-class 0.34-0.40.
- table: real_c_s16 R4 emis 0.283, within-class 0.565; 4.2 s, 2.7 GB.
- graph and shim load: 18 new jobs (15 PhiContentJob, 1 TableContentJob, 1 PartitionBoundsJob,
  1 reader). All inputs are finished (MfaFrameLabelsJob.22i2qtnJUOWm is the battery's own job).
  0 jobs shared with the phibattery, phiemis, em, a14 and em_table shims.

## Routing and projected time
- PhiContentJob and TableContentJob: gpupack, 1 slot, cpu 72, mem 16 GB, time 1 h, gpu 0. The
  heaviest job (A10 phi + 5 nulls, 500 items) projects to about 3.6e3 CPU-s, under 5 min wall.
- PartitionBoundsJob and the reader: mini_task on short (login).
- No job is over 1 h and none needs a GPU.

## Concerns / undetermined
- (a) Wording. "Majority MFA class", read literally (majority class by frames), gives 7-class 0.279
  and 13-class 0.367. The audit's numbers come from the class of the unit's majority phone, which
  gives 0.287 and 0.377. The primary uses the audit rule; the literal reading is printed as VARIANT.
- (b) The ABOVE MANNER reference is weak. The 7-class oracle's within-class accuracy (0.374) lies
  below that of random 40-way partitions (mean 0.370, max 0.429): with only one phone per class, its
  within-class accuracy is deflated. Almost any phi with real structure clears it (the ep48 nulls
  reach 0.34-0.40). PROPOSAL for the orchestrator: compare against the random-partition
  within-class max as well, or against a 40-way within-manner oracle. The reader prints both
  bounds, but the rule as registered is applied unchanged.
- (c) "A gain exceeds the max over the 5 nulls" is read as each column's value vs the null max per
  phi.
- (d) The null seed values 1-5 are my choice (the spec gives the count only; they follow T2_SEEDS).
- (e) The table phi uses the smoothed emission (0.1 uniform).
- (f) Unrelated pre-existing changes in the checkout were not staged: config_sae_1g_v1.py
  (modified) and config_sae_3e1_d6_swap_cont_v1.py (untracked).

## Amendment (2026-09-24)

Status: DONE. Commit 48ca9bc9 on haotian_modality_matching_jupiter (not pushed). No launch.

- phi_content_controls.py (+70/-17): PhiContentReadJob.above_manner / manner_line apply the amended rule.
  ABOVE MANNER LEVEL = within-class > max(7-class primary oracle, random-partition max, own permuted-null max).
  For phis without nulls it is max(first two), printed "no own null". Class share and R4 emis are printed
  against the same references, and only within-class sets the flag. The table phi uses the same rule.
  The docstring holds the amendment verbatim, with the original rule marked SUPERSEDED. The bounds lines
  are labelled "PRIMARY (class of the unit's majority MFA phone)", with VARIANT unchanged.
  PartitionBoundsJob docstring: one sentence changed to name its two references. This is outside the four
  items, but was needed for consistency. The docstring is not part of the hash.
- test (+75/-2): new part `rule` (synthetic): a phi above the oracle and random max (0.44) but below its own
  null max (0.45) reads NOT above. The control part runs the real reader on a permphi copy: 0.4388 vs a set
  null max of 0.4488 reads NOT above, and 0.4588 reads above. Gold prints "no own null".
- Checks: rule, unit, control and graph all PASS. Bounds 0.2872/0.3767, variant 0.2787/0.3667, 7-class
  within-class 0.3744, random max 0.4288. The graph is unchanged: 18 new jobs.
- Note: the real permphi nulls reach only 0.312 within-class, below the random max.
