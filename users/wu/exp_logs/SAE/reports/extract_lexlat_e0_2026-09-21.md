# Extract: Lexlat e0 Census 2026-09-21

## (1) Top-level keys and cell record fields

### Census.json top-level keys
arm, bars, checkpoint, curve, epoch, funding_rule, greedy_agreement, kill_read_a, kill_read_b, lattice, log_z_gap, max_candidates, monitors, name, notes, primary, resources, runtime_seconds, split, subset_seed, tau, unpruned, utterances

### Cell record field names
budget, contexts_per_frame_mean, exact_utterances, lam_lex, log_z_per_retained_pooled, max_contexts_reached, max_multiplicity, neg_inf_utterances, pruned_max_pooled, pruned_median_of_utterance_medians, pruned_median_pooled, pruned_p95_pooled, seconds_per_utterance_max, seconds_per_utterance_mean, utterances

## (2) Gap to ceiling field (C vs unpruned on 20 shortest utterances, per lam_lex)

ABSENT

The log_z_gap field provides comparisons of C=256 and C=1024 against the unpruned ceiling (C=16384), but does not include per-lam_lex breakdown:
- C=256: max_per_retained=0.34347884752383406, mean_per_retained=0.09317748723794433, pooled_per_retained=0.09134133063697251
- C=1024: max_per_retained=0.24140778009831262, mean_per_retained=0.0492923279611514, pooled_per_retained=0.05119381660860347

The funding_rule does reference C=4096 and provides pruned_median_per_lam breakdown, but log_z_gap_per_retained is null for the 4096 budget.

## (3) Per-cell logZ per retained frame and utterance count

### 100-utterance cells
- C256_lam0.3333: log_z_per_retained_pooled=-2.260379336555141, utterances=100
- C256_lam0.6667: log_z_per_retained_pooled=-2.4716862664784145, utterances=100
- C256_lam1.0000: log_z_per_retained_pooled=-2.5019348309896326, utterances=100
- C1024_lam0.3333: log_z_per_retained_pooled=-2.2131740009178995, utterances=100
- C1024_lam0.6667: log_z_per_retained_pooled=-2.431640772786171, utterances=100
- C1024_lam1.0000: log_z_per_retained_pooled=-2.6095047990008498, utterances=100

### 20-utterance cells (unpruned ceiling)
- Cunpruned_lam0.3333: log_z_per_retained_pooled=-2.1482422500608496, utterances=20
- Cunpruned_lam0.6667: log_z_per_retained_pooled=-2.3760768757906523, utterances=20
- Cunpruned_lam1.0000: log_z_per_retained_pooled=-2.565240016332681, utterances=20

## (4) Kill-read records

### lam_lex=1.0 (kill_read_a, augmented)
- paired_delta_mean: -0.050895221561152504
- paired_delta_median: -0.038098693759071156
- paired_delta_sd: 0.06460868737099335
- n_better: 76
- n_worse: 9
- pass: True

### lam_lex=0 control row (control_lam0 in kill_read_a)
- paired_delta_mean: -0.03302456474477096
- paired_delta_median: -0.02888425443169973
- paired_delta_sd: 0.05473839808512331
- n_better: 65
- n_worse: 13

### Per-frequency-bin or per-utterance-length breakdown
NOT STORED in census.json or per_utt.json (per_utt.json contains only per-utterance PER records, no aggregate breakdowns)

## (5) Monitor fields at primary operating point (C=1024, lam_lex=1.0)

- phones_per_word_mean: 2.630201842340586
- escape_phone_fraction_mean: 0.6493168131293668
- term_mean: -224.18933649087015
- term_mean_per_retained: -0.8957496681413449
- expected_words_mean: 5.086697385532248
- expected_escape_words: NOT IN MONITORS (stored in per_utt.json per-utterance records only, not aggregated in census.json)

## (6) Checkpoint path and subset selection

### Checkpoint path
/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_50/models/epoch.010.pt

### Subset seed and utterance count
- subset_seed: 0
- total_utterances: 100
- unpruned_utterances: 20 (20 shortest utterances)

## Metadata
- arm: ctrl_50
- name: ctrl_50_ep10/dev-other
- split: dev-other
- epoch: 010
- lattice: topology=blankfree, history=trigram, stride=3, band=25, float64=True, anchor_weight=0.0, prior_weight=1.0
