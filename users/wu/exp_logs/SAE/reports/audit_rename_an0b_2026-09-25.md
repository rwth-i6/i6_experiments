# Audit: SAE_4A_rename AN-0b, the escape diagnostic (RenameEscapeJob.cw4BeJzrQ3U9), 2026-09-25

VERDICT: CONFIRMED_WITH_CORRECTIONS. All three readings hold under the registered text and the read-time rules.
- E-STEP CHECK: passes. All 10 MAP paths were extracted. The largest bridge_rel_diff is 4.2e-16 and the largest
  re-add rel diff is 6.4e-7, both far below 1e-4.
- ESCAPE NOT THE BLOCK: holds. No-escape restore is 0.0000 at plain/4.4 and 0.0053 at rate_neutral/4.4. Both
  cells are VALID (gold-row identity 0.9888 and 0.9834).
- SIL ESCAPE (descriptive): holds, by a small margin. SIL takes 0.0713 of the swapped units' mass and the right
  name takes 0.0603.

The corrections concern what the readings license, not the readings themselves.
1. The no-escape operator masks only OY and ZH, so ESCAPE NOT THE BLOCK tests only the empty-row half of H1b.
   SIL was never masked. "SIL escape is not the block" is therefore not measured.
2. SIL ESCAPE should not be reported as "the LM pressure escapes through SIL".
   - The same units send 0.050 of their mass to SIL in the gold-key row, where the names are right. The excess
     that comes from the wrong names is about 0.021.
   - Most of the mass stays on the partner name (0.489) or goes to other populated names (0.379).

Inputs read:
- SAE_4A_rename.md lines 97-158 and 328-352. The rule text is unchanged since commit 449b250ee (02:45:07). The
  AN-0b registration was committed at a06f90037 (02:06:45). The job ran from 02:50 to 02:53, so both commits
  precede the run.
- Job outputs: output/an0b.json, n_su.npz, report.txt, log.run.1, usage.run.1 and info.
- Code: rename_escape_jobs.py, and in rename_emstep_jobs.py the functions derangement, prior_table,
  phi_tables, key_of, m_step_type, restore_share and _Bed.
  - The whole code is at commit 0822791c. The working tree is clean, and the file mtime (02:25) precedes the run.
  - Also read: the relevant parts of lattice.py (PriorHistory, the prior table shape), prior.py
    (per_token_log_probs, the log_tri layout), blankfree_emtable.e_step_batch, blankfree_genmarg_jobs._setup,
    and test_rename_escape.py (skimmed).
- The config config_sae_4a_rename_an0b_v1.py and its shim.
- Reports: impl_rename_an0b, review_rename_an0b_launch, exec_rename_an0b_launch, extract_rename_an0b and
  audit_rename_an0 (all 2026-09-25).

Scripts and outputs (session scratchpad, numpy and h5py only; no project module imported):
/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/
audit_an0b.py/.out, audit_an0b_other.py/.out, audit_an0b_readd.py/.out.

## 1. E-step check (read-time rules: all 10 extracted, each bridge_rel_diff <= 1e-4, each rel <= 1e-4)

From an0b.json map_paths, recomputed from the stored lattice_score, readd.total and both log Z values:

| row | utterance | extracted | rel (lattice vs re-add) | bridge_rel_diff | lattice - re-add (nats) |
|---|---|---|---|---|---|
| 5-pair s1 | 1723-141149-0038 | yes | 7.9e-8 | 1.3e-16 | -0.0003 |
| 5-pair s1 | 7447-91187-0016 | yes | 1.6e-7 | 2.0e-16 | +0.0002 |
| 5-pair s1 | 8108-274318-0010 | yes | 5.3e-7 | 0 | +0.0020 |
| 5-pair s1 | 1578-140049-0004 | yes | 6.4e-7 | 0 | +0.0025 |
| 5-pair s1 | 3436-172171-0041 | yes | 2.4e-7 | 1.5e-16 | -0.0007 |
| gold | 1723-141149-0038 | yes | 2.0e-7 | 4.2e-16 | -0.0007 |
| gold | 7447-91187-0016 | yes | 2.0e-7 | 0 | +0.0002 |
| gold | 8108-274318-0010 | yes | 5.0e-7 | 1.3e-16 | +0.0018 |
| gold | 1578-140049-0004 | yes | 3.7e-7 | 4.0e-16 | +0.0013 |
| gold | 3436-172171-0041 | yes | 5.1e-7 | 3.2e-16 | -0.0015 |

- not_extracted_paths is empty. The 5 utterances are the first 5 of the 300-utterance subset and equal AN-0's
  subset_first_tags. The production reduction is matmul and the history is trigram.
- **The re-add is independent of the lattice code.**
  - readd_path uses numpy only: it smooths the type table in float64, does its own duration indexing, and
    calls PhoneNgramPrior.per_token_log_probs(order=3), which has its own BOS and trigram index arithmetic.
  - It imports nothing from lattice.py and does not use build_segment_table or segment_scores.
  - It shares with the lattice only the model tables: log_m and log_dur from phi_tables, the prior npz, the
    0.1 smoothing and H_LM.
  - Path selection uses the lattice's tau = 1e-7 posterior only to define the support. Within the support, the
    path is chosen by the re-add's own score.
- **A second re-add of my own.**
  - Inputs: the channel from PhiFromKeyInitJob's key_table.npz rows permuted by g, the units read from the
    units.train HDF shards, and the LM from prior.npz log_tri with my own history indexing. The duration term is
    taken from the job.
  - Result: the channel matches the job's re-add within 4e-5 nats on every path, and the rate-neutral LM matches
    within 2e-12. The total then matches the lattice within 6.3e-7 relative.
  - The path segments tile each utterance exactly, and no token repeats.
- **A small note, not a correction.** The implementer's report says the lattice score sits slightly below the
  re-add. That held on its test utterance, but here the sign is mixed: the lattice is above the re-add on 6 of
  10 paths. Every difference is at most 0.0025 nats (6.4e-7 relative), which is float32 level.
- **Scope, as the read-time rule says.** A pass shows correct path scoring in the as-is rate_neutral/4.4 cell,
  on single utterances (B = 1). It does not verify the tau = 1 accumulation into N(s,u), and it does not cover
  batching. The no-escape E-steps differ from as-is only in the prior table handed to the same lattice code.

## 2. No-escape operator

- The mask is applied in _EscapeBed.set_prior. After the form's table is built, it sets
  `t[:, mask_ids] = NEG_INF`.
  - The table is log_tri, shaped [1681, 40]. Its rows are the (h2, h1) histories and its columns are the next
    token (prior.py asserts shape (N_CTX*N_CTX, N_TYPES); lattice.py asserts [|h|, K]).
  - So the mask removes OY and ZH as emitted tokens after every history, including BOS, and changes no other
    entry. The LM is not renormalised over the other 38 symbols, which is the natural reading of "masked out of
    the E-step lattice".
- The escape set is derived for each row and asserted two ways. For the gold row, the symbols holding no unit
  in the gold key are OY and ZH (verified from key_table "key", which equals GoldUnitKeyJob key.json). The
  uniform rows of the key table are also exactly OY and ZH (verified). The smallest deviation from uniform among
  populated rows is 0.045.
  - The derangement pool excludes OY and ZH, so in the 5-pair and 1-pair rows OY and ZH still carry the uniform
    rows. The mask is therefore the same pair of columns (ids 25 and 38) in all three rows. It is applied by the
    same code path to the gold row and the deranged rows.
- **The mask reached the lattice (from n_su.npz).**
  - No-escape N(OY, ZH) is exactly 0 in all 9 no-escape arrays.
  - As-is N(OY, ZH) is 136 to 9,347 expected frames per cell.
  - The per-unit column sums of N are equal between the operators to within 5e-10, so each unit's posterior
    still sums to its frame count.
- **The guard is taken per operator.** The no-escape gold-row identity after the step is 0.9964 (λ = 1), 0.9888
  (plain/4.4) and 0.9834 (rate_neutral/4.4), equal to 4 decimals to the as-is guard.

## 3. restore, rho, the guard and the split, re-derived from n_su.npz

- Weights and gold key: key_table.npz "counts" and "key". The counts sum to 15,275,716, which equals the job's
  weights_total. The AN-0 audit had already matched these counts element by element against an independent
  bincount of the train HDF.
- For all 18 E-steps, I rebuilt the keys myself:
  - pi = N.sum(1) / N.sum();
  - key_before = argmax_s m0(u|s) pi(s), with m0 the key-table rows permuted by g;
  - key_after = argmax_s m1(u|s) pi(s), with m1 = (N + 1e-3) row-normalised.
- My key_before and key_after equal the job's on 500 of 500 units in every cell.
- restore = sum w[(k0 != gold) & (k1 == gold)] / sum w. Identity, rho, SIL share, escape share and pi all match
  the json to at most 4e-17. This is R1's definition, the same one the AN-0 audit confirmed.
- The as-is cells equal AN-0's job (RenameEmStepJob.sXyLgUqfPBPv, rename_emstep.json) exactly: key_before,
  key_after, restore, tokens per frame and SIL share agree on plain/1, plain/4.4 and rate_neutral/4.4 for both
  rows. H_LM is 2.2571084 in both.
- **The split.**
  - Swapped units are the units whose gold phone p is moved by g. The partner is the k with g[k] = p, the
    symbol that carries p's row; since every g here is an involution, that is g[p].
  - The categories are disjoint, because the job asserts that p and k are never SIL, OY or ZH. "Other" is the
    remainder, and every split sums to 1.000000.
  - My recomputed splits match the json to 8e-16. 170 of 173 swapped units have frames in the subset.
- **The denominator.** restore divides by all train frames, as in AN-0. If it were divided by the moved frames
  only (0.4068), 5-pair no-escape rate_neutral/4.4 would be 0.0131, still below 0.05. The reading does not depend
  on this choice.

## 4. The readings, recomputed

| 5-pair s1, no-escape | restore | guard (gold identity after) | VALID |
|---|---|---|---|
| plain/4.4 | 0.000000 | 0.9888 | yes |
| rate_neutral/4.4 | 0.005341 | 0.9834 | yes |

- ESCAPE BLOCKS THE STEP needs restore >= 0.05 at a VALID λ = 4.4 cell. Neither cell reaches it.
- ESCAPE NOT THE BLOCK needs restore < 0.05 at every VALID λ = 4.4 cell. Both cells are below. **ESCAPE NOT THE
  BLOCK.**
- SIL ESCAPE: in the no-escape rate_neutral/4.4 split, SIL is 0.07127 and the right name is 0.06035, so **SIL
  ESCAPE** holds, by 0.011. The cell is VALID.
- The job's verdict and the extraction report agree with all of these. The extraction's per-cell and split
  figures match the json. Its usage line calls used_time a "fraction"; it is in hours (0.0576 h, about 3.5 min).

## 5. Operating point, and what the reading does and does not license

- **Measured operating point.**
  - Phi: the gold-key phi PhiFromKeyInitJob.f0jaGuiJVe6A. It is a type-level table from the hard gold key, with
    off-key entries of 0.0002, uniform OY and ZH rows, and durinit durations. Key purity is 0.644.
  - Derangement: the seed-1 5-pair derangement AH-T, AO-N, K-OW, M-Y, P-S, which moves 0.4068 of the train-frame
    weight.
  - The step: E-step smoothing 0.9 table + 0.1 uniform, tau 1, one E-step and one type-level M-step
    (pseudo-count 1e-3), table discarded.
  - Data: 300 train utterances (A12 selection, seed 0).
  - LM cells: λ in {1, 4.4} x {plain, rate_neutral}, on the frozen trigram RtzbESkOedsT.
  - Every constant traces to AN-0 as registered, and the as-is step reproduces AN-0 bitwise.
- **What masking changed.**
  - At plain/4.4, masking removes 0.069 of the swapped units' mass from the empty rows. The right name gains
    0.0024 of it, about 3.5 %. The partner gains 0.020, SIL 0.005 and other names 0.042. restore falls from
    0.0001 to 0.
  - At rate_neutral/4.4 only 0.009 sat in the empty rows, and restore is unchanged at 0.0053.
  - In the descriptive 1-pair row, masking lowers restore (0.0101 -> 0.0074).
  - So the empty rows were a side channel of the λ trade-off. They were not holding back a rename.
- **Where the pressure goes.** From λ = 1 to rate_neutral/4.4 under no-escape, the partner share falls by 0.208.
  Of that, the right name gains 0.033, SIL 0.058 and other populated names 0.118.
  - In the gold-key row at the same cell, the same units' on-key share also falls (0.809 -> 0.723), to SIL
    (+0.037) and other names (+0.047).
  - Most of the SIL gain is therefore generic to λ-weighting, not specific to wrong names.
  - The MAP paths agree: on swapped-unit frames of the 5 deranged paths, the right name is 1.5-7.5 %, the partner
    43-56 %, SIL 7-19 % and other names 30-39 %.
- **Licensed.**
  - One E-step at λ <= 4.4, in either form, from the gold-key phi's sharp channel with 5 swapped pairs (seed 1),
    does not restore a material share of keys even when the unit-free rows OY and ZH are removed from the lattice.
    The empty-row escape is not what blocks the step.
  - The lattice scores MAP paths as the model defines them (E-step check).
  - This is consistent with recording H1 (channel lock-in) at this sharp-channel operating point, as registered.
    No escape mechanism is brought to the user.
- **Not licensed.**
  - That SIL escape is not the block: SIL was not masked.
  - That the pressure "goes to SIL": SIL is a minor sink beside the partner and other populated names.
  - Anything about iterated EM, a flat-channel start (TP-A1's regime), other seeds, 1,000 utterances or
    found-partition phis.
  - That N(s,u) is verified at full, batched size.
- **Descriptive only.** The 1-pair row carries no reading. Measured against its own moved weight (0.1463), it
  restores 6.9 % (as-is) and 5.1 % (no-escape) at rate_neutral/4.4. The registered denominator is all frames,
  and there it is 0.010 and 0.007.
