# Code review: AN-0b launch (RenameEscapeJob.cw4BeJzrQ3U9), 2026-09-25

VERDICT: APPROVE_WITH_CONDITIONS. The job does what the AN-0b bullet registers. The as-is operator is AN-0's
step, and AN-0's finished job id has not moved. The no-escape mask reaches the lattice and changes nothing else.
The cells, rows, split and readings match the bullet. The rebuilt E-STEP CHECK still tests what the registered
check tests. Both conditions below are reading-time rules. Neither needs a code change or a second review.

Inputs read:
- SAE_4A_rename.md: Shared inputs, Amendment R1, AN-0, AN-0b, and the AN-0 Results entry.
- reports/audit_rename_an0_2026-09-25.md (sections 2 and 5) and reports/impl_rename_an0b_2026-09-25.md.
- Commit 0822791c:
  - rename_escape_jobs.py (826 lines, read in full);
  - test_rename_escape.py (the real and graph parts);
  - config_sae_4a_rename_an0b_v1.py and the shim config/sae_4a_rename_an0b.py.
- rename_emstep_jobs.py at d2bbd4c7: _Bed, prior_table, phi_tables, key_of, m_step_type, restore_share.
- blankfree_emtable.e_step_batch and emission_counts; blankfree_genmarg_jobs._setup.
- lattice.py: _arc_weights, _prior_term, _emit_context and PriorHistory.same_nonsil.

## Conditions (for the extraction and the read)

1. **A MAP path that is not extracted must block the escape readings.**
   - Where: rename_escape_jobs.py:332-353 (an0b_readings).
   - E-STEP CHECK FAILS is computed over extracted paths only. A path whose tau_MAP support holds no complete
     legal path is only counted.
   - Failure: with k of 10 paths extracted, the outcome still prints "E-STEP CHECK does not fail (k of 10 ...)",
     even at k = 0. The escape readings are then printed as readable. The deranged row, which is where the
     suspected defect would sit, could drop out of the check unflagged.
   - Rule for the read: if any of the 10 paths is not extracted, the E-STEP CHECK has not passed. Do not read
     ESCAPE BLOCKS / NOT THE BLOCK until the path is explained.
   - Risk is low. My probe extracted all 16 path-row pairs, and none of them is a registered MAP utterance (see
     E-step check below).
2. **The bridge must be read as part of the E-STEP CHECK.**
   - The check runs the lattice's elementwise reduction at tau 1e-7 (lines 440-443). The E-step ran the
     production reduction (matmul at the trigram) at tau 1.
   - The link between the two is the bridge: log Z at tau 1 under both reductions, on the same utterance
     (lines 436-439 and 463). The job prints it, but no reading uses it.
   - Rule for the read: "E-STEP CHECK does not fail" speaks for the E-step as run only if bridge_rel_diff
     <= 1e-4 on all 10 paths. The test measured 8.7e-16 on one utterance.

Recommendation (not a condition): record the path-extraction rule (implementer choice 1) in SAE_4A_rename.md
before the read. It is already pre-registered in the CONVENTION docstring. The verbatim rule is unchanged.

## 1. As-is is AN-0's step; AN-0's id did not move

- `_EscapeBed` overrides only `set_prior`. With `mask_ids = ()` it builds `R.prior_table(base64, form, lam, h_lm)`,
  exactly as `R._Bed.set_prior` does. The E-step, batches, subset selection, weights, key_of and m_step_type are
  AN-0's code, imported unchanged.
- Inputs (train_units, train_segments, model_args, max_frames, max_seqs, gold key, phi) are built the same way as
  in config_sae_4a_rename_an0_v1. The phi comes from AN-0's own `gold_key_phi()`.
- rename_emstep_jobs.py and its config have no commit after d2bbd4c7 and no working-tree change. Only the
  unrelated config_sae_1g_v1.py is modified, and nothing in the AN-0b import chain imports it.
- I rebuilt config.sae_4a_rename_an0 myself: RenameEmStepJob.sXyLgUqfPBPv, FINISHED.
- The implementer's real test (a) reproduced AN-0's json on the 4 shared cells to at most 1.4e-17, and
  key_before and key_after match exactly.
- Deltas from AN-0, all registered:
  - lambda {1, 4.4} only;
  - no S_1 (the holdout set is not loaded, which does not touch the E-step);
  - the 1-pair row;
  - the no-escape operator;
  - the split, N(s,u), and the MAP paths.

## 2. No-escape

- `_EscapeBed.set_prior` (lines 373-381) sets the escape columns of the form's table to NEG_INF. It does this
  after the rate-neutral shift, so the columns are exactly -1e30. Every other column is untouched.
- `_Bed.e_step` calls `self.set_prior`, so the override is what the E-step uses. `gm._setup` passes
  `model.prior_log_bi` to the lattice, and `reset_prior` restores the base table.
- The job asserts N(OY, ZH) == 0 and pi(OY) = pi(ZH) = 0 for every no-escape cell (line 573). A mask that failed
  to reach the lattice would crash the job, not pass silently. The as-is step puts mass there (test b).
- Test (b) showed that under the production reduction only the OY and ZH columns differ, and those are bitwise
  NEG_INF. Their seg_post is exactly 0, and coverage stays within 1e-6.
- Escape symbols are derived two ways, from the key and from the uniform rows, and the two are asserted equal and
  equal to R1's OY and ZH. Derangements never move OY or ZH (R1 pool), so they are the same for all 3 rows.
- Cells: lambda {1, 4.4} x {plain, rate_neutral}; the lambda-1 E-step is shared, after asserting the two tables are
  equal. Rows: gold (guard), 5-pair s1 (read), 1-pair s1 (descriptive; it enters no reading).
- Note: 1-pair s1 is AH<->T, the first pair of the 5-pair draw, because legacy RandomState.choice without
  replacement takes a permutation prefix. It is descriptive only.
- The guard is applied per operator, so the no-escape readings use the gold row's no-escape identity. That is
  the natural reading of "Printed per cell, row and operator: ... the gold-row guard".

## 3. Split and N(s,u)

- swap_split (lines 202-250) matches the bullet:
  - The swapped units are the units whose gold phone p is moved by g.
  - Right name = p. Partner = the k with g[k] = p, i.e. the symbol that carries p's row.
  - The other categories are SIL, the escape symbols and other.
  - Each unit's share is N(., u) / sum_s N(s, u), averaged with w(u), the train frame counts that identity uses.
    Units with no frames in the subset are excluded and counted. The shares are asserted to sum to 1.
- N(s,u) (the emission counts summed over the cells, [40, 500] float64) is saved as 18 arrays: 3 rows x 2 operators
  x 3 E-steps. The shared rate_neutral/1 cell equals plain/1.
- Caveat for the read: this split is w-weighted per unit. The audit's AN-0 "moves" (0.4-11 %) pool expected
  frames per symbol. The two are not the same number.

## 4. E-STEP CHECK

- **Lattice side.** `tau * log Z_tau` at tau 1e-7, from the unchanged `lattice_forward_backward`, on the E-step's
  own `gm._setup` inputs: the same segment table, the same form-applied prior table and the same history.
  - A sum-product lattice gives no path score at tau 1. At tau -> 0, tau log Z_tau tends to the max path score,
    and 0 <= tau log Z - max <= tau log(#paths).
  - The job computes that bound and asserts it below 1e-6 relative (line 471). So the lattice side is the
    Viterbi score of the lattice's own scoring, computed by the lattice's forward pass.
- **Re-add side.** It is independent of build_segment_table, segment_scores, the lattice and its table:
  - float64 exp and smoothing of the type table;
  - its own duration indexing;
  - `PhoneNgramPrior.per_token_log_probs(order=3)` from the prior npz loaded anew;
  - the form applied separately;
  - log q asserted to be 0.
  Shared with the lattice: log_m and log_dur (phi_tables), em.SMOOTHING and h_lm. The AN-0 audit checked the
  first three, and it re-derived H_LM.
- **Does choosing the path by re-add weaken it?**
  - The support at tau 1e-7 with mass > 0.01 holds only paths within about 5e-7 nats of the lattice optimum.
  - If the lattice scores correctly, the re-add maximum over the support equals the lattice maximum.
  - A lattice defect D = L - R can be hidden only if the support holds an exactly L-tied alternative path on
    which D is 0, so that R there equals L's maximum. The selection is by R, so the bias runs one way only.
  - A systematic defect affects every whole path, so it cannot vanish on one of them. Examples: LM order or
    history, BOS, a missing rate-neutral term, duration indexing, the emission table, SIL handling. Against
    these the check keeps its power.
  - The implementer's power check (a bigram re-add fails at 0.146 relative) perturbs the re-add of the same
    path. Since the support holds only L-ties, re-running the selection under the wrong scorer can only choose
    among those same ties.
- **Probe (this review).**
  - Setup: the login GH200, the job's own map_paths, as-is rate_neutral/4.4, both MAP rows. Utterances: 8 from
    the subset, tags[5:13], none of them one of the registered 5.
  - All 16 were extracted, with rel 1.2e-8 to 7.5e-7, at least 130x below 1e-4.
  - At SUPPORT_MASS 1e-9 the support grows, e.g. 171 -> 198 segments, so there are tie segments with mass
    < 0.01. All 16 were still extracted and rel is unchanged. SUPPORT_MASS is not decisive at these utterances.
  - Script and output: scratchpad probe_extract.py / probe_extract.out
    (/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/).
- **Exact ties are real.**
  - The durinit duration table is exactly geometric for d = 2..25: second differences are 0, and all non-SIL
    rows are identical.
  - Under the hard gold key, every off-key emission is equal.
  - So a boundary frame that is off-key for both neighbours can move at exactly equal score. This confirms that
    a "mass > 0.5" rule cannot work.
- **Latent divergence, cleared.**
  - extract_path line 279 forbids a repeat of any token. The lattice allows SIL after SIL (lattice.py:349,
    same_nonsil excludes SIL).
  - That cannot bite here, for two reasons. The trigram has 0 SIL-SIL counts and log P(SIL | x, SIL) is about
    -24 nats. And in the 5 MAP utterances no SIL-keyed run is longer than 15 frames, against D_sil = 50 (0 of
    the 300 have a run > 50).
  - Had it bitten, the result would have been a non-extraction, which condition 1 covers.
- **Scope of what "does not fail" licenses.** Neither the registered check nor this one tests the tau = 1
  posterior accumulation into N(s,u).
  - Small-shape tests already cover it: brute force (test_blankfree_emtable.py:110) and
    counts = tau d log Z / d log-table (:154). e_step_batch also asserts the frame, token and emission totals.
  - So a pass licenses "the lattice scores paths as the model defines them". It does not license "N(s,u) is
    verified at real shape".
  - Optional extension, if the campaign needs the posterior itself ruled out. On the 5 MAP utterances, take a
    central finite difference of the production log Z at tau 1 with respect to an additive constant on prior
    columns SIL, OY and ZH. Compare it with the E-step's expected tokens on those symbols. That tests exactly
    the mass the audit could not explain.

## 5. Readings and tolerances

- AN0B_RULE is verbatim in SAE_4A_rename.md and in the job docstring (checked).
- Every reading can fire, and each edge is unit-tested:
  - E-STEP CHECK FAILS: rel > 1e-4. When it fires, the other readings are marked "not to be read".
  - ESCAPE BLOCKS: restore >= 0.05 at a no-escape-VALID cell of plain/4.4 or rate_neutral/4.4.
  - ESCAPE NOT THE BLOCK. It prints "not read" if neither cell is VALID, instead of holding vacuously; this is
    disclosed in APPLIED.
  - SIL ESCAPE: the no-escape rate_neutral/4.4 split, SIL share > right-name share.
- UNIFORM_ATOL 1e-6 is appropriate:
  - float32 rounding puts OY and ZH within 6.4e-9 and 2.1e-9 of 1/500 (about 150x inside the tolerance).
  - Every populated row deviates by more than 0.08.
  - The set is cross-asserted against the key derivation and R1's names. A wrong tolerance would crash the job;
    it could not change the mask silently.
- New method constants: MAP_TAU 1e-7 (bound asserted) and SUPPORT_MASS 0.01 (probe: not decisive). Both are
  disclosed in CONVENTION. Neither enters restore, the guard or the split.

## 6. No training, no candidate, labels in readouts only

- Outputs are report.txt, an0b.json and n_su.npz. The M-step table is used only for key_after, and nothing is
  written as a phi.
- The gold key enters only the derangement pool (R1), identity, restore, the guard, the split and the key-side
  derivation of the escape set. That derivation is asserted equal to the label-free uniform-row derivation.
- MFA labels enter only the printed MAP segments.
- Gold labels load for the 5 MAP utterances with matching lengths (checked), so the post-E-step label load will
  not crash.

## 7. Launch

- Graph (rebuilt by me): 4 jobs. Three are FINISHED: PhiFromKeyInitJob.f0jaGuiJVe6A, GoldUnitKeyJob.sLnMRRd2qO0t
  and BlankfreeDurationPriorMeanJob.ReQtJKYpZgsN. The only UNFINISHED one is RenameEscapeJob.cw4BeJzrQ3U9, and
  its work dir does not exist yet.
- Only the AN-0b config and shim reference rename_escape_jobs, so no job is shared with keyarms (pid 3019650,
  running here on jpbl-s02-03).
- Resources: gpu 1, cpu 4, mem 24, time 0.5 h, gpupack, width 1. This is AN-0's request, and AN-0 used 0.059 h.
  AN-0b is 18 E-steps at about 9 s each plus 10 MAP paths at about 0.7 s each (the implementer's measurement).
- Launch line, from the setup dir on the login node, with no -co/-cio. It is correct: the PATH prefix reaches
  the manager through setsid and nohup.
  `PATH=/e/project1/spell/wu24/env/sis_env/bin:$PATH setsid nohup /e/project1/spell/wu24/env/sis_env/bin/python tools/sisyphus/sis --log_level 20 manager -r config/sae_4a_rename_an0b.py < /dev/null >> log/sae_4a_rename_an0b.manager.$(date +%Y%m%d_%H%M%S).log 2>&1 &`
