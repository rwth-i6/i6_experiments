# JUPITER's genmarg conventions: the R2 uniform-prior decode, and the E[d] average

This replies to `SAE_i6_P1.md` (i6 commit 3dbdfb3bf), Task A, "Read conventions against JUPITER's".
The sources are in the speech-llm repo on JUPITER:
- `src/speech_llm/sae/emc/phi_competence_battery.py` (last changed in commit 74ca2d09);
- `src/speech_llm/sae/emc/blankfree_genmarg_jobs.py`;
- `src/speech_llm/sae/emc/blankfree_phifirst_probe_read.py`;
- the entry config `prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_lexlat_v2_phibattery_v1.py`.

## 1. R2 decodes with a normalised 1/40 table at prior weight 1, not at prior weight 0

R2's prior comes from `UniformPhonePriorJob` (banked as `UniformPhonePriorJob.n7Yd9EbtVi13`). It is a
`PhoneNgramPrior` in which every unigram, bigram and trigram cell holds `log(1/40)`:

```python
v = -math.log(N_TYPES)          # N_TYPES = 40, SIL included
prior = PhoneNgramPrior(np.full(N_TYPES, v), np.full((N_CTX, N_TYPES), v),
                        np.full((N_CTX * N_CTX, N_TYPES), v), meta={"uniform": True, ...})
```

The job docstring states the choice: "A NORMALISED uniform distribution over the 40 types at every
history, so each token costs `log 40` -- not `prior_weight = 0`, which the lattice implements by
skipping the prior addend."

The R2 decode (`uniform_prior_decode` in the entry config) copies the bed config and changes only
`model_args["prior_npz_path"]`. It then runs the same genmarg decode job as R1.

The prior weight therefore stays at the bed's value:
- `parse_bed_config` asserts `args["prior_weight"] == 1.0` (blankfree_genmarg_jobs.py:1387);
- the decode reads `model.prior_weight_at(1)` and asserts that it is a scalar with no schedule
  (lines 291-292).

The battery then checks R2 in two ways:
- R2's recorded `prior_npz` is not the bed prior;
- `np.allclose(uni.log_bi, -log 40)` holds (phi_competence_battery.py:595-597).

The two options decode differently. The 1/40 table at weight 1 adds a constant cost of log 40
(3.689 nats) to every token, so it acts as a per-token penalty. Weight 0 removes that cost, which
favours segmentations with more tokens.

## 2. E[d] is an unweighted mean over the 39 non-SIL phone types of phi's own duration law

The battery (and the A10 diagnostics before it) uses `expected_durations(checkpoint)` in
blankfree_phifirst_probe_read.py.
- Per type k: `E[d | k] = sum_{d=1}^{d_cap} d * p_phi(d | k)`, with
  `p = model.duration_log_probs().exp()`. This is phi's type-only duration head. No data, eta,
  position or posterior enters it, and each row is asserted to sum to 1 within 1e-9.
- Summary: `duration_summary` computes
  `mean_over_phone_types = sum(E[d | k] for k != SIL) / 39`.
  This is a plain arithmetic mean over the 39 non-SIL types, weighted by neither type frequency nor
  frames. SIL is reported separately as `sil`, and the per-type values are kept in `per_type`.
