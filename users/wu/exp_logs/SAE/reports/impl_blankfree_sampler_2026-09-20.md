# Blank-free path sampler and the sampled-reward probe (SAE_4A_prior, training arm)

Implementer round, 2026-09-20/21.  Two new modules, one new config, two new test modules.  Nothing
existing was edited: every blank-free module named in the brief (`lattice.py`, `rate_term.py`,
`candidates.py`, `reverse.py`, `blankfree_*_jobs.py`, `train_steps/sae_blankfree.py`,
`definitions/sae_emc.py`) is untouched, and so are `prior_gap.py`, `neural_phone_lm.py`,
`prior_probe.py` and every live config.  `PackedBlankfreeTrainJob.ks7CbtlvpcIL` was RUNNING
throughout (10 h at the time of the commit), so the no-edit constraint mattered.

## Files

| file | what |
|---|---|
| `2025-10-speech-llm/src/speech_llm/sae/emc/blankfree_sampler.py` | forward-filtering backward-sampling over the blank-free trigram lattice (new) |
| `.../sae/emc/test_blankfree_sampler.py` | 9 tests, brute-force oracle (new) |
| `.../sae/emc/blankfree_probe_jobs.py` | `SampledRewardProbeJob`, the pre-launch falsifier (ii) (new) |
| `.../sae/emc/test_blankfree_probe_jobs.py` | 15 tests of the pure parts and the render (new) |
| `.../librispeech/configs/config_sae_4a_sf_probe_v1.py` | three job instances, ep1 / ep4 / ep10 (new) |
| `<setup>/config/sae_4a_sf_probe.py` | workspace stub (untracked workspace file) |

## Deliverable 1 -- the sampler

`sample_blankfree_paths(...)` draws `G` complete paths per utterance from the EXACT posterior
`q_theta(path | x)` of the lattice, at the lattice's own temperature, prior weight and anchor
weight.  It reuses `lattice.py`'s own internals by import (`_forward_ctx`, `_forward_step`,
`_prior_term`, `_use_matmul`, `forward_log_z`, `scaled_seg_pad`, `check_batch_budget`), so the
weights it samples under are bit-identical to the ones the training DP normalises -- no copied
code, no edit.  D4 checkpointing is reused as-is: the checkpoint table is filled by
`forward_log_z(..., table_out=table)`, and the backward pass replays each chunk with
`_forward_step` under the SAME `_forward_ctx`.  The replay buffer is sized `ck + 1` so the terminal
`fwd_{T_b}` -- which the stored table does not hold -- comes out of the same replay instead of a
per-utterance extra step.

Per draw it returns the frame path, the segment structure (phone, start, duration), the collapsed
string, and the path log weight DECOMPOSED into (a) the recogniser log-q gathers, (b) the
segment/reverse scores and (c) the prior term, plus `log Z`.  `path_score(...)` re-gathers (a)+(b)
DIFFERENTIABLY from a live `log_q` / `seg_table`, which is the Fisher-identity estimator amendment
A4 needs: the prior term is constant in theta and phi and `-log Z` cancels under centred
advantages, so (a)+(b) is the whole of `grad log w(path)`.

Validation (`test_blankfree_sampler.py`, 9 tests, all passing):

* **brute force.**  Two synthetic lattices, every legal path enumerated with its own log weight.
  T = 3 (60 paths) and T = 4 (564 paths); 20,000 draws each.  Per-path frequency within 4 standard
  errors on every path with >= 20 expected counts (all 60 resolvable at T = 3, 331 at T = 4), and
  chi-square 61.11 on 59 dof (bound 113.3).  `log Z` from the sampler equals the enumerated
  `logsumexp` to 1e-10.
* **decomposition.**  (a) + (b) + (c) equals the enumerated path score over s_len 6-9 x tau 1/8 x
  prior weight 1/0, and `log w - log Z <= 0` on every draw.
* **low temperature.**  No max-semiring decoder exists anywhere in `emc/` (there is no Viterbi over
  this lattice to compare against, and `lattice.py` was not to grow one), so the MAP path is taken
  from the same enumeration: at tau = 0.01 the top path leads by 33 nats and all 8 draws ARE that
  path, with `log w == log Z` to 1e-9.
* checkpointing invariance (ck 0 vs 1 / 2 / 3 / 32 bit-identical), stride-3 boundary lengths at
  band 1 and 2, padded batches, and G = 1 / 8 / 64 the same sampler.

Cost at the production shape (GH200, synthetic inputs at `B = 128, S = 687, T = 229, W = 25,
|h| = 1681, float64, ck = 32, matmul`, i.e. the bed's 88,000-frame batch):

| pass | time | peak allocated |
|---|---|---|
| `lattice_forward_backward` (the training DP) | 3.92 s | 15.67 GiB |
| `sample_blankfree_paths`, G = 8 | 2.33 s | 10.94 GiB |

0.59x the time and 0.70x the peak of the DP pass the arm already runs, so a sampled arm costs about
1.6x a plain one per step and fits the same 96 GiB GPU.

## Deliverable 2 -- `SampledRewardProbeJob`

One forward-only GPU job per checkpoint (`gpu 1, mem 64, time 2 h`, plus `gpu_mem 96` -- the value
every blank-free GPU job of this bed carries; the DP needs it).  It rebuilds the arm's model from
the arm's OWN written `returnn.config`, loads the checkpoint with `_load_state`, and:

1. takes the FIXED 300-utterance dev-other subset (seeded sample of the sorted gold tags, so it is
   the same 300 for every checkpoint and depends on no checkpoint);
2. EVAL mode: one recogniser forward per batch, the reverse segment table, G = 8 draws at the
   sub-epoch's tau (imported from `blankfree_budget_jobs.budget_temperature_schedule`), and the
   greedy decode, which is compared against the BANKED `greedy_phones.json` of the same checkpoint;
3. scores every draw, the greedy string and the gold string under the live prior's unigram and
   trigram and under the frozen neural phone LM, SIL dropped from every string first;
4. TRAIN mode: the A4 score-function term's gradient norm over `l_tau`'s, from one forward per
   batch, over theta and phi together.

`probe.json`, `per_utt.json` and `probe.md` are written; the render is tested off the record so no
banked field is missing from the report.

Pins (all frozen `tk.Path`s with a `hash_overwrite` whose key carries a sha of the realpath, the
`config_sae_4a_prior_gap_v1._pin` idiom; the decode, prior and VAD pins are reused from
`config_sae_4a_private_code_v1`, i.e. literally the files the Step 0 read consumes):
`PackedBlankfreeTrainJob.ks7CbtlvpcIL/output/ctrl_50/{returnn.config, models/epoch.00{1,4}.pt,
models/epoch.010.pt}`, `BlankfreeVadHdfJob.SAjz8y1cT06g` dev-other feats/units,
`BlankfreeGreedyPerJob.{XN6vhGGyKQu5, Vz6QOYliPU40, dG4n46xTRSl0}/output/greedy_phones.json`,
`PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz`,
`NeuralPhoneLmTrainJob.Iv6P6YVPNWmB/output/model.pt`, `GoldPhonesJob.ZGSp0hxyd2YP`.
No banked job is reconstructed, so loading the config cannot re-fund anything.

**Checkpoints that exist**: `epoch.001.pt`, `epoch.004.pt`, `epoch.010.pt` (also 025 and 050/050.opt)
-- the arm keeps `[1, 4, 10, 25, 50]`, so ep1 IS on disk and all three instances are buildable.
Their temperatures, taken from the schedule and matched exactly against the arm's written
`temperature_schedule`: ep1 8.0, ep4 5.039684199579493, ep10 2.0 (the anneal ends at sub-epoch 10).

**Job hashes** (graph loads, three distinct jobs, nothing launched):

    ep1   SampledRewardProbeJob.d1NoUQJ2EXN5   alias sae/4a/sf_probe/ctrl_50/ep1/dev-other
    ep4   SampledRewardProbeJob.Y81PrZ6fWWKu   alias sae/4a/sf_probe/ctrl_50/ep4/dev-other
    ep10  SampledRewardProbeJob.HVHIlaUkIkVi   alias sae/4a/sf_probe/ctrl_50/ep10/dev-other

## Checks run

* `test_blankfree_sampler.py` 9 passed, `test_blankfree_probe_jobs.py` 15 passed,
  `test_lattice.py` + `test_blankfree_lattice.py` 24 passed (untouched) -- 48 in 23 s.
* graph load of the new config: three jobs, hashes above.
* **one real batch, end to end** (a 15-minute `srun`, NOT a sisyphus launch): the job object built
  outside the graph with a scratch WORK_DIR at 8 utterances and G = 4 on the ep4 checkpoint,
  `run()` finished in 33.7 s.  It reproduced the BANKED greedy decode on 8 of 8 utterances (the
  rebuild / checkpoint / HDF / feature path is right), `max log w - log Z = -137.5`, no Z = 0
  utterance, every scorer row rendered.  This is a SMOKE at 8 utterances -- its reward numbers are
  not a read and are not reported as findings; the read is the 300-utterance job.

## Decisions, and what the spec leaves open

1. **The lexicon ESCAPE row is OMITTED** (the brief's own escape clause).  `best_segmentation`,
   `Lexicon`, `KenLMWordLM`, `ESCAPE_WORD` and `drop_sil` import fine from `prior_gap.py`, but the
   word trigram LM is fitted inside `PriorGapAnalysisJob._lmplz` (a job METHOD) and neither it nor
   the lexicon is a job OUTPUT -- `PriorGapAnalysisJob.Gct95xZHe0zt/output/` holds only
   `per_utt.json`, `prior_gap.json`, `prior_gap.md`.  Adding the row would mean editing
   `prior_gap.py` (forbidden, and it is another read's live module) or re-deriving the lmplz flags,
   which is an experimental constant I will not choose.  `r_nn` is scored and is the row the
   falsifier's rule is stated on.
2. **The sf term's per-utterance normalisation is NOT pinned by the design.**  `l_tau` is
   `mean_b(-log Z_b / retained_b)`, but A4 states the sf term per utterance with no batch
   normalisation, and the two candidates differ by exactly the `1 / retained` factor.  The job
   reports BOTH (`utterance_mean` and `per_frame`) with their own `lam_sf` at 0.1x and 0.3x of
   `l_tau`, and prefers neither.  **The planner has to pick one before `lam_sf` is set** -- on the
   8-utterance smoke they differ by a factor of ~285.
3. Module mode: sampling and scoring in EVAL (the mode the banked greedy decode and the posterior
   dumps are in, and the 8/8 agreement confirms it), the gradient ratio in TRAIN (the mode the
   training step and `BlankfreeGradNormProfileJob` run in).  Both are stated in `probe.json`.
4. The ratio is measured on EVERY batch of the subset and the MEDIAN is reported, so `lam_sf` does
   not depend on which utterances landed in one batch.  Per-batch values are banked too.
5. Advantages are CENTRED ONLY (`A_g = r_g - mean_G r`), never divided by the group std --
   `grpo.py:11-41` and the recorded pitfall; the within-group std is REPORTED, not applied.
6. Gold is scored as a disclosed label-using diagnostic and is stated as such in the job docstring,
   in `probe.json` and in the first lines of `probe.md`.
