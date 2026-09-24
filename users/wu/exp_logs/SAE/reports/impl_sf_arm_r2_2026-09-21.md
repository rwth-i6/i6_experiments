# SAE 4a prior -- implementer round 2: the score-function arm `sf_20`

Binding spec: `SAE_4A_prior.md` "Training arm" (A1 / A3 / A4 / A5) and the **user override of
2026-09-21 ("do it, try 8 samples as well")**, which funds `sf_20` in the slot `soft_20_r03` held.
Built on round 1 (speech-llm `f99f9f6` + `0474d7f`); branch `haotian_modality_matching_jupiter`.

## What is on the branch

| file | new? | what it is |
|---|---|---|
| `src/speech_llm/sae/emc/sf_scorer.py` | new | the term: G = 8 exact FFBS draws, A1 reward, centred advantages, A4 path score, A5 normalisation |
| `src/speech_llm/sae/emc/sf_scorer_jobs.py` | new | `SfLamProbeJob` -- the override's own lam_sf probe |
| `src/speech_llm/sae/emc/test_sf_scorer.py` | new | 11 tests (below) |
| `src/speech_llm/prefix_lm/model/definitions/sae_blankfree.py` | edit | the default-off `sf_*` keyword block |
| `src/speech_llm/prefix_lm/model/train_steps/sae_blankfree.py` | edit | the guarded `sf` loss + monitors |
| `.../librispeech/configs/config_sae_4a_soft_pack_v1.py` | edit | `LAM_01` filled, `soft_20_r03` -> `sf_20`, pairings, probe, disclosed read |
| `config/sae_4a_sf_lam_probe.py` (setup dir, untracked) | new | the probe shim |

### The term (`sf_scorer.py`)

    term = -lam_sf * mean_b[ (1 / retained_b) * mean_g[ A_{b,g} * pathscore_{b,g} ] ]
    A_{b,g} = r(y_{b,g}) - mean_G r      (CENTRE ONLY -- no std division, grpo.py:11-41)

* **the draws** are falsifier (ii)'s own sampler, `blankfree_sampler.sample_blankfree_paths`, at
  the schedule's tau, G = 8, vectorised over G, inside the DP's own checkpointed backward
  recomputation.  Nothing keeps autograd through the lattice DP (memory
  `lattice-dp-autograd-memory`).  The per-step RNG is `stream_seed(SF_SAMPLE_SEED, epoch, step)`,
  so no two steps draw the same paths and a resumed sub-epoch redraws its own stream.
* **the reward** is round 1's, imported from `soft_scorer.py` (that module is NOT edited):
  `r(y) = log p_strong(y) - log p_uni(y)` summed over the SIL-dropped string, the same frozen
  3.3 M scorer and the same window unigram.  Strings over the scorer's 512 positions are masked
  out of the group mean and counted in `sf_masked_long`; they are never truncated, and they are
  never handed to `soft_log_prob` (which asserts the cap).  A group with fewer than two live draws
  is not scored.
* **the path score** is A4's Fisher identity, `grad log q(y|x) = E_{path|y}[grad log w(path)]`:
  the sampled path's own log weight, a gather of the scaled per-frame log-q terms plus the segment
  scores along the path.  `log Z` is not in it and does not need to be: it cancels under centred
  advantages.  Rewards are detached; the score is differentiable in theta and phi.

**Monitors** (`as_error=True`, so they never enter the loss): `sf_reward_mean` (the greedy decode's
reward per token -- the quantity the soft arm's `-0.67 -> +0.95` dev band is in),
`sf_reward_mean_sample` (the draws' mean), `sf_reward_std_within` (median over utterances, nats per
utterance), `sf_unique_strings`, `sf_masked_long`, `sf_adv_absmean`, plus `sf_lam`,
`sf_num_samples`, `sf_scored`, `sf_term_mean`, `sf_tokens`, `sf_decode_tokens`.

### Why a new module and a new path score

`blankfree_sampler.path_score` gathers on an **expanded** view (`acoustic.unsqueeze(1).expand(...)`
and the same for the segment table).  `GatherBackward` allocates `zeros(input.sizes())` -- the
EXPANDED shape -- which is the 224.61 GiB OOM the round-1 probe died of.  `sf_path_score` gathers
on the un-expanded tables instead and transposes G into place.  It is NOT a rewrite of the
sampler's own function: `blankfree_sampler.py` is untouched so falsifier (ii)'s banked reads stay
reproducible, and a test asserts the two agree.

Measured on GPU (test 3): backward of the expanded spelling allocates 8.67 x the table bytes
(the G = 8 copies), the flat spelling 1.67 x.

## Tests -- `test_sf_scorer.py`, 11 passed

1. the path-score gradient equals the autograd gradient of the fixed-string log weight, taken as
   the posterior-weighted mean over paths (the Fisher identity), `atol = 1e-12`;
2. it equals the sampler's own `path_score` in value (`torch.equal`) and in gradient
   (`atol = 1e-15`; the residual is scatter-add REASSOCIATION over the G draws, the same caveat
   `reports/impl_soft_arm_fix1_2026-09-21.md` recorded for the segment gather);
3. the backward does not materialise the expanded tables (CUDA, the numbers above);
4. the reward on a known string is `soft_scorer`'s reward for that string;
5. `kept_strings` drops SIL and counts distinct draws;
6. centred advantages sum to zero and exclude masked draws;
7. **with the block off the loss and every gradient are `torch.equal` to the banked path**;
8. eight draws are eight distinct strings at tau 5 and tau 8 (`sf_unique_strings == 8`);
9. a draw over the scorer's positions is masked out and counted;
10. an utterance whose group collapses to one live draw is not scored;
11. the probe's `summary.txt` renders every field it banks.

`test_soft_scorer.py` (23) re-run: passed.  `test_blankfree_permute.py` (7),
`test_bt_blankfree.py` (8), `test_blankfree_trigram.py` (10): each passes alone.  Run as one
pytest invocation in that order, two trigram tests fail on a leaked `rf` run context
(`run_ctx.py:279`); **the same two fail identically with both edited files reverted to `HEAD`**, so
it is a pre-existing test-isolation artefact of that module pair, not this change.

## The probe (`SfLamProbeJob`, hash `SfLamProbeJob.OHrcN9pEuXni`)

The override: lam_sf comes from "its OWN probe at ctrl_20 ep4 with THIS scorer, not the first-pass
ratio of falsifier (ii)" (which was ctrl_50, another temperature, another scorer instance).
Operating point = the soft probe's, so the two ratios are on one axis: the frozen `ctrl_20` ep4
seed-0 checkpoint, 3 batches at 88,000 padded frames / max_seqs 128, tau from the arm's own anneal
at sub-epoch 4, the A5 per-frame convention (ONE ratio column -- the open question of
`reports/impl_blankfree_sampler_2026-09-20.md` item 2 is settled by what the pack runs).
`rqmt = {cpu 16, mem 64, gpu 1, gpu_mem 96, time 1.0}`.

It runs **two real backwards at the run shape per batch** (the sf term's and `l_tau`'s, over theta
+ phi together) and records the peak allocated memory and the step seconds beside the ratio -- the
round-1 OOM was invisible to every CPU test, so the probe is the memory read as well as the weight
read.  Output: the median ratio and `lam_sf = target / ratio` for 0.1 x (preferred) and 0.3 x
(ceiling), plus every `sf_*` monitor.

## The pack config

* `LAM_01 = 0.326639` -- read off the FINISHED `SoftLamProbeJob.wAJQ26T7iZzX`
  (`output/summary.txt`: median ratio 0.306148, so 0.1 / 0.306148), the row `SAE_4A_prior.md`
  banks.  `LAM_03 = 0.979918` is recorded from the same file; **no arm takes it** (the override
  drops `soft_20_r03`).  `LAM_SF_01` is `None` and is now the only unfilled weight; `_lam` refuses
  it by name and points at the probe shim.
* arms: `soft_20` (LAM_01, s0), `soft_20_s1` (LAM_01, s1), `sf_20` (LAM_SF_01, G = 8, s0),
  `softshuf_20` (LAM_01, s0, derangement seed 0).  `ARMS[arm]["kind"]` picks which default-off
  block the arm states; the model asserts the two are mutually exclusive.
* pairings: `soft_20 - ctrl_20`, `soft_20_s1 - ctrl_20_s1`, `softshuf_20 - ctrl_20`,
  `soft_20 - softshuf_20`, **`sf_20 - ctrl_20`**, **`sf_20 - soft_20`** (the `soft_20_r03` row
  goes with the arm).
* disclosed reads for `sf_20`: the prior-gap rerun at ep10 / ep20 (`PRIOR_GAP_ARMS = tuple(ARMS)`,
  so it comes with the arm, as the override asks) **and** `_register_sf_reads` -- falsifier (ii)'s
  own `SampledRewardProbeJob` at the arm's ep10 / ep20 checkpoints, the same 300-utterance seeded
  dev-other subset, G = 8, but THIS scorer and the arm's own prior.  The UNINFORMATIVE rule
  (fraction >= 0.95 at both) stays pre-registered in that job's docstring; it is not re-typed.
  Read-side only: nothing it writes is read by training or selection.
* efficiency: a comment at the allocation assert states the override's rule -- one extra DP pass
  per step, the ep1 sec-per-sub-epoch read applies PER ARM against 2.00 x 601 s, and if `sf_20`
  alone exceeds it the pack relaunches without it rather than slowing the three soft arms.

## Census -- nothing banked moved

| graph | jobs | note |
|---|---|---|
| prepro pack | 133 | unchanged |
| budget pack | 775 | unchanged |
| lexlat probes | 3 | unchanged |
| falsifier (ii) | 3 | `d1NoUQJ2EXN5`, `HVHIlaUkIkVi`, `Y81PrZ6fWWKu` -- unchanged |
| soft lam probe | 1 | **`SoftLamProbeJob.wAJQ26T7iZzX`, unmoved** (`soft_scorer.py` was not edited) |
| sf lam probe | 1 | `SfLamProbeJob.OHrcN9pEuXni` (new) |

`py()` (the pack) **refuses to build**: `LAM_SF_01` is `None` and `_lam` fires by name.  With a
throwaway placeholder injected in a scratch script (0.123456 -- NOT a chosen weight, never written
to the file) the whole graph builds: 196 jobs, one `PackedBlankfreeTrainJob`, two new
`SampledRewardProbeJob` instances for the sf read.  **Those ids are functions of the placeholder,
so there is no pack hash to report until the probe reads out.**  A separate check confirms the
`sf_20` arm's written config carries `sf_scorer / sf_lam / sf_unigram_npz / sf_num_samples /
sf_sample_seed` and no `soft_*` key, and the three soft arms carry no `sf_*` key.

## Deviations from the dispatch, and open points

1. **The shim is `config/sae_4a_sf_lam_probe.py`, not `config/sae_4a_sf_probe.py`.**  The latter
   exists and is the FINISHED falsifier (ii) graph; overwriting it would orphan three banked reads.
2. `LAM_03` was filled (0.979918) although no arm takes it, so that `LAM_SF_01` is the only `None`
   left, as the coordinator asked.
3. **No pack hash yet** -- see above; it is a function of `LAM_SF_01`.
4. `sf_reward_mean` carries the **decode** value and `sf_reward_mean_sample` the **draws'** mean
   (the override asks for "hard decode and sample mean"); the decode is the banked greedy-argmax
   collapse, not a second Viterbi pass, so the step still costs exactly one extra DP pass.
5. Per-arm read names keep this pack's `soft/` phase prefix (`PREFIX` / `ALIAS` are the pack's), so
   the sf arm's reads are named `soft/sf_20/...`.  Cosmetic; renaming would churn every arm's
   alias.
6. `soft_scorer.py` and `blankfree_sampler.py` are NOT edited, by design (hash and reproducibility).

## Checks run

* `pytest test_sf_scorer.py test_soft_scorer.py` -> **34 passed**.
* `pytest test_blankfree_permute.py test_blankfree_trigram.py test_bt_blankfree.py` -> 23 passed,
  2 failed; reproduced identically at `HEAD` without this change (see above).
* six graph loads (the census table) + the placeholder pack load + the model-args check.

Loading a graph is not a result: nothing here has run on GPU.  The first real read is
`SfLamProbeJob.OHrcN9pEuXni`.
