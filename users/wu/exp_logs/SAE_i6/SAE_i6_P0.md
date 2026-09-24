# SAE_i6 P0 — port to i6: reproduce the banked baselines and verify every component

## State

Active: environment build (conda, k2 from source, KenLM) and the test-coverage survey; then the i6
`settings.py`, the reproduction config, code review, design review, launch. No job has run yet.
Next experimental action: launch the P0 reproduction graph (section "Runs") after the pre-launch
checks pass.

## Objective

Establish a trusted i6 baseline for the SAE_i6 campaign before any new experiment: (1) the ported
package (`recipe/i6_experiments/users/wu/experiments/unsupervised_asr`) builds its inputs from raw
public sources and trains on i6 hardware; (2) its banked JUPITER numbers reproduce within the bed's
known noise; (3) every component, above all the loss terms, computes what `SAE_i6_ref_objective.md`
says. Everything later compares against the i6 controls produced here, not against JUPITER numbers.

## Constraints

- Label quarantine and the other standing constraints of `SAE_i6_ref.md` section 2.
- No change to the method. Permitted i6 adaptations, each disclosed below when made: interpreters and
  tool paths, Slurm resources (partition, memory, time), the audio generation label if the ffmpeg pin
  check fails, and a batch-shape change ONLY if the reference shape (88,000 padded frames, 128 seqs)
  does not fit a 46 GB L40S; a batch change is a new operating point and voids the step-1 clause of
  G0.R1.
- GPU pool: gpu_48gb L40S (5 GPUs per user). The reference was one GH200 96 GB per arm.

## Runs

| run | entry point | why |
|---|---|---|
| input graph | `config/base.py` inputs (download, ffmpeg pin, ogg, rVAD, w2v2 L15, units, eta, CV split, phone text, prior, duration prior, gold) | everything downstream |
| `ctrl_20` | `config/base.py` | the control of every later pack; banked PER 0.874568 |
| `k2lat_20_ma3000` | `config/k2_word_lm.py` preset | the only banked arm clearly off the control (-0.0560); exercises the k2 path |
| gold-phi supervised reverse init | `config/supervised_init.py` `gold_phi()` (analysis only, disclosed label use) | exercises the reverse model and its DP on a deterministic target; banked NLL 3.2888 |

`ctrl_20_x60`, `k2lat_20_ma3000_x60`, `off4_k2lat_20` and the never-run default `k2_word_lm` are
not part of P0 (E60 already showed a plateau; they cost 3x). Reads: the package's registered reads
(greedy PER at kept epochs, derangement and decode gaps and JS rows at the final epoch, paired delta).

## Gates (pre-registered 2026-09-24, before any job)

The banked per-utterance hypotheses are not available on i6, so the reproduction reads are
tolerance reads against banked scalars. The tolerance 0.03 PER is the measured run-to-run spread of
identical cold configs by sub-epoch 4 (0.01-0.03; `SAE_i6_ref.md` section 3).

- **G0.R1 `ctrl_20`.** dev-other greedy PER within +-0.03 of banked at each kept epoch (1 / 4 / 10 / 20:
  0.855 / 0.875 / 0.869 / 0.874568), AND, if the batch shape and audio are the reference ones,
  step-1 l_tau within +-0.01 of -0.350 and expected tokens within 2 % of 63.821. The PER clause alone is
  weak (any content-free arm lands in the 0.83-0.91 band); the step-1 clause is the sharp one.
- **G0.R2 `k2lat_20_ma3000`.** PER at 20 within +-0.03 of 0.818615, AND paired `k2lat_20_ma3000` minus
  i6 `ctrl_20` at 20 with CI upper bound < 0 and point estimate in [-0.0760, -0.0360] (banked
  -0.0560 +- 0.02).
- **G0.R3 gold phi.** held-out NLL per frame at epoch 8 within +-0.02 of 3.2888.
- **G0.V components.** Every priority-1 test of the verification plan
  (`reports/test_plan_2026-09-24.md`) passes, including the package's existing suite. A check, not a
  result: it licenses "the tested instances compute the objective note's quantities", nothing about
  the reproduction numbers.

Verdict: REPRODUCED if G0.R1-R3 and G0.V all pass; a failed R-clause goes to the debugger before any
rerun; P0 closes on REPRODUCED or on a user decision.

## Deviations from the reference (filled as they are made)

- Hardware: L40S 46 GB (sm_89) instead of GH200 96 GB (sm_90); x86_64 instead of aarch64.
- Env (`reports/env_build_2026-09-24.md`): every environment.yml pin unchanged; BLAS MKL instead of
  OpenBLAS; k2 source build of the pinned commit for sm_70/sm_86 (SASS sm_86 runs on the L40S);
  librosa 0.11.0 added (i6_core imports it; the spec omitted it). Recipe checkouts newer than the README
  pins: i6_core 4537aaf (pin ca161b7 is an ancestor; three later commits: JAX checkpoint support in
  ReturnnTrainingJob, a new ExtractOovWordsFromTextJob, an optional prettify), sisyphus a567fa7 (pin ddcd028 plus later fixes); RETURNN for jobs
  is cloned at the pinned commit with the shipped patch.

## Results

None yet.
