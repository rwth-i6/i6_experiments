# Review: speech-llm b19f2ba, the k2 arm backend (`SAE_4A_lexlat.md` amendment 9)

Reviewer context: fresh; read the commit diff, the two reference paths (`lexlat_k2.run_probe`,
`lexlat_train`), the DP pack config and the new test suite; re-ran the suite myself in the k2 env.
Verdict: **DONE_WITH_CONCERNS** -- no silent no-op, no wrong number found in the term; the concerns
are one launch hazard and two disclosure items. Nothing in this commit is launched.

## What I verified

**(1) The term.** `lexlat_k2_train.py:305-327` builds the dense tensor, the supervision segments and
the pruned intersection line for line as `lexlat_k2.py:905-921` (`run_probe`): blank column 0 at
`NEG_INF`, emissions `e / temperature`, `seg = [arange(b), 0, feat_lens]`, `search_beam=20`,
`output_beam=8`, `min_active_states=30`, `max_active` from the config, `get_tot_scores(
log_semiring=True, use_double_scores=True)`. The graph's own scores are divided by the same
temperature (`graph()`, rescaled from a saved raw copy when the anneal moves) -- the probe's rule.
`Z_H` goes through unpruned `k2.intersect_dense` with the segments re-sorted by decreasing duration
and the scores scattered back (`log_z_h`); k2's own `dense_fsa_vec.py:69` requires exactly that
ordering for `intersect_dense`, and only for it, so the unsorted pruned call matches the probe.
It is a re-implementation of the probe's lines, not a call into them, so I checked the equality the
brief asks for: the enumeration test (`test_lexlat_k2_train.py:256`) prices `L_lex` against a brute
force over frame strings at tau 1.0 and 2.0 to 1e-6, and `test_log_z_h_is_the_frame_wise_logsumexp`
checks the `Z_H` plumbing against the closed form. Emissions are `dp_log_q`, the same tensor the
bed's lattice consumed (`train_steps/sae_blankfree.py:161`); `retained` is the 50 Hz unit-frame
count, the `l_tau` divisor; the term is `mean_kept[(-L_lex)/n_unit_frames]` with the lattice loss's
own `keep.sum()` denominator (`lexlat_k2_train.py:403-407`) and reaches `mark_as_loss` at
`scale = lam_lex` (`train_steps/sae_blankfree.py:164`). Gradient: only the dense tensor carries it;
the graph's scores are detached constants and the runtime is a plain object, both asserted
(`test_gradient_reaches_the_emissions_alone`), plus a finite-difference check of `dL/de`.

**(2) Default-off and hash.** Both edited model sources enter a config by `code_object_path`
(`blankfree_train_jobs.py:222,225`), so their contents do not hash; the ten new keywords are
default `None`/absent and only the k2 pack states them, so `get_model`'s `hashed_arguments` for
every banked arm is unchanged. The identity test is meaningful: same batch, same seed, both paths
executed, `torch.equal` on the loss and on both gradients, for every sub-epoch before the on-set and
for two on-sets, with a companion test that the on-set really moves the loss. Residual (disclosed by
the implementer): no test imports the real `train_step`; I read that block instead and its call
matches `LexlatK2Runtime.step`'s signature and the runtime's units.

**(3) Empty lattices.** `~torch.isfinite(z_hlg)` -> dropped from the numerator, counted, >10 %
raises; the `torch.where` keeps -inf out of the loss and the test asserts finite gradients and a
zero gradient row for the empty utterance, and that the term is exactly halved on a 2-of-2 batch.
`EMPTY_UNINFORMATIVE_FRAC = 0.02` is a reading rule on RETURNN's sub-epoch mean of the
`as_error` monitor `lexlat_k2_empty_frac`, as amendment 9.6 writes it.

**(4) Monitors.** All nine of amendment 9.9 present, 0-dim, finite (tested on both an escape graph
and one that produces an empty lattice). `_sec` is CUDA-synchronised on both sides of the two
intersections and excludes the arc-posterior pass. `_expected_words` / `_expected_escape_words` come
from `get_arc_post(log_semiring=True)` over the ragged aux labels with the graph's own convention
(`w+1` is word-LM id `w`, `#0` is `n_words+1`), cross-checked against the resource npz's
`n_words`/`unk_word`.

**(5) Rate term.** Untouched: the commit changes no DP file, and the k2 arm states no
`lexlat_resources`, so `lex_params is None` and the banked `lattice.py` / `_fd_passes` path carries
`l_tau`, the rate term and every `blankfree_*` column.

**(6) Null arm.** `shuffled_hlg()` builds `LexlatHLGBuildJob(shuffled=True)` with every other
argument taken from `config_sae_4a_lexlat_k2_v1`'s own build (same resources, bed, ladder, escape,
allocation); the job class is unmodified. Arm-to-graph mapping is by `ARMS[arm]["shuffled"]`.

**(7) Over-count job.** Exact host `k2.intersect` (no beam) in both semirings against
`lexlat.string_best_segmentation + sil_model_log_prob`, three per-token columns, median/p95/max/min,
the 0.05 acceptance read on `over_count`, partial every 50 strings. SIL: I read
`step0_strings.json` on disk -- both sets are SIL-free (0 SIL tokens in 177,275 gold and 167,510
private tokens), so `(n_words+1) log(1-p_sil)` is exactly the graph's silence-free path price and
the known constant is handled; `tropical_residual` is the detector if it ever is not.
Coverage note: 772 of 2,864 gold strings (27 %) and 22 private are excluded as adjacent repeats;
this is principled (the run-collapse topology emits none) and printed in the summary, but the gold
median is over 73 % of the set and should be read that way.
**Launchable now: yes.** `py_overcount() -> overcount()` touches `_hlg()` only and never
`_max_active()`, so filling `HLG_PATH` / `HLG_STATS_PATH` (the on-disk
`LexlatHLGBuildJob.rtX44PBJFNy1/output/{HLG.pt,build.json}`) is sufficient; `MAX_ACTIVE` may stay
`None`. Verified by reading the call graph, not by building the sisyphus graph.

**(8) Pack config.** Function-level diff against `config_sae_4a_lexlat_pack_v1`: `_flat_init`
identical; `_arm_train_config` differs only in the graph argument; `_register_epoch_reads` only in
the `name` prefix; arms, on-sets, seeds (`prepro._seed()`), schedules, N, kept epochs, mixing
(`LAM_LEX`, `RAMP` imported), the six paired rows, `PRIOR_GAP_EPOCHS = (10,20)` (ruling 6),
`SELECTION_EPOCHS = KEEP_EPOCHS` and every `PriorGapAnalysisJob` argument are the DP pack's.
`returnn_exe = K2_RETURNN_EXE` is set for the training job alone; every decode keeps the shared
`RETURNN_EXE`. The substituted phone LM is really the same model: I computed
`md5 = 61eae5635495b66cec12ef638bafa6d4` on
`PriorGapAnalysisJob.l0p0srBryKrs/work/lms/phones_o4.bin`, which is the digest the DP pack's own
comment records; the three original copies are gone from disk, and the pin is a content sha, so the
statistic's hash is where any copy would put it.

**Test re-run (mine).** `/e/scratch/spell/wu24/envs/sae_k2/bin/python -m pytest
src/speech_llm/sae/emc/test_lexlat_k2_train.py -q` -> **18 passed** (3.6 s), with
`PYTHONPATH=<setup>/recipe/2025-10-speech-llm/src:<setup>/tools/sisyphus:<setup>/recipe:<setup>/returnn`.

## Findings

1. `configs/config_sae_4a_lexlat_k2_pack_v1.py:553-560` -- the allocation assert prices the arm at
   `E1_TIME_FACTOR_BAR = 2.00`, the **DP route's** bar, which the DP route then missed by 41x. The
   assert therefore passes for any k2 cost; if the settling probe's per-step time is not read
   against the same bar before `py()` is launched, the pack burns a 4-GPU exclusive node and is
   killed at the 11.5 h clamp with no resume path. Fix: no code change required -- read the
   settling probe's seconds per step into the 1202 s / sub-epoch bar as a launch precondition, or
   turn the assert into one over the probe's measured number.
2. `lexlat_k2_train.py:481` -- `_expected_words` counts word ids `1..n_words`, which INCLUDES the
   escape word (`unk+1`), so `lexlat_k2_expected_words` is total word transitions, not lexical
   ones. Documented in the docstring; the Gate's engagement read must be the ratio
   `_expected_escape_words / _expected_words`, never `_expected_words` alone. No code change needed.
3. `config_sae_4a_lexlat_pack_v1.py:187` (NOT this commit, but blocking) -- the DP pack still lists
   only the three auto-cleaned phone-LM copies, so that config now raises at graph-build time. Fix:
   append the `l0p0srBryKrs` path there as the k2 pack does (one line, DP pack's file scope).

## NOT CHECKED

* `lexlat_k2_jobs._child_env` (reused unchanged by the over-count job) and the shared-env/scratch
  handover in a real run.
* `lexlat_k2.py`'s `compile_hlg` / `lexicon_to_fst` / escape pricing internals (reference code,
  unchanged by this commit; exercised only through the toy graphs).
* `get_arc_post` row-id mapping on a real ragged `aux_labels` (toy graphs only).
* `_pin(HLG_PATH, ...)` -- cannot resolve while the constant is unfilled.
* The full `py()` / `py_overcount()` graph build (the implementer's census claim is not
  independently reproduced here).
