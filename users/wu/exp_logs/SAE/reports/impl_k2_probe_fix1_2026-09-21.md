# impl -- k2 settling probe, review round 1 fixes (2026-09-21)

Dispatch: fix the five findings of `reports/review_k2_probe_2026-09-21.md` in the k2
lexicon-in-lattice settling probe. Nothing of this graph has run, so hash moves are free.
Code: `recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`.
Nothing was launched.

Status: **DONE_WITH_CONCERNS** -- all five deltas are implemented and exercised, but delta 4
(ESCAPE in the graph) forced one implementation decision that changes what route A prices and that
the orchestrator should see before the job is funded: **the escape graph cannot be determinized,
so the determinization stage of icefall's recipe is skipped for it** (section 4).

---

## 1. Finding 1 -- `int(hlg.shape[1])` on an FsaVec is `int(None)`

`k2.Fsa.shape` is `(num_states, None)` for an Fsa and `(num_fsas, None, None)` for an FsaVec, so
`hlg.shape[1]` was never a state count and `lattice.shape[1]` was `None` (verified in the
`sae_k2` env on a real FsaVec). `lexlat_k2._sizes` now reads the ragged arc structure, uniformly
for both ranks:

```python
axes = fsa.arcs.num_axes()            # 2 for an Fsa, 3 for an FsaVec
{"n_fsas": 1 if axes == 2 else int(fsa.arcs.dim0()),
 "states": int(fsa.arcs.tot_size(axes - 2)),
 "arcs":   int(fsa.num_arcs)}
```

`run_probe` reports `graph` (the HLG) and, per cell, `lattice_n_fsas / lattice_states /
lattice_arcs`. Test `test_sizes_read_a_saved_fsa_vec` saves `HLG.as_dict()`, reloads it the way
`run_probe` does (`Fsa.from_dict` -> `/ tau` -> `create_fsa_vec`), asserts `hlg.shape[1] is None`
and checks `_sizes` against the FSA's own `num_arcs` and a state count taken independently.

## 2. Finding 2 -- no partial output and no step abort

`run_probe` gained `partial_path` and `step_abort_sec` (CLI: `--partial-json`,
`--step-abort-sec`), and `LexlatK2ProbeJob` gained `out_partial` (`partial.json`, registered in
the config) and passes `step_abort_sec = LexlatEfficiencyProbeJob.STEP_ABORT_SEC` -- E1's own
constant, read, never re-typed.

* `partial.json` is rewritten after **every timed cell** and again after every batch, with
  `partial: true`, the ladder, the abort table and the cells so far (E1's `_write_partial`
  pattern, `lexlat_train_jobs.py:465-477`); `probe.json` is still written only by a completed run.
* A cell whose `seconds` exceed `STEP_ABORT_SEC` = 1200 s (one step above the bar for a WHOLE
  sub-epoch) sets `over_step_abort`, **aborts that `max_active` rung** (no later batch is timed at
  it), and the other rungs continue; when every rung is aborted the loop stops. The job reads an
  aborted rung as `time_read = FAIL` with the basis string marked `LOWER BOUND`, and prints the
  abort in `summary.txt`. A measured FAIL is the deliverable; a Slurm timeout is not.

## 3. Finding 3 -- the max-plus pass sat inside the timed region

k2 launches asynchronously, so a max-plus `get_tot_scores` called before the timer stopped drained
its kernels into `seconds_tot_scores` and its buffers into the peak. The timed region is now
exactly: `intersect_dense_pruned` -> `get_tot_scores(log_semiring=True)` -> `sum().backward()`,
each CUDA-synchronised, `seconds` their sum, `peak_reserved/allocated` read at its end. The
max-plus check is taken **after** it, behind a synchronisation, after `reset_peak_memory_stats()`,
with its own timer and peak: `seconds_max_plus`, `peak_reserved_gib_max_plus`. The job reports
`seconds_max_plus_per_batch_mean/sum` and states in `summary.txt` that the max-plus pass is
outside the timed region and outside the bar.

## 4. Finding 4 -- ESCAPE was missing from the graph

Implemented as `lexlat.py` does it, constants imported and never re-declared
(`ESCAPE_LENGTH_LOG_PROB`, `ESCAPE_WORD` from `lexlat`; `escape_prices` = `LexResources.build`'s
`escape_price`, i.e. the banked `escape_phone_log_prob` + `log 0.5`, SIL floored to `NEG_INF`).
In `L`: one extra state, an opening arc per non-SIL phone from the word-boundary state carrying
the word `<unk>`, a self-loop per non-SIL phone that extends the span, and a free exit through the
escape's own disambiguation symbol back to the word boundary with the usual optional silence. The
`#0` self-loops are added **before** the escape arcs, so the escape state carries no back-off loop
(the LM state does not move while a span is open). `G` keeps its `<unk>` arcs (they are the span's
word cost); only the strict build drops them (`NON_EMITTABLE_WORDS_STRICT`).

Equivalence test (`test_hlg_with_escape_equals_the_lexlat_string_scorer_on_an_oov`): on
`AA AE S K R IY M`, `AA AY S` and `Z F AO R` (each with an out-of-lexicon leading word) the HLG
max-plus total equals `lexlat.string_best_segmentation(..., escape=True) +
sil_model_log_prob(n_words)` to < 1e-6 (tolerance 1e-4), and the strict graph has no path at all
for them.

### The concern: the escape graph is not determinizable

`k2.determinize` **does not terminate** on the escape LG. Measured on the test fixture: the strict
build completes in 0.009 s end to end (HLG 100 states / 519 arcs); the escape build gets through
`L_disambig` (16/118), `G` (21/64), `compose` and `connect` (82/271) and then sits in
`determinize` for > 240 s with no result (killed). The cause is structural, not a size effect: the
escape loop and the lexical branch read the same phone with different weights, so the weighted
transducer violates the twins property and weighted subset construction never converges. Neither
LM pruning nor the theta ladder can fix it -- every rung would burn its 4 h and the job would
deliver no graph.

Decision taken (disclosed, and reversible at config level only by `escape=False`):
`compile_hlg` gained `determinize: bool = True` and `_cmd_build_hlg` passes `determinize=not
escape`. The escape build therefore runs icefall's recipe **minus** that one stage.

* It does not change the graph's scores: determinization is tropical-equivalent, so both the
  max-plus best path and the log-semiring total are the same object; the equivalence test above is
  taken on the non-determinized graph.
* It does change the graph's SIZE and hence the cost being measured -- the graph priced is the
  larger, ambiguous one. On the fixture: strict determinized HLG 100 states / 519 arcs vs escape
  non-determinized 137 states / 4340 arcs (the two effects, escape arcs and no determinization,
  are not separated by that number). Whether the full-lexicon escape LG compiles inside the 4 h /
  200 GB attempt is now an open empirical question the build job answers; the ladder still runs.
* `build.json` carries `determinized`, and both `summary.txt` files (build and probe) print it
  beside the "graph priced" line.

Observation, outside the dispatch and not acted on: for a MARGINAL, determinization is the wrong
reduction in any case -- a determinized graph keeps, per phone string, the max-scoring reading, so
the strict build's log-semiring total is a sum over strings of the best reading of each, not a sum
over segmentations. The strict build inherits that from icefall unchanged.

## 5. Finding 5 -- `n_neg_inf` double-counted

A genuine `-inf` is both non-finite and below the finite floor, and the two tests were added:
`int(((~finite) | (tot_cpu <= NEG_INF / 2)).sum())` now counts each utterance once. Exercised in
the GPU smoke below: the strict graph on an OOV utterance gives `tot_scores = [-0.333, -inf]` and
`n_neg_inf = 1` (it was 2).

---

## Checks run

| check | result |
|---|---|
| `test_lexlat_k2.py`, `sae_k2` env (`/e/scratch/spell/wu24/envs/sae_k2/bin/python`) | **25 passed, 1 skipped** (the skip is the pre-existing `kaldilm` SIGABRT on aarch64) |
| `test_lexlat_k2.py`, shared env (`.../conda/envs/speech_llm`) | **6 passed, 20 skipped** ("k2 lives only in the scratch env sae_k2"), rc 0 |
| `sis console -s` graph load of `config/sae_4a_lexlat_k2.py` | loads; 10 jobs |
| new hashes | `LexlatHLGBuildJob.rtX44PBJFNy1`, `LexlatK2ProbeJob.Tkl94t85j4pV` (both `__sis_version__ = 2`; nothing of this graph has run) |
| census, unchanged | `sae_4a_prepro_pack` **133**, `sae_4a_budget_pack` **775**, `sae_4a_lexlat_probes` **3**, `LexiconTrieBuildJob.rlMsnTBSZXsB` |
| GPU smoke of the new probe bookkeeping (toy HLG, 2 utterances, seconds of GPU on the login node) | partial.json written per cell with `partial: true`; `step_abort_sec = 0` aborts both rungs at batch 0, stops after one batch, `over_step_abort = true`, `aborted_max_active = {'1000': 0, '3000': 0}`; `seconds` == int+tot+bwd exactly, `seconds_max_plus` separate; `n_neg_inf = 1` on the `-inf` utterance; `_sizes(lattice)` = 2 fsas / 39 states / 67 arcs |

Not covered by any check: `LexlatK2ProbeJob.run()` itself (it needs RETURNN, the arm checkpoint
and a GPU node), so the read block and `summary.txt` render are reviewed, not executed. The full
`H . L . G` build has never been run -- everything above is on the test fixture.

One more fix was needed on the way: the new constant import had to be ABSOLUTE
(`from speech_llm.sae.emc.lexlat import ...`), because the jobs run `lexlat_k2.py` **as a script**
with the recipe `src` root as the child's whole `PYTHONPATH`, where a relative import has no
parent package; `test_module_runs_as_a_script` caught it.

## Files touched (all in `recipe/2025-10-speech-llm`)

* `src/speech_llm/sae/emc/lexlat_k2.py` -- `_sizes` off the ragged structure; `escape_prices`;
  escape arcs in `lexicon_to_fst`; `NON_EMITTABLE_WORDS(_STRICT)`; `compile_hlg(determinize=)`;
  `run_probe` timed region, partial writes, per-rung abort, single-count `n_neg_inf`; CLI
  `--strict-lexicon`, `--partial-json`, `--step-abort-sec`; docstring (ESCAPE, non-determinizability).
* `src/speech_llm/sae/emc/lexlat_k2_jobs.py` -- `LexlatHLGBuildJob(escape=True)`, `__sis_version__`
  2, "GRAPH PRICED" / "determinized" block in `summary.txt`; `LexlatK2ProbeJob` `out_partial`,
  step-abort wiring, aborted-rung reads, max-plus columns, `__sis_version__` 2.
* `src/speech_llm/sae/emc/test_lexlat_k2.py` -- `_build_hlg`, `_string_score`, the escape/OOV
  equivalence test, `test_escape_prices_are_lexlats_own`, `test_sizes_read_a_saved_fsa_vec`.
* `.../librispeech/configs/config_sae_4a_lexlat_k2_v1.py` -- `ESCAPE = True` (Design 1),
  `escape=ESCAPE`, `partial.json` registered.

Commit: staged by explicit path on `haotian_modality_matching_jupiter`, not pushed. Two other
working-tree changes in that checkout (`config_sae_1g_v1.py`, untracked
`config_sae_3e1_d6_swap_cont_v1.py`) are someone else's and were left alone.
