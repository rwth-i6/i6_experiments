# Code review: the `#0` back-off self-loop placement as an HLG build option (6bd6f1f), 2026-09-21

Reviewed: `recipe/2025-10-speech-llm` commit `6bd6f1f` (parent `f240391`, a separate change, not
reviewed here).  Implementer report: `reports/impl_k2_hlg_loops_2026-09-21.md`.  Context:
`reports/audit_k2_overcount_2026-09-21.md` 2-3, `SAE_4A_lexlat.md` Results "Over-count read,
amendment 9.5" and line 386 (the decision).  Read-only; I ran the fixture tests, my own census and
my own SIL checks, and changed nothing.

Verdict: **DONE_WITH_CONCERNS**.  The delta does what the dispatch asks and nothing else that I can
find; the two concerns below are about what has to be true for the resulting numbers to be read as
intended, not about the graph construction.

## Findings

**F1 (the deciding column depends on another worker's UNCOMMITTED edit to the same file).**
`SAE_4A_lexlat.md:386` fixes the read that decides amendment 9.5 as `over_count_exact` on the
rebuilt graph.  At commit `6bd6f1f`, `config_sae_4a_lexlat_k2_pack_v1.py:787` `py_overcount()`
registers only `overcount()`, `overcount_exact()` on the BANKED graph, and
`overcount_word_boundary()` -- the max-plus-referenced column on the new graph.  The exact column on
the new graph exists only in the working tree, through the other implementer's uncommitted change to
the SAME file (`overcount_exact(job, *, tag="")` and
`out["word_boundary_exact"] = overcount_exact(out["word_boundary"]["overcount"], tag=...)`, working
tree lines 407 and 800-802) plus uncommitted `lexlat_k2_overcount_exact_jobs.py`.  `git status`
confirms both are `M`, not committed.  Concrete failure: if that working tree is lost or the edit is
committed without this one, loading `config/sae_4a_lexlat_k2_overcount.py` launches the rebuilt
graph's 16-CPU build plus a ~77-min over-count run and still does not produce the column the phase
says decides the gate.  Two workers on one config file is also what the concurrency rule forbids.
Not a defect of 6bd6f1f's own code; it is the seam between the two, and it needs one commit that
contains both halves.

**F2 (the "differ by the loop placement and by nothing else" claim is not enforced).**
`config_sae_4a_lexlat_k2_pack_v1.py:384` (docstring of `overcount_word_boundary`) states that the two
`over_count` columns "differ by the loop placement and by nothing else".  Both builds carry
`prune_ladder = (0.0, 0.5, 2.0)` and `LexlatHLGBuildJob` walks that ladder until a rung compiles
inside `attempt_hours = 4.0`.  The banked build settled at `chosen_theta_nats = 0.0`
(`LexlatHLGBuildJob.rtX44PBJFNy1/output/build.json`, 173 s, 26.0 GiB peak).  Nothing asserts that the
`word_boundary` build settles there too; if it falls back to theta = 0.5 the two columns differ by LM
pruning as well and the difference cannot be attributed to the loop placement.  The new L is strictly
smaller (it loses one `#0` loop per word-interior state, ~10^6 of them over 151,731 pronunciations),
so a fallback is unlikely -- but it is one line to check, and it must be checked before the columns
are compared: `chosen_theta_nats` in the new build's `build.json` must read `0.0`.

## What I checked and found sound

**(1) No back-off route is lost under `word_boundary`.**  I rebuilt the fixture `L` under both
placements and read the loop set off the arcs: `"all"` -> states `[1..13]`, `"word_boundary"` ->
`[1]` (`loop_state`) exactly, as the implementer states.  Every state that loses its loop reaches
`loop_state` without emitting a word, so every back-off route stays reachable there:
`sil_state`'s only outgoing arc is `(2 -> 1, token SIL, word eps)` (measured); the escape exits go to
`loop_state` or to `sil_state` and thence to `loop_state`; `start_state` -> `loop_state` directly or
through `sil_state`; and the `-1` arc leaves `loop_state`, so G's end-of-string closure is taken
there.  `lexicon_to_fst` asserts `loop_state` carries the loop in either placement
(`lexlat_k2.py:513-519`).  Word labels leave `loop_state` alone (the first arc of every entry; the
escape's opening arcs), so the placement resolves to the trie root and to nothing else.  Structurally,
the set of G label sequences is the same under both placements -- an interior `#0` is emitted after
its word's label, exactly as a `loop_state` `#0` is -- so the MAX-PLUS total is unchanged by
construction, not only by measurement.

**The fixture does NOT exercise the SIL cases the dispatch names**, so I ran them myself.  The new
tests use `_LOGSUM_STRINGS` plus `AA AE S K R IY M`, and `_string_score` pins one phone per frame,
so no path can take a SIL arc; `test_hlg_max_plus_equals_the_lexlat_string_scorer` is SIL-free too.
On five SIL strings (`AY S SIL K R IY M`, `SIL F AO R`, `F AO R SIL AY S`, `F AO R AY S SIL`,
`AA AE SIL S K R IY M`) x the strict and the escape graph, the max-plus difference is `+0.00e+00`
exactly and every score stays finite, so nothing is lost at a word-after-SIL, SIL-initial,
SIL-final or escape-plus-SIL boundary.  (This is a coverage note, not a defect; also, per the audit,
0 of 2,864 strings in either over-count set contains SIL, so it cannot move the full-scale column --
it can move the arm.)

**(2) No residual double position.**  `loop_state` is visited exactly once between consecutive word
labels: every cycle back to it emits either a word (lexical arcs, the escape opening arc) or `#0`
itself, and the `#0` chain walks G's back-off arcs strictly downward in order, so no route is
consumed twice.  Measured confirmation: under `word_boundary` the graph's log-sum is INVARIANT to
inserting a SIL at a word boundary (`F AO R SIL AY S` = `F AO R AY S SIL` = `F AO R AY S` =
-2.788230; `SIL F AO R` = `F AO R` = -2.078779, which is the implementer's -2.102869 + 0.024090),
while under `"all"` it is not (-2.758682 / -2.755154 against -2.766935) -- the extra `sil_state`
position showing up exactly where the mechanism predicts.

**(3) The default reproduces the banked graph.**  My own census (three shims, sis env, nothing
launched) on the live k2 graph (12 jobs), the over-count graph and the official graph (17 jobs):
`LexlatHLGBuildJob.rtX44PBJFNy1`, `LexlatK2ProbeJob.Tkl94t85j4pV`,
`LexlatK2OvercountJob.f5Ljn6twbc4b`, `LexlatK2OvercountExactJob.4yKkB64ZDvJ4`,
`LexlatOfficialHLGBuildJob.GxCkDk90bQpT` / `YJsdBTcEQJz9` / `NrghE6fnf5hc` and the three official
probes all reproduce, and every one of them that has run is on disk at that id.  New ids, exactly
three from this change: `LexlatHLGBuildJob.cdcxYJMjiYj5` (the same id from both configs),
`LexlatK2ProbeJob.by7UYEjtYdel`, `LexlatK2OvercountJob.x4a6MX6ZgJ5n`.  (My over-count census shows 5
jobs, not the report's 4, because the working tree carries the other worker's second exact job
`LexlatK2OvercountExactJob.vNM59xVEnC2G`; see F1.)  `_add_self_loops` at `col = 2` is the old
expression character for character, and `lexlat_k2_official.py` and `test_lexlat_k2_train.py` call
`lexicon_to_fst` without the argument, so the graphs themselves are byte-identical.  `build.json`
gains a `backoff_loops` key; every reader of it tolerates its absence on a pre-option build
(`lexlat_k2_jobs.py:273` `build.get("backoff_loops", "all")`, `backoff_loops_of` in the exact job).

**(4) The decomposition test computes the route sum independently of the graph code.**
`test_lexlat_k2._route_sum_score` / `_backoff_chain` / `_explicit_arc` read `arc_key`, `arc_logp`,
`arc_next`, `backoff`, `backoff_state` off the LM tables with `np.searchsorted` and a forward pass
keyed on the successor state; they call nothing in `lexicon_to_fst`, `g_fsa` or `compile_hlg`
(only `K.sil_model_log_prob`, a three-term formula, and `K.DEFAULT_LM_ORDER`).  The end-of-string
closure matches `g_fsa`'s "one `-1` arc from EVERY state at score 0".  I reran the suite:
`test_lexlat_k2.py` + `_official` + `_train` = **67 passed, 1 skipped**, matching the report.

**(5) The new registrations change nothing but the graph.**  Per-argument `sis_hash_helper`
comparison: the two `LexlatK2ProbeJob`s differ in `name`, `hlg`, `hlg_stats` and in nothing else
(`returnn_config`, `checkpoint`, `max_active`, `search_beam`, `output_beam`, `min_active_states`,
`max_timed_batches`, `epoch`, the three stream paths, `time_rqmt` all hash-equal).  The two
`LexlatK2OvercountJob`s likewise differ only in `name`, `hlg`, `hlg_stats` (`strings`, `resources`,
`env_python`, `tolerance=None`, `max_utterances=None`, `time_rqmt`, `mem_rqmt`, `cpu_rqmt` equal), so
the new read is on E-1's own gold and private sets with the same exclusions and the same
pre-registered 0.05 tolerance.  The two `LexlatHLGBuildJob`s differ only in `backoff_loops` and
`name`.  The two configs use different `PREFIX`/`ALIAS`, so the shared build job picks up two
aliases (a set) and no output path collides.

**Constants.**  +0.287 / +0.162 trace to `SAE_4A_lexlat.md:377-378` (the registered read);
+0.013/+0.019, +0.033/+0.017, +0.240/+0.110 to the audit's section 3 table; 74/63/62 % and the
per-string fixture numbers are reproduced by the tests I ran.  No number in the new docstrings is
unsourced.

## Evidence

* census / kwarg diffs: `/e/project1/spell/wu24/2026-07-13_unsupervised` under
  `/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python`, scripts in this session's
  scratchpad (not kept).
* fixture runs: `/e/scratch/spell/wu24/envs/sae_k2/bin/python -m pytest`, with
  `PYTHONPATH=<setup>/tools/sisyphus`.
