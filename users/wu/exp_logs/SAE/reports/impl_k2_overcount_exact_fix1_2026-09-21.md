# Implementer report: the exact read on the word_boundary graph + review fixes F1/F2, 2026-09-21

Commit `3c12dad` in `recipe/2025-10-speech-llm` (parent `6bd6f1f`, the concurrent implementer's
loop-placement commit), three explicit paths, not pushed. **Nothing launched.** No project document
touched. `lexlat_k2.py`, `lexlat_k2_jobs.py`, `lexlat_k2_arm_jobs.py`, `config_sae_4a_lexlat_k2_v1.py`
and the other implementer's `word_boundary_hlg()` / `overcount_word_boundary()` were read and NOT
edited; `config_sae_1g_v1.py` and the untracked `config_sae_3e1_d6_swap_cont_v1.py` were left alone.

## (1) The deciding read is registered

`py_overcount()` now builds a SECOND `LexlatK2OvercountExactJob`, consuming
`LexlatK2OvercountJob.x4a6MX6ZgJ5n`'s `overcount.json` -- the graph-side read on
`LexlatHLGBuildJob.cdcxYJMjiYj5`, the `word_boundary` build. `overcount_exact(job, *, tag="")` puts
the tag into the job's `name` (`sae_4a_lexlat_k2/k2lat_20_word_boundary`), its alias
(`sae/4a/lexlat_k2_pack/overcount_exact_word_boundary`) and its output prefix
(`.../overcount_exact_word_boundary/{overcount_exact,partial}.json`, `summary.txt`), so the two
exact reads are distinguishable in every place a label is taken from -- including the `name` field
inside the json. The empty tag reproduces the first registration argument for argument, which is
why `4yKkB64ZDvJ4` does not move.

## (2) F1, traceability: the summary names its graph

`summary.txt` now opens with

```
graph: <.../LexlatHLGBuildJob.<hash>/output/HLG.pt>
       #0 back-off self-loops: <placement>
       escape ..., shuffled ..., determinized ..., sil_prob ...
graph numbers read off: <.../overcount.json>
```

The placement is read for PROVENANCE ONLY by the new `backoff_loops_of()`, out of the `build.json`
whose path the consumed `overcount.json` records; `hlg`, `hlg_stats` and `backoff_loops` also go
into the json. **No parameter was added to the job**, so no hash moved and the old registration
did not have to be kept twice. It never guesses: a build from before the option (the banked
`rtX44PBJFNy1`, verified on disk) renders `not recorded in build.json -- the build predates the
option, i.e. icefall's "all" placement`; a missing or unreadable `build.json` says so with the path.

## (3) F2, the rendered verdict

`_write` now sets a `verdict` field -- `INCOMPLETE` when the run stopped on its own clock, else
`PASS` / `FAIL` -- and the summary prints that word. `pass` keeps its old meaning (False on a
partial run), so nothing downstream can read a prefix as a PASS, but the summary no longer prints
`VERDICT: FAIL` above per-set lines that read `PASS`. The `PARTIAL RUN` line now says the per-set
lines are not a read against the acceptance.

## Census of `config/sae_4a_lexlat_k2_overcount.py`, before and after

Before = commit `6bd6f1f` checked out into a detached `git worktree` in the scratchpad (the shared
checkout was never stashed or reset); after = the working tree. **Additions only, 4 jobs -> 5.**

| job | id | before | after |
|---|---|---|---|
| `LexlatK2OvercountJob` (banked graph, FINISHED) | `f5Ljn6twbc4b` | yes | unmoved |
| `LexlatK2OvercountExactJob` (banked graph) | `4yKkB64ZDvJ4` | yes | unmoved |
| `LexlatHLGBuildJob` (word_boundary) | `cdcxYJMjiYj5` | yes | unmoved |
| `LexlatK2OvercountJob` (word_boundary) | `x4a6MX6ZgJ5n` | yes | unmoved |
| `LexlatK2OvercountExactJob` (word_boundary) | `vNM59xVEnC2G` | -- | NEW, the deciding read |

No hash moved anywhere. Census script: `scripts/sae_4a_lexlat_k2_overcount_census.py` (setup dir,
updated to print all five ids).

## Checks

| check | result |
|---|---|
| `test_lexlat_k2_overcount_exact.py`, training env | 11 passed (7 before + 4 new) |
| same + `test_lexlat_k2.py`, k2 scratch env | 39 passed, 1 skipped |
| `test_lexlat.py` + `test_lexlat_k2.py`, training env | 60 passed, 24 skipped (unchanged) |
| ruff on the three touched files | clean (the pre-existing `F401` on `probes` in the config is untouched) |
| smoke of the reader on the real banked inputs, 3 strings/set | summary's new graph block renders; the banked build.json genuinely has no `backoff_loops`, and the line says so |

New tests: the summary names the HLG path and the placement; `backoff_loops_of` on an absent path /
a missing file / a build.json without the key / with it; a budget abort yields
`verdict == "INCOMPLETE"`, `pass is False`, `VERDICT: INCOMPLETE` and no `VERDICT: FAIL`; and the
review's exact shape -- `READ: PASS` per-set lines under `VERDICT: INCOMPLETE`.

## Notes for the orchestrator

* The word_boundary exact read (`vNM59xVEnC2G`) depends on `x4a6MX6ZgJ5n`, which depends on the
  build `cdcxYJMjiYj5`: launching it compiles the graph (~3 min, 26 GiB peak on the banked build)
  and runs the k2 host intersection (~77 min on 16 CPUs) before the ~36 min CPU read.
* F3 of the review is a scope note for you, not a code defect: the banked-graph exact read will
  report FAIL (the reviewer's 60-row cross-check gives +0.274 gold / +0.152 private against 0.05).
  It is the baseline and the `legit_multi_parse` split, not the gate.
* The review's observation that the committed enumeration identity is STRICT-only still stands; the
  reviewer closed that gap in his own context at `escape=True` (agreement 3.6e-15) but no test in
  the repository covers the escape branch by enumeration. I did not add one, as it was not in this
  dispatch.
