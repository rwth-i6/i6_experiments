# Implementer: the `#0` back-off self-loop placement as an HLG build option, 2026-09-21

Dispatch: make the placement of `L`'s `#0` back-off self-loops a build option of the k2 HLG,
verify it on the fixture, and register (without launching) a `word_boundary` build, a settling
probe on it and a second over-count read on it.  Motivation and the decomposition it rests on:
`reports/audit_k2_overcount_2026-09-21.md` sections 2-3, amendment 9.5 of `SAE_4A_lexlat.md`.

Status: **DONE**.  Committed as `6bd6f1f` on branch `haotian_modality_matching_jupiter` of
`recipe/2025-10-speech-llm` (not pushed; five explicit paths staged, nothing else).

## 1. The change

`src/speech_llm/sae/emc/lexlat_k2.py`

* new constants `BACKOFF_LOOPS_ALL = "all"`, `BACKOFF_LOOPS_WORD_BOUNDARY = "word_boundary"`,
  `BACKOFF_LOOP_PLACEMENTS`;
* `_add_self_loops(..., placement=...)`.  `"all"` is icefall `prepare_lang.add_self_loops`
  verbatim and unchanged: a loop at every state with a non-epsilon outgoing INPUT label (the arc
  tuple's column 2).  `"word_boundary"` selects on the outgoing WORD label instead (column 3);
* `lexicon_to_fst(..., backoff_loops="all")` passes it through, with an assert that the word
  boundary state ends up carrying a `#0` loop in either placement;
* `build-hlg` gains `--backoff-loops {all,word_boundary}` (default `all`), prints which placement
  it compiled, and writes `backoff_loops` into `build.json`;
* the module docstring's DISAMBIGUATION section states the option and why it exists.

WHICH STATES `"word_boundary"` RESOLVES TO, on this `L`: only `loop_state`, the trie root.  Word
labels are emitted on the FIRST arc of a word and on the escape's opening arcs, all of which leave
`loop_state`; word-interior arcs, the SIL closure `sil_state -> loop_state` and the escape's exit
arcs all carry an epsilon word label, and the escape state never carried a loop in either
placement.  The SIL and ESCAPE closures both return to `loop_state` (the escape's exit to
`sil_state` reaches it through the SIL closure), so backing off there is still reachable exactly
once, and G's end-of-string closure is taken at `loop_state` before the `-1` arc.  This is the
reading of the dispatch's parenthetical that the arc structure supports; it is stated here as the
one assumption of the delta.

`src/speech_llm/sae/emc/lexlat_k2_jobs.py`

* `LexlatHLGBuildJob(backoff_loops="all")`, validated against `BACKOFF_LOOP_PLACEMENTS`, passed to
  the child, recorded in `build.json` and reported in `summary.txt` with a sentence saying what
  the placement does to the log-semiring total;
* `__sis_hash_exclude__ = {"backoff_loops": "all"}`.  `__sis_version__` is NOT bumped.  Excluding
  a NEW parameter at its default is hash-neutral (sisyphus `Job.hash` drops the key when the
  parsed value equals the excluded one), so every existing build keeps its id and a non-default
  value enters the hash and is a different job.

`src/speech_llm/sae/emc/lexlat_k2_official.py` is NOT touched: it has no loop code of its own, it
calls `lexicon_to_fst` without the new argument, and so it builds at the default.  The three
official builds are unchanged by census (below).

## 2. Fixture verification (`test_lexlat_k2.py`)

Run under the k2 scratch env
(`/e/scratch/spell/wu24/envs/sae_k2/bin/python -m pytest`, from the setup dir):

    test_lexlat_k2.py + test_lexlat_k2_official.py + test_lexlat_k2_train.py
    -> 67 passed, 1 skipped

`_build_hlg` and `_logsum_over_parses` take `backoff_loops`; two tests are new, and the former
`test_hlg_log_semiring_equals_the_sum_over_parses` (`xfail(strict=True)`) is replaced by the
second of them.  `test_hlg_log_semiring_exceeds_the_sum_over_parses` is untouched and still
measures the `"all"` placement.

**(i) `test_word_boundary_loops_leave_the_max_plus_total_unchanged`.**  On the STRICT and on the
ESCAPE graph, over `AY S K R IY M`, `F AO R`, `F AO R AY S` and `AA AE S K R IY M`: the two
placements' max-plus totals agree to **0.0e+00** (exactly, on every string checked), and the
`word_boundary` graph still equals `lexlat.string_best_segmentation` plus `sil_model_log_prob` to
1e-4 -- the E-1 equivalence, same reference and same tolerance as the two existing equivalence
tests.

**(ii) `test_word_boundary_loops_leave_only_the_back_off_route_sum`.**  The test computes the
"plain back-off route sum" itself, as the audit describes it: per word transition, the log-sum of
its up to three routes (the explicit arc; back off once or twice and take the lower-order arc),
each with its own successor state, plus G's end-of-string back-off closure, plus the silence
constant, log-summed over the enumerated parses.  Graph log-sum MINUS `logsumexp` over parses, in
nats per string (fixture, undeterminized, strict graph, 2026-09-21):

| string | parses (absolute) | `"all"` | `word_boundary` | route sum |
|---|---|---|---|---|
| `AY S K R IY M` | -2.802934 | +0.063811 | +0.016357 | +0.016358 |
| `F AO R`        | -2.102869 | +0.064746 | +0.024090 | +0.024090 |
| `F AO R AY S`   | -2.801069 | +0.034135 | +0.012839 | +0.012839 |

The `word_boundary` column and the route-sum column are the same number; the largest
`|word_boundary - route sum|` over the three strings is **6.4e-8**, comfortably inside the
dispatch's 1e-6 (the residual is the graph's float32 arc scores).  The positional multiplicity
that is removed is 74 % / 63 % / 62 % of what icefall's placement adds on this fixture.  The
over-count does not go to zero: the plain back-off double count is a property of an
undeterminized G, not of the loop placement, and the test asserts that `word_boundary` is still
strictly above the parses.

These are FIXTURE numbers on a toy LM.  They license nothing about the full-scale column; that is
what the registered job below is for.

## 3. Registrations (nothing launched)

`config_sae_4a_lexlat_k2_v1` (`config/sae_4a_lexlat_k2.py`): `hlg_build` is split into a pure
`hlg_job` (constructs, does not register -- THE one place the build's arguments are written down)
and `hlg_build` (alias + outputs).  `settling_probe` now registers two probes through one inner
helper, so every argument but `name` / `hlg` / `hlg_stats` is shared between them by construction.

`config_sae_4a_lexlat_k2_pack_v1.py_overcount` (`config/sae_4a_lexlat_k2_overcount.py`): new
`word_boundary_hlg()` (which calls `k2cfg.hlg_job`, so the graph is the SAME job, not a copy at a
second hash) and `overcount_word_boundary()`, a second `LexlatK2OvercountJob` with the registered
read's own strings, resource, scorer and tolerance.  `lexlat_k2_arm_jobs.py` was not edited.

| new job | id |
|---|---|
| `LexlatHLGBuildJob` (`word_boundary`) | `cdcxYJMjiYj5` |
| `LexlatK2ProbeJob` on it | `by7UYEjtYdel` |
| `LexlatK2OvercountJob` on it | `x4a6MX6ZgJ5n` |

The build job id is identical from both configs (verified by censusing both graphs).  The banked
build took 174 s at a 26 GiB peak; this one is the same compile with a smaller `L`.

## 4. Census

`scripts/sae_4a_lexlat_k2_census.py live|official` and
`scripts/sae_4a_lexlat_k2_overcount_census.py`, before and after, diffed:

* **live k2 graph**: 10 -> 12 jobs, ADDITIONS ONLY.  `LexlatHLGBuildJob.rtX44PBJFNy1` and
  `LexlatK2ProbeJob.Tkl94t85j4pV` unchanged.
* **over-count graph**: 2 -> 4 jobs, ADDITIONS ONLY.  `LexlatK2OvercountJob.f5Ljn6twbc4b` and
  `LexlatK2OvercountExactJob.4yKkB64ZDvJ4` unchanged.
* **official graph**: byte-identical, 17 jobs.  `LexlatOfficialHLGBuildJob.GxCkDk90bQpT`,
  `YJsdBTcEQJz9`, `NrghE6fnf5hc` and their three probes unchanged.

A hash mismatch was caught by this census during the work and fixed: `word_boundary_hlg()` first
passed the k2 pack's own `TREATMENT` ("k2lat_20") as the arm, which put the same build at a second
id (`dpxvgcwyN9iF`).  It now takes `hlg_job`'s default arm, the DP pack's `TREATMENT`, which is the
arm name the banked build carries.

## 5. Deviations and things left undetermined

1. **`lexlat_k2_official.py` was not plumbed.**  It has no loop code of its own; plumbing the
   option through its `build_hlg` would be hash-neutral but is outside the delta, so the official
   builds are fixed at `"all"`.  If the official-LM leg should also be measured under the new
   placement, that is a separate one-line change plus a config registration.
2. **`scripts/sae_4a_lexlat_k2_census.py` still prints only the banked hlg/probe pair by name**
   (the full sorted id list does cover the new jobs, so the diff is complete).  Extending its
   header lines is a setup-dir change outside the assigned files and was not made.
3. **The name of the new over-count job** is `sae_4a_lexlat_k2/k2lat_20_word_boundary`; the new
   build is `lexlat_20/hlg_word_boundary` and the new probe
   `lexlat_20/ctrl_50_ep10_word_boundary`.  The dispatch fixed no naming, so these follow the
   existing pattern (the placement as a suffix).
4. **Nothing was launched**, per the dispatch.  The full-scale numbers -- whether the
   `word_boundary` graph's over-count clears the 0.05 nats/token tolerance, and what the
   `word_boundary` graph costs in the settling probe -- are measurements that these three jobs
   have yet to produce.  The fixture result is a construction check, not a read on the gate.
5. **Concurrency**: `lexlat.py` and `lexlat_k2_overcount_exact_jobs.py` were being edited by
   another implementer and were not touched; their commit `f240391` landed mid-round and all
   tests were re-run against it afterwards.  `config_sae_1g_v1.py` and an untracked
   `config_sae_3e1_d6_swap_cont_v1.py` are someone else's working-tree changes and were left
   unstaged.
6. This report file was written but NOT committed: the dispatch named only the
   `recipe/2025-10-speech-llm` checkout for commits.
