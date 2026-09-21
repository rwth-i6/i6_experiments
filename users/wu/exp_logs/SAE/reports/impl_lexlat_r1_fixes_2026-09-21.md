# SAE 4a lexlat -- ROUND 1 review fixes (C1, C2, C3), 2026-09-21

Status: **DONE**.  Three fixes from `reports/review_lexlat_r1_2026-09-21.md` applied to the
speech-llm checkout (`recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`),
tested and committed as **4c451a3** (no push).  Review item C4 (a pruned-path autograd case) was
NOT touched: deferred to round 2 by the dispatch.

ONE FILE CHANGED: `src/speech_llm/sae/emc/lexlat_jobs.py` (+92 / -9).  `lexlat.py`, `test_lexlat.py`
and `configs/config_sae_4a_lexlat_probes_v1.py` are untouched -- every fix was possible inside the
job.  Nothing in the DP core moved, so no banked number of any other phase can move.

## Fix 1 (C1, blocking) -- the arc guard now scales with the ceiling

`lexlat_jobs.py:690` (`max_candidates: Optional[int] = None` on `LexlatCensusJob`), `:719` (stored),
`:812-825` (derived in `run()` and printed), `:869`, `:934`, `:941` (handed to every `LexlatParams`).

`None` = derive from the resource and the largest budget the job actually runs:

```
n_cand(C, K, wmax) = C*K (CONT) + C*(1+wmax)*(2K+1) (START + SIL + ESC-open)
                   + C*K (ESC-ext) + C (REP)
                   = C * (2K + 1) * (2 + wmax)
max_candidates      = 1.25 * n_cand(max(budgets, unpruned_ceiling), cfg.n_phones,
                                    int(res.word_cnt.max()))
```

`wmax` is the resource's own largest homophone count (`lexlat._frame_slots`' `n_es = 1 + wmax`,
taken over ALL trie nodes, which bounds the per-frame maximum over LIVE contexts); the job runs at
B = 1, so no batch factor enters.  The expression is already an upper bound; the 1.25 is head-room.
The guard REMAINS A GUARD: `lexlat.py:964` still raises with the frame's own arc count and the
budget in the message -- nothing is silently truncated.  The derived value is printed at the top of
the log and banked in `census.json` under `max_candidates` (`used`, `handed_in`, `phones`,
`max_homophones`, `largest_budget`).

**The ceiling 16384 stands; it was not lowered.**  Numbers, at C = 16384, K = 40, B = 1, float64,
with `wmax = 33` -- the largest homophone group of the phase's phonemization lexicon, measured with
`prior_gap.load_phonemization_lexicon` on the pinned bliss + g2p files (584,893 words, 472,131
distinct pronunciations, largest group `SH UW` = 33; restricting to the 151,731 LM words can only
lower it, so this is the worst case):

| quantity at C = 16384 | value |
|---|---|
| arcs per frame `n_cand` (wmax 33 / 10 / 4) | 46.4 M / 15.9 M / 8.0 M |
| derived `max_candidates` (wmax 33) | 58.1 M (the old fixed 6 M would have aborted) |
| slot columns, 58 B per arc (rank/off/raw f64, src/k/key/word i64, live/flags 1 B) | 2.51 GiB |
| `_group_by_key` sort + grouping transients, ~58-70 B per arc | 2.5-3.0 GiB |
| pre-`cat` per-family tensors, `[1, n_e, K]` f64 with `n_e <= C*(1+wmax)` = 557 k (178 MB each) | ~1.4 GiB |
| persistent `fwd [1, 51, 16384, 2]` f64 | 13 MB |
| **peak, one frame** | **~7 GiB** |

This is an ANALYTIC bound from the code's own dtypes and shapes, not a measured peak: the real
resource does not exist on disk yet, so no run at C = 16384 with the real trie is possible before
the build job runs.  It is a per-FRAME peak -- the cells call `lexlat_log_z` with no checkpoint
store, so utterance length (the 20 shortest utterances) does not enter it.  ~7 GiB is far below the
60 GiB line the dispatch set and below the job's `gpu_mem 96`, so the "unpruned" cell keeps the
declared ceiling of 16384 and the report of record stays the per-utterance `exact` flag.

## Fix 2 (C2) -- the `exact` flag now implements its docstring

`lexlat_jobs.py:899`: `"exact": bool(pruned.max() <= 0.0 and max(stats["n_ctx"]) < ceiling)`, with
`ceiling` (`:870`) the cell's own budget, or `unpruned_ceiling` for the unpruned cell.  A cell that
reaches its ceiling is NOT exact even when the discarded mass rounds to a bit-exact 0.  Docstring
`:647-649` states the flag.  The smoke shows the fix biting: the tiny-lexicon C = 256 and C = 1024
cells have `pruned max 0.0000` and `max_contexts_reached == ceiling` and now report `exact = False`
(they would have read "exact" before), while C = 4096 (2129 contexts reached) reports `exact = True`.

## Fix 3 (C3) -- one disclosed control row for kill read (a)

Per utterance (`lexlat_jobs.py:936-956`): a second `lexlat.lexlat_viterbi` at `lam_lex = 0` with
the SAME `max_contexts = primary_budget`, the same `escape_budget`, the same `escape = True` and
the same code path as the augmented decode, scored by the same `eval_jobs.edit_counts` against the
same gold (`:957`, row key `per_lam0`, decode kept under `control_lam0_decode`).  Aggregated at
`:1081-1097` as `kill_read_a.control_lam0` (paired mean / sd / median against the PLAIN greedy
decode, n better / n worse, pooled PER and pooled delta) and rendered at `:1181-1189` beside the two
existing rows.  The docstring (`:627-633`) and the `disclosed` string in the json both say it is a
CONTROL, not a rule: the pre-registered funding rule is unchanged and still reads the
augmented-vs-greedy row only (`kill_read_a.pass` is computed from that row alone).

Reading of the ambiguous phrase "same pruning off": taken as "the same pruning setting as the
augmented decode" (C = `primary_budget`), since the control's purpose is to differ from the
augmented decode in `lam_lex` and in nothing else.

Cost: one extra Viterbi decode per utterance (the primary decode's own cost).  `rqmt` untouched
(time 6.0 h against the Design's "~1 h" estimate).

## Checks

* `pytest sae/emc/test_lexlat.py -q`: **31 passed, 1 skipped** in 67 s (the skip is the
  real-resource test, skip-guarded since round 1).  No test was changed.
* **CPU smoke of the census path, end to end** (`PT_DEVICE=cpu`, login node, 3 dev-other
  utterances, `n_unpruned=2`, outputs redirected to a scratch dir so no job dir was created):
  the real `ctrl_50` `returnn.config` + ep10 checkpoint + VAD streams + gold + banked greedy
  decode, with a 5-word synthetic resource.  Ran to completion, **1183 s** (the login node was
  loaded; round 1's equivalent was 21.8 s).  It exercised all three fixes: the guard printed
  `max_candidates = 4976640 (C <= 16384, K = 40, wmax = 1)` = 1.25 x 16384 x 81 x 3 and no cell
  tripped it; the `exact` flags came out as above; the control row printed, e.g.
  `CONTROL (disclosed, NOT the rule) -- ... lam_lex = 0, C = 1024 ...: paired delta +0.0064
  (sd 0.0441, median +0.0227, 1 better / 2 worse), pooled PER 0.8609 (+0.0000)`, and `per_lam0` /
  `control_lam0_decode` are in `per_utt.json`, `kill_read_a.control_lam0` in `census.json`.
  **EVERY NUMBER THE SMOKE PRINTED IS A SMOKE ARTEFACT** of a 5-word synthetic lexicon on 3
  utterances (its kill read (a) "FAIL", its PER 0.86, its monitors): it measures nothing.
* Graph load: `config_sae_4a_lexlat_probes_v1.build()` still constructs exactly the three jobs and
  eight outputs.

## Hashes -- the dispatch's premise corrected

The dispatch expected all three jobs in the file to re-hash (file-sha `__sis_version__`).  They do
NOT: this class carries a plain `__sis_version__ = 1`, so only the job whose PARAMETERS changed
moves.  Measured by re-loading the config after the edit:

| job | before | after |
|---|---|---|
| `LexiconTrieBuildJob` | `rlMsnTBSZXsB` | `rlMsnTBSZXsB` (unmoved) |
| `LexlatEquivalenceProbeJob` | `lq61PSAg1DcC` | `lq61PSAg1DcC` (unmoved) |
| `LexlatCensusJob` | `hJKXcRjKieh3` | **`RDMTvt4ngAPB`** (the new `max_candidates` parameter) |

That is the better outcome and it is also the correct one: the build job has already FINISHED on
disk (`finished.tar.gz`, ~4 CPU-h with `lmplz`) and is preserved, E-1 never ran and is unaffected,
and E0 -- whose behaviour changed -- gets a new hash, so the fixed code cannot be confused with the
reviewed code.  The old `LexlatCensusJob.hJKXcRjKieh3` directory (never run to completion) is now
an orphan and can be removed by the executor.

## Undetermined / assumptions

1. `wmax` for the REAL resource is not known until `LexiconTrieBuildJob` runs; the code takes it
   from the resource at run time, and the memory table above uses the measured upper bound 33 from
   the full phonemization lexicon.  Nothing in the code depends on the number I measured.
2. The 1.25 head-room factor on an expression that is already an upper bound is an implementation
   choice, stated in the code comment; it enters no rule and no measured quantity.
3. The peak-memory figure is analytic (dtype x shape accounting), NOT a measurement; the first real
   evidence will be the E0 run itself, whose unpruned cells report `exact` and
   `max_contexts_reached` per utterance.
