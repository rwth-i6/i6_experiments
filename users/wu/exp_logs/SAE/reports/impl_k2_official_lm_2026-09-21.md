# impl: the OFFICIAL LibriSpeech LMs as G in the k2 cost probe (2026-09-21)

Round dispatched 2026-09-21 on the user's decision to benchmark the official LibriSpeech LMs as
`G` beside the phase's own 1M-line word trigram, plus a word-perplexity comparison. Status:
**DONE**. Nothing launched; nothing of the live k2 graph moved.

## What was built

| file | delta |
| --- | --- |
| `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lexlat_k2.py` | EDITED: generalised to LM order n. `read_lm_tables` reads an optional `order` / `state_order`; new `DEFAULT_LM_ORDER = 3` and `state_orders(lm)`; `_lookup` takes `levels`, defaulting to the tables' own `order`; `prune_lm_tables` walks every order and reports `n_{o}gram_after` for each. Every default is order 3, so the banked npz path is untouched. NOT a Job module: hash-neutral. |
| `.../sae/emc/lexlat_k2_official.py` | NEW: `parse_arpa_lm` (ARPA of any order -> the CSR back-off automaton), `save_lm_npz`, `unk_word_id`, `read_escape_prices`, `read_official_lexicon`, `official_prons`, `build_trie_multi`, `build_official_hlg`, and the `arpa-to-npz` / `build-hlg` sub-commands. Everything downstream of the tables is `lexlat_k2`'s, imported and never re-implemented. |
| `.../sae/emc/lexlat_k2_official_jobs.py` | NEW: `LexlatOfficialHLGBuildJob` (`__sis_version__ = 1`), `WordLmPerplexityJob` (`__sis_version__ = 1`). |
| `.../sae/emc/test_lexlat_k2_official.py` | NEW: 13 tests. |
| `.../librispeech/configs/config_sae_4a_lexlat_k2_official_v1.py` | NEW: three HLG builds, three probes, one perplexity job, two downloads. |
| `config/sae_4a_lexlat_k2_official.py` | NEW shim (workspace, untracked). |
| `scripts/sae_4a_lexlat_k2_census.py` | NEW hash census, `live` / `official` (workspace, untracked). |

## Hashes

New, from `scripts/sae_4a_lexlat_k2_census.py official` (17 jobs total):

```
off3g_1e7  hlg   speech_llm/sae/emc/lexlat_k2_official_jobs/LexlatOfficialHLGBuildJob.GxCkDk90bQpT
off3g_1e7  probe speech_llm/sae/emc/lexlat_k2_jobs/LexlatK2ProbeJob.5dP7W2NMFq4T
off3g_3e7  hlg   speech_llm/sae/emc/lexlat_k2_official_jobs/LexlatOfficialHLGBuildJob.YJsdBTcEQJz9
off3g_3e7  probe speech_llm/sae/emc/lexlat_k2_jobs/LexlatK2ProbeJob.R7QzD6vYBLD3
off4g      hlg   speech_llm/sae/emc/lexlat_k2_official_jobs/LexlatOfficialHLGBuildJob.UMLMzgsG4aF4
off4g      probe speech_llm/sae/emc/lexlat_k2_jobs/LexlatK2ProbeJob.zeXxiYy0VBnD
perplexity       speech_llm/sae/emc/lexlat_k2_official_jobs/WordLmPerplexityJob.ddxHefQ0vl2D
3-gram.pruned.1e-7.arpa.gz  i6_core/tools/download/DownloadJob.n2emmqhiewWZ
3-gram.pruned.3e-7.arpa.gz  i6_core/tools/download/DownloadJob.Zf61w9skB0va
```

The remaining eight jobs of the official graph are the shared upstream ones (VAD, features,
split ids, cv holdout, flat recognizer init, prior, text sample) and they come out at the SAME
ids they have in the live graph.

**Census, live graph:** `scripts/sae_4a_lexlat_k2_census.py live` before and after the round
diffs to nothing (13 lines, 10 jobs). `LexlatHLGBuildJob.rtX44PBJFNy1` and
`LexlatK2ProbeJob.Tkl94t85j4pV` are unchanged, and neither is in the new graph. This was proved,
not assumed: `tools/sisyphus/sisyphus/job.py:1240-1256` computes a job hash as
`sis_hash((filtered_parsed_args, cls.__sis_version__))` with no file sha, so editing the
non-Job module `lexlat_k2.py` cannot move them; the new Job classes went in a NEW module anyway.

## Deliverable 1 -- G from an ARPA of any order

`parse_arpa_lm` keeps `lexlat.parse_arpa_word_lm`'s layout and extends it: state 0 is the null
context, `1 + w` the single-word contexts, then one block per listed k-gram order for
`k = 2 .. order - 1` in file order. Two new stored fields, `order` and `state_order[s]` -- above
order 3 the block boundaries are the ARPA's own counts and cannot be derived from the vocabulary
size, which is all the order-3 layout ever needed. Back-off stays an epsilon (`#0`) arc exactly
as before, and n-grams over an unlisted context are dropped and counted (`n_unreachable`), the
rule the order-3 reader already applied.

* **Order 3 is bit for bit.** `test_order3_reader_is_the_banked_one` compares `arc_key`,
  `arc_logp`, `arc_next`, `backoff`, `backoff_state` with `np.array_equal` (not a tolerance) and
  the scalars and word list besides. One bug was found and fixed getting there: storing the
  ARPA's log10 probability in a `float32` array before the multiply by `log 10` moved the stored
  nats by up to 2.4e-7. The reader now keeps it in `float64` until the single multiply and cast,
  which is what the banked function does; the comment in the code says so.
* **Order 4 against KenLM.** `test_lookup_order4_matches_kenlm` (numpy walk, no k2) and
  `test_g_fsa_order4_matches_kenlm` (max-plus single path through the compiled `G`) both agree
  with the `kenlm` module on the same LM to better than 1e-4 nats per sentence, over six
  sentences of 1 to 4 words. The four-level walk is what makes it pass: a three-level walk backs
  off one context too early and loses the 4-gram terms.

## Deliverable 2 -- L from the official lexicon

`read_official_lexicon` strips stress digits, asserts every mapped phone is in the bed's
`prior.PHONES` (39 + SIL), and drops with its own counter: an entry with a remaining unmappable
phone, an entry carrying SIL, a line with no pronunciation, and the duplicates stress-stripping
creates. `official_prons` restricts to `G`'s vocabulary and drops `<s>` / `</s>` / `<unk>` /
`<UNK>` **as words** while the unknown word keeps its LM transition for the ESCAPE loop. Several
pronunciations per word are kept (`build_trie_multi`).

The ESCAPE convention is NOT re-declared and NOT refitted: `read_escape_prices` reads the
per-phone price vector out of the banked `lexlat_resources.npz`, asserts the inventory, and hands
it to `lexlat_k2.escape_prices`. `SIL_PROB` unchanged, `determinize=False` always. So an official
graph differs from the banked one in `L` and `G` alone.

`build.json` carries `n_phones`, `sil_id`, `d_min`, `recognizer_stride` (the four
`LexlatK2ProbeJob` asserts against the live model) plus `min_frames`, `order`, `ngram_counts`,
`vocab_size`, `n_words_with_pronunciation`, `n_pronunciations`, `n_lexicon_entries_dropped`,
`lexicon_stats`, `restrict_stats`, `lexicon`, and an `lm_provenance` block with the ARPA path,
its listed and kept per-order counts, the unknown-word spelling and the parse time.

## Deliverable 3 -- three graphs, three probes

`LexlatK2ProbeJob` is **unchanged**: same class, same arm, same streams, prior and recognizer
init, same pinned `ctrl_50` checkpoint at `probes.EPOCH`, same E1 batches
(`pack.E1_MAX_TIMED_BATCHES`, `MAX_CONTEXTS = pack.E1_CONTEXTS_LOWER`), same temperature, same
`max_active (1000, 3000, 10000)`, `search_beam 20.0`, `output_beam 8.0`,
`min_active_states 30`. Every one of those constants is IMPORTED from
`config_sae_4a_lexlat_k2_v1` rather than re-typed, as are `ENV_PYTHON`, `GPU_CHECK` and all four
`BUILD_*` allocations, so the three official reads sit on the live read's own axis.

### The pruning ladders (a disclosed implementation choice)

The dispatch fixes the rule -- full LM first rung, the ladder applied only if the full graph does
not compile in budget -- but not the rungs. (a) and (b) take the live config's ladder unchanged,
`(0.0, 0.5, 2.0)`: both pruned 3-grams are smaller than the banked trigram the live build already
compiles. (c), the full 4-gram, gets `(0.0, 0.5, 2.0, 5.0, 8.0)`.

The extra rungs were chosen from a MEASURED curve, not guessed. Running
`lexlat_k2.prune_lm_tables` on the banked trigram (13,964,716 arcs over 3,545,312 states) keeps:

| theta (nats) | 0.10 | 0.25 | 0.50 | 1.00 | 2.00 | 3.00 | 5.00 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arcs kept | 99.8 % | 98.3 % | 93.4 % | 79.9 % | 52.8 % | 32.8 % | 12.0 % |

The official 4-gram is 145.1 M n-grams (1: 200,003, 2: 38,229,161, 3: 45,941,329,
4: 60,975,692) over 84.2 M states -- about ten times the banked trigram. `5.0` is the measured
x0.120 point: it would take the 4-gram to roughly 17.4 M arcs, the SCALE OF THE BANKED TRIGRAM
the live build is already funded to compile, so a graph that fails there has failed for a reason
other than size. `8.0` is one rung below (the curve loses about a further factor 3 per 2 nats
above theta = 3, so roughly 6 M arcs) and is the last rung that still carries 4-gram context at
all. **Caveat, stated in the config and here:** the ratios are the TRIGRAM's and a 4-gram's
back-off gains are not the same distribution. The ladder is a budget ladder, not a prediction;
the rung actually used is recorded in `build.json` and in `summary.txt`.

### Pruning drops ARCS, not STATES -- disclosed

`prune_lm_tables` removes n-gram arcs and leaves the state set in place, so the 4-gram `G` keeps
its 84.2 M states at every rung (about 2 extra arcs per state for the back-off and the final
transition). No state-compaction pass was written: `k2.compose` explores the product forward from
the two start states and `connect` drops the rest, so the states no surviving arc reaches cost
`G`'s own memory (a few GB) and nothing else. Every official `summary.txt` says this.

### The max-plus vs `string_best_segmentation` check -- ADAPTED, my call

It is **not** run inside the three production build jobs, and that is a stated choice. Running it
there would need a second full-size object beside the graph being compiled -- a
`lexlat.LexResources` over the official automaton, up to 84 M states for the 4-gram with its
per-state unknown-word arc precomputed -- inside the same 200 GB attempt the pruning ladder exists
to protect, and the equality is a property of the CONSTRUCTION, not of the LM's size. Every
official `summary.txt` says so in as many words.

Instead it is run at fixture scale on the IDENTICAL code path (`read_official_lexicon` ->
`official_prons` -> `build_trie_multi` -> `build_official_hlg`), twice:

* **order 3, ESCAPE on**, against `lexlat.string_best_segmentation` plus
  `lexlat_k2.sil_model_log_prob`, over six phone strings including two that only the escape loop
  makes finite, to 1e-4. Order 3 is where that reference is exact: `lexlat.lm_score` answers
  THREE levels, so it cannot be the reference above order 3, and pretending otherwise would have
  been the silent error here.
* **order 4, strict graph**, against every enumerated segmentation scored with the four-level
  `_lookup` walk plus the silence constant, to 1e-4. Same property, with a reference that is
  valid where `string_best_segmentation` is not.

The same test also asserts the four `build.json` fields the probe checks and that
`determinized` is `False`.

## Deliverable 4 -- `WordLmPerplexityJob`

CPU, shared env, `kenlm` in process; a `.bin` is read as is, an `.arpa` as text, an `.arpa.gz`
gunzipped into the work dir first. `CreateBinaryLMJob` was not used: the ARPA path avoids a
second copy of the 4.4 GB LM on disk and kenlm's own probing structure for 145 M n-grams is a few
GB (its `lmplz` estimate is about 23 B per n-gram). LMs are loaded ONE AT A TIME and the
per-token scores kept, so peak memory is the largest single LM's, not the sum. Allocation
`mem 64 GB, time 4 h, cpu 2`.

The four LMs are our `word_lm.bin`, `3-gram.pruned.1e-7`, `3-gram.pruned.3e-7` and the full
4-gram; the sets are dev-clean and dev-other, from `LibriSpeechWordRefsJob`'s `word_refs.json`
(the reference file the PER jobs already use). **The conventions are pre-registered in the job's
own docstring** -- the producing job, not only this report -- and every one of them is named in
the printed table:

* text: uppercased, whitespace-collapsed, apostrophes kept; the token charset is asserted to be
  `A-Z` plus the apostrophe;
* tokens: one sentence per utterance, scored from `<s>` with a final `</s>`; `<s>` never scored;
  `n_tokens = n_words + n_sentences`;
* OOV: the LM's own `full_scores` flag; `oov_rate = n_oov / n_words`;
* `ppl_all_oov_as_unk`, `ppl_all_excl_oov`, `ppl_common_oov_as_unk`, `ppl_common_excl_oov`, where
  COMMON is the token set whose word is in all four vocabularies (plus every `</s>`). The last
  two are equal by construction; the job computes both and ASSERTS the equality rather than
  assuming it, and prints both so the pre-registered pair is complete.

The docstring also carries the disclosure in its first paragraph: reference transcripts are
LABELS, used here for text-only LM scoring; no model, checkpoint, audio or decode enters the job
and nothing it prints selects anything. The same disclosure heads `summary.txt`.

## Checks run

* `test_lexlat_k2_official.py` under the k2 env: **13 passed** in 41 s.
* `test_lexlat_k2.py` under the k2 env: **26 passed, 1 skipped, 1 xfailed** -- identical to the
  pre-round baseline.
* `test_lexlat_k2.py` + `test_lexlat.py` together: 80 passed, 2 skipped, 1 xfailed.
* **Both new jobs' `run()` bodies were exercised end to end** at fixture scale (an `lmplz`
  order-4 ARPA, the official-format lexicon fixture, a small escape-resources npz, a four-utterance
  refs json, and three tiny LMs one of which is gzipped). The build job parsed the ARPA through
  its subprocess, compiled at rung `theta = 0`, wrote `HLG.pt` / `build.json` / `lm.json` /
  `summary.txt` with every asserted field present and `order 4, escape True, determinized False`;
  the perplexity job produced the full table, the gz and binary readings of the same LM agreed to
  1e-6, and the `ppl_common_*` equality assertion held on both splits. This is a check of the job
  wiring, not of the graphs the round will price.
* Hash census, live graph: before == after (see above).
* The `lexlat_k2.py` edit cannot reach a live run: `LexlatHLGBuildJob.rtX44PBJFNy1` is
  FINISHED on disk (`finished.tar.gz`), and `run_probe` -- the only entry point
  `LexlatK2ProbeJob` calls -- touches none of `read_lm_tables`, `state_orders`, `_lookup`
  or `prune_lm_tables` (grepped). Were the build job ever rerun, its `prune` stats block
  would gain an `order` key with every number unchanged; the probe reads `theta_nats`
  from it and nothing else.
* No Greek letters and no assistant name in any new or edited file (grep for U+0370-03FF and the
  names): none.

Loading and these checks do not show that the three official graphs compile at production scale
or what they cost. That is what the jobs are for.

## Open / undetermined

1. **The 4-gram build's wall clock.** The dispatch fixes "same rqmts as the live config", so all
   three builds carry `time 6 h, mem 200 GB, cpu 16`. But the live build job has no ARPA-parse
   stage -- its `G` arrives as the banked npz -- and the official 4-gram adds one: 145.1 M lines
   through a python loop before the ladder starts. `PARSE_HOURS = 2.0` bounds it inside the six
   hours and the job then gives each rung whatever wall is left (it skips a rung rather than
   overrun). If the parse alone exceeds two hours the `off4g` build will fail visibly with the
   attempt record in its log, and it will need a larger allocation -- a deviation from "same
   rqmts" that I did not take unilaterally. Flagged for the orchestrator.
2. **`off4g` rung 0 may not fit 200 GB.** That is what the ladder is for, and the ladder's later
   rungs are the disclosed choice above. A ladder that bottoms out is a recorded result, not a
   crash.
3. **Nothing was launched.** No manager was started and no job submitted, per the dispatch.

## Commit

`recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`, explicit paths only
(other agents have untracked work in this checkout -- `lexlat_k2_train.py`,
`config_sae_3e1_d6_swap_cont_v1.py`, `config_sae_1g_v1.py` -- which was NOT staged). Not pushed.
The workspace shim `config/sae_4a_lexlat_k2_official.py` and `scripts/sae_4a_lexlat_k2_census.py`
live in the setup dir, which is not a git repository, as the live k2 shim does.

---

# Fix round, after `reports/review_k2_official_lm_2026-09-21.md` (DONE_WITH_CONCERNS)

All four findings fixed, tests and both censuses re-run, nothing launched. Status: **DONE**.

## 1. The order >= 4 back-off hole (`lexlat_k2_official.py`)

The review's own diagnosis was right and its first repair was not enough. Assigning the state
block from the unfiltered mask (so a state whose n-gram arc was filtered out still carries its
back-off weight and target) removes the 2.07-nat error on a walk that ENTERS a holed context
through a 4-gram, but it still disagrees with KenLM by **0.46 nats** on `A B C` from the begin
state: KenLM CREATES the missing context `A B`, so its `p(C | A B)` is the listed 3-gram, while
the arc-dropping reader answered `p(C | B)`. Measured, not argued (`kenlm.Model` on the fixture
ARPA: -2.3 vs -2.5 log10).

So the second repair the review offered is the one implemented: `parse_arpa_lm` now runs KenLM's
**blank-context insertion**. Top down for k = order .. 3, every listed k-gram whose (k-1)-prefix
the ARPA does not list gets that context created; a created context has back-off weight 0 and
backs off to its own longest listed suffix, and the arc entering it carries the BACKED-OFF
`p(w | shorter context)` read off the levels already built (`_lookup` with `levels = o`, whose
first probe is always a miss by construction). `n_{o}gram_blank` and `n_blank` are reported in
`lm.json` / `build.json`. On an ARPA with no hole -- every `lmplz` output, the banked trigram
included -- nothing is created and the tables do not move, which is what keeps
`test_order3_reader_is_the_banked_one` bit-for-bit.

Two new tests, both on a hand-written order-4 ARPA with exactly the hole (`A B` unlisted under
`A B C` / `A B D`, and `B D` unlisted under `A B D` so one back-off target is two levels down):
`test_order4_backoff_hole_is_filled_the_way_kenlm_fills_it` asserts the structure (one blank,
back-off 0, `p(B|A) = backoff(A) + p(B)`, the 3-grams now reachable) and
`test_order4_backoff_hole_matches_kenlm` asserts 11 walks against `kenlm` to 1e-4 nats.

## 2. Compaction (`compact_lm_tables`)

New in `lexlat_k2_official.py`, called at EVERY rung including theta = 0: forward reachability
from the begin state over `arc_next` and `backoff_state`, unreachable states dropped, monotone
renumbering (so state 0 stays the null context, which `g_fsa`'s begin-state swap needs), per-order
counts recomputed. `build.json` gets a `compact` block with states and n-gram arcs before and
after and `g_arcs_before` / `g_arcs_after` = `n-gram arcs + 2 * states - 1`, and `summary.txt`
prints one line per rung tried. The `PRUNE_LADDER_4G` docstring is rewritten in G-FSA terms; the
claims "scale of the banked trigram" and "failed for a reason other than size" are gone.

Tests: `test_compact_lm_tables_cannot_move_a_score` (every walk keeps its exact value at
theta = 0 / 0.5 / 2 / 5, sortedness, state 0, monotone shrinkage),
`test_compact_lm_tables_is_idempotent_and_keeps_the_hole`, and
`test_g_arcs_before_and_after_are_what_g_fsa_emits`, which pins the formula to `g_fsa`'s own arc
count -- and showed it is an UPPER BOUND by the arcs of the words L can never emit (`<s>`,
`</s>`), now said so in the docstring. The guard itself prices the COMPILED `G`, not the formula.

## 3. The pre-compose size guard

`predict_hlg_cost(g_arcs)` extrapolates linearly from the reference build
(`LexlatHLGBuildJob.rtX44PBJFNy1/output/build.json`: 20,578,578 G arcs, 143,874,554 HLG arcs,
26.0406 GiB peak -> 6.99 HLG arcs per G arc, 194 bytes per HLG arc). `build_official_hlg` takes
`mem_guard_gib`, prices `G` after compaction, and returns `(None, record)` without composing when
the prediction is over budget; the CLI writes `build.json` anyway and exits **3**
(`EXIT_SIZE_GUARD`), which the ladder reads as "declined" rather than "crashed" and continues to
the next rung. The budget is `MEM_GUARD_FRACTION (0.8) * mem_rqmt`, from the review.
`PRUNE_LADDER_4G` is now `(0.0, 0.5, 2.0, 5.0, 8.0, 12.0, 16.0)`; the guard admits a rung up to
about 126 M G arcs after compaction at 200 GB.

Tests: `test_predict_hlg_cost_reproduces_the_reference_build`,
`test_size_guard_skips_the_rung_without_composing` (no `hlg` / `stages` in the record, and the
same call with a large guard builds the identical graph), and
`test_build_hlg_exits_with_the_size_guard_code` (a real subprocess: exit code 3, no `HLG.pt`
written, the skip in `build.json`). The job smoke script gained a rung-declined case: a build with
an unusable allocation declines all three rungs and raises with the three attempt records instead
of crashing.

## 4. The ladder's budget clock

`LexlatOfficialHLGBuildJob.run()` starts the ladder's clock AFTER the ARPA parse and gives it
`time_rqmt * 3600 * 0.95 - parse_seconds`; `parse_seconds`, `ladder_budget_seconds` and
`ladder_seconds` are recorded separately in `build.json` and printed in `summary.txt`. The parse
is charged against the same allocation (it must be, or Slurm kills the job mid-rung); what the
fix removes is a rung's attempt budget being charged for it.

`BUILD_BUDGET` in the config gives `off4g` `attempt_hours = 1.0` and `time_rqmt = 8.0 h`
(`ATTEMPT_HOURS_4G` / `TIME_HOURS_4G`), a disclosed deviation from the live config's 4 h / 6 h,
which the two pruned 3-grams keep. Seven rungs at 4 h could never have fitted in 6 h.

## Checks re-run

* `test_lexlat_k2_official.py` **21 passed** (13 before, 8 new), `test_lexlat_k2.py` **26 passed,
  1 skipped, 1 xfailed** (unchanged), all three modules together **101 passed, 2 skipped,
  1 xfailed**, in `/e/scratch/spell/wu24/envs/sae_k2`.
* The in-process job smoke (both `run()` bodies at fixture scale) passes, including the new
  size-guard case; `summary.txt` renders the compaction, guard, per-rung and parse/ladder lines.
* `scripts/sae_4a_lexlat_k2_census.py live` still diffs to nothing against the pre-round file
  (13 lines, 10 jobs): `LexlatHLGBuildJob.rtX44PBJFNy1` and `LexlatK2ProbeJob.Tkl94t85j4pV` have
  not moved.

None of this is evidence that a production graph compiles or what it costs.

## Hashes after the fix round

Only `off4g` moved (new constructor values: ladder, `attempt_hours`, `time_rqmt`):

```
off4g      hlg   LexlatOfficialHLGBuildJob.NrghE6fnf5hc   (was UMLMzgsG4aF4)
off4g      probe LexlatK2ProbeJob.DtWYToXTPh6w            (was zeXxiYy0VBnD)
```

`off3g_1e7` (`GxCkDk90bQpT` / `5dP7W2NMFq4T`), `off3g_3e7` (`YJsdBTcEQJz9` / `R7QzD6vYBLD3`), the
perplexity job (`ddxHefQ0vl2D`) and the two downloads are unchanged. No job directory exists on
disk for any of these ids, so the 3-gram builds keeping their hash while changing behaviour
(blank contexts, compaction, guard) cannot reuse anything stale; `__sis_version__` stays 1.

## Open, after the fix round

* Item 1 of the first round's Open list is resolved for the 4-gram: its allocation is now 8 h with
  the parse charged separately. `PARSE_HOURS = 2.0` still bounds the parse alone, and the blank
  pass adds two `np.unique` / `np.isin` passes over the 4-gram's n-gram table inside it.
* The size guard is a ONE-POINT linear fit. It decides only which rungs are ATTEMPTED; it is not a
  measurement and nothing downstream reads it.
