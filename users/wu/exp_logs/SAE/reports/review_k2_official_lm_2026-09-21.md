# review: the OFFICIAL LibriSpeech LMs as G in the k2 cost probe (speech-llm 8d31839)

Reviewed: `git show 8d31839` (5 files), read against the baseline `lexlat_k2.py`,
`lexlat_k2_jobs.py`, `config_sae_4a_lexlat_k2_v1.py`, the reference build
`LexlatHLGBuildJob.rtX44PBJFNy1/output/summary.txt` + `build.json`, the on-disk official ARPA /
lexicon / word_refs, and `lexlat.parse_arpa_word_lm`.  Verdict **DONE_WITH_CONCERNS**: two
findings, both in `off4g` only; the three-probe wiring, the order-3 bit-identity and the
perplexity job are sound.

## Findings

### F1 (medium) config_sae_4a_lexlat_k2_official_v1.py:105-107 -- the theta = 5 rung is NOT "the scale of the banked trigram"

The config records, and every `off4g/hlg/summary.txt` will repeat, that theta = 5 "takes the
4-gram to about 17.4 M arcs, the SCALE OF THE BANKED TRIGRAM the live build is already funded to
compile, so a graph that fails there has failed for a reason other than size."

`g_fsa` (lexlat_k2.py:704) builds `n_arcs = n_ngram_arcs + 2 * n_states - 1` (one back-off arc and
one `-1` final arc per state), and `prune_lm_tables` drops n-gram arcs only -- `n_states` is
untouched at every rung.  Numbers:

| | n_states | n-gram arcs | G FSA arcs |
| --- | --- | --- | --- |
| banked trigram (reference, measured) | 3,545,313 | 13,964,716 | 20,578,578 |
| off4g rung 0 | 84,370,494 | 145,346,185 | ~313.9 M |
| off4g rung 5.0 (x0.120) | 84,370,494 | ~17.4 M | ~186.1 M |
| off4g rung 8.0 | 84,370,494 | ~6 M | ~174.7 M |

So the ladder moves the G the compiler actually sees by a factor 1.8 (313.9 M -> 174.7 M), not the
factor 10 the arc-ratio argument implies, and the last rung is still 9.0 x the reference G's arcs
and 23.8 x its states.  The inference "a graph that fails at theta = 5 has failed for a reason
other than size" does not follow, and it is written into the artifact that would be read if
`off4g` bottoms out.

The mitigation the module docstring offers (lexlat_k2_official.py:39-44, "the states no surviving
arc reaches ... are removed by the composition's forward exploration") is right in kind --
`compile_hlg` calls `k2.compose(L, G)` then `k2.connect`, and the host intersection explores
forward -- but it is a claim about the PRODUCT, not about G, and nothing measures how many G
states stay reachable at a rung.

Fix (either): (a) restate the rung justification in G-FSA terms (`n-gram arcs + 2 * n_states`) and
delete the "failed for a reason other than size" clause; and/or (b) compact G before `g_fsa` --
compute forward reachability over `arc_next` on the pruned tables, drop unreachable states and
renumber -- so the rung actually shrinks the graph; and/or (c) record the reachable-state count
per rung in `build.json` so a bottom-out is interpretable.

Expected consequence if unfixed: `off4g` is the only arm at risk; rung 0 should OOM fast
(product ~1.5 G arcs vs the reference's 143.9 M at 26 GiB peak, i.e. ~10 x 26 GiB just at
`remove_epsilon`), so the ladder will proceed, but a bottom-out would be recorded with an
unsupported conclusion.

### F2 (conditional, order >= 4 only) lexlat_k2_official.py:312-330 -- a dropped n-gram's STATE keeps backoff = 0, backoff_state = 0

At `o >= 3` the parser filters `ok` to the rows whose (o-1)-prefix is a listed (o-1)-gram
(line 313-319), and then at line 326 assigns the state block's back-off from the FILTERED mask:

```python
states = blocks[o].base_state + np.nonzero(ok)[0] if o >= 2 else 1 + last
backoff[states] = gram_bo[o][ok]
backoff_state[states] = ... _context_state(..., w[ok][:, 1:], ...)
```

`_Block` numbers a state for EVERY listed o-gram (line 288), including the dropped ones, and such
a state is still reachable as the `arc_next` of a surviving (o+1)-gram.  Its `backoff` stays 0.0
and its `backoff_state` stays 0 (the null context), so `_lookup` from it skips every intermediate
back-off level and pays no back-off weight -- a wrong `log p(w | h)` in G for those contexts.

At order 3 this cannot fire: `blocks` = {2} only, and the o = 2 block has no `o >= 3` filter, so
every state gets its back-off.  That is why the banked path and
`test_order3_reader_is_the_banked_one` are unaffected.  At order 4 the 3-gram block is both a
state block and a filtered one, so the trigger is `lm.json`'s `n_unreachable > 0` for `off4g`.
For an lmplz/SRILM-built full 4-gram every sub-n-gram is normally listed and `n_unreachable`
should be 0, but nothing asserts it.

Fix: assign the state block from the UNFILTERED mask (keep a separate `state_ok` before the
`o >= 3` filter), or -- cheapest -- `assert n_unreachable == 0` in `parse_arpa_lm` when
`order >= 4`, or at minimum fail the build when `lm.json`'s `n_unreachable > 0`.

Same class, not fixed by either: `backoff_state` = the longest LISTED suffix
(`_context_state`), so if an intermediate suffix context is absent the walk skips that level.  The
reference (`lexlat.parse_arpa_word_lm:250-255`) documents this as score-equivalent only when the
skipped level has back-off 0, which holds for a listed-without-back-off context but not for an
absent one.

### Non-findings worth recording

* `lexlat_k2_official.py:648` -- `assert int(lm["n_phones"]) == n_phones and int(lm["sil_id"]) ==
  sil_id` can never fire: `save_lm_npz` (line 377-378) writes exactly `len(PHONES)` /
  `PHONE2ID[SIL]`.  Harmless tautology.
* Prose rounding in the config docstring (lines 100-101) and the report: the ARPA header gives
  200,003 + 38,229,161 + 45,941,329 + 60,975,692 = 145,346,185 n-grams and n_states =
  84,370,494; the text says "145.1 M" and "84.2 M".  No computed number depends on it.
* `n_{o}gram` changed meaning between the reference and the new parser: `parse_arpa_word_lm`
  stores the LISTED count, `parse_arpa_lm` stores the KEPT count (`per_order[o]`, line 324/355)
  and adds `n_{o}gram_listed`.  Self-consistent, both printed in `summary.txt`.
* The two `DownloadJob`s carry `checksum=None`, so only the URL is in the hash; if openslr
  reissues the file the hash does not move.  Standard idiom in this recipe.

## The six dispatch checks

### (1) order-3 bit-identity and the live hashes -- PASS

Code path, not just the test.  `read_lm_tables` (lexlat_k2.py:501-529) reads the banked npz's own
arrays unchanged and only ADDS `out["order"] = 3` (`"order"` is absent from the banked file);
`child` / `word_start` / `word_id` / `escape_phone_log_prob` / `n_nodes` / `n_entries` merely
moved from mandatory to optional and are all present in the banked file.  `state_orders`
(line 533-554) with no `state_order` field and order 3 returns exactly the expression
`prune_lm_tables` used to inline (`where(s==0,1,where(s<=V,2,3))`): `out[0]=1`,
`out[1:V+1]=2`, rest 3.  `keep = (st_order[src] == 1)` is `(src == 0)`; `orders = st_order[ksrc]`
is the old `np.where` chain.  `_lookup`'s `levels` defaults to `lm["order"]` = 3.  The only
observable change on the banked path is a new `"order"` key in the `prune` stats block, and the
live build job is FINISHED (`finished.tar.gz`) so it never reruns.

Hashes: `Job.hash` (tools/sisyphus/sisyphus/job.py:1239-1256) is
`sis_hash((filtered_parsed_args, cls.__sis_version__))` -- no file sha -- and the commit touches
neither `lexlat_k2_jobs.py` nor `config_sae_4a_lexlat_k2_v1.py` (`git show --stat`: 5 files, none
of them either).  `LexlatHLGBuildJob.rtX44PBJFNy1` and `LexlatK2ProbeJob.Tkl94t85j4pV` cannot
move.  `run_probe` -- the only entry point `LexlatK2ProbeJob` calls -- references none of
`read_lm_tables` / `state_orders` / `_lookup` / `prune_lm_tables` (grepped over the function
body).  I did not re-run `scripts/sae_4a_lexlat_k2_census.py`; the argument above is structural
and does not need it.

Tests re-run by me under the k2 env:
`test_lexlat_k2_official.py` **13 passed**, `test_lexlat_k2.py` **26 passed, 1 skipped,
1 xfailed** -- the implementer's numbers reproduce.

### (2) ARPA parsing -- PASS except F2

Checked line by line against `lexlat.parse_arpa_word_lm`:

* log10 -> nats: `gram_lp[o]` is float64, multiplied by `LN10` once and cast to float32 at
  line 322/338, the reference's order of operations.  Back-off: `backoff * np.float32(LN10)`
  (line 340), identical to reference line 353.
* missing back-off column: `gram_bo` defaults 0.0, taken only when `len(fields) == cur + 2`
  (line 274).  Highest order never reads `gram_bo` (`if o < order`, line 325).
* `<s>`: `begin_state = 1 + word2id["<s>"]` (line 346) = the reference's `state_of([<s>])`;
  `g_fsa` swaps it with state 0.  `<s>` / `</s>` arcs are dropped by
  `drop = (BOS_WORD, EOS_WORD)` (line 587) = `lexlat_k2.NON_EMITTABLE_WORDS`; strict uses the
  LM's own unk spelling instead of the hard-coded `<unk>`, which is the correct generalisation.
* `<unk>`: `unk_word_id` accepts `<unk>` or `<UNK>` (line 382-389) and raises if neither; the
  official ARPA lists `<UNK>` (verified in the file header).  The spelling is recorded in
  `build.json` as `escape_word`.
* every state's back-off chain terminates: `backoff_state` is always the longest listed SUFFIX,
  whose `state_order` is strictly smaller, and 0 is the base -- no cycle.  The correctness gap is
  F2 (weight/level skipped, chain still terminates).
* arc ordering is the reference's: concatenate 1..order then `argsort(kind="stable")` over keys
  asserted unique, so the permutation is identical at order 3.
* `_encode`'s int64 guard (line 131) allows at most 3 context words at V = 200,003; order 4 needs
  exactly 3.  Order 5 would abort with a clear message.
* Parse-time edge: a row whose word has no unigram is stored as NaN and still occupies a state
  slot with partially-zero words (line 262-272).  If it ever happened the `_Block` uniqueness
  assert would abort the parse ("the ARPA lists the same n-gram twice").  `n_unlisted_token` is
  recorded.  Not reachable on the official ARPAs.

### (3) the official lexicon -- PASS

Checked against the real file
(`.../FetchLibrispeechLmJob.bTGUMQQciD5S/output/librispeech-lexicon.txt`, 206,508 lines):

* the stress-stripped phone inventory of the file is EXACTLY the 39 ARPAbet symbols of
  `prior.ARPABET_39` -- no SPN, no SIL, no `<UNK>` / `!SIL` entry -- so `oov_phone`, `has_sil`,
  `no_pron` and `dropped_marker` will all be 0 in production; the counters are correct but the
  drops are exercised only by the fixture.
* `_STRESS = re.compile(r"\d+$")` strips trailing digits only; duplicates created by stripping are
  counted and removed via the `seen` set (line 449-452).
* multi-pronunciation: kept (`A -> AH`, `A -> EY` both survive); the `(word id, pron)` list is
  sorted `(t[0], t[1])` (line 493), the same order `prons_from_trie` returns, so
  `add_lex_disambig` hands out indices the same way.
* restriction to G's vocabulary via `word2id` from `lm["words"]` (line 477), counted as
  `oov_word`.  `n_lexicon_entries_dropped` sums oov_phone + has_sil + no_pron + oov_word +
  dropped_marker (jobs line 674-676); `duplicate` is deliberately not in that sum.
* ESCAPE: `read_escape_prices` (line 392-405) reads `escape_phone_log_prob` out of the banked
  `lexlat_resources.npz`, asserts `n_phones` and `sil_id`, and calls `lexlat_k2.escape_prices` --
  the SAME vector and the same `ESCAPE_LENGTH_LOG_PROB` / SIL-floor the banked build used, so
  `escape_price_min/max` must come out at the banked -8.5624 / -3.1246.  One `<unk>` transition
  per non-SIL span is `lexicon_to_fst(escape_word=unk, escape_price=...)`, imported, not
  re-implemented.
* `SIL_PROB`: the job default is `sil_prob = 0.5`, byte-identical to `LexlatHLGBuildJob`'s
  (lexlat_k2_jobs.py:129) and to `lexlat_k2.SIL_PROB`.  `determinize=False` is passed literally at
  lexlat_k2_official.py:591 and recorded as `determinized: False`.
* `build_trie_multi` is TEST-ONLY -- `_cmd_build_hlg` goes `read_official_lexicon` ->
  `official_prons` -> `build_official_hlg` and never builds a trie.  Disclosed in its docstring.

### (4) probe wiring -- PASS

`settling_probe` (config official:225-245) passes exactly the same kwarg set as the live
`settling_probe` (config live:178-198): `arm`, `returnn_config`, `checkpoint`, `features`,
`units`, `originals`, `env_python`, `gpu_check`, `epoch`, `max_active`, `search_beam`,
`output_beam`, `min_active_states`, `max_timed_batches`, `time_rqmt`.  `name` gains `_{tag}` and
`hlg` / `hlg_stats` point at that tag's `LexlatOfficialHLGBuildJob` -- the only differences.
`train_cfg` is built by the identical call
(`attrib._control()` -> `attrib._priorshuf_prior` -> `pack._arm_train_config(..., max_contexts=c)`
with `c = MAX_CONTEXTS = k2cfg.MAX_CONTEXTS = pack.E1_CONTEXTS_LOWER`), so tau, the batches and
the ordering are E1's; `_priorshuf_prior`'s `prefix` / `alias` reach only `add_alias` and
`register_output`, never a job hash (config_sae_4a_attrib_v1.py:159-170), so the prior job id is
the live one's.  The checkpoint tk.Path is built with `probes._pin` and the live key string, so it
hashes identically.  Every operating-point constant is `k2cfg.<NAME>`, imported (official:72-84),
including `PRUNE_LADDER_3G = k2cfg.PRUNE_LADDER`.

`build.json` carries `n_phones`, `sil_id`, `d_min`, `recognizer_stride` (from
`build_official_hlg`'s `record`, line 593-595) plus `min_frames` and `prune.theta_nats` -- the
four the probe asserts at lexlat_k2_jobs.py:469-476 and the two it prints at :476.  With
`theta = 0` `prune_stats = {"theta_nats": 0.0}` (line 643), so the print does not KeyError.

`_pin` in the official config uses key prefix `sae4a_lexlat_k2off_` where the live one uses
`sae4a_lexlat_k2_`, so the same `lexlat_resources.npz` enters the two graphs under different
`hash_overwrite` strings.  Harmless -- different Job classes -- but worth knowing if the two
configs are ever merged.

### (5) rqmts for `off4g` -- feasible for the parse, at risk for the ladder

ARPA header (read off disk): 200,003 / 38,229,161 / 45,941,329 / 60,975,692 = 145,346,185
n-grams; `n_states = 1 + V + c2 + c3 = 84,370,494`; the file is 4.40 GB plain text (already
banked uncompressed, so no gunzip).

* **Parse.** A pure-python line loop with per-word dict lookups and numpy row stores, ~4-5 us per
  line -> ~11-14 min for 145.3 M lines, plus ~4 min of I/O.  `PARSE_HOURS = 2.0` is comfortable.
  Peak RSS: `gram_words` 1.83 GB + `gram_lp` float64 1.16 GB + `gram_bo` 0.58 GB + the two
  `_Block`s (~2 GB transient) + the concatenated key / logp / next arrays and their sorted copies
  (~7 GB) -> order 15-20 GB.  Well inside 200 GB.  The npz is ~4 GB on the job's work dir.
* **Budget clock.** `started` is taken BEFORE the parse (lexlat_k2_official_jobs.py:171) and
  `remaining = time_rqmt * 3600 * 0.95 - (now - started)` (line 214), so the parse IS charged
  against the ladder.  Answer to the dispatch's question: the clock does NOT restart after the
  parse.
* **Ladder budget -- risk.** `attempt_hours = BUILD_ATTEMPT_HOURS = 4.0` and
  `time_rqmt = BUILD_TIME_HOURS = 6.0` (5.7 h effective).  After a ~0.4 h parse, rung 0 gets 4.0 h;
  if it TIMES OUT, rung 1 gets 1.3 h and rung 2 is then skipped at `remaining < 300`, so three of
  the five rungs (2.0, 5.0, 8.0) never run and the job dies with "no rung of the pruning ladder
  compiled" after burning 6 h x 16 CPU x 200 GB.  The ladder only gets to walk if the early rungs
  fail FAST, which for rung 0 they should (an OOM kill of the subprocess, caught at line 231).
  Cheap insurance: pass `attempt_hours = 1.0` for `off4g` (its reference compile is 173 s at
  20.6 M G arcs, so a rung that has not finished in an hour is not going to) or raise
  `time_rqmt`.  Note this arithmetic is inherited verbatim from the live job, not introduced here
  -- but the live ladder has 3 rungs and this one has 5.
* **Does a rung fit?** Not established -- see F1.  Rung 0 almost certainly does not (product
  ~10 x the reference's 143.9 M arcs / 26 GiB peak).  Rungs 5.0 / 8.0 plausibly do, because what
  the composition costs is the REACHABLE G state count and heavy arc pruning does strand most of
  the 45.9 M 3-gram-context states -- but that is an argument, not a measurement, and the config's
  own justification for those rungs is the wrong one.
* The two pruned 3-grams are 34 MB / 13 MB gz on openslr (URLs at official:117-118 verified
  against the openslr resource listing), i.e. a few M n-grams over a few M states -- smaller than
  the banked trigram on both axes, so `PRUNE_LADDER_3G` should terminate at rung 0.

### (6) `WordLmPerplexityJob` -- PASS

* Source: `LibriSpeechWordRefsJob.1EsLvSbyl06D/output/word_refs.json`, the file the PER reads
  already use; 2,703 dev-clean + 2,864 dev-other utterances (read off disk).
* Normalisation: `.upper().split()`, apostrophes kept.  I verified on the actual file that every
  token is already within `A-Z` plus apostrophe and that no utterance is empty, so the
  `n_bad_char == 0` assert (jobs:445) passes and the `if s` filter (jobs:442) drops nothing --
  the text matches librispeech-lm-norm.
* Tokens: `full_scores(sentence, bos=True, eos=True)` for ALL four LMs (jobs:469), length asserted
  `len(sentence) + 1`, `is_word` False for the last row, `n_tokens = n_words + n_sentences`.
  `</s>` is included identically for every model and `is_word` is asserted stable across LMs
  (jobs:480).
* OOV: the LM's own `full_scores` flag; `oov_rate = n_oov / n_words`, matching the docstring.
  `ppl_all_excl_oov` drops the OOV terms from sum AND denominator.  (Wording nit: the docstring
  says "the context still contains the OOV word"; kenlm in fact conditions on `<unk>`.  No number
  changes.)
* Common subset: `common[i] = (not word) or all(not oov[tag][i] for tag in self.lm_order)` over
  ALL FOUR tags (jobs:511-514) -- the intersection of the four vocabularies, as specified.
  `ppl_common_excl_oov == ppl_common_oov_as_unk` is asserted, not assumed (jobs:523, 539).
* Text only: no checkpoint, model, feature or decode path enters `run()`; the only inputs are
  `word_refs.json` and four LM files.  Disclosure is in the docstring's first paragraph and
  heads `summary.txt`.
* Inputs exist and are readable: `word_lm.bin` is a genuine KenLM binary
  (`mmap lm http://kheafield.com/code format version 5`), and `4-gram.arpa` is the plain 4.40 GB
  ARPA (no gunzip needed).  Loaded one at a time with `del model`; a 145 M-n-gram probing model is
  a few GB, so `mem 64 GB / time 4 h / cpu 2` is ample.

## What I did not check

* I did not execute `scripts/sae_4a_lexlat_k2_census.py` (live or official) -- the hash argument
  in (1) is structural and independent of it.
* I did not verify empirically that `n_unreachable == 0` on the official 4-gram (F2's trigger);
  that needs a 38 M-entry bigram index over the 4.4 GB file.
* No production graph was compiled, so no rung's real memory or wall clock is measured.
