# Code review -- SAE 4a lexlat ROUND 1 (commit 319ce90), 2026-09-21

Verdict: **PASS_WITH_CONCERNS**.  Nothing found that changes a reported number of E-1 or of E0's two
kill reads; two items can cost a run or mislabel a cell.  Reviewed read-only against
`SAE_4A_lexlat.md` Designs 1, 2 (+amendment 3), 4, 6, 8, the design-review amendments and the
orchestrator's probe-checkpoint note; the implementer report's "Undetermined/deviations" was read
first and every one of its six items was checked against the code.

Paths (speech-llm checkout `/e/project1/spell/wu24/2026-07-13_unsupervised/recipe/2025-10-speech-llm`):
`src/speech_llm/sae/emc/lexlat.py`, `.../lexlat_jobs.py`, `.../test_lexlat.py`,
`src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_lexlat_probes_v1.py`,
workspace shim `/e/project1/spell/wu24/2026-07-13_unsupervised/config/sae_4a_lexlat_probes.py`.

## Concerns (each with fix and blocking status)

**C1 (non-blocking for correctness, costs a run) -- the unpruned cell can trip the candidate guard
and abort the whole census.**  `lexlat.py:964` asserts `n_cand * B <= params.max_candidates`
(default `6_000_000`, `lexlat.py:643`).  Per frame the arc count is about
`C*K (CONT) + 2*C*(1+wmax)*K (START + ESC-open)`, where `wmax` is the largest homophone count over
LIVE contexts (`lexlat.py:840`).  At the primary C = 1024 this is ~0.5 M and at C = 4096 ~2 M, both
safe; at the declared `unpruned_ceiling = 16384` (`lexlat_jobs.py:842`, report deviation 4) it is
~6.5-7.9 M for `wmax` 4-5, which a 151,731-word ARPAbet lexicon reaches easily.  The assert is
uncaught inside the per-utterance loop (`lexlat_jobs.py:836-866`), so it kills the job after the
cells already computed, and `LexlatCensusJob` has **no** constructor argument for `max_candidates`:
repairing it needs a code edit, not a parameter.  Fix before launch: expose `max_candidates` on the
job (one line, passed into every `LexlatParams`), or lower `unpruned_ceiling` to 8192.  Does not
change any number; it risks ~1 GPU-h and a round trip.

**C2 (non-blocking, reporting) -- the `exact` flag does not implement its own docstring.**
`lexlat_jobs.py:865` sets `exact = pruned.max() <= 0.0`; the docstring (`lexlat_jobs.py:641-646`)
says "a cell that hits the ceiling is reported as NOT exact".  When `n_groups <= C` the discarded
mass is bit-exactly 0 (`lexlat.py:1061-1068`, `sel` = all groups -> `kept == total`), so the flag is
sound in that direction; but a cell that DOES hit 16384 and discards mass below float resolution
gets `kept - total == 0.0` and is printed as exact.  `max_contexts_reached` is recorded beside it
(`lexlat_jobs.py:863`), so the fix is `exact = pruned.max() <= 0 and max_contexts_reached < ceiling`.
The flag enters no funding clause (clause 2 compares C = 1024 with C = 4096, `lexlat_jobs.py:968-985`),
so this is a summary-wording risk only.

**C3 (non-blocking, interpretation; spec-conformant) -- kill read (a) conflates decode mode with the
lexicon.**  `lexlat_jobs.py:899-908` scores the augmented max-plus decode against the *frame-greedy*
decode, exactly as amendment 6 words it.  A negative paired delta can therefore come from
max-plus-over-the-lattice versus greedy, with no lexical content.  The cheap control is one more
`lexlat_viterbi` call at `lam_lex = 0` (same machinery, one extra decode per utterance) reported as
a third column.  Orchestrator's call; the implementation follows the pre-registration.

**C4 (non-blocking, round-2 coverage) -- the gradient check never sees the pruning path.**
`test_lexlat.py:559` runs the autograd comparison with `max_contexts=None, escape_budget=0`, so the
3.3e-16 residual certifies the unpruned backward only; the toy is `t_len = 2-3` frames
(`test_lexlat.py:238-247`), so `checkpoint` 2/3 exercise at most two D4 blocks and 32 exercises one.
What does cover pruning is `test_lexlat.py:476-511`: at the production shape with `max_contexts=256`
the posterior rows still sum to 1 (<= 1e-4 fp32), which is a real consistency check of the pruned
backward but not a gradient identity.  E0 needs only `expected_tokens` and the monitors, so this is
not blocking for E0; before round 2 launches training, add one autograd case at `max_contexts` small
enough to prune on the toy.

## What was checked and is sound

1. **Escape parity with `prior_gap.py`.**  `lexlat.py:929-937` opens an escape with one
   `res.unk_logp[state]` (`<unk>` through the same back-off walk) plus `escape_price[k]`, and
   `lexlat.py:938-947` extends at `escape_price` only with the word-LM state unchanged -- the same
   two-table CLOSED/OPEN structure as `prior_gap.py:477-490`.  `escape_price` is order-1 Witten-Bell
   plus `log 0.5` with SIL at NEG_INF (`lexlat.py:446-449`; built from
   `witten_bell_utt_log_prob(prior, [p], 1)` at `lexlat_jobs.py:277`, the same call as
   `prior_gap.py:1556`).  SIL is excluded from START, ESC-open and ESC-ext (`!= sil` masks at
   `lexlat.py:922, 934, 943`) and is emitted only from a boundary entry with `desc = 0`
   (`lexlat.py:925-928`), i.e. it closes an escape, matching `prior_gap.py:461-467`.  Escape contexts
   carry `node = ROOT`, so the free close is the boundary entry at `lexlat.py:843`.  Homophones are
   separate branches over `word_start/word_id` (`lexlat.py:847-858`) and a word close exists only
   where `word_cnt[node] > 0` (trie word-end).  The gold/private E-1 numbers are not re-derived by
   assumption: `LexlatEquivalenceProbeJob` scores every gold and private string through both
   implementations and `asserts` at 1e-4 per token (`lexlat_jobs.py:586-588`), which stops the phase.
2. **CSR back-off trigram.**  `lexlat.py:498-507` walks three levels, adding `backoff[state]` per
   dropped order and taking the longest matched suffix -- ARPA back-off as KenLM computes it.
   Successors keep the listed 2-gram context (`lexlat.py:311-317`); where KenLM would truncate, the
   dropped context's back-off is absent-hence-zero in an unpruned modified-KN ARPA, so the scores
   agree, and the agreement is measured by E-1 and by `test_lexlat.py:146` against real `kenlm`.
   `<unk>` is priced by the same walk (`lexlat.py:474-476`), the start state is the `<s>` context
   (`lexlat.py:357`, `lexlat.py:1184`) and no `</s>` term appears at finality
   (`lexlat.py:1120-1145`) -- `prior_gap.KenLMWordLM`'s own convention (`prior_gap.py:333`).
   The 3,302,936 assertion is computed on the TEXT (`lexlat_jobs.py:177-218`, `_bigram_types` on the
   replayed `window["words"]`, the survey's source) under three conventions with the matched one
   recorded (`lexlat_jobs.py:258-263, 315`); `|V| = 151,731` is asserted twice
   (`lexlat_jobs.py:253-257`).  The `lmplz` call is `prior_gap.py:1477-1479` verbatim including
   `DISCOUNT_FALLBACK`; only `-S` differs (24G vs 8G), which does not change lmplz's output.
3. **Pruning.**  Ranking is exact pre-band forward mass per destination (`lexlat.py:1032`,
   justified at `lexlat.py:806-812`); the escape reserve is a true reservation, not a count --
   the top-`C_esc` live escape groups are boosted by 1e25 before the global top-k
   (`lexlat.py:1050-1060`), and the reserve is not wasted when fewer escape groups exist.
   `pruned = 1 - exp(kept - total)` (`lexlat.py:1067-1068`) is discarded mass over the frame's total
   destination mass, before any renormalisation, and it is read on active frames only
   (`lexlat.py:1427-1432`; `lexlat_jobs.py:852` slices to `out_lens`).  Finality is applied at the
   utterance end only (`lexlat.py:1238-1241` forward, `lexlat.py:1347-1350` backward) and gives mass
   to word-end, open-escape and root contexts only (`lexlat.py:1120-1145`).  NEG_INF utterances are
   counted per cell (`lexlat_jobs.py:857, 959`) and clause 3 requires zero (`lexlat_jobs.py:999`).
4. **Gradient path.**  The lexical increment is an additive per-arc constant scaled like the bed's
   trigram (`lam = lam_lex / tau`, `lexlat.py:1163`; added to `beta log P3` at `lexlat.py:835, 917`),
   and nothing on the lexicon side carries grad.  `post_q` / `seg_post` have `lattice.py`'s shapes
   and semantics (`lexlat.py:654-657`, `1425-1434`), verified at `lam_lex = 0` against
   `lattice_forward_backward`'s own tables (`test_lexlat.py:573`).  Numerics are imported, not
   copied (`NEG_INF`, `_logmm`, `_exact_fp32_matmul`, `_band_matrix`, `scaled_seg_pad`,
   `trigram_history`).  D4 replay indexes the checkpoint store consistently (`lexlat.py:1232-1233`,
   `1341-1345`).  See C4 for the coverage gap.
5. **E0 plumbing and reads.**  Model rebuild is line-for-line the falsifier (ii) job's:
   `lexlat_jobs.py:764-794` against `blankfree_probe_jobs.py:426-459` (`_load_state`, `tau` asserted
   against the arm's own anneal, `permute_frames_seed is None`, `lattice_float64`), and the forward
   is `lookup_eta` + `recognizer` + `build_segment_table` (`lexlat_jobs.py:815-818` against
   `blankfree_probe_jobs.py:496-499`).  The checkpoint is the banked ctrl_50 ep10 pack file
   (`config_sae_4a_lexlat_probes_v1.py:153-155`), `tau = temperatures()[EPOCH-1]` from the pack's own
   schedule.  The 100 utterances come from `select_tags(sorted(gold), 100, seed 0)`
   (`lexlat_jobs.py:800`; rule at `blankfree_probe_jobs.py:92-103`) -- a recorded rule, no stored
   list.  Grid is C in {256, 1024, 4096} x lam in {1/3, 2/3, 1} plus the unpruned cell on the 20
   shortest (`lexlat_jobs.py:806-808`), as Design 6 + amendment 4.  PER: both strings are SIL-free
   and the gold is used as stored (`lexlat_jobs.py:899-908`), which is exactly
   `blankfree_eval_jobs.py:86-90`, the convention the banked PER rows are scored under; the paired
   delta of record is the mean of per-utterance differences with the pooled delta beside it and the
   bar at -0.010 (`lexlat_jobs.py:1002-1024`).  Kill read (b) uses `expected_tokens` (SIL included)
   with the non-SIL count beside it and the 10 % bar (`lexlat_jobs.py:1025-1044`).  Both rules are
   quoted in the job docstring (`lexlat_jobs.py:619-637`) and printed in `summary.txt`.  The rebuilt
   model's greedy decode is compared against the banked decode and a mismatch is noted
   (`lexlat_jobs.py:826, 913-916`).  `retained` is the unit-frame count (`lexlat_jobs.py:805`, with
   `_to_tensors` asserting units length == feature length, `blankfree_probe_jobs.py:739`).
6. **Sisyphus hygiene.**  `git show --stat 319ce90` = 4 added files, 0 modified; no banked module
   touched.  All three jobs are `tk.register_output`ed (8 outputs) inside `build()`, which `py()`
   returns (`config_sae_4a_lexlat_probes_v1.py:139-179`); `add_alias` is used only beside a
   registration.  Every input is a `_pin`ned frozen path whose hash key carries a sha of the
   realpath (`config_sae_4a_lexlat_probes_v1.py:68-77`); `pc._vad` / `pc._decode`
   (`config_sae_4a_private_code_v1.py:104-124`) return pinned paths and construct no Job, so no
   banked job is reconstructed or re-hashed.  Rqmts: build CPU 4 / mem 48 / time 4 for `lmplz` on
   1 M lines (Step 0 ran the same replay on 4 CPU / 16 GB in 302.85 s); E-1 CPU 4 / mem 32;
   E0 gpu 1 / gpu_mem 96 / mem 64 / cpu 8 / time 6.0 against a "~1 h" estimate -- headroom, inside
   the clamp.  The workspace shim imports `py` as `run`.
7. **Tests vs Design 8's list.**  (a) is run in its realisable form (one-phone-word lexicon, STRICT,
   4.4e-16) plus the ambiguous-lexicon half; the deviation is correctly argued and does not touch
   Design 5's step-1 check, since the arms run the banked `lattice.py` path before the on-set
   (amendment 2).  (b) is covered by the TINY_LEXICON string comparison with real `lmplz` + real
   `kenlm`, by a brute-force path/duration/analysis oracle, and -- for the real 151,731-word
   resource -- NOT by a test: `test_step0_strings_through_the_real_trie` is skip-guarded
   (`test_lexlat.py:174`) because no banked output carries the resource, and the check of record is
   E-1, which raises.  Acceptable, and the E-1 assert is present.  (c), (d) (pruned and unpruned,
   fp32 and fp64) and (e) are all present.  Gaps: C4, and (b)'s real-resource half is only closed
   when E-1 runs -- i.e. E-1 must be read before E0 is trusted, which is the Design's own ordering.

## Not checked
No job was launched, no graph was loaded (console load is out of scope for this review); the
report's "3 new jobs / 8 outputs" was verified by reading the config, not by loading it.
