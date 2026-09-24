# Survey: the lexicon scorer of `prior_gap.py` and what a GPU trie DP could reuse (2026-09-20)

Read-only survey. Scope: how `lexicon_strict` / `lexicon_escape` are computed today, which pieces a
batched GPU port could reuse, whether a torch lattice DP already exists to mirror, how the KenLM word
trigram is queried and whether it can be pre-tabulated, and what the tests cover.

Source files (absolute):
`/e/project1/spell/wu24/2026-07-13_unsupervised/recipe/2025-10-speech-llm/src/speech_llm/sae/emc/{prior_gap.py,prior_probe.py,lattice.py,blankfree_sampler.py,blankfree_probe_jobs.py,neural_phone_lm.py,test_prior_gap.py}`
Banked read: `work/speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.2RkbKYl0v1XK/output/prior_gap.json`
(paths below are relative to `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/` unless absolute).

## 1. How the lexicon rows are computed today

**Algorithm: exact Viterbi (max-plus) DP, no beam in the banked run, no lattice object.**
`prior_gap.py:399-514` `best_segmentation`. State = `(position i, KenLM word-LM state, escape-open
flag)`; two dicts per position, `closed[i]` and `open_[i]`, each `state -> (score, back position,
back state, back table, emitted word)` (`prior_gap.py:441-443`). The `beam` parameter
(`prior_gap.py:447-460`) truncates states per position by score; the job default and the operating
point of the banked read is `beam=None` (exact) — `prior_gap.py:1443` (`beam` param doc),
`prior_gap.json:beam = null`.

Per position the DP does three things:
- SIL is a free optional word boundary, never consumed inside a word, and it closes an open escape
  (`prior_gap.py:461-467`).
- every pronunciation starting at `i` is expanded, each of its homophone words scored separately
  (`prior_gap.py:468-476`, using `Lexicon.matches`, `prior_gap.py:385-397`);
- ESCAPE only: open a new escape (one `<unk>` transition + per-phone cost) or extend an open one
  (`prior_gap.py:477-490`).
Backtrace at `prior_gap.py:491-514` recovers the word sequence, `n_escape_words`, `n_escape_phones`.

**Trie.** `class Lexicon`, `prior_gap.py:356-397`. Pure Python: `by_pron: bytes(phone ids) ->
(word, ...)` plus a `set` of every proper prefix (`prefixes`) so a walk stops early, plus `max_len`.
Keys are `bytes` of phone ids, i.e. a hash-set trie, not an explicit node structure.

**Lexicon source and size** (banked `prior_gap.json:lexicon`, printed at `prior_gap.py:1557` and in
the job log):
- source = the phonemization's bliss lexicon (first pronunciation per word) + the Sequitur g2p
  lexicon for the words it misses — `load_phonemization_lexicon`, `prior_gap.py:836-848`, delegating
  to `i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm._load_lexicon_word_to_phon` /
  `_load_g2p_lexicon`;
- 584,893 phonemization words in total; the trie holds only words the word LM knows
  (`restrict_to_word_lm`, `prior_gap.py:850-859`; called `prior_gap.py:1552-1553`), so
  **151,731 words / 132,049 distinct pronunciations** (=> 19,682 words share a pronunciation with
  another, i.e. homophony is ~13%), **max pronunciation length 33 phones**; 433,162 words dropped as
  outside the word-LM vocabulary. Pronunciations containing a symbol outside 39+SIL are skipped
  (`prior_gap.py:374`).

**Word LM.** KenLM order 3, modified Kneser-Ney, built by the job itself with
`lmplz -o 3 --interpolate_unigrams True --discount_fallback ...` then `build_binary`
(`prior_gap.py:1471-1486`, called `prior_gap.py:1543`). Training text = the replayed word side of the
same 1,000,000-counted-line window the live prior is fitted on
(`replay_window_texts`, `prior_gap.py:873-...`; window meta in `prior_gap.json:window`).
Measured on the surviving text `PriorGapAnalysisJob.pg14aEYJyiva/work/lms/window.words.txt`:
**1,000,000 lines, 19,629,091 tokens, 151,731 word types** (identical to the trie word set — every
corpus line is lexicon-covered by construction, so vocab == trie words, plus KenLM's `<unk>/<s>/</s>`),
**3,302,936 distinct word bigram types**. Binary `words_o3.bin` is 274,266,991 bytes
(a word-bigram variant job `PriorGapAnalysisJob.m6lUxAO65f6A` has `words_o2.bin`, 66,356,227 bytes).

**ESCAPE vs STRICT.** Same trie for both (`prior_gap.py:1581-1586`). ESCAPE adds ONE escape word
`<unk>` (`ESCAPE_WORD`, `prior_gap.py:235`) that may cover any span of non-SIL phones at
`log p_wordLM(<unk> | state)` once for the span, plus per phone
`order1_phone_log_prob[phone] + log(0.5)` (`ESCAPE_LENGTH_LOG_PROB`, `prior_gap.py:237`;
cost table built at `prior_gap.py:430-433`; the order-1 prices come from the live Witten-Bell unigram,
`prior_gap.py:1555`). A longer escape is therefore strictly dearer, so the length exploit is closed;
every string gets a finite score. STRICT passes `escape_log_probs=None`: trie words only, `-inf` and
`segmentable=False` when no full segmentation exists (`prior_gap.py:491-494`), so it is defined only
on its own subset and is read against a trigram gap recomputed on that subset
(`SUBSET_VERDICT_KEYS`, `prior_gap.py:225`). Banked: STRICT segmentable 2509/2864 gold vs 678/2864
decode; ESCAPE 2864/2864 both.

**Runtime.** No per-utterance or per-row timer is logged. Only the whole-job wall clock is banked:
`prior_gap.json:seconds = 302.85` for 4 CPUs / 16 GB, covering three `lmplz` builds (phone o4, o6,
word o3 over 1M lines) plus 6 string sets x 2 lexicon rows x 2,864 utterances = 34,368 exact
segmentations; `usage.run.1` reports max RSS 2.55 GB, max CPU 101%. The informative cost proxy that
IS logged is `max_states` — the largest number of live `(word-LM state, table)` entries at any
position: **23-38 over all 12 (row, set) pairs** (`prior_gap.py:458`, `prior_gap.py:1602`,
`prior_gap.json:priors.lexicon_*.sets.*.max_states`).

**Where SIL is dropped.** Two different places, deliberately:
- for scoring, the string sets come SIL-free from the banked decode: `private` is `greedy_phones.json`
  and the job asserts `drop_sil(raw) == hyps` (`prior_gap.py:1506-1510`; `drop_sil` at
  `prior_gap.py:286-288`); `gold` is SIL-free by construction. The `private_sil` set keeps SIL and is
  scored separately.
- inside the DP, SIL is NOT dropped: it is the free word boundary (`prior_gap.py:461-467`) and it is
  excluded from the escape cost table (`prior_gap.py:432-433`, `if p != sil`).
The sampled-string path does the same drop: `blankfree_probe_jobs.py:760-767` `_sample_strings`
maps `BlankfreePathSamples.phones` to phone NAMES and applies `drop_sil`.

## 2. What a GPU port could reuse, and what is pure Python

Reusable as-is (host side, build once per job/epoch):
- `Lexicon` (`prior_gap.py:356-397`) already normalises pronunciations to `bytes` of phone ids and
  carries `prefixes` + `max_len` — everything needed to emit a flat trie (node arrays
  `child[node, phone]`, `is_word[node]`, `word_id[node]` as int32 tensors). It is a hash set today,
  so the node numbering itself does not exist yet.
- `load_phonemization_lexicon` / `restrict_to_word_lm` (`prior_gap.py:836-859`) — the exact word set
  the banked numbers use; reuse verbatim so a GPU row stays comparable.
- The escape cost vector (`prior_gap.py:430-433`, `prior_gap.py:1555`) is already a per-phone scalar
  table, i.e. a length-39 tensor.
- `prior_probe.py:371-403` `score_strings` is the existing "score a list of token lists under every
  scorer" wrapper (it calls `best_segmentation` twice per string, escape then strict) — the natural
  reference for the arm's black-box reward interface, and `prior_probe.py:126-146` shows the import
  discipline (reuse `prior_gap`'s scorers, do not fork them).
- `blankfree_probe_jobs.py:641-659` already implements the G=8 score-function loop: draw with
  `sample_blankfree_paths`, convert to strings, compute per-string rewards on the HOST
  (`_batch_rewards`, `blankfree_probe_jobs.py:841-...`), centre them (`group_advantages`,
  `blankfree_probe_jobs.py:135-143`), and multiply by `path_score` (`blankfree_sampler.py:406-...`).
  A GPU lexicon scorer plugs in exactly where `_batch_rewards` sits; no gradient is required.
- `blankfree_sampler.py:78-96` `BlankfreePathSamples.phones [B, G, U_max] long` / `n_tokens [B, G]`
  is already the padded batched phone-id tensor a trie DP would consume — no CPU round trip needed if
  the scorer is on device (today it goes through `.cpu().tolist()`,
  `blankfree_probe_jobs.py:764-767`).

Pure Python / not portable as written:
- the whole DP body `prior_gap.py:443-514`: per-position `dict` keyed by opaque `kenlm.State`
  objects, `lexicon.matches` returning Python lists, per-word `word_lm.score` calls, and a Python
  backtrace. Cost is O(positions x live states x matching prons x homophones) with a Python-level
  inner loop.
- `KenLMWordLM` (`prior_gap.py:332-354`): one `model.BaseScore(state, word, out)` call per
  (state, word) candidate arc, returning a new opaque `kenlm.State`. This is the single hardest
  dependency to port — there is no tensor form of the state anywhere in the repo.
- `Lexicon.matches` (`prior_gap.py:385-397`) does up to `max_len = 33` byte-slice hash lookups per
  position.
No numpy/torch tensor is used anywhere in the lexicon path; the only numpy in `prior_gap.py` is the
null shuffle (`prior_gap.py:291-304`) and the prior npz.

## 3. Torch lattice DP already in the repo to mirror

Yes — `sae/emc/lattice.py` (86 KB) is exactly a batched GPU DP over a state set with per-step
transition tensors, and it is the model a trie DP should copy.

- State layout: `fwd [B, O, H, 2]` = batch x band offset `o = s - t + W` in `[0, 2W]`
  (`LatticeConfig.n_offsets = 51`, `lattice.py:191-246`) x phone history `H = n_outer * n_ctx`
  (bigram H=41, trigram H=1681, `PriorHistory`, `lattice.py:254-325`) x CTC repeat flag.
- Hot loop: `_forward_step`, `lattice.py:861-902` — one frame, vectorised over `(B, band, h, f) x
  (k, d)`; emit via `torch.gather` + `torch.logsumexp` and a `index_copy_` placement of the
  destination history (`lattice.py:874-885`), blank/repeat as a shift + add. `reduction="auto"`
  switches the same recursion to log-semiring matrix products (`_logmm`, `lattice.py:634-666`),
  which never materialises the reduced axis.
- Numerics: tables carry the caller's dtype (float32 in training), but `_logmm` accumulates in
  **float64 always** (`lattice.py:642-665`), with TF32 explicitly disabled (`_exact_fp32_matmul`,
  `lattice.py:611-633`) and a finite `NEG_INF` floor imported from `..psi_align` (`lattice.py:114`)
  so an all-masked row cannot produce NaN.
- Checkpointing: `checkpoint = S` keeps every S-th forward frame and replays the rest inside the
  manual backward (`lattice.py:1040-1160`, replay at `lattice.py:1154-1161`; 32 at the training bed).
  The backward is hand-written on detached tables — autograd through the expanded transition tensor
  was rejected as ~9 GB/utterance (module docstring, `lattice.py:78-92`).
- `blankfree_sampler.py:134-...` `sample_blankfree_paths` shows how to ride the same tables for G
  draws per utterance with everything `[B, G, ...]`.

A trie DP maps onto this cleanly: replace `(offset, history, flag)` by `(trie node, word-LM history,
escape flag)`, the phone position replaces the frame index, and the per-step tensors become
`child[node, phone]` gathers plus a word-emission scatter. The banked `max_states` 23-38 is the live
word-LM history count per position, so the state set per step is tiny compared to lattice.py's
51x1681 — the limiting factor is the word-LM state, not the trie.

## 4. KenLM query path and whether it can be pre-tabulated

Queried through the **python `kenlm` module**, imported lazily inside functions (`prior_gap.py:321`,
`prior_gap.py:336`); there is no ARPA loader in this repo. `KenLMWordLM.score`
(`prior_gap.py:347-350`) calls `model.BaseScore(state, word, out)` and returns
`(log10 * ln10 = nats, successor kenlm.State)`; `begin_state` is `BeginSentenceWrite`
(`prior_gap.py:342-345`); `word in model` is the vocabulary test used by `restrict_to_word_lm`.
The ARPA is deleted after `build_binary` (`prior_gap.py:1484`), so only the binary survives.

Pre-tabulation verdict: **a dense `|V| x history-state` table is impossible; a sparse one is the
only option.**
- `|V| = 151,731`, and the trigram's history states are the surviving bigram contexts: the training
  text contains **3,302,936 distinct bigram types** (plus 151,731 unigram back-off states), i.e.
  ~3.45M states. Dense table = 151,731 x 3.45e6 ≈ 5.2e11 entries (~2 PB fp32). Out of the question.
- What IS feasible on GPU is the standard back-off representation, which is what the 274 MB
  `words_o3.bin` already is: CSR arrays for the explicit (context, word) trigram and bigram arcs plus
  a per-state back-off weight, i.e. ~20M explicit n-grams -> a few hundred MB of int32/float32 that
  fits in device memory. A GPU port then needs (a) an integer state id in place of `kenlm.State`
  (state = the longest matched suffix, which the CSR gives directly) and (b) a batched binary search
  or hashed lookup per (state, word) arc. Nothing of this exists in the repo today.
- The per-position candidate word set is bounded by the trie: at most `max_len = 33` prefix matches
  per position times the homophone multiplicity (~1.15 on average, 151,731/132,049), so the arc count
  per phone position is small — the pressure is entirely on state lookup, not on table size.
- Cheaper fallback worth naming: the word bigram (`words_o2.bin`, 66 MB, already built by
  `PriorGapAnalysisJob.m6lUxAO65f6A`) has 151,731 states, which is a plausible dense-row regime if a
  reduced state set is acceptable for the reward.

## 5. Tests

`sae/emc/test_prior_gap.py` (948 lines). It builds a REAL KenLM word LM with `lmplz` on a tiny
synthetic corpus (`_word_lm`, `test_prior_gap.py:74-90`) over a tiny ARPAbet lexicon
(`TINY_LEXICON`, `test_prior_gap.py:61`) and calls `best_segmentation` directly
(`_segment`, `test_prior_gap.py:91-94`). Lexicon coverage:
- `test_lexicon_segmentation_and_sil` (`:101`): the best segmentation is found, SIL is a free
  boundary (mid-string and surrounding), a SIL inside a word makes the string unsegmentable, an
  out-of-inventory phone gives `-inf`, and the empty string scores 0.
- `test_escape_makes_every_string_finite_at_the_registered_price` (`:158`): the escape price is
  exactly `<unk>` transition + per-phone `order1 + log 0.5`, a contiguous span takes ONE escape
  rather than one per phone, SIL is never escaped and closes an open escape, the empty string is
  untouched.
- `test_escape_never_loses_to_strict_and_is_not_taken_when_it_is_dearer` (`:190`): escape >= strict
  always; a dear escape reproduces the strict path bit-for-bit; a dominating escape swallows the
  whole string.
- `test_the_trie_holds_only_words_the_word_lm_knows` (`:219`): `restrict_to_word_lm` semantics.
- `test_strict_row_is_read_against_the_trigram_gap_on_its_own_subset` (`:390`) and
  `test_per_utterance_dump_carries_every_score_and_recovers_the_subset` (`:558`): the subset read and
  the `-inf`/`segmentable` plumbing through the JSON.
- Render / decision-table tests (`:445`, `:470`, `:502`) assert the lexicon rows and the lexicon
  clause of the decision table are printed.
Not covered anywhere: beam vs exact equivalence, any timing/complexity bound, and any batched or GPU
variant (none exists). `test_prior_probe.py` re-tests the same scorers through `score_strings`.

## Bottom line for a GPU arm

The port is a (trie node x word-LM state x escape flag) forward DP over phone positions; `lattice.py`
supplies the batching, NEG_INF, float64-accumulation and checkpointing idioms, and
`blankfree_sampler.py` + `blankfree_probe_jobs.py:641-659` supply the G=8 sampling and the
score-function plumbing (no gradient through the scorer needed). The two missing pieces are (1) a
flat int32 trie built from `Lexicon` and (2) an integer-state, CSR back-off word trigram to replace
`kenlm.State` — the LM cannot be pre-tabulated densely (151,731 x ~3.45M states), but its ~20M
explicit n-grams fit on device.
