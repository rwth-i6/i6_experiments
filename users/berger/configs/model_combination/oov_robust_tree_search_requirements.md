# Requirements for an OOV-robust RASR tree-search prototype

## Purpose and scope

Implement open-vocabulary fallback inside a lexicon-based search while retaining
normal word-level LM scoring for known words. Support both continuation-marked BPE
and Granite CTC's SentencePiece-style, word-start-marked tokenization. This is an
implementation brief, not a description of functionality already implemented.

The motivating Granite results on LibriSpeech dev-other are 2.8% WER for greedy
lexicon-free decoding versus 3.0% for the tree without an LM. Tuned word-level LMs
reach 2.7% with a 4-gram and 2.6% with a Transformer. These rounded results do not
prove that OOV words explain the gap, nor that an LM experiment would recover the
same absolute improvement. The prototype should let us test that hypothesis.

Previous reduced-lexicon BPE experiments already used an unknown-word subtree.
One concern is that known words can take that route and update the word LM with
`<UNK>` instead of their actual word. Increasing unknown probability can therefore
damage both the current decision and subsequent LM context. Prevent that failure
structurally before relying on tuning.

Deliver a working CTC prototype for both tokenization modes, retain the existing
BPE RNA/transducer behavior, and exercise shared changes in RNA and AED builders.
Do not require retraining an acoustic model or a neural LM. Keep the existing
closed-vocabulary and legacy fallback configurations available for comparison.

## Existing implementation and relevant files

RASR checkout used in these experiments:

`/work/asr4/berger/rasr_dev/label_scorer/rasr`

Relevant paths relative to that checkout:

- `src/Search/TreeBuilder.cc`: `SharedBaseClassTreeBuilder`,
  `CtcTreeBuilder::addUnknownWordStates`, and corresponding RNA/AED builder code.
- `src/Search/TreeTimesyncBeamSearch/TreeTimesyncBeamSearch.cc`: word exits,
  syntactic-token scoring, recombination, pruning, and sentence-end handling.
- `src/Test/Search_TreeBuilder.cc`: existing fallback topology tests.
- `doc/search_v2.rst`: current open-vocabulary fallback documentation.

Current special lemma groups are multi-valued:

```xml
<lemma special="unknown-continuation">
  <orth>ra@@</orth>
  <phon>ra@@</phon>
  <synt/>
</lemma>
<lemma special="unknown-final">
  <orth>word</orth>
  <phon>word</phon>
  <synt><tok>&lt;UNK&gt;</tok></synt>
</lemma>
```

Each piece has its own lemma so its identity survives traceback. Continuation
pieces loop through a non-final OOV root; final pieces apply one word-LM unknown
event and return to the normal root. Both types are reachable at the normal root.
CTC blanks remain available inside an OOV. Legacy singleton-unknown pronunciation
fallback is also supported. This code does **not** implement the known-path
exclusion or word-start-boundary semantics required below. The illustrated LM
token spelling is not universal; resolve the actual configured unknown token.

Related experiment-repository files:

- `data/librispeech/lexicon.py`: BPE fallback lexicon construction; final-piece
  orthographies currently carry `&&` for diagnostics.
- `data/huggingface.py`: Granite vocabulary mapping, sanitized RASR label names,
  pronunciation variants, and punctuation lemmas.
- `data/librispeech/lm_text_normalization.py`: normalized word-search lexica.
- `model_pipelines/common/recog.py`: traceback formatting and marker stripping.
- `experiments/librispeech/recognition/ctc_granite.py`: recognition conditions.

These paths are relative to `example_setups/seq2seq_rasr_2025`. Do not change
acoustic vocabulary indices when adding fallback metadata or lemmas.

## Required invariants

1. Preserve every previously accepted ordinary lexical path and its score when
   fallback is enabled, apart from explicitly requested scoring changes.
2. Accept every supported tokenizer-valid lexical token sequence through either
   ordinary paths or fallback. Preserve the underlying CTC collapse semantics.
   Define and report any unsupported tokenizer/control-token cases explicitly;
   do not equate "all token sequences" with arbitrary blank/control emissions.
3. One completed lexical word causes exactly one word-LM event: its known LM
   token, or the unknown token. Piece exits and CTC blanks are not word events.
4. An exact known pronunciation must not also complete as an unknown word.
5. Fallback must preserve acoustic pieces in traceback and transcription; output
   the recognized spelling, not the literal LM symbol `<UNK>`.
6. Genuine unknown words update each word LM with its own unknown-token ID once.
   Do not reset the history, silently keep the old history, or insert one unknown
   per piece.
7. No future reference transcription or evaluation-set vocabulary may influence
   topology, scoring, or known/OOV classification at decode time.

## Word-boundary semantics

### Continuation-marked BPE

For the current BPE setup, a piece ending in `@@` continues the word; a lexical
piece without that suffix ends it. Apply this classification to original token
metadata, not to an orthography decorated with `&&`.

```text
ra@@ re@@ word | next
<--- word ---> | word
```

- Accumulate continuation pieces without a word-LM update.
- At the final piece, resolve known versus unknown and apply one word event.
- Permit single-piece unknown words.
- Do not allow ordinary words to start in the middle of an unfinished word.
- Reject dangling continuation pieces at sentence end in strict mode; configure
  `sentence-end-fall-back = false` and verify the new code honors it.
- Keep the configured CTC blank/repeat behavior throughout the word.

### Granite CTC: word-start-marked pieces

Granite uses a `▁` word-start/space marker, not an explicit word-final marker.
Any ordinary lexical piece can be the last piece, but a piece emission by itself
does not imply word completion.

```text
▁un familiar ▁word
<-- word --> <word>
```

The next word-start boundary closes the preceding word; sentence end closes a
pending nonempty word. Thus `▁un familiar` receives one word-LM event, not two.

The implementation may use delayed boundary resolution or equivalent graph
states with constrained exits. Required semantics are:

- A word-start piece with lexical content begins a word. If another word is
  pending, resolve and score that word before updating the word LM for anything
  belonging to the new word.
- A non-word-start piece extends the pending word. It must not create a new word
  merely because a lexical prefix is already a complete dictionary word.
- Consume the boundary-bearing acoustic token exactly once. Do not duplicate its
  acoustic or subword-LM score while closing the previous word and opening the
  next one.
- Sentence end finalizes the pending word once, then applies the normal LM EOS
  score. Empty input and trailing separators do not create an unknown word.
- Blanks and framewise repetitions do not create word boundaries. Boundary
  processing operates on emitted/collapsed labels, not every nonblank frame.

Inspect the actual tokenizer vocabulary and detokenizer before implementing the
boundary classifier. Handle standalone `▁`, repeated separators, punctuation,
and a first lexical piece without `▁` explicitly. A valid beginning-of-sequence
unmarked piece must not be rejected just because ordinary pronunciations usually
start with `▁`. If any token contains internal whitespace/boundaries, support its
actual decoded semantics or reject that configuration with a clear diagnostic;
never silently classify it as a single ordinary piece.

**Do not put every Granite label into both existing continuation and final lemma
groups.** That admits arbitrary within-word segmentation and repeated unknown
LM events. The current two-group lexicon interface is insufficient by itself.

### Punctuation and normalization

Preserve existing empty-syntactic punctuation/separator paths without charging an
unknown score for each symbol. Do not universally treat punctuation as a boundary:
apostrophes and decimal punctuation can belong inside words. Establish the policy
against the tokenizer and configured lexicon, and test it explicitly.

Evaluation normalization is a separate operation. Do not run the full HF scoring
normalizer on partial prefixes or assume that a normalized scoring word equals
one surface word. Expansions, numbers, and contractions can change word counts.
The prototype must preserve existing lexical interpretations, including any
pronunciation spanning multiple surface words, instead of silently removing them.
If a mode cannot support such entries, fail validation and name the limitation.

The current timesynchronous search requires at most one syntactic token per
lexical exit. Do not reintroduce multi-token exits such as `ZYL + S`; use the
existing compatible lexica, or implement and test an explicit extension.

## Known-word priority: exclude duplicate unknown realizations

For a complete word token sequence `p`, let `K(p)` be the set of ordinary lexical
exits representing that exact sequence under the configured boundary policy.

- If `K(p)` is nonempty, allow those known exits, and disallow the unknown exit
  for `p` regardless of LM scale or unknown reward.
- Otherwise permit one fallback word event, provided `p` is a valid lexical word.
- Preserve multiple genuine known lexical interpretations if they already exist;
  this proposal removes duplicate **unknown** interpretations, not all ambiguity.

Use pronunciation-prefix tracking, e.g. a trie/automaton state carried by the
fallback hypothesis. A missing prefix enters an OOV-only state which can continue
until a valid word boundary. If several lexical paths/neutral transitions can
match, retain the necessary state set; one trie pointer is not always sufficient.
Classify against the same acoustic-label mapping used by the ordinary tree.

An accepting prefix must not block continuation: if `▁cat` is known but
`▁cat xyz` is not, the longer word still needs a fallback path. Likewise, an
unfinished proper prefix can be a valid OOV at a legal boundary even if a longer
known word exists.

Known-path exclusion is defined by the configured lexicon, **not by which normal
beam hypotheses happen to survive**. Do not reinterpret a known sequence as OOV
because its ordinary hypothesis was pruned. Prefer a unified prefix representation
or a design that can resolve the fallback candidate into its known exit with the
correct score/history. Never drop both realizations through independent pruning
plus a late exclusion check. Test this explicitly.

The exact-token-sequence rule is the required prototype baseline. It does not
recognize an unseen tokenization of a known spelling as known automatically. Log
such cases separately. A subsequent extension may resolve completed detokenized
spellings to known LM tokens using a declared, deterministic lexical mapping.
That mapping must handle casing/normalization ambiguities explicitly, not choose
an arbitrary word or apply evaluation normalization as an unconditional LM map.

For pronunciations spanning surface boundaries or neutral punctuation paths,
exclusion must agree with the complete ordinary path, not just a naive whitespace
split. Document the supported construction and add a counterexample test. If
necessary, start with a validated single-surface-word lexicon mode and treat
support for the more general lexicon as a separate required integration milestone.

## LM and other scorer interaction

### Word-level LMs

In negative-natural-log cost notation, a completed unknown word has cost

```text
C_unknown(h) = sum_j lambda_j * [-ln P_j(UNK_j | h_j)] + beta
```

Here each `j` is a configured word LM. Each LM advances to its successor history
for `UNK_j` once. `beta` is an additive per-unknown-word penalty outside the LM
scales: positive discourages fallback, negative rewards it. Specify conversion
to RASR's actual internal score/log convention; ARPA log10 numbers must not be
treated as natural logs without conversion. Apply `beta` once even with several
word LMs, never once per scorer or acoustic piece.

Known words receive their normal word scores and history updates, with no unknown
penalty. A real unknown updates an n-gram history by appending UNK and a Transformer
history/cache by the corresponding token step; it does not erase all prior words.
Do not substitute the reconstructed OOV spelling into a closed-vocabulary LM.

With no word LM, fallback remains available and `beta` is independently usable.
For the topology equivalence tests use zero LM scales and `beta = 0`.

A constant penalty cannot guarantee that an OOV beats deletion while being worse
than every competing known word in every context. Acoustic evidence and context
vary. Structural exclusion solves the duplicate-known-path problem; tune `beta`
for genuine OOV insertion/substitution/deletion tradeoffs.

Support existing original and OOV-adapted LMs. Adapted ARPA models may be estimated
after mapping removed vocabulary words to UNK. A neural LM wrapper may collapse
removed-word probability mass into UNK using log-sum-exp over log probabilities
(including the old UNK), suppressing the removed outputs. Verify normalization
and token-index mapping; never sum logits. These adaptations are independent of
the tree change and must remain separate experimental factors.

The prototype's additive `beta` is a decoding bias, not necessarily a normalized
probability modification. Multiplying only the scored UNK probability by a factor
`f` without renormalization gives `-lambda * ln(f)` for one LM. Renormalizing the
distribution also changes known-word scores and is a different experiment. Do not
silently conflate these modes. A finite additive bonus cannot revive a zero-probability
UNK; validate LM support and fail clearly, or require an explicitly configured,
documented smoothing/floor policy.

Validate each LM's actual UNK spelling and vocabulary ID (`<unk>` versus `<UNK>`
is not interchangeable). For ONNX models, verify output columns, vocabulary maps,
and history-input IDs agree. Fail before decoding when a required unknown token
is missing. Do not assume acoustic-label IDs equal LM-token IDs.

### Subword LMs and transition scorers

- A subword LM scores the actual emitted pieces on both known and OOV paths and
  retains their actual history. It must not receive word-level UNK replacements.
- Known and fallback realizations of identical pieces must receive identical
  acoustic and subword-LM contributions. Frame repetitions/blanks follow existing
  scorer semantics; they must not be scored again as new lexical pieces.
- Apply a word-exit reward once per completed word, on both routes according to
  the configured policy. Internal piece-lemma exits are bookkeeping, not words.
- Audit transition scorers, insertion penalties, and word-end pruning so the
  fallback is not accidentally rewarded, penalized, or pruned once per piece.
- Preserve correctly sized per-scorer beam/threshold lists when adding scorers.

## Search-state, pruning, and caching requirements

Word-start tokenization delays boundary confirmation. Audit known and unknown
score timing: do not let one route advance its word LM earlier merely because
its prefix is a known word. Prefixes that can grow must remain represented.

If lookahead/provisional costs are needed for pruning, keep them distinct from
final scores, and cancel/replace them exactly at completion. Do not charge the
word LM twice. A positive final unknown cost can otherwise leave long unfinished
OOV hypotheses artificially cheap before completion; document and measure this.

Recombination keys must preserve all state that changes legal continuations,
future scores, or required output: known-prefix matching state, pending-word and
boundary state, word-LM histories, subword-LM histories, and CTC repetition state.
Define explicitly when different OOV spellings may recombine under the existing
Viterbi or full-sum semantics, and retain a valid traceback. Do not merge states
merely because their current word-LM histories are equal.

Keep all normal acoustic scoring/prior settings fixed in topology comparisons.
With exhaustive search, every original path must retain its original score and
the expanded space cannot have a worse optimum. Finite-beam results need not be
monotonic; record score and pruning diagnostics instead of assuming WER measures
search correctness. Distinguish Viterbi best-path and CTC summed-path objectives.

Version persistent trees/LM images as necessary. Include tokenization mode,
fallback/exclusion policy, and relevant lexicon metadata in cache dependencies.
Runtime-only score parameters should affect search-job identity even if they need
not rebuild the tree. Old cached trees must not silently activate the new mode.

## Traceback and instrumentation

Record enough information to inspect a word without inferring its route from
capitalization:

- Raw emitted token IDs and original tokenizer pieces, plus frame/time spans.
- Word boundaries, reconstructed surface spelling, and known/fallback route.
- Selected known lemma/LM token, or the exact UNK IDs used by each word LM.
- Acoustic, each LM, word-exit, and unknown-bias score contributions separately.
- Prefix-match/exclusion diagnostics and sentence-end finalization events.

Prefer explicit route metadata. Preserve the existing `&&` diagnostic convention
where used; strip only diagnostic markers during CTM conversion and never feed
them to an LM. Setting lemma `eval` fields is not required for these experiments.
Do not detokenize one piece at a time if doing so changes spaces or punctuation;
the complete recovered token stream must match the tokenizer's detokenization.

Aggregate counts should distinguish true lexicon OOVs, known spellings with missing
pronunciations, and exact known pronunciations. Exact known pronunciations ending
through the UNK route must have count zero in exclusion mode.

## Required tests and acceptance criteria

Use tiny artificial vocabularies, deterministic toy word/subword LMs, and short
exhaustively enumerable score tensors before running full recognition jobs.

1. **BPE boundaries:** one-piece OOV; several `@@` pieces plus final; successive
   OOV words; dangling continuation rejected; no normal-word escape mid-OOV.
2. **Granite boundaries:** `▁un familiar ▁word` produces two word events; initial
   unmarked text, standalone/repeated space markers, final pending word, empty
   input, trailing separator, punctuation, apostrophe and decimal cases obey the
   declared tokenizer policy without phantom unknown words.
3. **Known exclusion:** an exact known pronunciation never emits UNK, even with
   an extreme unknown reward. Test single-piece and multi-piece known words,
   multiple pronunciations, genuine known ambiguities, and known-prefix extension
   into an OOV. Also test alternate tokenization of a known spelling and report
   it as a distinct limitation/classification.
4. **Pruning interaction:** when parallel ordinary/fallback representations are
   used, a pruned ordinary representation must not cause both loss of an otherwise
   surviving candidate and an invalid unknown reinterpretation.
5. **LM accounting:** for `KNOWN OOV KNOWN`, the word-LM sequence is
   `[KNOWN, UNK, KNOWN, EOS]`; the subword LM sees the actual pieces. Check both
   summed costs and successor histories with two word LMs and one subword LM.
   `beta` and any word reward occur once per word of the appropriate class.
6. **CTC topology:** repeated frame labels collapse normally; identical emitted
   pieces separated by blanks remain two pieces; blanks within a word and around
   boundaries cause no LM updates. Compare path acceptance/scores to a reference
   enumerator, including repeated word-start labels.
7. **Existing paths:** ordinary paths and scores remain intact with fallback on;
   exercise neutral lemmas and multi-surface-word pronunciations where supported.
   With LM/bias/rewards disabled and exhaustive Viterbi search, the greedy CTC
   path is accepted whenever it is in the declared tokenizer-valid domain and
   has the expected acoustic score. Ties may choose different output identities.
8. **Vocabulary mapping:** missing UNK fails clearly; uppercase/lowercase spelling
   and ONNX indices are checked; sanitized acoustic label names round-trip.
9. **Artifacts:** old configuration compatibility, tree-cache invalidation,
   serialization/reload, and traceback-to-CTM reconstruction are tested.
10. **Other topologies:** retain BPE RNA behavior and test shared AED changes;
    unsupported new tokenization/builder combinations fail explicitly rather than
    silently falling back to the legacy graph.

## Experiment interface and evaluation plan

Expose explicit choices for tokenization/boundary mode, fallback disabled/legacy/
known-excluding mode, and unknown-word cost offset. Parameter names are up to the
implementation; document their defaults and exact score semantics. Prefer an
opt-in prototype and keep old experiment identities reproducible.

Initial comparisons:

- Granite, full standard lexicon: closed tree versus new fallback, with no LM,
  4-gram, and Transformer LM. Keep a lexicon-free baseline.
- BPE CTC and transducer: baseline lexicon and the existing reduced-lexicon
  targets 1%, 1.5%, 2%, and 3% OOV. Compare legacy and known-excluding fallback.
- Separate original versus adapted word LMs and presence/absence of a subword LM.
  Do not bundle these changes into a single comparison.

First establish correctness at zero unknown bias. Then sweep both penalties and
rewards; an illustrative initial natural-log cost grid is
`beta = [-4, -2, -1, 0, 1, 2, 4]`, subject to checking actual cost magnitudes.
Tune jointly with word-LM scale on development data, not on test data. Keep
acoustic prior, blank penalty, pruning, and word reward fixed initially; measure
runtime and active hypotheses as well as WER. A larger search space may need a
separate search-budget comparison.

Report insertion/deletion/substitution counts, paired newly introduced/corrected
errors, accuracy on reference IV versus OOV words, known-spelling/missing-token-path
cases, and fallback use on IV words. Use the same reference partition and scoring
normalizer for every compared condition. Inspect adjacent IV errors after OOVs
to assess remaining LM-context damage. Select settings on development sets and
then run held-out evaluation.

Successful implementation means the invariants and tests hold and the experiment
can measure the remaining gap. It does not mean promising a particular WER gain
or assuming that every error made by the original tree was a lexicon error.
