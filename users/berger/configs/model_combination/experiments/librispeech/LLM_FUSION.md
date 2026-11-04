# Qwen Base shallow fusion: closed-tree Granite prototype

`llm_fusion.run()` is registered by the LibriSpeech entrypoint. Its preflight
builds RASR commit `8c1ac793675fb7aba0198cc165dca7a2e624375d` independently of
the existing `6e07a2e5` experiments. The approved versioned separator XML patch
still applies and is included in the source job hash. No existing recognition
or build identity is changed.

Outputs are under `tools/rasr-oov-8c1ac793/`:

- `binaries`: pinned CMake build;
- `multi-token-smoke` and logs: 34 previous accounting checks plus 12 new
  sequence-scoring checks;
- `delayed-penalty`, `first-piece`, their logs, and `pruning-acceptance`:
  re-run the existing four/eight pruning regression conditions on this build.

The new tests use a normalized six-token bigram LM. The ordinary word KAT
maps to the syntactic sequence KA T. A following DOG has a different score
after T than after KA, so the oracle detects incorrect history advancement,
not just missing token costs. The Granite cases force entry into the fallback
region before KAT, checking known-word resolution there and completion of KAT
at segment end. LM scales 1 and 0.3 check scaling exactly once; beta remains
outside the LM scale. Positive alpha also exercises refund on known resolution.

These are scheduled native tests, not a claim that the new binary has already
passed them. The local Python graph/oracle tests do not execute the decoder.

## Merge semantics and next integration

The merge brings in ordinary LM look-ahead as well as multi-token scoring.
Look-ahead remains disabled for this initial comparison; it is explicitly off
in the added smoke configurations and off by default in this RASR revision.
Existing word-Transformer recognition experiments stay on 6e07a2e5. Matched
closed-tree baselines are registered on the new binary for the Qwen comparison.

The commit charges beta once per **UNK token** in a syntactic sequence, not
once per lemma irrespective of its tokens. Existing single-UNK experiments
are unchanged. Do not manufacture an UNK mapping for an LLM that has no such
linguistic token. Start with closed-tree fusion; dynamically tokenizing the
actual spelling of a genuine fallback OOV remains a separate extension.

## Checkpoint, text policy and index mapping

User-approved checkpoint: `Qwen/Qwen3-0.6B-Base`, pinned at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. No chat template, generation,
sampling, fine-tuning, or subword acoustic-label scorer is used. RASR scores
each lemma's complete Qwen syntactic sequence at word exit.

Each word is lowercased and prefixed with exactly one space, **including the
first word**. This is an explicit static-lexicon approximation: the scored text
is `BOS + " hello world" + EOS`, not an unprefixed or sentence-capitalized text.
Neutral punctuation/separator lemmata retain their empty syntactic sequences.
Original orthographies, pronunciations, and the complete acoustic inventory
are unchanged. Scoring still uses the existing HF normalizer.

`QwenLexiconJob` checks exact encode/decode round trips for every word and checks
concatenated lexical tokenizations against whole-string tokenization, both in
dictionary batches and on dev-other text. Development references only validate
the policy; they do not add vocabulary or pronunciations. It fails on special
token injection, out-of-range ids, multiword lexical orthographies, or fallback
lemmata. The original single-token OOV validator is not relaxed.

Synthetic names `llm_000000`, etc. map explicitly to the model's output column.
All 151936 output columns, including padded unused columns, remain represented
in the vocabulary file. BOS and EOS both map to model id 151643. The unreachable
singleton unknown anchor has empty synt; there is no linguistic Qwen UNK and
no substitution of EOS for unknown words.

## Export and numerical gates

`Qwen3RasrLm` exports the pretrained float32 weights with full causal attention,
Q/K normalization, grouped-query attention, and default RoPE. It implements the
RASR transformer state-manager contract: int32 tokens/lengths, right-aligned
prefix K/V, absolute positions from prefix lengths, float32 negative-log
probabilities, and **suffix-only** K/V outputs with STATE metadata. No history
truncation is configured. This adapter is intentionally restricted to this
Qwen3 architecture; unsupported RoPE/sliding-window variants fail explicitly.

Mandatory gates are:

1. New-build accounting and pruning preflight; fresh acoustic state-tying
   validation on all boundary flags, before the syntactic rewrite.
2. Full pretrained model's HF NLLs vs exported ONNX NLLs, including all output
   columns, empty prefix, unequal lengths and right-aligned cache reuse
   (maximum absolute tolerance 0.002 nats).
3. Native RASR forced word paths vs HF teacher-forced sums, including BOS/EOS,
   multi-token words, contractions, and history advancement; four utterances
   at LM scales 1.0 and 0.3 (total-cost tolerance 0.01 nats).

Recognition consumes the lexicon published by the successful native gate,
not the unchecked lexical rewrite. Local tests use a tiny randomly initialized
Qwen3 model; they are **not** a claim that the full pretrained export or native
build has already passed. The real pinned tokenizer has additionally been
checked locally on all 200000 words of the completed closed Granite lexicon.

2026-09-24 validation: the completed pretrained export passed its HF comparison.
The first native gate exposed a test-fixture bug: EOS had no pronunciation, so
Python RASR omitted that final trace item and the test read the pre-EOS score.
The fixture now includes an empty EOS pronunciation and asserts EOS is present
in the returned trace. All eight cases were rerun manually against the pinned
binary and completed export; the largest total-cost discrepancy was 0.000162
nats. The updated Sisyphus gate still runs before recognition.

Preparation outputs:
`datasets/LibriSpeech/llm-8c1ac793/qwen3-0.6b-base/` contains the pinned repo,
model, tokenizer report, explicit vocabulary, native score report/logs, and
gated lexicon. ONNX external weight files remain alongside the model in the
export job's output directory; keep that directory intact when relocating it.

## Eight development experiments

Descriptors start with `llm-8c1ac793_`; only **dev-other** is scheduled:

- Closed Qwen lexicon: LM scales 0, 0.1, 0.3, 0.5. Scale zero skips LM execution
  and checks that the syntactic rewrite alone does not change recognition.
- Original closed lexicon: no LM, published 4-gram at 0.3, and the existing
  word-level Transformer at 0.3 (same completed model/vocab, not retrained).
- Lexicon-free greedy control on the same new binary.

All trees use beam 256 / threshold 16 for both within-word and word-end pruning,
zero acoustic prior and blank penalty, recombination on, no sentence-end
fallback, and unknown fallback disabled. No word-exit reward is introduced.
The Qwen LM runs via ONNX CUDA alongside the Granite encoder, with a maximum
LM batch size of 16 and a 24-GB GPU / 48-GB host-memory request. Initial scales
are exploratory; multi-token word exits may require subsequent pruning tuning.
No held-out evaluation or LLM OOV fallback is scheduled in this first comparison.

2026-09-25 runtime update: the word-Transformer control is replaced by
`closed-word-trafo-lm0.3_gpu-stateless`, using the same checkpoint re-exported
for full-history CUDA inference (no KV-cache input tensors). It shares the
stateless export with the 18-run OOV Transformer sweep and has its own
lexicon/UNK and cached-vs-stateless score-equivalence gate. It requests 24 GB
GPU memory and 32 GB host memory, with pruning and scale unchanged. Qwen's
separate export/recognition conditions are unchanged by this word-LM update.

## 2026-09-28: alternative Qwen text spellings

The original eight conditions and their preparation identities remain unchanged.
Three additional **dev-other only** conditions use
`closed-qwen3-0.6b-base-text-variants-v1-lm{0.1,0.3,0.5}`. Pruning, acoustic scores,
checkpoint, closed vocabulary and HF scoring normalization are unchanged. The
new conditions request 64 GB host memory for the larger lexical representation;
GPU memory remains 24 GB. These are configured jobs, not completed WER results.

`data/qwen_text_variants.py` constructs distinct lexical routes with identical
output orthography, acoustic pronunciations and evaluation sequences:

- lowercase, initial-capital and uppercase LM spellings, deduplicated;
- the ordinary leading-space form of each spelling;
- no-leading-space forms **restricted to the first LM token after BOS**;
- existing acoustically supported punctuation lemmata now contribute their
  punctuation to Qwen history instead of being LM-neutral. The underscore
  separator and structural special lemmata remain neutral.

`capitalize()`, not `title()`, gives `Don't`, not `Don'T`. All-caps alternatives
allow acronyms without deciding in advance whether a word is a name or acronym.
No vocabulary is added, no references select preferred variants, and no
subtoken averaging, word reward or probability renormalization is introduced.
Search chooses among surface realizations by their ordinary joint scores.

RASR has no static lexical restriction for sentence-initial ordinary lemmata.
Simply making unspaced variants available everywhere would allow neighboring
words to be concatenated in LM history. To prevent this, a separately exported
ONNX adapter adds an alias column for each possible initial token. After BOS
it copies the original Qwen token's NLL; at every later position its cost is
1e30. Alias inputs map back to original token ids before embedding/cache updates.
All original vocabulary columns remain identical. Aliases are alternative names
for existing events, not extra probability mass to renormalize; this is Viterbi
selection among routes, not summation of duplicate routes. Acoustic vocabulary
indices are entirely independent and unchanged.

The completed local full-dictionary audit produced 1,199,955 lemmata from
200,017 source lemmata and 15,608 additional LM alias columns. Every variant
round-trips through the pinned tokenizer; all source pronunciations/evaluation
forms and the phoneme inventory were checked unchanged. The initial aliases
are illegal in medial positions even when their underlying Qwen token is a
valid within-word continuation. Original, space-prefixed first words remain
available alongside the new alternatives.

Because librasr's Python trace exposes lemma **names**, duplicate `<orth>`
values alone do not preserve transcription: auto-generated names gain suffixes.
We use explicit `qwen-v1:<source-index>:<variant-index>:<original-orth>` names.
`QwenTextVariantTracebackFormatter` restores original words for transcription
and CTM while preserving timings; raw tracebacks retain the chosen route.
The emitted lexical XML supplies that route's exact Qwen token sequence.

Preparation is under
`datasets/LibriSpeech/llm-8c1ac793/qwen3-0.6b-base/text-variants-v1/`.
The HF/export gate tests the full output vocabulary, alias masks, alias input
mapping and KV-cache reuse. The native gate compares RASR against an exhaustive
HF spelling oracle for small utterances (including EUGENIUS, contractions and
a comma). Recognition depends on both gates. Local tests additionally execute
a tiny-model native route competition on the pinned 8c1ac793 build.

Limits: this is not unrestricted truecasing or punctuation restoration. We do
not insert punctuation without acoustic support, generate arbitrary mixed-case
spellings, or allow unspaced words after an opening quote. Standalone punctuation
can also give a noncanonical BPE segmentation at adjacent punctuation boundaries;
its literal surface is preserved, but whole-text retokenization is not performed
during search. The previously observed pruning losses are not repaired by this
change; keeping pruning fixed isolates the text-policy experiment first.

## 2026-09-29: bounded Qwen search and stateless control workaround

Replace the three text-variant runs at beam 256 / word-end beam 256 with **six**
dev-other runs: LM scales **0.1 / 0.3 / 0.5**, within-word beam **64**, word-end
beam **8 / 16**. Both thresholds stay **16**. Their descriptors append
`_beam64_wordbeam{8,16}_thr16`, so old outputs are not reused for the new search.
The lexicon, positional alias adapter, ONNX export, native scoring gates,
24-GB GPU / 64-GB host request and scoring policy remain unchanged.

Qwen's own cached ONNX LM keeps `max-batch-size = 16`; it is not the stateless
backend with the leaking speculative queue. The matched **word-Transformer**
control now uses stateless LM batch size **1**, with descriptor suffix
`_gpu-stateless_lm-batch1`, to bypass that queue while retaining GPU full-history
inference. The original lowercase Qwen/no-LM/4-gram/greedy controls are unchanged.
The full active Qwen comparison driver registers 14 conditions. These changes
do not cancel already-running jobs or delete their outputs.

## 2026-10-05: LibriSpeech-only focused follow-up

Completed original and alternative-text runs remain around 2.8% dev-other;
the alternative-text runs have RTF around 6–7. Spelling alternatives alone
have not established a benefit. Before expanding Qwen to the benchmark suite,
`llm_fusion.run()` adds **ten** `qwen-focus-v2` conditions, all on dev-other:

| Text policy | LM scales | Within-word / word-end threshold |
| --- | --- | --- |
| Alternative spellings | 0 (no LM execution) | 8 and 16 |
| Alternative spellings | 0.05, 0.1, 0.2, 0.3 | 8 |
| Alternative spellings | 0.05, 0.2 | 16 |
| Original lowercase | 0.1, 0.3 | 8 |

Every new condition uses within-word beam 64 and word-end beam 8. The completed
alternative-text scales 0.1 and 0.3 at threshold 16 supply the other matched
threshold comparisons without rerunning them. Zero-scale controls verify the
lexical rewrite/formatter and bound the pruning-only loss at both thresholds.
The original-lowercase controls isolate the spelling policy at the same budget.
New jobs retain 24-GB GPU and request 64-GB host memory, with the existing pinned
model, export and native numerical gates. Original 14 conditions and their
recognition identities are unchanged; the entrypoint now registers 24 total,
only ten of which are new.

Keep ordinary summed Qwen token log probabilities, no length averaging, word
reward, subword LM, acoustic-prior change, or test-set tuning. Compare I/D/S and
RTF alongside WER before increasing the search budget. If the lower-cost runs
cannot preserve quality, do not interpret a tighter-pruning loss as model quality.

These are **closed-tree** Qwen experiments: they do not yet have the arbitrary
OOV support of the word-LM extended-tree runs. Qwen has no linguistic UNK token;
open fusion requires a separate RASR mechanism to reconstruct and tokenize the
actual hypothesized surface words, then score those tokens in the correct Qwen
history. Do not reuse the word-LM fallback by mapping OOVs to EOS or an invented
UNK. Resolve the closed-tree score/text-policy and runtime questions first,
then integrate dynamic spelling scoring before drawing open-vocabulary conclusions.
