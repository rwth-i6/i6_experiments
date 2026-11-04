# Pinned OOV tree prototype experiments

Entry point: `oov_prototype.run(ctc_model=...)`, also called by LibriSpeech
`run_all`. Reuses the existing trained BPE CTC checkpoint and pinned Granite
model. All 109 recognition conditions run on **dev-other only**. No new acoustic
model training or test-set tuning is introduced.

## Decoder and compatibility

`tools.get_oov_prototype_rasr_binary_path()` creates a separate Git clone and
CMake job at `5083eddece2982cb3b55c09daf1368e6d460f467`. The approved
`rasr_oov_separator.patch` adds only the XML spelling
`special="unknown-word-start+nonword"`, registering the same separator lemma in
both groups. It does not change fallback scoring/topology. Patch content and
commit are part of the build hash; no production hash overrides are reused.
The source checkout under `/work/asr4/berger/rasr_dev` is not modified.

The new binary is used for Python `librasr`, LM-image generation, and state-tying
checks. This is a master-based decoder without berger_dev LM look-ahead, so all
closed baselines are rerun. Existing experiments leave the new options unset.
No shared writable tree archive is configured: recognition jobs have separate
working directories, avoiding cross-policy cache contention.

## Inputs and checks

- Before any full-corpus tree recognition, a CPU-only `decoder-smoke` job runs
  12 forced-path cases in the freshly built `librasr`: known-word exclusion under
  a strong UNK reward, positive/negative per-word unknown costs, two unknown words,
  CTC repeated frames/blanks, and Granite standalone/leading/trailing separators.
  Each case has its own process and logs. This checks real startup/decoding, not
  just Python config construction; it remains a small smoke test, not a proof.
- BPE uses one full-lexicon fallback file for disabled, legacy, and known-excluding
  search. Existing `@@`/`&&` formatting is retained.
- Granite's inspection job writes the classification consumed by its lexicon job:
  original token, model ID, sanitized phoneme, boundary role, punctuation flag.
  It rejects unsupported internal boundaries/whitespace and checks tokenizer
  round trips on 200 dev utterances and explicit punctuation/spacing examples.
- One acoustic inventory, unchanged and in model output order, is used throughout.
  Granite pieces are marked with `&&` for unambiguous piece-versus-lemma formatting.
  A mixed formatter joins complete piece runs and preserves word time spans.
- Granite's matched disabled/known-excluding pair shares a lexicon restricted to
  single-surface-word pronunciations. This avoids an unvalidated corner of the
  pinned prototype, but removes cross-word normalization aliases. The report lists
  every removed pronunciation and fails if a word loses all pronunciations.
  Two additional closed-tree controls (no LM and 4-gram scale 0.3) retain the
  original lexicon to quantify this restriction independently.
- A real `allophone-tool` job dumps all four boundary-flag combinations for every
  label under monophone tying with one state per phone. A validation job checks
  every resulting emission index against the acoustic vocabulary. BPE's existing
  EOS name alias `</s>` -> `<s>` is explicit; it changes no index.
  Allophones are enumerated inside RASR (`add-all` with zero context), not read
  from a list: its list reader treats the legitimate `#` phoneme as a comment.
  The validator distinguishes expected `#` states from comment lines as well.
- The same validation job rejects multi-syntactic-token exits and verifies the
  original 4-gram contains a finite usable `<UNK>` unigram. Recognition consumes
  its output lexicon, so a failed validation blocks expensive tree decoding.
  RASR still performs its own LM/startup validation.

## Sweeps

Common settings: prior scale 0, blank penalty 0, no word-exit reward, strict
sentence end, one acoustic label scorer, Viterbi recombination (`on`). The no-LM
condition truly has no word LM. The original, unadapted LibriSpeech 4-gram is used
throughout; LM adaptation is not bundled with the tree change.

| Axis | BPE CTC | Granite CTC |
| --- | --- | --- |
| Boundary mode | continuation-marked | word-start-marked |
| Main comparison | disabled / legacy / known-excluding | disabled / known-excluding |
| Word LM | none; 4-gram scales 0.2, 0.4, 0.6 | none; 4-gram scales 0.1, 0.3, 0.5 |
| Known-excluding beta | -4, -2, -1, 0, 1, 2, 4 | -4, -2, -1, 0, 1, 2, 4 |
| Positive-penalty follow-up | beta 6, 8, 10, 12 at LM scales 0.4 and 0.6 | unchanged |
| Closed-limit follow-up | beta 10, 20, ..., 100 at fixed LM scale 0.6 | unchanged |
| Main beam / threshold | 256 / 16 | 256 / 16 |
| Additional beams | 64, 1024 | 64, 1024 |
| Wide no-LM smoke pair | 4096 / 30 | 4096 / 30 |

Word-end beam and threshold equal the corresponding main beam and threshold,
avoiding the old very tight word-end pruning in this first exploration. The wider
and smaller beam pairs compare disabled versus known-excluding at beta 0, both
without an LM and at the middle 4-gram scale. Each model also gets a same-build
lexicon-free greedy control.

Counts: 64 BPE conditions, 43 matched/control Granite conditions plus 2
original-lexicon Granite controls. The wide smoke condition is finite beam search,
**not an exhaustive correctness proof**. It can be expensive with Granite's 16k
inventory; all tree jobs request 32 GB RAM. These are setup defaults, not a claim
that the search is converged or that resource use has already been measured.

`beta` is a cost outside the LM scale, charged once per unknown word. Positive
penalizes unknown words; negative rewards them. It is not a renormalized change
of LM probabilities. Closed/legacy conditions use beta 0; the legacy decoder
ignores the new penalty.

The BPE follow-up adds eight conditions at beam 256 / threshold 16, retaining
all original conditions and their configurations. Together with the existing
beta 4 points, it covers beta 4, 6, 8, 10, 12 at both LM scales. The initial
results showed excessive novel spellings/word merges, rising deletions with LM
scale, and no WER improvement from the tested larger beams. This follow-up changes
only the per-unknown-word cost; Granite, LM training, acoustic scores and pruning
remain unchanged.

The next BPE sweep fixes the LM scale at the closed-vocabulary-tuned value 0.6
and spans beta 10, 20, ..., 100. Beta 10 reuses its existing condition; nine new
conditions are appended, with beam 256 / threshold 16 and all other settings
unchanged. Retain the earlier scales as completed controls, not new tuning axes.
At high beta, compare hypotheses and IV/OOV errors against the same-build closed
tree, not just rounded WER. This tests convergence toward closed-vocabulary
behavior; it does not assume exact equivalence at a finite penalty/beam, since
pending fallback prefixes can still affect pruning before the completion cost.

Select on dev-other using both IV and OOV reference-word accuracy against the
same-scale closed tree and greedy baseline, alongside total WER and its S/D/I
breakdown. Use one fixed reference partition from the standard lexicon for every
condition. Suppressing all fallback and returning to closed-tree accuracy is a
control outcome, not successful OOV recovery. Inspect newly introduced IV errors
(especially word merges) and retained/corrected OOVs before selecting a penalty;
neither aggregate WER nor fallback frequency alone establishes the desired trade-off.

## Outputs and interpretation

Build alias/output: `tools/rasr-oov-prototype/{source,cmake,binaries}`.
Runtime checks: `tools/rasr-oov-prototype/decoder-smoke-{report,logs}`.
Input audits: `datasets/LibriSpeech/oov-prototype-5083edde/...`.
Recognition descriptors start with `oov-prototype-5083edde_` and encode mode, LM,
beta, beam, and threshold. CTM, raw RASR log, scoring reports, runtime and
search-space statistics use the existing recognition output registration.

The standard setup's HF normalizer is used for Granite scoring; BPE retains its
existing scoring convention. Compare improvements within each model, not raw
cross-model WER as if their scoring normalization were identical.

Piece traceback means the fallback representation was used; it does **not** mean
the LM scored UNK. The prototype can resolve exact known pronunciations to their
known LM tokens while keeping piece traceback. Its per-segment unknown/known-
resolved counters count attempts across the beam, not final-transcript words.
Do not use those counters or `&&` alone as an OOV-word accuracy measure.

After completion, first inspect validation reports and same-build baselines,
then beta/LM sweeps and beam sensitivity. Prefer dev-selected settings for later
held-out evaluation. No WER improvement or successful full prototype decode is
claimed merely by constructing this job graph.
