# SAE 4a Step 0: `PriorGapAnalysisJob` (implementer report, 2026-09-20)

Status: **DONE_WITH_CONCERNS** (one material convention ambiguity resolved by computing both
readings, named below).  Nothing was launched.  No `*.md` under `exp_logs/SAE/` was edited except
this new report file.

## What was built

| file | what |
|---|---|
| `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/prior_gap.py` | new module: scorers, aggregates, render, `PriorGapAnalysisJob` |
| `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_prior_gap.py` | 9 tests (a)-(d) of the brief plus the window replay |
| `.../librispeech/configs/config_sae_4a_prior_gap_v1.py` | the config: pinned inputs, one job, two registered outputs |
| `config/sae_4a_prior_gap.py` (setup dir, not in the checkout) | two-line entry point, same pattern as `config/sae_4a_private_code.py` |

A **new module** on purpose: a file-sha `__sis_version__` moves every job in a file, so this could
not be an addition to `private_code.py` or `prior.py`.

Job: CPU, `rqmt = {cpu: 4, mem: 16, time: 4}`, everything in `run()` in-process.
Outputs `output/prior_gap.json` and `output/prior_gap.md`, alias `sae/4a/prior_gap/ctrl_50_ep10/dev-other`,
registered under `exp2025_11_06_speech_llms/librispeech/sae_4a_prior_gap/ctrl_50_ep10/dev-other/`.

The pre-registered **read rule** and **decision table** of `SAE_4A_prior.md` "Step 0" are verbatim in
`prior_gap.py`'s module docstring (pre-registration lives with the code), and both are evaluated and
printed by the job with their convention (`verdict()`, `decision()`, threshold
`IS_SD_THRESHOLD_NATS = 3.0` taken from the spec's "< 3 nats").

## Inputs, resolved on disk 2026-09-20 (all exist)

| role | path |
|---|---|
| decode, SIL kept | `work/speech_llm/sae/emc/blankfree_eval_jobs/BlankfreeGreedyPerJob.dG4n46xTRSl0/output/greedy_raw.json` |
| decode, SIL dropped (the PER string) | same job, `output/greedy_phones.json` |
| gold phones | `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/eval/GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json` (`s0b.GOLD_PHONES`, split `dev-other`) |
| live prior | `work/speech_llm/sae/emc/prior/PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz` (`pc.prior_npz()`) |
| prior window (phones) | `work/speech_llm/sae/emc/text_sample/SampleLinesJob.orN768ARKwlt/output/text.phn.gz` -- the seeded uniform 1,010,000-line sample of `PhonemizeWithSilJob.DbFgvZOGZQ8F`, i.e. the priorshuf refit's own window, **not** the alphabetical head window |
| word corpus | `work/i6_core/tools/download/DownloadJob.g4jClO48cAvP/output/librispeech-lm-norm.txt.gz` |
| bliss lexicon | `work/i6_core/lexicon/modification/MergeLexiconJob.qKaOAPqURCkK/output/lexicon.xml.gz` |
| g2p lexicon | `work/i6_core/g2p/apply/ApplyG2PModelJob.myTIGtmrUIFq/output/g2p.lexicon` (its entries count as lexicon words) |
| KenLM binaries | `phoneme_lm.kenlm_binaries()` |

The decode and the prior come from `config_sae_4a_private_code_v1` (`pc._decode(10, "dev-other")`,
`pc.prior_npz()`), so the strings and the trigram are literally the private-code read's own inputs.
Every frozen file is pinned with `hash_overwrite=` so no producing job enters this graph (rebuilding
one would re-hash any class that gained a parameter and show up as RUNNABLE, i.e. re-fund it).

**Word-level twin of the window.**  The window is phone text; the word trigram needs the same lines
as words.  `PhonemizeWithSilJob` drops corpus lines containing an out-of-lexicon word, so phone-window
line *j* is the *j*-th kept line of `librispeech-lm-norm.txt`.  The job replays that filter (pass 1:
count kept lines; then `text_sample.sample_line_indices(n_kept, 1_010_000, seed 0)`; pass 2: lockstep
selection) and **asserts per line** that re-phonemising the recovered word line reproduces the phone
line.  Both the happy path and a deliberately misaligned window are covered by tests.

**Training-data parity.**  The phone KenLMs and the word trigram are fitted on exactly the lines
`prior.count_ngrams` counts (`index % HELD_STRIDE != 0`, non-empty) -- 1,000,000 lines -- so no prior
in the table sees a line the live trigram has not.

## Scorer conventions (all numbers are natural log, per token = total / token count)

* **Orders 1, 2, 3** -- the live interpolated Witten-Bell estimator loaded from the arm's `prior.npz`
  (`PhoneNgramPrior`).  BOS-padded context, **no end-of-sentence term**.  Order 3 therefore
  reproduces the live prior's per-token convention exactly (test (b) asserts equality with
  `prior.log_prob` and with `private_code.prior_score_per_token`).
* **Orders 4, 6, 8** -- KenLM, modified Kneser-Ney.  `BeginSentenceWrite` for the `<s>` context, then
  `BaseScore(in, tok, out)` per token, **`</s>` is never scored**, log10 x ln(10) -> nats.  Same
  convention as the live prior, so per-token numbers are comparable across rows.
* **Lexicalised prior** -- exact Viterbi over (position, KenLM word state): every lexicon
  pronunciation matching at the current position is a transition scoring `log p(word | state)` under
  the word trigram (again `<s>` context, **no `</s>`**); SIL is a free epsilon that may be skipped at
  a word boundary only; a string with no segmentation scores `-inf` and is excluded from the
  aggregates, which are reported over the segmentable subset together with `frac_segmentable`.
  `beam=None` = exact (no pruning constant had to be invented: the measured DP cost is ~1e5 state
  extensions per gold string).  Per-token normalisation uses the **phone** count, so the lexicon row
  is on the same denominator as the n-gram rows.
* **Shuffled null** of each set: tokens permuted within the utterance, `np.random.default_rng(0)`
  (length and unigram counts preserved; test (c)).

## KenLM build command (in the job, `lms/` inside the job dir)

```
lmplz -o <order> --interpolate_unigrams True -S 8G -T <job>/lms --discount_fallback 0.5 1.0 1.5 < <text> > <arpa>
build_binary <arpa> <bin>          # the ARPA is deleted after the binary is written
```

`-S` is half the job's memory.  `--discount_fallback 0.5 1.0 1.5` is the campaign's canonical value
(`config_sae_1g_h4_matched_lm_v1`); a 40-symbol vocabulary has no singleton unigrams, so modified
Kneser-Ney needs it.  The word trigram is the same command on the recovered word text.

## The one material ambiguity (reported, not silently resolved)

Gold is SIL-free; the collapsed decode carries SIL.  A per-token number over different token
inventories confounds the gap (most of all for the lexicon row, where SIL is free).  The spec does
not say which string the private code is.  The job therefore computes **both pairings** and prints
the read rule and the decision table for each:

* `like_for_like` (**primary**): gold vs the decode with SIL dropped -- the string the banked PER is
  scored on, and the pairing whose trigram numbers reproduce the banked `decipher.json` E5 rows.
* `sil_kept` (disclosed, labelled "NOT comparable to gold"): gold vs the decode with SIL kept.

If the planner wants a single pairing in the log, `like_for_like` is the one anchored to the banked
numbers.  Anchors it must reproduce (banked trigram per token, 2864 dev-other utterances):
gold **-3.1992**, private SIL-free **-4.6287**, private SIL-kept **-3.9865**.

## Checks run

* `pytest recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_prior_gap.py -q` -> **9 passed**
  (0.64 s, conda env `speech_llm`).  Covers the brief's (a)-(d): the lexicon scorer picks the correct
  best segmentation against a *real* lmplz word trigram ("ICE CREAM" beats "I SCREAM", the score
  equals the LM's own sum), returns `-inf` for an unsegmentable string / SIL inside a word / an OOV
  phone; the order-3 scorer equals `prior.py`'s own scoring and `private_code.prior_score_per_token`
  (asserted at the production dtype, float64); the null preserves length and unigram counts; the
  render prints **every** row of the json, with the fixture built *through* `assemble_record` so the
  render is checked against the producer's own record shape (the computed-but-never-rendered trap).
  Two further tests cover the window replay and its misalignment failure, one covers all four
  decision-table outcomes, one the paired halves / spread / log-weight sd.
* Config imports under the sis env; `sis console -s config/sae_4a_prior_gap.py` loads the graph with
  **no traceback** and exactly one job: `speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.l0p0srBryKrs`,
  plus the two registered outputs.  The hash is stable across the last refactor.  The cluster was not
  touched and nothing was submitted.
* `ruff` line length 120 respected; `py_compile` clean.

Loading and the unit tests do not prove the job's numbers; the job has not been run.

## Commit

`recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`, **`1431dbf`**
"SAE 4a step 0: the gold-minus-private prior gap against prior strength".  Three explicit paths
staged (`prior_gap.py`, `test_prior_gap.py`, `config_sae_4a_prior_gap_v1.py`); other people's dirty
files in that checkout (`config_sae_1g_v1.py`, `config_sae_3e1_d6_swap_cont_v1.py`) were left alone.
Not pushed.  `config/sae_4a_prior_gap.py` lives in the setup dir and is outside that checkout.
This report is not committed (the dispatch named only the speech-llm repository).

## To launch (planner's call, not done here)

```
cd /e/project1/spell/wu24/2026-07-13_unsupervised && sis m config/sae_4a_prior_gap.py
```

## Notes for the planner

* Runtime is dominated by `lmplz` at order 8 on 1e6 lines and by the lexicon DP over 6 string sets
  x 2864 utterances; 4 h / 16 GB is an estimate, not a measurement.  If the order-8 build is tight,
  `-S` (half of `mem`) is the knob.
* `version=1` is a hash-relevant parameter, as in the private-code jobs: bump it to force a re-read
  after a convention change.

---

# v2 (2026-09-20, after the code review and the v1 build_binary failure)

Status: **DONE**.  Nothing was launched.  The v1 job directory
(`PriorGapAnalysisJob.l0p0srBryKrs`) was not touched.

New job hash: **`speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.2RkbKYl0v1XK`**
(`version=2`, a hash-relevant parameter; the graph holds exactly this one job and three registered
outputs).  Launch is unchanged: `sis m config/sae_4a_prior_gap.py`.

## What changed, and why

| # | change | source |
|---|---|---|
| 1a | `lexicon_strict`: the trie holds ONLY lexicon words the word trigram has in its vocabulary; a string with no such segmentation is -inf.  Its gap and verdict are on the subset segmentable on BOTH sides, read against the **trigram gap recomputed on that same subset** (`pairings.*.subset_trigram_gaps`), printed as `(subset)` and disclosed with its fraction; it fires no row of the decision table. | review F1, F2 |
| 1b | `lexicon_escape`: the same trie plus ONE escape word for any span of non-SIL phones, at `log p_wordLM(<unk> \| state)` + per phone `log p_1(phone) + log(0.5)`.  Every string is finite, so its gap and verdict are on the FULL paired set; **this is the row the decision table's lexicon clause reads** and the only one a training reward could use. | review F2 |
| 2 | `output/per_utt.json`: per utterance tag, per string set, per prior -- log-probability (`null` = -inf), token count, segmentable flag.  Any subset read is recoverable without a rerun. | review F1 |
| 3 | the report renders EVERY field of `priors.*.sets.*`, `pairings.*.priors.*` and `pairings.*.subset_trigram_gaps.*` (dynamic columns), `n_paired` also in the main table. | review F3 |
| 4 | `_pin()` keys end in a 12-hex sha of `os.path.realpath`, so repointing the window / word corpus / either lexicon moves the job hash (verified: swapping the window's producing-job hash gives `FiSGeYizEaly`). | review F5 |
| 5 | the docstring's prediction line now reads "the nulls' segmentable fraction and lexicon score fall well below gold's", marked as an amendment (34 of 39 phones have a single-phone word). | review F6 |
| 6 | phone orders are **4 and 6**, not 4/6/8: `tools/kenlm` is compiled with `KENLM_MAX_ORDER` 6 and `build_binary` refused the order-8 ARPA, where v1 died (slurm 1916086, `log.run.1`).  KenLM was not recompiled.  The 6-gram alone carries the "table-lookup proxy" clause. | v1 failure |

Everything the review marked sound is unchanged: the anchors (-3.1992 / -4.6287 / -3.9865), BOS /
no end-of-sentence, the window and its counted-line split, the `lmplz` flags, the per-phone
denominators, both pairings, the null, the IS bracket.

The escape's per-phone model is the **live Witten-Bell unigram** (the `order1` row of this table),
restricted to the 39 phones: SIL is not an escaped phone, it stays the free word boundary and it
closes an open escape.  One convention, no tuning; `escape_length_log_prob` is a job parameter only
so that it is in the hash.

## Checks run

* `pytest .../sae/emc/test_prior_gap.py -q` -> **16 passed** (0.75 s, conda `speech_llm`).  New:
  the escape's price against a hand-computed `<unk>` + per-phone term, one escape per contiguous
  span (and that one span beats two), SIL not escaped, escape never below the strict score, the
  strict trie excluding a word the LM never saw (and that word being a cheap one-`<unk>` cover if
  it is not excluded), the subset read (a strict row that beats the trigram only on its own subset
  is found, and fires no row of the table), the per-utterance dump round-tripping through strict
  JSON and reproducing `paired_tags`, every banked aggregate field appearing in the rendered `.md`,
  and the python `kenlm` module loading an order-6 binary built by these binaries.
* Graph loads under the sis env from the setup dir: **one job**, `PriorGapAnalysisJob.2RkbKYl0v1XK`,
  three registered outputs (`prior_gap.json`, `prior_gap.md`, `per_utt.json`).  `version=1` and a
  changed `escape_length_log_prob` both give different hashes (the parameters are hash-relevant).
* Nothing submitted; no cluster call.  Loading and the unit tests do not prove the job's numbers.

## Commit

`recipe/2025-10-speech-llm`, branch `haotian_modality_matching_jupiter`, **`291dab1`**
"SAE 4a step 0: prior-gap read v2 after the code review".  Three explicit paths staged; other
people's dirty files (`config_sae_1g_v1.py`, `config_sae_3e1_d6_swap_cont_v1.py`) left alone.  Not
pushed.  This report is not committed (the dispatch named only the speech-llm repository).

## For the planner

* `rqmt` is unchanged at `{cpu: 4, mem: 16, time: 4}`, but v2 runs TWO lexicon DPs per string set
  and the escape row never dies out early on a null, so the DP cost is roughly 2-3x v1's.  4 h is
  an estimate, not a measurement; a timeout is resumable (the engine doubles `time`) but reruns
  from the start.
* The decision table's lexicon clause now reads `order6_kenlm` and `lexicon_escape` only.
  `lexicon_strict` is disclosed (`decision.disclosed_subset_rows`) and closes nothing.
* `SAE_4A_prior.md` "Step 0" pre-registers orders "1, 2, 3, 4, 6 and 8"; the 8 cannot be built with
  this KenLM.  That is an amendment to the plan and belongs in the phase file, which was not edited.
