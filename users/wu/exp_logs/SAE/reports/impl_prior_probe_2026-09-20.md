# Implementer report: SAE 4a pre-launch neighbourhood probe + word-LM ESCAPE rows (2026-09-20)

Status: **DONE_WITH_CONCERNS** — deliverable 1 complete; deliverable 2 complete for the word
**bigram** row only, the word **unigram** row is **BLOCKED** (exact change named below).

Spec: `SAE_4A_prior.md` Design "Training arm", "Design review amendments" item "Pre-launch
falsifier (i)" and A4's last sentence; `reports/design_review_prior_arm_2026-09-20.md` sections
1, 5, 6.

Hard constraint honoured: `sae/emc/prior_gap.py`, `sae/emc/neural_phone_lm.py` and
`configs/config_sae_4a_prior_gap_v1.py` are **byte-identical** (`git diff --stat` on the three
paths is empty after the commit). All new code is in new files; the only use of the protected
files is `import`.

## Files

| path | what |
|---|---|
| `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/prior_probe.py` | new module, `NeighbourhoodProbeJob` (CPU, in-process `run()`), ~860 lines incl. the pre-registration docstring |
| `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_prior_probe.py` | 16 tests (edit construction, reward arithmetic vs hand computation, render completeness, config hashes, end-to-end `run()` on a tiny bed) |
| `recipe/2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_prior_probe_v1.py` | new config: two probes + one word-bigram analysis instance |
| `/e/project1/spell/wu24/2026-07-13_unsupervised/config/sae_4a_prior_probe.py` | workspace shim (setup dir is not a git repo, so not committed) |

Commit `d22d81d` on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm`, staged by
explicit path, **not pushed**, no assistant attribution. Other people's uncommitted work in that
checkout (`config_sae_1g_v1.py`, `neural_phone_lm_v2.py`, `text_filter.py`, …) was left untouched.

## Hashes

New (registered by `config/sae_4a_prior_probe.py`):

```
ctrl_50_ep4/dev-other               speech_llm/sae/emc/prior_probe/NeighbourhoodProbeJob.ZhYOeCltF4wW
ctrl_50_ep10/dev-other              speech_llm/sae/emc/prior_probe/NeighbourhoodProbeJob.6JZvwE7T1UmA
ctrl_50_ep10/dev-other_neural_wlm2  speech_llm/sae/emc/prior_gap/PriorGapAnalysisJob.m6lUxAO65f6A
```

Banked (loaded from `config/sae_4a_prior_gap.py` **after** the new files existed, unchanged):
`2RkbKYl0v1XK`, `Iv6P6YVPNWmB`, **`Gct95xZHe0zt`**, `xObXEwRpvmzd`, `5wNIQs2lpC5P`, `pBozvj6c3l16`,
`pg14aEYJyiva`. The banked read `Gct95xZHe0zt` is untouched, as required.

`m6lUxAO65f6A` differs from `Gct95xZHe0zt` in exactly two `__init__` kwargs, `word_lm_order`
(3 → 2) and `name`; a test asserts that (comparing every hash-relevant attribute, excluding the
`out_*` outputs). The `neural_lm` input is the **same object**: the config reconstructs
`NeuralPhoneLmTrainJob` exactly as `config_sae_4a_prior_gap_v1.neural_phone_lm()` does and
**asserts** its id is `Iv6P6YVPNWmB`, so no finished job is re-funded and the two instances differ
in one input, not two (a frozen `tk.Path` to the same model file would have hashed differently).

## Second-choice dump: EXISTS, nothing to re-decode

`posteriors.hdf` of the banked PER jobs is `[frames, 40]` float32 **normalised log-posteriors**
(not an argmax), already pinned by `config_sae_4a_private_code_v1._decode`:
ep4 `ReturnnForwardJobV2.AY6gwxHxHcUR`, ep10 `ReturnnForwardJobV2.QMYeLWX1G7Na`. The probe reads it
with `blankfree_eval_jobs._hdf_sequences` (needs a `tk.Path`, not a `str`).

Real-data smoke over **all 2864 dev-other utterances × both checkpoints** (scratch script, no job
launched):

* the collapsed argmax reproduces the banked `greedy_raw` / `greedy_phones` on **every** utterance
  → the probe's y0 is the banked decode, not a re-derivation;
* 22,892 (ep4) and 22,912 (ep10) edits drawn, i.e. ≈ 8 × 2864 (a handful of utterances have fewer
  than 8 distinct edit sites);
* mean edit length 35.2 / 57.8 tokens, max 266 / 305 — well under the neural LM's 512-position
  limit, so A4's mask-and-count policy is implemented but never fires at this bed.

## What the job computes (all of it pre-registered in the module docstring, with the code)

Per utterance: y0 = SIL-dropped collapsed decode; K = 8 single-token edits drawn without
replacement with `np.random.default_rng(edit_seed)` (`edit_seed=0`, the module idiom) consumed in
sorted-tag order — **substitutions** of a non-SIL frame run by that run's frame-posterior second
choice (argmax with the argmax masked to −inf, majority over the run's frames) and **deletions** of
a non-SIL run. Edits are applied at frame level and then collapsed and SIL-dropped, so an edit can
merge with a neighbour, exactly as a decode would.

Scorers (all from `emc.prior_gap`, unchanged): Witten-Bell phone unigram (`order1`) and trigram
(`order3`), the banked neural phone LM, lexicon ESCAPE and lexicon STRICT (exact Viterbi over
(position, word-LM state, escape-open)). Conventions printed in both outputs: BOS context, **no**
end-of-sentence term, nats, SIL dropped.

Rewards, per string as sums and (for the report) per token:
`r_nn = neural − unigram`, `r_lex = escape − unigram` (amendment A1), plus the **disclosed
original** `r_nn_orig = neural − trigram`, `r_lex_orig = escape − trigram`.

Reported per checkpoint, for each of the four rewards: within-neighbourhood **std** (ddof 1,
nats/utterance, median and mean), **Pearson and Spearman** of `r − r(y0)` against
`trigram − trigram(y0)` pooled over all edits (the anti-trigram check), the **fraction of edits
raising the strict segmentable word count**, and the **fraction of utterances with
`r(gold) > max over {y0, edits}`** (A3, replacing the dead band). Outputs `prior_probe.json` and
`prior_probe.md`.

Gold is a **disclosed label-using diagnostic**: it enters the gold-above-max fraction and nothing
else, and nothing the probe prints enters training, checkpoint choice or selection. Stated in the
module docstring, the config docstring and the rendered report.

## BLOCKED: the word-UNIGRAM ESCAPE row

`PriorGapAnalysisJob`'s word-LM "input" is not a path but an **order**
(`word_lm_order: int = 3`); the model is built inside the job by
`self._lmplz(window["words"], self.word_lm_order, …)` + `build_binary`. `lmplz -o 1` writes a valid
ARPA, but KenLM cannot represent an order-1 model at all: `build_binary` **and** `kenlm.Model` on
the ARPA both refuse it with *"This ngram implementation assumes at least a bigram model"* (checked
2026-09-20 with this campaign's own `tools/kenlm` binaries; a test records it).

Exact change needed, **inside `sae/emc/prior_gap.py`** (deliberately not made — banked jobs and a
live manager's running jobs re-import that file):

1. an optional `word_lm: Optional[tk.Path] = None` parameter with
   `__sis_hash_exclude__ = {"word_lm": None}` (hash-neutral for the banked instances, since
   excluding a NEW parameter is one-directional), used instead of the internal `_lmplz` call when
   given; **or**
2. a `UnigramWordLM` class with `KenLMWordLM`'s interface (`begin_state` / `score` /
   `__contains__` / `order`) selected when `word_lm_order == 1`.

Either is a one-input change; both need `prior_gap.py` edited, so they are for whoever owns that
file once the live manager's jobs are done.

## Checks run

| check | result |
|---|---|
| `pytest src/speech_llm/sae/emc/test_prior_probe.py` | 16 passed |
| `pytest src/speech_llm/sae/emc/test_prior_gap.py` (regression on the untouched module) | 24 passed |
| `ruff check` on the three new files (line-length 120) | All checks passed |
| config load from the setup dir, new hashes printed | the three ids above |
| config load of `config/sae_4a_prior_gap.py` after the change | `Gct95xZHe0zt` and the other six banked ids unchanged |
| real-data neighbourhood smoke, 2864 utts × 2 checkpoints | collapsed argmax == banked decode everywhere; lengths ≪ 512 |
| end-to-end `run()` on a tiny bed (tiny corpus, tiny neural LM, synthetic posteriors) | both outputs written, every reported field rendered |

Not run: the jobs themselves (the dispatch forbids launching). **Loading and the tiny-bed `run()`
do not prove the probe's numbers are right at the real bed** — they prove the code executes and
renders.

## Choices made where the spec left room (all named in the code)

* Neighbourhood std reported **both** over {y0} ∪ edits (primary) and over the edits alone.
* Strict word count is 0 when a string is unsegmentable; an escape-real-word count
  (`n_words − n_escape_words`) is reported alongside, since the strict count is degenerate on most
  decodes.
* Edit sites are **non-SIL** runs only (a SIL substitution/deletion is a decode-level change with no
  phone content).
* `edit_seed = 0`, `version = 1`, both hash-relevant.
* Word-LM order for the probe's lexicon scorers is the banked `WORD_LM_ORDER = 3`, so the probe's
  ESCAPE column is the banked table's quantity.
