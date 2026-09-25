# LexiconTrieBuildJob: banked identity checks made report-only (2026-09-25)

Status: DONE. The edit is uncommitted; no job was launched, cleared or touched.

## Change
File: `recipe/i6_experiments/users/wu/experiments/unsupervised_asr/lm/word_lm.py`, function `LexiconTrieBuildJob.run`.
- The three asserts that compared against the banked JUPITER constants are now recorded and printed, not asserted: trie words == BANKED_WORDS (151,731); window word types == BANKED_WORDS; one bigram-counting convention == BANKED_BIGRAM_TYPES (3,302,936).
- They are printed as `IDENTITY vs banked: <key> <got> (banked <expected>) MATCH|MISMATCH`.
- They are written to the existing output `build.json` (`out_stats`) under the new key `identity_vs_banked`. That key holds `{trie_words, window_word_types, bigram_types}`, each as `{expected, got, pass}`. The summary line in `summary.txt` (`out_summary`) now says "matched by NONE: MISMATCH" when no convention matches, plus one IDENTITY line.
- A new internal assert: `len(in_vocab) == text["n_types"]` (trie words == window word types). Before the edit this held only implicitly, through both counts equalling 151,731. The logic behind it: the replay writes only lines whose every word is in the phonemization lexicon, and the LM vocabulary is the window types plus `<s>`, `</s>` and `<unk>`. So the lexicon restricted to the LM vocabulary is exactly the window type set.
- These asserts are unchanged: derangement bit-identity (child, word_start, is_word_end) and "the null moved nothing".
- Only docstrings and comments were updated (the module comment on BANKED_* and the class docstring's "PRE-REGISTERED CHECKS").
- These are unchanged: the `__init__` signature, defaults, kwargs and outputs, and the content of the npz, word_lm.bin, lexicon and shuffled-lexicon outputs.

## Matched bigram convention
Nothing downstream reads `bigram_convention_matched`; it is only written to build.json and summary.txt. So no fallback convention is needed. When nothing matches it is recorded as an empty list, and all three counts are recorded under `identity_vs_banked.bigram_types.got`.

## Other banked asserts in the lexlat/HLG chain
None that abort on the i6 window.
- `lm/hlg.py`, `LexlatHLGBuildJob.run`: no comparison against banked graph sizes. The G0.R0 ranges (HLG states 23.8M-24.0M, arcs 98.1M-99.1M) appear nowhere in the code (grep).
- `LexlatOfficialHLGBuildJob`: its only reference-derived logic is the memory size guard (`predict_hlg_cost`). That guard skips pruning rungs instead of asserting, and the official graphs are built from the openslr lexicon and ARPA, not from the i6 window.
- `LexlatOfficialResourcesJob`: its asserts are internal (against its own build.json and inventory).
- `model/lexlat_k2.py` and `model/lexlat_k2_train.py` (`facts`, `check_bed`, `check_expected_build`): internal only (n_words npz vs build.json, d_min, stride, min_frames, expected_build fields). I read them and did not edit them.
- `lm/lexlat_k2_official.py`: internal only.

## Checks
- Import of `lm.word_lm` and `lm.hlg` under the sae python: OK.
- Hash neutrality: I listed the sorted `_sis_id()` of all jobs in `tk.sis_graph.jobs()` for `config/sae_i6_p0.py` before and after the edit. Both lists have 164 ids and are IDENTICAL. `LexiconTrieBuildJob.W0e4no47Crfu` and `LexlatHLGBuildJob.avjHv1Xvjyqd` are both in the list.
- pytest `tests/test_lm_word_lm.py` and `tests/test_lm_hlg_size_guard.py`: 11 passed. These tests do not exercise `run()`.
- Behavioural check on the real i6 inputs. I ran the edited `run()` of the graph's own W0e4no47Crfu job object in a sis console, with the scratch dir as cwd and `lexlat.parse_arpa_word_lm` stubbed to stop right after the checks. Result: trie_words 182215, window_word_types 182215 (the internal assert holds), and bigram types in_line 3,414,449 / with_bos 3,448,655 / with_bos_eos 3,510,073 against banked 3,302,936. All three were reported MISMATCH and there was no abort. The trie build, save and summary write were not run in this check.

## Undetermined / for the orchestrator
- The failed job dir keeps its error state. The live manager picks up the new code only when that job is re-run: the worker imports the current module. Clearing the error is outside this dispatch.
- The i6 trie has 182,215 words and about 3.41M in-line bigram types, against JUPITER's 151,731 and 3.30M. The in-house HLG will therefore be larger than the banked graph that G0.R0's ranges come from. Whether 200 GB and the ladder (0.0, 0.5, 2.0) still suffice has not been checked. The G0.R0 size ranges become a recorded comparison, not a pass criterion, for the phase file.
- The working tree also has an unrelated modification to `SAE_i6_P0.md` from another session. I did not touch it.
