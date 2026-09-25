# impl: window_word_types.py (2026-09-25)

Status: DONE_WITH_CONCERNS. The script is written and ran end to end on a smoke subset. The full run has not been submitted.

## Files (new; existing files untouched, nothing committed)
- `analysis/prior_gap/window_word_types.py` (about 330 lines): for each window (i6_seed0, emu0, emu1, emu2) it computes the word types, tokens, g2p-only types and tokens, and the distinct bigram types under the three `_bigram_types` conventions. It also runs a self-check against the replay job's output and computes the lexicon x word-LM restriction.
- `analysis/prior_gap/run_word_types.sh`: a copy of `run.sh` with only these changes: the script name, the job name `window_word_types`, and `--time=03:00:00` (run.sh uses 02:30:00). It keeps cpu_modern, hlt, 2 CPUs and 64G.

## How the code is reused
- Imported and not copied:
  - from `prior_window_spread`: `joint_pass`, `sample_and_fit` (which writes and fits each window), the paths, `T2B_EMU_SEEDS` and `T2B_DROP_FRACTION`;
  - from `prior_gap`: `load_phonemization_lexicon`, `window_line_flags`, `replay_window_texts`, `KenLMWordLM` and `restrict_to_word_lm`. Together these are the body of `WordWindowReplayJob.run`;
  - `LexiconTrieBuildJob._bigram_types`.
- The line mapping is replayed exactly as `WordWindowReplayJob` does it:
  - `window_line_flags` selects the counted lines (the held stride is 101, so 1,000,000 of the 1,010,000 lines are counted);
  - `replay_window_texts` maps window line j to the keep[j]-th kept word line;
  - for an emu window, the phonemisation map is passed without the dropped types. The replay's own kept-line filter then equals the filtered corpus. The replay asserts that every window phone line phonemises to its word line and that the kept count equals the sample's.
- CONCERN: the drop-type selection and the per-line drop mask are inline in `prior_window_spread.main`, so they cannot be imported without editing that file. About 6 lines restate them. Every window is checked against the prior_window_spread output on three counts: kept lines, the sha256 of the decompressed sample, and ppl3. Any mismatch aborts the run.

## Checks (smoke: first 300,000 word lines, n_out 10,100)
- The run finished with rc 0 in 43 s, with a maximum RSS of 2.3 GB.
- Identity against a smoke run of prior_window_spread.py (`--smoke-lines 300000`) passed for all 4 windows: the line counts, the sample sha256 and ppl3 were all equal.
- The smoke type counts (i6_seed0 21,090; emu0 20,464) were recounted independently with `sort -u` and matched.
- The self-check was SKIPPED in smoke mode because the smoke window is not the job's window. In the full run it compares the i6_seed0 type count, a byte-identical words file, and the token, line and bigram counts against `WordWindowReplayJob.YLcmHAGAbZ1j/output/window.words.txt`. That code path has not been executed yet.
- Reference counts on the job output, computed in both modes (these are real numbers, not smoke numbers):
  - 1,000,000 lines, 19,885,331 tokens and 182,215 types (a `sort -u` recount agrees);
  - 44,061 of those types are g2p-only;
  - bigram types are 3,414,449 in-line, 3,448,655 with `<s>`, and 3,510,073 with `<s>` and `</s>`. JUPITER banked 3,302,936.
- Lexicon x word-LM restriction, using the trie job's own KenLM binary:
  - `restrict_to_word_lm` keeps 182,215 words, which matches the trie;
  - all 182,215 window types are in the LM vocabulary and in the lexicon;
  - so on the i6 window, the trie size equals the window type count.
  - This is not computed for the emu windows, because no LM exists for them. By construction every replayed word has a pronunciation.

## Full run
- Submit command (run from the setup dir):
  `analysis/prior_gap/run_word_types.sh /work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/window_word_types`
- Expected runtime is about 70 min, estimated from measured parts:
  - joint pass: 690 s;
  - sample and fit: 4 x about 165 s;
  - replay: 4 x about 10 min (the replay job took 10:22);
  - counts: 5 x about 0.5 min.
- Expected peak RSS is about 13-15 GB (the joint pass alone was 12.4 GB), well within the 64G request.
- Outputs:
  - `window_word_types.json` and `window_word_types.txt`;
  - `samples/` and `priors/`;
  - `replay/<case>/window.words.txt`, about 106 MB each.

## Undetermined
- None of the experimental constants were chosen here: all of them are read from the job info files or imported.
