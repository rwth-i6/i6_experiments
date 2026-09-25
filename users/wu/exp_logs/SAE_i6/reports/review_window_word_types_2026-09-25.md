# review: window_word_types launch (2026-09-25)

APPROVE_WITH_CONCERNS. The code measures what the dispatch asks for, on exactly the windows of the
prior_window_spread run, with the replay job's line mapping and the trie job's bigram counter. I found
no defect that changes a number it writes. There is one reading hazard on the bigram comparator, and
one conditional abort risk.

Launch reviewed: `analysis/prior_gap/run_word_types.sh /work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/window_word_types`
(script `analysis/prior_gap/window_word_types.py`; implementer report `impl_window_word_types_2026-09-25.md`).

## Findings

1. `analysis/prior_gap/window_word_types.py:103,105-112,197`: the three bigram columns are printed
   against one banked number, 3,302,936, and the JSON stores no convention with it. The convention is
   known. JUPITER's own `LexiconTrieBuildJob.rlMsnTBSZXsB` gave in-line 3,302,936 (the banked value),
   with `<s>` 3,335,328, and with `<s>` and `</s>` 3,393,577 (`exp_logs/SAE/SAE_4A_lexlat.md:244`, survey
   s4). The same `_bigram_types` produced them.
   - Failure: an emu window whose `with_bos` or `with_bos_eos` column lands near 3.30M reads as "near
     3.30M". The like-for-like JUPITER values for those columns are 3.34M and 3.39M.
   - Only `in_line` compares to 3,302,936. On i6 that is 3,414,449, not 3.51M.
   - A fourth comparator exists and is printed but not compared: JUPITER's window has 19,629,091 tokens
     (`exp_logs/SAE/reports/survey_lexicon_scorer_2026-09-20.md:54`); i6 has 19,885,331.
   - Action for whoever reads the result: compare column by column against 3,302,936 / 3,335,328 /
     3,393,577, and compare tokens against 19,629,091.
2. `analysis/prior_gap/window_word_types.py:240`: the abort compares ppl3 with exact float `==`.
   - The reference was computed on cn-604, and the i6 prior job ran on cn-601. cpu_modern also holds
     cn-801..832, whose CPUs I could not check.
   - numpy 2.4.6 dispatches float64 `log` to AVX-512 kernels when the CPU has them. A node with
     different SIMD support can move ppl3 by a few ULPs and abort a correct window about 15 min in, after
     the joint pass and the first fit.
   - The decompressed sample sha256 in the same line already proves the window is identical.
   - Unverified. The cost is one relaunch. Pinning to a cn-60x node avoids it.

## Checks done, no finding

- Q1, restated lines: lines 213-220 and 228-231 are token-for-token equivalent to
  `prior_window_spread.py:385-408` and `389-393`.
  - They use the same sorted type list, `random.Random(e).sample(types_sorted, int(n_types*0.5))`, the
    same reduceat mask and the same compress stream.
  - The guards read the real files of Slurm 4349993 in `analysis_out/prior_window_spread/`, not a
    re-run: the json rows `lines_in_corpus` / `held_ppl_order3` of T1_seed0 and T2b_emu{0,1,2},
    `samples/<case>.phn.gz` decompressed sha256, and `non_bliss.types` 773,672.
  - A mismatch aborts before any replay.
  - In that run T1_seed0 matched SampleLinesJob.CrPgeKXsOosb's sha and the job's ppl3 exactly, so the
    chain reaches the job's window.
- Q2, line mapping: `prior_gap.replay_window_texts` is called with `WordWindowReplayJob.run`'s
  arguments (`word_window.py:73-82`).
  - The job's info records n_window_lines None, held_stride None and sample_seed 0. These are asserted
    at line 163 and resolve to 1,010,000 and 101.
  - For emu windows the filtered map makes `count_kept_lines` equal the joint-pass keep minus drop lines:
    the `plen` keys equal the `load_phonemization_lexicon` keys, and the dropped types are non-bliss types
    of kept lines.
  - This is asserted at line 259 (map size) and line 266 (kept count against the spread row), and the
    replay's per-line phone == word assert covers it too.
  - The self-check (lines 290-297) compares a freshly replayed i6_seed0 words file with the sha of
    `WordWindowReplayJob.YLcmHAGAbZ1j/output/window.words.txt` (106,131,742 B, finished). It is not
    vacuous: it tests the harness inputs, flags and n_out. The emu-only filtered map is covered by the
    asserts above.
- Q3, counting: `LexiconTrieBuildJob._bigram_types` is imported (`word_lm.py:229-269`). The
  working-tree diff of `word_lm.py` (another worker, 10:27, uncommitted) does not touch it.
  - g2p-only means window types not in the bliss lexicon `MergeLexiconJob.qKaOAPqURCkK`. That lexicon
    is asserted identical to PhonemizeWithSilJob's, so the count is correct under setdefault priority.
- Q4, wrapper and side effects:
  - It differs from `run.sh` only in the script, the job name and `--time=03:00:00`.
  - Output and chdir go to /work shared storage, which does not exist yet, so nothing is stale. It uses
    the env python, cpu_modern and hlt, 2 CPU, 64G.
  - Writes happen only under OUT. work/ is only read: job info files, the replay output and
    `CreateBinaryLMJob.4gueCR4UnpLG/output/lm.bin`.
  - Package modules are only imported (lexicon, phone_prior, phone_text, prior_gap, word_lm,
    model/prior); nothing edits them.
  - Trie job W0e4no47Crfu is in error state. Only its info is read, at startup.
  - The time estimate (about 70 min against the 3 h limit) and memory (12.4 GB joint pass against 64G)
    are consistent with the parts measured.
- Q5, the type count (other wrong-reason paths are in finding 1):
  - Paths that would give "near 182k" by mistake all abort: an emu replay with the unfiltered lexicon
    (266), an empty drop set (240), or a wrong window (240, sha).
  - The types column is like-for-like: JUPITER's 151,731 is both its window type count and its trie size
    (survey s1).
  - The audit's known mis-specification of the emulation biases emu window types slightly low. JUPITER
    lost rarer types: 788,092 lines, against the emulation's 798-800k. To first order the window types
    removed scale with the lines dropped, so the bias is about 1.4% of the removed types, a few hundred.
    That is small against the 30k gap.

## Note, not a finding
- The result JSON records no code version. `word_lm.py` is dirty in the working tree. The round
  commit must include the tree state the run imported.
