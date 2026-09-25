# JUPITER phone-prior text gap: why the banked g2p lexicon covers half of the OOV types (2026-09-25)

Status: DONE. Read-only diagnosis. No job was run and nothing under work/ was modified. Logs were
unpacked from the finished.tar.gz archives into a session scratch dir.

## Verdict
The defect is on the JUPITER side, in the text. The banked g2p lexicon
`work/i6_core/g2p/apply/ApplyG2PModelJob.myTIGtmrUIFq/output/g2p.lexicon` holds exactly 8 of the
job's 16 parallel chunks: chunks 1-4 and 13-16 are 100% covered, and chunks 5-12 are 0% covered.
The g2p did not fail on those words and the model was not the problem. The merge task concatenated
8 chunk files that were empty at that moment. They were empty because a duplicate re-run of those
already finished `run` tasks had reopened them for writing (`"wt"` truncates) a few minutes earlier.

The duplicate re-runs come from the sisyphus engine layer. A known upstream bug made the manager
resubmit 'short'-engine (LocalEngine) tasks on every cycle, and LocalEngine then executed every queued
copy, including copies of tasks that had already finished. The i6_core `ApplyG2PModelJob` owns a latent
contract bug that turned those duplicates into data loss: `run()` writes its chunk non-atomically in
place, and `merge()` does not check coverage.

So the i6/JUPITER prior difference is not evidence of a port defect. The port's own T7 already
reproduced the banked held-out ppl 9.561056 byte-for-byte from JUPITER's window (see Q2). The JUPITER
bar is a number measured on a text with an alphabetical hole.

Owning layer: launcher/engine (sisyphus tools/sisyphus at ddcd028, pre-#314), together with the
non-atomic write in `recipe/i6_core/g2p/apply.py`. The owner is not the dataset/vocab code of the port,
not the g2p model, and not the prior fit.

## Q1. Why the lexicon covered only part of the non-bliss types

### Jobs
- OOV list: `work/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/CollectOovWordsJob.q0KCWART4B2t`
  (inputs: `DownloadJob.g4jClO48cAvP` librispeech-lm-norm.txt.gz and `MergeLexiconJob.qKaOAPqURCkK`).
  Its code (`recipe/.../posterior_hmm/data/phon_lm.py:145-176`) emits *every* corpus type that is not in
  the bliss lexicon, with no count threshold. Output: 773,673 types (`output/num_oov`, `oov_words.txt`).
- G2P model: `work/i6_core/g2p/train/TrainG2PModelJob.pD4nbqFLWtbi`, where model-best -> model-4 (err-4 = 5.36).
- Apply: `work/i6_core/g2p/apply/ApplyG2PModelJob.myTIGtmrUIFq`, concurrent = 16, variants_number 1,
  filter_empty_words True. Engine 'short' (the login-node LocalEngine).
- Phonemize: `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/text/PhonemizeWithSilJob.DbFgvZOGZQ8F`
  (inputs: myTIGtmrUIFq g2p.lexicon, MergeLexiconJob.qKaOAPqURCkK, DownloadJob.g4jClO48cAvP).
  `output/stats.txt`: `lines_in=40418261 lines_out=39630169 dropped_oov=788092`.

### Counts (observed)
| quantity | value |
|---|---|
| types in the g2p input list (oov_words.txt) | 773,673 |
| entries in the output lexicon (= words, 1 variant each) | 384,893 |
| bliss words (MergeLexiconJob.qKaOAPqURCkK, non-special, first pron) | 200,000 |
| input types with no lexicon entry | 388,780 |
| corpus types absent from bliss+g2p (full-corpus scan) | 388,780, all in the missing band; 0 outside it |
| lines dropped by PhonemizeWithSilJob | 788,092 |
| g2p "failed to convert" lines in g2p.untranslated | 0 (the file holds only 8 `stack usage:` lines) |

Per-chunk coverage. The 16 chunks were re-created with the job's own `split --number=l/16
--numeric-suffixes=1 --suffix-length=2` on oov_words.txt in scratch:
```
words.01..04  n=49384,48659,46610,46469  covered=100%
words.05..12  n=49432,48947,47883,49483,49138,49470,47805,46622  covered=0
words.13..16  n=48616,47965,46623,50567  covered=100%
```
The covered chunks sum to 384,893, which is exactly the lexicon's line count. The missing chunks sum
to 388,780. Because oov_words.txt is sorted, the hole is the contiguous band from `DITCHLIKE` (first line
of chunk 05) to `RIVAW` (last line of chunk 12). Every non-bliss corpus word in that band is missing, and
no word outside it is.

### Timeline (log.run.N and the merge/filter logs from the finished.tar.gz)
- All 16 run tasks completed at least once with `Job finished successfully`. Tasks 1-4 finished
  15:39-15:40, tasks 5-8 finished 15:42-15:55, tasks 9-12 finished 15:57-16:11, and tasks 13-16 finished
  16:13-16:27. Each ran about 15 min, and many ran twice concurrently.
- Tasks 1-4 and 13-16 were then re-run again and *completed* before the merge (1-4: 16:41:14 to 16:42:45;
  13-16: 16:43:43 to 16:44:09).
- Tasks 5-12 were re-run again and were *still running* at merge time:
  `log.run.5  [2026-07-15 16:41:15,782] Starting subtask for arg id: 4 args: [5]`, with no finish line;
  tasks 6, 7 and 8 started 16:41:50 to 16:42:46;
  `log.run.9  [2026-07-15 16:43:44,162] Starting subtask for arg id: 8 args: [9]`, whose last line is
  `16:47:44 Run time: 0:04:00`, with no finish; tasks 10, 11 and 12 started 16:43:49 to 16:44:10.
- `log.merge.1  [2026-07-15 16:46:05,679] Start Job ... Task: merge` then `Job finished successfully`.
- `log.filter.1 [2026-07-15 16:46:08,679] ... Task: filter` (host jpbl-s02-04).
- `usage.run.9` still shows the re-run alive at `'current_time': 'Wed Jul 15 16:49:50 2026'`.

Mechanism of the empty files. The empty files are *observed*; the explanation below is *inferred* from
code and the filesystem.
- `recipe/i6_core/g2p/apply.py:82-101`: `run()` opens `g2p.lexicon.{task_id}` with `uopen(..., "wt")`
  (truncating it) and streams g2p.py's stdout into it.
- `merge()` (`:103-108`) is a plain `cat` of all 16 files, with no check.
- Sequitur writes through `sys.stdout.buffer` (`misc.gOpenOut("-")`). On this GPFS, st_blksize is
  8,388,608, so CPython's stdout buffer is 8 MiB, while a chunk's output is about 1.8 MB. A chunk file
  therefore stays at 0 bytes until g2p.py exits. This is why the result is all-or-nothing per chunk
  rather than partial.

### Why the finished tasks re-ran (the tooling layer)
- `submit_log.run` holds 611 submissions: 311 x `[13,14,15,16]`, 190 x `[9..16]`, 87 x `[5..16]` and
  so on. Every one of them carries `'engine': 'short'`, which was injected by `check_engine_limits`, and
  `engine_name 'local'`. `engine_info` alternates between `jpbl-s01-04` and `jpbl-s02-04`, so at least
  two manager processes on two login nodes were driving this job (observed).
- This is a known upstream bug: rwth-i6/sisyphus PR #314 "Fix task_state engine selection, avoid
  duplicate LocalEngine tasks", merged 2026-09-14, commit d9e1ede. `EngineSelector.task_state` routed by
  the raw `task.rqmt()`, which has no 'engine' key, so it asked the default (Slurm) engine, got UNKNOWN,
  and resubmitted every cycle. `LocalEngine.submit_call` appended every copy, and each copy later
  started (`localengine.py` run loop; it never checks finished.run.*).
- The local `tools/sisyphus` is at ddcd028 (2026-06-24), which does not contain d9e1ede.
  `settings.py.bak_2026-09-23` shows the stock `sisyphus.engine.EngineSelector` with
  `"short": LocalEngine(...)`.
- The current `settings.py` uses the setup's own `RoutingEngineSelector` (`gpupack_engine.py:531ff`,
  dated 2026-09-24), which routes task_state through `get_rqmt()`. That fixes the selector half. The
  LocalEngine dedupe half of #314 is still missing locally.
- The same bug also hit the g2p training: `TrainG2PModelJob.pD4nbqFLWtbi/log.run.1` shows two concurrent
  starts on jpbl-s01-04 (13:35:02 and 13:36:34) writing into one output dir, then two more starts that
  died on `FileExistsError: ... model-best`.

### The obvious wrong answers, and why they are wrong
- **Truncated g2p model (np.sometrue).** The defect is present on JUPITER too. The speech_llm env has
  numpy 2.4.6 and an unpatched `sequitur.py:431` (`if not num.sometrue(...)`), and the train log has 4 x
  ``AttributeError: `np.sometrue` was removed in the NumPy 2.0 release`` (two concurrent processes, each
  aborting the ramp-ups that ended in model-1 at about 14:01 and model-3 at about 15:19; Sequitur's bare
  `except` prints the traceback, then "iteration failed." and breaks). It lowers pron quality, and it
  applies to both sites. It cannot produce a 100%/0% pattern over contiguous chunks, and
  g2p.untranslated reports no failed word.
- **Filtered word list.** There is none: the CollectOovWordsJob code applies no filter, and the list
  holds 773,673 types.
- **Words the g2p could not handle.** No "failed to convert" line exists, and every word of chunks 1-4
  and 13-16 is covered.
- **Crash or timeout of a task.** Every task has a successful completion and finished.run.1..16 exist.
  The loss is in the merge input, not in the task.

## Q2. Does the port's T7 already count as T0?
For the code, yes. It ran on JUPITER's banked window, in JUPITER's environment.
- `port_reports/impl_lm.md:54`: "T7, the prior fit. The ported `PhoneNgramPriorJob` was compared with
  the banked prior. All arrays are array_equal with max abs diff 0.0, and `prior.json` and
  `prior.stats.txt` are byte-identical: T7 EQUAL (`t7_prior.log`)." The re-run after the nits round is
  at `impl_lm.md:87` ("6 keys array_equal ... prior.json and stats byte-identical").
- `port_reports/review_lm.md:56-71` (Q3 "is T7 real?"): it checks all 6 keys (log_bi, log_tri,
  log_uni, meta, phones, tri_counts) for dtype, shape and array_equal, finds prior.json and
  prior.stats.txt byte-identical, and notes that it used the real `phones.py`.
- The banked `work/speech_llm/sae/emc/prior/PhoneNgramPriorJob.RtzbESkOedsT/output/prior.stats.txt`
  contains `held_ppl_order3 = 9.561056344111362` and
  `corpus = .../SampleLinesJob.orN768ARKwlt/output/text.phn.gz`. If the port's stats file is
  byte-identical, it reproduced 9.561056 exactly, and it read the banked window orN768ARKwlt. So T7
  is the T0 test ("port code on JUPITER's window gives 9.561056"), and it passed.
- Related: t_text (`impl_lm.md:62-63`) shows the ported PhonemizeWithSilJob on the banked inputs,
  including the myTIGtmrUIFq lexicon, equal to the DbFgvZOGZQ8F 200k-line prefix, and the ported
  SampleLinesJob equal to orN768ARKwlt (n_in 39,630,169).
- "All 22 npz arrays identical" is a different check (`impl_fix_round.md:165-173`,
  `review_fix_round.md:81`). It covers LexiconTrieBuildJob's `lexlat_resources.npz` (word LM and trie),
  run full-scale on the banked `LexiconTrieBuildJob.rlMsnTBSZXsB` inputs, with word_lm.bin sha
  8c30483f. It is not the phone prior. It does establish the same property for the word LM and trie:
  the port reproduces |V| = 151,731 from JUPITER's window.
- Limits:
  - The scripts and logs (`/e/project1/spell/wu24/worktrees/port_checks/lm/t7_prior.py`, `.log`,
    `.nits.log`) were deleted at task end (PORT_WORK.md:22), so only the report text survives and I
    could not re-read them.
  - T7 ran on the JUPITER login node in the JUPITER speech_llm env. On i6 it would still test the i6
    runtime (python/numpy), but not the port code.

## Q3. Is the drop exactly the lines that contain a word missing from the lexicon?
Yes. This was observed with a full-corpus scan (pigz | python, read-only, same loader logic as
`_load_lexicon_word_to_phon`/`_load_g2p_lexicon`, scratch script `scan.py`):
```
lines_in=40418261 empty=1 dropped=788092 dropped_with_band_word=788091 dropped_without_band_word=0 lines_with_band_word=788091
absent_types=388780 absent_types_in_band=388780 absent_types_outside_band=0 absent_tokens=935346 total_tokens=803288729
```
The 788,092 dropped lines are 788,091 lines that each contain at least one band word (every such line
was dropped) plus 1 empty line. By code (`w2vu2/text.py:39-55`), a line is dropped iff it is empty or
has a word missing from w2p. The phone text therefore changed only through the g2p hole. Because the
band is alphabetical (non-bliss words from DITCHLIKE to RIVAW), the drop removes rare-word lines
selectively by spelling.

## What a fix has to change
- **Interpretation.** Everything downstream of myTIGtmrUIFq on JUPITER carries the hole: DbFgvZOGZQ8F,
  window orN768ARKwlt, prior RtzbESkOedsT (9.561056), and the lexlat word LM/trie/HLG (|V| 151,731,
  rlMsnTBSZXsB). This is a defect affecting experimental validity and belongs in SAE_ref.md. The Tier-A
  bar 9.561056 is not a valid port target for a clean text.
- **Confirming on i6 that the gap is only the hole.** Copy JUPITER's 14 MB myTIGtmrUIFq g2p.lexicon to
  i6, run the port's PhonemizeWithSilJob with it, then sample and fit. It must give lines_out 39,630,169
  and held ppl 9.561056 exactly. The cheaper pure-env T0 is to copy the 50 MB orN768ARKwlt text.phn.gz
  and fit it.
- **Checking i6's own lexicon.** Check it for the same failure: lexicon words should equal input types,
  with 0 "failed to convert", and every chunk covered.
- **Tooling.**
  - Bring tools/sisyphus up to d9e1ede or later (the #314 LocalEngine dedupe), and run one manager per
    work dir.
  - Make `ApplyG2PModelJob.run` write to a temp file and rename it on success (`i6_core/g2p/apply.py:82-101`).
  - Give merge a coverage assert: the number of lexicon words must equal the number of input words
    minus the untranslated words.

## Blast radius (partial list)
These are job `info` files dated 2026-07-15 or later that name myTIGtmrUIFq. The search was cut at 30
hits, so the list is not exhaustive:
- LexiconTrieBuildJob.rlMsnTBSZXsB;
- LexlatPositionalUnigramJob.F4m7uZm4hRRH;
- LexlatWordCountsJob.1ZJy5dFbOAHD and X2YnVYfqN7aV;
- NeuralPhoneLmTrainJob.Iv6P6YVPNWmB, pBozvj6c3l16 and xObXEwRpvmzd;
- at least 20 PriorGapAnalysisJob.* (all under work/speech_llm/sae/emc/).

Every job that reads DbFgvZOGZQ8F, orN768ARKwlt or RtzbESkOedsT inherits the band hole.
