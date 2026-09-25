# Review: LexiconTrieBuildJob banked identity checks made report-only, and the rerun on the live manager (2026-09-25)

Verdict: APPROVE_WITH_CONCERNS. The code change is sound, hash-neutral and safe for the live processes.
The concern is the gates. Clearing the task starts an automatic chain (trie, then HLG, then k2lat) with
no further check point. That chain ends in a G0.R0 HLG-size read that will FAIL by construction.
Any amendment must be written into the phase file BEFORE the task is cleared.

Inputs reviewed:
- `git -C recipe/i6_experiments diff` at HEAD be7fbe447. Only `lm/word_lm.py` is modified in the package,
  and the package has no commits since 51f4def2d (the live manager's code).
- Implementer report `reports/impl_trie_identity_reportonly_2026-09-25.md`.
- The failed job dir `LexiconTrieBuildJob.W0e4no47Crfu`.
- `lm/hlg.py`, `model/lexlat_k2*.py`, `training/arms.py`, `settings.py`.
- `SAE_i6_P0.md` (working tree, which another session is editing).
- Frozen JUPITER logs.

## 1. The diff is exactly the stated delta

- `lm/word_lm.py:307-309`: a new internal assert, `len(in_vocab) == text["n_types"]`.
- `lm/word_lm.py:314-327`: the three banked comparisons are recorded and printed, not asserted.
- `lm/word_lm.py:382`: the new build.json key `identity_vs_banked`.
- `lm/word_lm.py:413-415`: the summary lines.
- Everything else in the diff is comments and docstrings.
- Unchanged: `__init__`, the defaults, `__sis_version__`, the outputs, and every line that computes the
  trie, CSR, escape and derangement. The derangement asserts (`:335-338`) are kept verbatim.

The new internal assert is correct, and it holds on the real inputs. I checked the counts independently:
- `window.words.txt` of `WordWindowReplayJob.YLcmHAGAbZ1j` has 182,215 distinct tokens (counted with
  `tr | sort -u`).
- The ARPA header has `ngram 1=182218`, which is the types plus `<s>`, `</s>` and `<unk>`.
- The failed run's traceback prints `len(in_vocab) = 182215`.

The replay keeps only lines whose every word is in the lexicon, so the in-vocabulary set contains every
window type. Because the counts are equal, the two sets are identical. So `tok[w]` at `:358` cannot
raise a KeyError.

The new code is JSON-safe: plain ints and bools, and string keys under `sort_keys`. When `matched` is
empty, the summary line handles it.

Readers of the trie's `build.json` and `summary.txt`: none.
- The only consumer of this job is `trie.out_resources` (`lm/hlg.py:1092`, `1107`, `1122`, `1145`).
- Every `build.json` reader in the package reads an HLG job's `build.json`, never the trie's:
  `model/lexlat_k2_train.py:320-340` and `lm/hlg.py:831`.
- No config and no analysis module registers or reads the trie's `out_stats` or `out_summary`.
- `bigram_convention_matched` is read by nothing.

## 2. Hash neutrality (verified independently)

I ran a scratch script that loads `config/sae_i6_p0.py` twice through sisyphus's own loader:
- once with the working-tree `word_lm.py`;
- once with the HEAD `word_lm.py`, injected under the same module name.

Both runs give 164 sorted `_sis_id()`s, and the two lists are IDENTICAL. The list includes:
- `LexiconTrieBuildJob.W0e4no47Crfu`
- `LexlatHLGBuildJob.avjHv1Xvjyqd`
- `ReturnnTrainingJob.GiT88bxzoZbZ` and `llSFybyKXkbL`

The id lists are in the scratchpad (`ids_cur.txt`, `ids_head.txt`); nothing under `work/` was touched.

## 3. Live-process safety

No RETURNN training imports `lm/word_lm.py`:
- The training config imports only `model.blankfree_model`, `model.train_step` and `model.param_groups`
  (`GiT88bxzoZbZ/output/returnn.config`).
- `model/` imports only `..phones` from the rest of the package.
- `lm/__init__.py` and the package `__init__.py` import nothing.
- `lm/hlg.py` imports `.word_lm` only lazily, inside `get_hlg`, which runs at graph time.

So ctrl_20, ctrl_20_rc, ctrl_20_s1 and k2lat cannot see the edit.

The live manager keeps its already-loaded job objects, whose hash is unchanged. The resubmitted worker
unpickles `job.save` and imports the current file, so it runs the new `run()`. Every attribute that
`run()` reads was set in `__init__` and is in the pickle.

The clear procedure (move `log.run.1` aside, remove `error.run.1` and `submit_log.run`) takes the task to
RUNNABLE (`sisyphus/task.py:306-310`, `389-419`). Do it in one command. If `error.run.1` is removed
while `log.run.1` still exists, the task reads INTERRUPTED_NOT_RESUMABLE until the next poll.

## 4. Downstream capacity with the i6 vocabulary

N-gram counts:

| | 1-grams | 2-grams | 3-grams |
|---|---|---|---|
| JUPITER (banked) | 151,734 | 3,393,577 | 10,419,405 |
| i6 (ARPA header) | 182,218 (+20.1 %) | 3,510,073 (+3.4 %) | 10,623,445 (+2.0 %) |

- The size of G follows the n-gram states, not the vocabulary. G states grow from 3,545,313 to about
  3,692,293 (+4.1 %), and G arcs grow about 3-5 %.
- HLG states are roughly G states times the pronunciation length. So the i6 word-boundary HLG is
  predicted at about 24.9-25.1 M states and 101-104 M arcs (banked 23.9 M / 98.6 M).

HLG build (`lm/hlg.py:1042-1047`):
- Request: 200 GB, 16 CPUs, 6 h, ladder (0.0, 0.5, 2.0) with 4 h per attempt.
- Routing: 200 > `CPU_MODERN_MAX_MEM` = 180 (`settings.py:31`, `398-399`) sends it to gpu_32gb with no
  GPU. That partition is cn-32/33 with 1.5 TB and 96 CPUs each, and its QoS shows no per-user cap.
- The banked all-states build peaked at 26.0 GiB in 173.5 s. Linear scaling predicts about 27-28 GiB,
  a few minutes, on rung 0.0.
- So theta = 0.0, which matches `expected_build` (theta 0.0), and `check_expected_build` will not trip.
- No failure is expected.

k2lat GPU memory, whether the training still fits a 46 GB L40S with the larger graph:
- The HLG resident on the GPU is about 20-24 B per arc, about 2.3 GiB, so the vocabulary adds about
  0.1 GiB.
- The pruned lattice is set by max_active 3000 and the beams. Only the fan-out of the unigram/back-off
  state grows 20 %, so the leg grows by a few percent at most. An upper bound is +20 % of the leg's
  lattice memory, about 2 GiB or less.
- The i6 bed alone reads 34.3 GiB of 45 GiB. The banked k2lat whole step at rung 3000 was 31.77 GiB,
  including the bed. The in-run leg peak was 17.4-21.9 GiB reserved.

Estimate: the i6 k2lat step is about 35-38 GiB. It likely fits, and the vocabulary is not the deciding
factor. This is NOT measured on i6. Whether the k2 leg fits on the L40S at all is a question that
predates this change. It would show at the k2 on-set, sub-epoch 8 (`training/arms.py:53`, `K2_ONSET = 8`),
about 7 h into the run. The cost would be those GPU-hours plus a resubmit from the last checkpoint. A
batch-shape fix would change the hash and mean a full rerun.

k2 int32 window sum: low risk.
- The largest out-degree grows about 20 %.
- On JUPITER, chunk 16 was verified at rung 10000, whose active set contains rung 3000's.
- `LEXLAT_K2_CHUNK_SEQS` can be lowered without a hash change.

## 5. Findings for this launch

- `SAE_i6_P0.md:89-90` (G0.R0, Tier A: HLG in [23.8, 24.0] M states and [98.1, 99.1] M arcs). The i6 graph
  is predicted at about 24.9-25.1 M states and 101-104 M arcs, so this clause FAILS by construction
  minutes after the trie finishes. A Tier-A FAIL mandates the debugger. Relabelling the clause after the
  number exists would be a gate rewritten to fit its result. Register the amendment (for example, record
  it as a comparison under the i6-text deviation) before clearing the task, because there is no later
  check point.
- `SAE_i6_P0.md:105-108` (G0.R2: PER 0.818615 +-0.03, the paired window, expected words [24, 38], escape
  share <= 0.01). These were registered on the banked 151,731-word graph. k2lat now trains on a
  different lexicon and G, so a miss cannot be attributed to the port. This is the same confound as the
  prior clause. Record the confound before the gate is read.
- Provenance: the rerun executes uncommitted code (HEAD be7fbe447 lacks the diff). The project rule is to
  commit it with the round that used it.

No code findings.
