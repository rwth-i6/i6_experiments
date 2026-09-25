# Review: T3 (ctrl_20 step 0 with only the phone prior swapped), before launch (2026-09-25)

Verdict: PASS_WITH_FIXES. The compute is sound as built. Part A emulates JUPITER's g2p defect exactly and
only that. The Part B arm configs differ from ctrl_20 only in `model` (probe dir) and, in arm ii,
`prior_npz_path`. Resources and paths are right. Both jobs can be launched with the commands at the end.
`read.py` does NOT implement the registered T3 reading (`SAE_i6_P0.md:437-442`), so its verdict line must be
fixed before `read.txt` is used as the T3 result. `read.py` only reads, and it is re-runnable after the fix
(the fix needs no compute).

Inputs reviewed: `reports/impl_p0_prior_t3_2026-09-25.md`; `analysis/prior_t3/{common.py, part_a_prior.py,
run_part_a.sh, build_config.py, probe.sbatch, read.py}`; ctrl_20 `ReturnnTrainingJob.GiT88bxzoZbZ` (config,
`log.run.1:670`, `work/rnn.sh`); the i6 jobs `ApplyG2PModelJob.3eJqzOadjOqw`, `PhonemizeWithSilJob.NpoY1pGJWNUJ`,
`SampleLinesJob.CrPgeKXsOosb`, `PhoneNgramPriorJob.qJxXHgXLe31S`; the package code `lm/{phone_text,phone_prior,
lexicon}.py`, `model/{emc_model,blankfree_model,agg,train_step,prior}.py`, `training/config.py:316`, and
`i6_core/g2p/apply.py`; the JUPITER sources cited in `common.py`; `SAE_i6_P0.md:95,115-122,430-443`.

## Findings

1. **`analysis/prior_t3/read.py:154-172` (with `common.py:86-87`): the T3 outcome is not the registered
   one.** The registered reading (`SAE_i6_P0.md:439-442`) has three outcomes:
   - arm ii within -5.657 +-0.005: the miss is owed to the prior;
   - |share| <= 0.005: the prior does not explain the miss;
   - anything between: a partial share.

   `read.py` instead classifies arm ii as WITHIN/OUTSIDE at `STEP1_TOL`, which for prior per token is +-0.01
   (the G0.R1 audio-label tolerance). It never tests |share| <= 0.005 and never prints "partial". It also adds
   WITHIN/OUTSIDE clauses for l_tau (+-0.005) and expected tokens (+-3 %) that the reading does not have. The
   implementation report's rule ("If arm ii is WITHIN ... the prior accounts for the G0.R1 miss") inherits the
   +-0.01.
   - Concrete failure: arm i -5.637 and arm ii -5.650 (share -0.013) print "prior WITHIN", which the report's
     rule reads as "the prior accounts for the miss". The registered outcome is a partial share, because
     |-5.650 + 5.657| = 0.007 > 0.005 and |share| = 0.013 > 0.005. Every arm ii value in [-5.667, -5.662) or
     (-5.652, -5.647] gets the wrong outcome.
   - Fix: classify on prior per token only, with +-0.005 for "owed to the prior", |share| <= 0.005 for "not
     explained", and otherwise "partial".

2. **`read.py:137-139`: CONTROL demands exact equality on every non-wall-clock field.** That is about 20
   fields, including `blankfree_rate_fd_check 7.701e-06` and both agg KLs. The registered control
   (`SAE_i6_P0.md:437-438`) is l_tau +-0.002, prior per token +-0.005 and expected tokens +-1.0 against
   -0.352 / -5.637 / 63.854, on the same batch.
   - Concrete failure: a V100 line that differs from ctrl_20's L40S line in one printed digit, but stays
     inside those tolerances, prints "T3 INVALID". That voids a run the registered reading counts as valid.
     `SAE_i6_P0.md:115-117` itself names a possible source: TF32 cuDNN on the L40S, and none on the V100.
   - The error is conservative: exact equality implies the registered pass, so there is no false VALID.
   - The likelihood is low: k2lat's ep-1 step 0 on a V100 matched ctrl_20 in every field
     (`reports/extract_step1_paired_2026-09-25.md:28`).
   - Fix: apply the three registered tolerances plus the BATCH identity. Print any other field that differs,
     but do not gate on it.

3. **`read.py:171` (and 114-125): the verdict line does not carry the Part A exactness label.**
   `SAE_i6_P0.md:433-434` requires the reading to say "JUPITER-like" unless lines_out, tokens_counted,
   held_raw_tokens and ppl3 are all EXACT. `read.py` prints EXACT/DIFF per quantity, but the "T3 VALID ..."
   line is identical in both cases. It is still printed when `partA_summary.json` is missing.
   - Concrete failure: the expected case is that Part A is not exact, because the i6 Sequitur pronunciations
     differ (debug report section 4). That case yields a verdict line indistinguishable from the result of an
     exact JUPITER prior.
   - Fix: carry the `PART A CORE` label (ALL EXACT, or "JUPITER-like: <quantities>") on the T3 line, and
     print NOT_READY when the summary is missing.

4. **`analysis/prior_t3/run_part_a.sh` has mode `-rw-r--r--`.** The documented launch
   `analysis/prior_t3/run_part_a.sh` (impl report line 88) therefore fails with "Permission denied". Launch it
   with `bash`, as in the commands below.

## Check 1: Part A emulates the defect exactly and only that. SOUND

- **The lexicon (verified on the files).** Chunk line counts are words.NN == g2p.lexicon.N for all 16 chunks.
  - Chunks 5-12 hold 49,432 + 48,947 + 47,883 + 49,483 + 49,138 + 49,470 + 47,805 + 46,622 = **388,780**
    words, running from DITCHLIKE (first line of chunk 5) to RIVAW (last line of chunk 12). These per-chunk
    counts are identical to JUPITER's re-split (`exp_logs/SAE/reports/debug_jupiter_prior_gap_2026-09-25.md:54-58`).
  - The only non-4-field line in any chunk is chunk 7, `HHH\t0\t0.785715\t`, an empty pronunciation. The
    filter drops it (3 fields after strip), so 388,779 of the removed words are in the i6 output. The claim
    is verified.
  - The emulated lexicon has 191,122 (chunks 1-4) + 193,771 (chunks 13-16) = 384,893 entries, which equals
    JUPITER's lexicon line count. Bliss 200,000 plus g2p gives a union of 584,893, equal to JUPITER's
    (`review_prior_gap_2026-09-20.md:40`).
  - `part_a_prior.py:127-162` builds the emulation as i6_core's merge (cat 1..16 with 5-12 omitted) plus its
    filter (4 tab fields), identical to `i6_core/g2p/apply.py:103-126`. It asserts that the result equals the
    i6 `output/g2p.lexicon` minus the chunk 5-12 words, with the order preserved.
- **The same code and parameters.** The script calls the package's own `PhonemizeWithSilJob.run`,
  `SampleLinesJob.run` and `PhoneNgramPriorJob.run` on stand-in objects whose fields are read from the i6
  `info` files: text, bliss, sil_prob 0.5, surround True, seed 0, max_lines None; n_out 1,010,000, seed 0;
  n_count 1,000,000, n_held 10,000.
  - It asserts that the i6 jobs are chained: g2p -> phonemize -> sample -> prior.
  - The trigram order, Witten-Bell smoothing and the held stride of 101 are fixed in code, not parameters, so
    no hidden parameter can differ.
  - Only `g2p_lexicon` is replaced.
- **Code-path checks.**
  - (a) The emulation's output is byte-identical to the i6 output up to the first emulation-only drop (the
    RNG streams coincide until then). The scan logic `:219-222` computes this correctly.
  - (b) A control on the first 1,000,000 lines with the i6 lexicon reproduces the i6 prefix.
  - The script aborts on either failure (`:351-352`, `:366-367`).
- **JUPITER references (`common.py:33-54`), each traced to its frozen source.**
  - lines_out 39,630,169, tokens 3,232,620,004 and SIL 448,460,735: `extract_sil_rate_full_2026-09-21.md:47-51`.
  - tokens_counted 81,559,944: `review_neural_phone_lm_2026-09-20.md:105`.
  - held raw tokens 808,146 (raw, including `<SIL>`, i % 101 of the first 1,010,000):
    `review_phone_lm_v2_2026-09-20.md:17`. `held_raw_tokens()` at `:253-260` matches `count_ngrams`.
  - ppl3 9.561056344111362: `debug_jupiter_prior_gap_2026-09-25.md:135`.
  - The scan values: ibid. :158-159.
  - The core gate at `:438-443` is exactly the four registered quantities; ppl is compared at 1e-9 relative.
- **What remains different is disclosed, not a defect.** The g2p pronunciations of chunks 1-4 and 13-16 come
  from the i6 Sequitur model (numpy-2 defect). This can move tokens and ppl but not the lines kept, the SIL
  draws or the sampled line indices, because the sample indices depend on n_in only. Hence finding 3.
- **The run itself.** Imports work from a cwd without `settings.py` (tested; sisyphus only warns). The memory
  is small next to 16 GB. Runtime is about 2 h against a 5 h limit, judged from the i6 phonemize time
  (1:29:34). The script refuses to overwrite an existing prior.

## Check 2: Part B arm configs and first batch. SOUND

- **The delta.** `build_config.py` replaces exactly two lines, `model` -> `"./models/epoch"` and, in arm ii
  only, `prior_npz_path`. It asserts the resulting unified diff. `read.py` re-checks both diffs.
  - Nothing else changes: not num_epochs, not the seeds, not the datasets.
  - Stopping is external: a process-group kill after the step-0 line appears.
- **No other key carries prior or text statistics.**
  - `rate_rho_hz 9.6619373279` is a literal (`training/config.py:316`) equal to JUPITER's banked value, so
    both arms already use JUPITER's rho.
  - The agg unigram and bigram targets are derived from the same npz (`emc_model.py:395-412` ->
    `agg.text_target_counts`), so the swap moves them too, consistent with JUPITER.
  - `eta_table_path` is audio-side and stays i6, by T3's design.
  - There is no dev prior and no other default path in `EmcModel.__init__`, and `PhoneNgramPrior.load` has no
    cache.
  - Result: T3 swaps the whole text side, not part of it.
- **Flat init and first batch.** The `models/` dir is asserted empty, and there is no
  `load`/`preload_from_files`/`import_model_train_epoch1`. The flat init comes from the config's
  `recognizer_checkpoint_path`. The seeds and `laplace:.1000` ordering are unchanged.
  - The model code on ctrl_20's path (model/, lm/phone_*, training/config.py) has had no commit since
    ctrl_20 started (2026-09-24 23:53), and the working tree is clean there.
  - The RETURNN root is the same `CloneGitRepositoryJob.KQ3NuCaDE6QH`.
  - The environment matches ctrl_20's log: CUDA_VISIBLE_DEVICES, OMP/MKL 16, PYTORCH_KERNEL_CACHE_PATH.
  - The arms run in sequence on one GPU, so the share carries no hardware term.

## Check 3: Resources. SOUND

- **Part A:** `cpu_modern`, `--account=hlt`, 2 CPUs, 16G, 5 h. Slurm output and all files go to
  `/work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/prior_t3/partA` (shared). It uses
  the sae python with PATH first, and no apptainer.
- **Part B:** `-A hlt -p gpu_32gb --gres=gpu:1`, 16 CPUs, 64G, 2:30. That is enough for 2 x 60 min arm
  timeouts plus overhead. `--chdir` and `-o` point under `/work/asr4/hwu/sae_i6_probes/p0_prior_t3_2026-09-25`,
  which must be created before submission (see the commands).
  - It uses the `env -i` sae environment, as the k2lat probe does, with no apptainer.
  - It aborts if the Part A prior is missing, and afterok is in the launch.
  - If Part A fails, B stays pending with DependencyNeverSatisfied and must be cancelled.
- The setsid process-group kill is correct in a non-interactive batch shell: the background child is not a
  group leader, so setsid does not fork and `$!` is RETURNN's pid.

## Check 4: read.py against the registered reading. FAIL, see findings 1-3

- The control verdict is gated: no share verdict is printed unless CONFIG, BATCH and CONTROL all pass;
  otherwise it prints "T3 INVALID/NOT_READY".
- The share (arm ii - arm i) and the gap are computed correctly.
- The thresholds and outcomes are wrong (finding 1), and so are the control tolerance (finding 2) and the
  Part A label (finding 3).

## Check 5: Nothing under recipe/, config/ or a job dir is written. SOUND

- Part A writes only into `--out-dir`. The stand-in `out_*` paths all point there, and the job dirs are only
  opened for reading.
- Part B writes into the arm dirs (config, `models/`, `learning_rates`, `returnn.log`, `rnn.out`). ctrl_20 is
  only read.
- The only incidental writes are Python bytecode caches under `recipe/**/__pycache__`, which any manager run
  produces too.

## Launch commands (from the setup dir; compute is cleared now, and read.py must be fixed before read.txt is used)

```
cd /u/hwu/setups/librispeech-960/2026-09-24-unsupervised
bash analysis/prior_t3/run_part_a.sh                       # -> "Submitted batch job <A>"
mkdir -p /work/asr4/hwu/sae_i6_probes/p0_prior_t3_2026-09-25
sbatch --dependency=afterok:<A> analysis/prior_t3/probe.sbatch
```
After the read.py fix: `/work/asr4/hwu/conda/envs/sae/bin/python analysis/prior_t3/read.py` (re-read at any time).
