# Implementation: T3, ctrl_20 step 0 with only the phone prior swapped to a JUPITER rebuild (2026-09-25)

Status: DONE. Everything is built and smoke-checked locally; nothing was submitted to the cluster.

Why: ctrl_20 fails G0.R1 on one clause only. Its step-0 prior per token is -5.637 against JUPITER's
-5.657 +-0.01. Two inputs differ from JUPITER at step 1: the phone prior and the audio. T3
(`reports/debug_prior_ppl_2026-09-25.md` section 9) swaps only the prior, which isolates the prior's
share of the gap.

## Files (all new, under `analysis/prior_t3/`; nothing under recipe/, config/, settings.py or any job dir touched)

- `common.py`: paths, the arm table, and the banked JUPITER values, each with its source.
- `part_a_prior.py`: Part A (CPU), rebuilding JUPITER's prior on i6.
- `run_part_a.sh`: sbatch submit for Part A on `cpu_modern` (2 CPUs, 16 GB, 5 h limit).
- `build_config.py`: writes one arm's `returnn.config` and `config_delta.diff`, and asserts the delta.
- `probe.sbatch`: Part B (V100 `gpu_32gb`, 16 CPUs, 64 GB, 2.5 h limit). Runs both arms in sequence on
  one GPU, then runs the read.
- `read.py`: the T3 read.

## Part A: what it does

It rebuilds JUPITER's prior by emulating JUPITER's g2p defect exactly. Each step calls the job's own
code.

1. **The g2p lexicon.** It applies `ApplyG2PModelJob.merge` (`cat g2p.lexicon.1..16`) to the i6 chunk
   files of `ApplyG2PModelJob.3eJqzOadjOqw/work/`, with chunks 5-12 empty. Then it applies
   `ApplyG2PModelJob.filter`, which keeps the lines with 4 tab fields.
   - Asserted: the result equals the i6 `output/g2p.lexicon` minus every word of chunks 5-12.
   - Measured on the real files:
     - Chunks 5-12 hold 388,780 words, running from DITCHLIKE to RIVAW.
     - 388,779 of those words are in the i6 output. The missing one is `HHH`, whose g2p pronunciation
       is empty, so the filter drops it on both clusters. This resolves the open one-word band
       difference.
     - The emulated lexicon has 384,893 entries.
     - The bliss + g2p union has 584,893 words. JUPITER's banked union is also 584,893, so this
       matches exactly.
2. **Word-text scan (no RNG).** It counts the dropped lines by reason and compares them with JUPITER's
   full-corpus scan (empty lines 1, dropped 788,092, lines with a band word 788,091, absent types
   388,780, absent tokens 935,346).
3. **Phonemisation.** It calls `PhonemizeWithSilJob.run` itself on a stand-in `self` that carries the
   parameters from `NpoY1pGJWNUJ/info`, with only `g2p_lexicon` replaced. There are two code-path checks,
   and the script aborts if either fails:
   - (a) The emulation's output up to its first extra drop is byte-identical to the i6 job's output.
   - (b) Control: the same `run` with the unmodified i6 lexicon on the first 1,000,000 word lines
     reproduces the i6 output prefix byte for byte.
4. **Sampling.** It calls `SampleLinesJob.run` itself, with n_out and seed taken from
   `CrPgeKXsOosb/info` (1,010,000 and 0).
5. **Prior fit.** It calls `PhoneNgramPriorJob.run` itself, with the parameters from
   `qJxXHgXLe31S/info` (1,000,000 counted, 10,000 held). This writes `prior/prior.npz`, `prior.stats.txt`
   and `prior.json`.
6. **Checks against JUPITER.** Each of the following is printed as EXACT or DIFF and never forced:
   - lines_out against 39,630,169;
   - tokens counted against 81,559,944;
   - held raw tokens against 808,146;
   - ppl3 against 9.561056344111362;
   - tokens against 3,232,620,004 and SIL tokens against 448,460,735.

Expected inexactness: the pronunciations of the g2p words outside chunks 5-12 come from the i6 Sequitur
model, which was trained under the numpy-2 defect (debug report section 4). If they differ from
JUPITER's, the phone tokens and the ppl will differ. The lines kept and the SIL tokens will not, because
the SIL draws depend on word counts only. The summary shows which quantities differ.

Output goes to `/work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/prior_t3/partA/`:
`partA_summary.{txt,json}`, `prior/prior.npz`, `slurm-*.out`, and the intermediate files. The script
refuses to run if a prior already exists there.

## Part B: what it does

There are two arms. `i_i6prior` is the control and uses ctrl_20's prior. `ii_partA` uses the Part A prior.

- **Config.** Each arm's config is ctrl_20's `GiT88bxzoZbZ/output/returnn.config` with two changes:
  - In both arms, `model` becomes `"./models/epoch"`, relative to the arm dir. The models dir starts
    empty, so RETURNN starts at sub-epoch 1 from the config's own flat init (`FlatRecognizerInitJob`),
    like ctrl_20.
  - In arm ii only, `prior_npz_path` points at the Part A prior.
- **Why only one key changes.** `prior_npz_path` is the only key that references the prior. The build
  asserts that the i6 prior path occurs once, and that the only lines mentioning a prior are
  `prior_weight` (a scalar) and `prior_npz_path`.
- **Run.** The environment is ctrl_20's (the `env -i` worker environment of the k2lat probe). RETURNN
  runs in its own process group. As soon as the `ep 1 train, step 0,` line appears in `returnn.log`, the
  group is killed, so no checkpoint is written.
- **Output** goes to `/work/asr4/hwu/sae_i6_probes/p0_prior_t3_2026-09-25/{i_i6prior,ii_partA}/` and
  `read.txt`.

## Commands (run from the setup dir S)

```
analysis/prior_t3/run_part_a.sh          # A: prints the Slurm job id <A>
mkdir -p /work/asr4/hwu/sae_i6_probes/p0_prior_t3_2026-09-25
sbatch --dependency=afterok:<A> analysis/prior_t3/probe.sbatch     # B; aborts if the Part A prior is missing
/work/asr4/hwu/conda/envs/sae/bin/python analysis/prior_t3/read.py  # re-read at any time
```

Expected runtimes:
- Part A takes about 1 h. From the smoke timings, the scan runs at about 1 s per 300k lines, and
  `PhonemizeWithSilJob.run` at about 21 s per 300k lines, so about 47 min for 40.4 M lines. The i6 job
  itself took 1:29.
- Part B takes about 10-15 min per arm on a node with a warm HDF cache (ctrl_20 reached step 0 5 min
  after start). On a cold node it takes about 30 min for the first arm, because `cf` copies 32 GB of
  features (k2lat's restart needed 21 min before step 0). The per-arm timeout is 60 min.

## How the read decides (`read.py`, last line)

- **CONTROL:** arm i's step-0 line must equal ctrl_20's `log.run.1:670` in every field except wall-clock.
  The reference values are l_tau -0.352, prior per token -5.637 and expected tokens 63.854.
- **BATCH:** ctrl_20 and both arms must show the same num_seqs (128), max frames (330) and
  `ep 1 train num_seqs` (7066).
- **CONFIG:** diff(arm i, arm ii) must be exactly the `prior_npz_path` line, and diff(ctrl_20, arm i)
  exactly the `model` line.
- **Verdict:**
  - If all three pass, the read prints `T3 VALID` with three numbers: the prior share (arm ii - arm i),
    the gap (arm i - JUPITER's -5.657), and the residual (arm ii - JUPITER). It also marks arm ii as
    WITHIN or OUTSIDE JUPITER's step-1 values at the G0.R1 audio-label tolerances (l_tau +-0.005,
    prior +-0.01, tokens +-3 %).
  - Otherwise it prints `T3 INVALID`, or `NOT_READY` if a line is missing.
- **Reading the result:**
  - If arm ii is WITHIN and the Part A core checks are EXACT, the prior accounts for the G0.R1 miss.
  - If arm ii is OUTSIDE, the residual belongs to the audio, or to any Part A inexactness shown in the
    same read.

## Checks run (local desktop, no cluster)

- **Part A smoke** (`--smoke-lines 300000`, output in the scratchpad): rc 0. The g2p numbers above are
  measured on the full files. The prefix check passed on 27 lines, and the control passed on 299,999
  output lines, byte-identical. The scan found no dropped line without a band word. The Part A npz has
  the same keys, phones, shapes and dtypes as the i6 prior. The smoke numbers are not results.
- **build_config for both arms** (arm ii on the smoke prior, probe root redirected to the scratchpad):
  the delta assertions passed, and the diffs are as specified.
- **RETURNN CPU dry load of both arm configs:** the start epoch is 1, with no checkpoint. Each model
  (`SaeBlankfreeModelV1`) builds through `get_model`, with its prior loaded by the model's own loader.
- **read.py on fabricated arm logs** (copies of ctrl_20's lines, with arm ii's prior edited): CONFIG,
  BATCH and CONTROL all PASS, and the T3 line is formatted as intended.
- **The process-group kill pattern**, tested with a dummy process tree: no process was left.
- `bash -n` on both shell scripts: OK.

These checks exercise loading and the plumbing only. No step-0 value and no full Part A number has
been produced.

## Open points

- **Exact CONTROL equality.** CONTROL demands exact equality on a V100, while ctrl_20 ran on an L40S.
  k2lat on a V100 gave an identical line, but if a field differs, the read prints FAIL with the
  mismatches. Whether to accept that difference under the registered cross-hardware tolerance
  (+-0.002 / +-0.005 / +-1.0) is the orchestrator's decision.
- **V100 memory at step 0.** The step-0 batch is 128 x 330 frames, well under the 88,000-frame batches
  that peaked at 34.3 GiB on the L40S. The k2lat run already ran this step 0 on a V100. The peak was not
  measured here.
