# Implementation: G0.K2M probes at k2 chunk 2 and chunk 1, and the matching live-change variants for J (2026-09-25)

Role: implementer. S = `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised`;
J = `S/work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl` (Slurm 4359832).

## Status: DONE_WITH_CONCERNS

Both parts are ready. Nothing was launched, installed, cancelled or committed. J was only read: its
`output/returnn.config` sha256 is still `dff329b6...cdc`, and `output/models` still holds epoch.001, epoch.004 and epoch.004.opt.
The recipe package has no git changes. `config/sae_i6_p0.py` is unchanged (sha `0f609369...`). No existing
chunk-16 or chunk-8 file was modified: mtimes and hashes were checked after the work. The dirs of probe A and probe B hold no file newer than 17:20.
The concerns are at the end: wall time at chunk 1, and seeding skew between the two probes.

## Part 1: probes at chunk 2 and chunk 1

**Files**
- `S/analysis/p0_k2lat_v100_probe/read.py` was edited (+107/-5; new sha `ce0736bb...`). This is the only probe script touched.
  `build_config.py`, `common.py`, `dry_check.py`, `seed.py` and `probe.sbatch` are unchanged (hashes checked).
- New dir `/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs2` holds the pre-flight outputs.
- New dir `/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs1` holds the pre-flight outputs.

**The two probes differ from B only in chunk size and directory.** They use the same `probe.sbatch`.
`PROBE_DIR` and `PROBE_CHUNK_SEQS` are set at submit time, and the job exports `LEXLAT_K2_CHUNK_SEQS` to RETURNN. Both configs differ from B's `returnn.config` only in the `model` path (diff checked).

**Launch commands** (both were validated with `sbatch --test-only`; nothing was submitted):
```
PROBE_DIR=/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs2 PROBE_CHUNK_SEQS=2 sbatch -t 3:00:00 -J sae_i6_p0_k2lat_v100_probe_cs2 --chdir=/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs2 -o /work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs2/slurm-%j.out /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/p0_k2lat_v100_probe/probe.sbatch
PROBE_DIR=/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs1 PROBE_CHUNK_SEQS=1 sbatch -t 3:00:00 -J sae_i6_p0_k2lat_v100_probe_cs1 --chdir=/work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs1 -o /work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25_cs1/slurm-%j.out /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/p0_k2lat_v100_probe/probe.sbatch
```
To read a probe, run `/work/asr4/hwu/conda/envs/sae/bin/python S/analysis/p0_k2lat_v100_probe/read.py <dir>`. The job also writes `<dir>/read.txt`.

**Seeding from epoch 4 (the check the brief asked for).** I ran the sbatch's own pre-flight on the desktop for both dirs: `probe.sbatch` lines 1-56 verbatim, that is, build_config, `seed --force` and the dry check.
- The seed is J epoch 4 (checkpoint internal epoch 4, step 228), presented as epoch 7. The sha256 values are `f649c354...` (model) and `21f39662...` (opt), the same in both dirs.
- The learning-rate file has J's entries for epochs 1-4, placeholder train/dev scores for 5-7, and the lr only for 8-20 (1e-4 each). The two dirs' lr files are identical.
- `DRY CHECK OK` in both dirs, 22 of 22 checks:
  - get_epoch_model is 7; the train start epoch is 8 and the final epoch is 8;
  - lr(8) is 1e-4 (ConstantLearningRate), and `_check_missing_eval` finds no gaps;
  - the state_dict loads strictly;
  - temperature is 2.0, anchor weight 0.0 and prior weight 1.0; lam is 1/3 and the leg is active; over sub-epochs 7..11 lam runs 0, 1/3, 2/3, 1, 1;
  - the train data is partition [3] of 0..3;
  - train_step is `rt_chunked_backward.train_step`, and `torch_log_memory_usage` is True.
- The per-chunk runtime reports chunk_seqs = 2 in the `_cs2` dir and 1 in the `_cs1` dir.
- These values are identical to B's epoch-3 pre-flight. Each job re-seeds from J's latest checkpoint at its own start. `seed.py` refuses once J has `epoch.008.pt`.

**read.py changes** (from review SHOULD 2 and 3). Everything is additive; the existing output is unchanged.
1. The stability print lines are now read from `returnn.log` and from `slurm-*.out`. A new section prints the
   `rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = N)` lines from `slurm-*.out`, next to the
   intended chunk size parsed from `<dir>/dry_check.txt`.
2. The last line is one verdict for G0.K2M. Its clauses are copied from `SAE_i6_P0.md`, and GATE_GIB = 28.0.
   - Peak = max(per-step max of pre_peak_allocated and mem_usage, epoch-end alloc peak), compared with 28 GiB.
   - **FAIL** if any of these holds:
     - peak > 28 GiB;
     - an OOM line or an int32 line;
     - `lexlat_k2_ABORT.json` exists;
     - the stability read FAILED, is not finite, or covers 0 utterances;
     - the sub-epoch did not complete because the rnn exit code was non-zero or the Slurm time limit was hit.
   - **NOT_DECIDED** if nothing is violated but a clause is still missing. That covers: the run is unfinished or cancelled; no epoch-end line; no stability line; other FAILED/ABORT text lines are present; or the installed chunk size is missing or differs from the intended one.
   - **PASS** otherwise.
- The script only reads. It writes nothing into A's or B's dirs; the only output is stdout.

Tests of read.py (outputs in the scratchpad):
| input | verdict |
|---|---|
| B (real run, chunk 8) | FAIL: rnn exit 139, 2 OOM lines. It now shows the stability line (16 of 16) and `installed (chunk_seqs = 8)`. The output is B's old `read.txt` plus the new lines only |
| A (real run, cancelled) | NOT_DECIDED: job cancelled, no step line; chunk 16 installed, 16 intended |
| EexT85vdfx25's real rt log, relabelled to ep 8, with its `log.run.1` as the Slurm output | PASS: peak 27.256 GiB (step 6), epoch-end 13.2, stability 0.0826 over 16 of 16, chunk 4 = 4 |
| the same with the stability line replaced by a FAILED line | FAIL |
| the same cut at 15 steps, intended chunk 2 | NOT_DECIDED |
| the cut run plus a Slurm time-limit line | FAIL |

## Part 2: live-change variants for J at chunk 2 and chunk 1

The route is the same as cs8's. The kwarg `lexlat_k2_chunk_seqs` is bound in the unhashed epilog:
`get_model = __import__("functools").partial(get_model, lexlat_k2_chunk_seqs=N)`. All files are new, in `S/analysis_out/`:

| file (N = 2, 1) | delta |
|---|---|
| `k2lat_perchunk_csN_sae_i6_p0.patch` | NOT applied. It is the cs8 patch with 3 lines changed: `K2LAT_CHUNK_SEQS = N`, the docstring count, and the evidence path pointing to this report. `patch --dry-run -p1 -d S` is clean |
| `k2lat_perchunk_returnn_csN.config` | 12345 bytes. The cs2 sha is `3a53ad6c0ad127c66cbed724a37063e439ea7de287b523e24b92f2ca0a3792a2`; the cs1 sha is `5d9848cfb0af95978e630c1157267a3b12d1b7ad1f7997d94acf3650fbbba8a0`. The file equals the cs8 config with `=8)` changed to `=N)` on the partial line |
| `k2lat_perchunk_returnn_csN.config.orig` | a copy of J's current config (`dff329b6...cdc`) |
| `k2lat_perchunk_returnn_csN.config.diff`, `.vs_cs16.diff` | the diff against the original is +8 lines; the diff against chunk 16 is +3 lines |
| `k2lat_perchunk_returnn_csN_install.sh` | the cs8 script with only the file names, NEW_SHA and comments changed |
| `k2lat_perchunk_p0_jobids_after_csN_2026-09-25.txt` | 164 ids |
| `k2lat_perchunk_csN_test_k2lat.py` | a scratch copy of `tests/test_rt_chunked_backward_k2lat.py` with 5 lines changed, the same pattern as cs8 |
| `k2lat_perchunk_csN_checks_2026-09-25.log`, `k2lat_perchunk_test_k2lat_csN_2026-09-25.log` | check outputs |

The check scripts `k2lat_perchunk_cs8_dryrun_driver.py` and `k2lat_perchunk_cs8_path_check.py` were reused unchanged. Both are generic.

**Job ids.** The cs8 driver ran through `sisyphus/sis console --skip_config -s` on scratch copies of the config: the current one and the ones patched to cs2 and cs1. All three give 164 ids, byte-identical to `k2lat_perchunk_p0_jobids_after_2026-09-25.txt`, and `jcKXbLMDk4hl` is present in each. The control's k2lat config is byte-identical to the chunk-16 config. The five other training configs are byte-identical between the control and each patched build. The epilog length is 408 and `python_epilog_hash` is `''`.

**Chunk size in the model.** I ran `path_check` with `env -i`, the sae python, the pinned RETURNN, and `CUDA_VISIBLE_DEVICES=` (no GPU):
| config | env | kwarg | spec.chunk_seqs / install print | calls for 128 seqs |
|---|---|---|---|---|
| cs2 | unset | 2 | 2 / `installed (chunk_seqs = 2)` | 64 |
| cs2 | `LEXLAT_K2_CHUNK_SEQS=16` | 2 | 2 (the kwarg wins) | 64 |
| chunk 16 | `LEXLAT_K2_CHUNK_SEQS=2` (the probe's route) | absent | 2 | 64 |
| cs1 | unset | 1 | 1 / `installed (chunk_seqs = 1)` | 128 |
| cs1 | `LEXLAT_K2_CHUNK_SEQS=16` | 1 | 1 (the kwarg wins) | 128 |
| chunk 16 | `LEXLAT_K2_CHUNK_SEQS=1` (the probe's route) | absent | 1 | 128 |
The `k2 lexicon term: {...}` spec from each csN config is byte-identical to the one from the probe's route at the same N. It differs from cs8's only in `chunk_seqs`, so a probe result at N transfers to the csN config. This shows that the file builds the intended runtime. It does not show that J uses it; J's own `installed (chunk_seqs = N)` line after the restart would show that.

**Exactness (CPU, scratch copies; the recipe tests are untouched).** The test runs 40 sequences, which is 20 chunks at chunk 2 and 40 at chunk 1. Both give **6 passed**. At sub-epochs 8/9/10/11 and in the no_grad pass, max |dlogZ| = |dmonitor| = |dgrad log_q| = |dgrad param| = 0.000e+00. The term is 0.0860062307299628, the same value as in the chunk-16 and chunk-8 runs. The fixture limits are the same as in cs8 (stability 0.0 on the fixture).

**Install script.**
- On a scratch mock of J, each script:
  - installs from J's original config (prints the 8-line diff and `INSTALLED`);
  - says `already installed` on a re-run;
  - refuses (`ABORT ... epoch.008.pt exists`) when the config is the original and `epoch.008.pt` exists;
  - refuses when the chunk-16, cs8 or other-csN config is live.
- The `.orig` files were left unchanged.
- J's real state, checked read-only: I ran the script's own lines 1-15, which are the sha checks and the epoch.008 guard and contain no cp, mv, rm or redirect, against the real J, then stopped. Both scripts print `PRECONDITIONS OK: would install`, with J's config `dff329b6...` and no `epoch.008.pt`.
- Install and restart otherwise follow the chunk-16 report, section 3, using `bash S/analysis_out/k2lat_perchunk_returnn_csN_install.sh`.

## Concerns / undetermined
1. **Wall time at chunk 1 is unmeasured.** A step with 115 sequences makes 115 k2 forward calls and 115 per-chunk backward calls, against 15 at chunk 8. The 3 h limit covers the roughly 57 steps only if the k2 leg averages under about 2 min per step (bed about 45-51 s/step plus 0-23 min startup). If the time limit hits, read.py gives FAIL for incompleteness together with the peak over the steps completed. The memory answer usually comes from step 0 (the 115 x 764 batch), within minutes.
2. **Seeding skew.** Each probe seeds at its own start. J's epoch.005 is due at about 17:55. If the two jobs start on either side of it, cs2 and cs1 differ in theta as well as in chunk size (review SHOULD 4). Compare `seed:` lines in both `read.txt` files. It also means the live sub-epoch 8 runs on epoch-7 weights, not 4 or 5, which is the gate's registered conservative premise.
3. **Not added:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the debugger suggested. The brief fixed the delta at chunk size and directory. Whether to add it, to both the probe and J's environment, is the orchestrator's decision.
4. The cs8 report's concerns 1-4 apply unchanged to the cs2 and cs1 variants. Each variant accepts only J's original config. The chunk size is visible only in the unhashed epilog. The manager restart and job.save caveat stands. The patch must be regenerated if `config/sae_i6_p0.py` changes.
5. The earlier chunk-16 and chunk-8 install scripts, and the new ones, all refuse once any other per-chunk config is live. So only one of the variants can be installed from J's original config, unless the orchestrator decides otherwise.
