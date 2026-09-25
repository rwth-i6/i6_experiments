# Training routing moved to V100 (gpu_32gb), plus a V100 k2 test wrapper (2026-09-25)

Status: DONE_WITH_CONCERNS. The concern: the existing unit test file `gpu_route_test.py` now has 4 failures, because those tests encode the old training routing. That file was outside my edit scope and I did not change it.

Nothing was submitted, restarted or committed. I also used `sbatch --test-only` twice to validate requests: it creates no job, and `squeue` afterwards showed only the 4 jobs that were already there.

## Background (from the dispatch)
On 2026-09-25 the user decided to run the SAE trainings on V100 (`gpu_32gb`, nodes cn-32 and cn-33). The evidence is the ctrl_20 step benchmark in `reports/review_v100_bench_2026-09-25.md`: the reference batch fits (20.28 GiB allocated), and a step is 1.32x faster than on L40S.

## Files changed

### 1. `settings.py` (setup dir)
The file grows from 460 to 468 lines. Hunk ranges refer to the new file.

- **Lines 52-53 (comment only).** New note: ReturnnTrainingJob `run` tasks bypass the flexible routing.
- **Lines 69-71.** New constant `GPU_ROUTE_TRAIN = "gpu_32gb"`, with a comment giving its source.
- **Lines 381-401, `check_engine_limits`:**
  - The docstring now describes the new rule.
  - The existing training test is kept in a variable, `is_train_run`. The 72 h time rule is unchanged.
  - An explicit `-p` in `sbatch_args` still wins, as before.
  - New rule: `if is_train_run and gpu > 0: sbatch_args = ["-p", "gpu_32gb"]`. It runs before the gpu_mem rules, so gpu_mem is ignored for trainings (96 is fine) and the training does not query Slurm at all. That includes the sticky logic, the sacct lookups and the reservations.
  - The rest of the function is unchanged (`if gpu > 0` became `elif`).
- **Unchanged:** `_gr_training_sticky` and `_gr_is_training` are still there. They are now unreachable for training `run` tasks. I did not refactor them.

### 2. `analysis/v100_bench/run_k2_tests.sh` (new, executable)
See the section on the test wrapper below.

### 3. `./gpu_route`
Not changed. It uses only the flexible-routing helpers, and `--rebalance` already skips ReturnnTrainingJob. Read-only runs after the edit work:
- the scores print normally;
- `--rebalance --dry-run` reports "would move 0".

## Resources the trainings request
All four trainings request the same resources:

| | cpu | mem | time | gpu | gpu_mem |
|---|---|---|---|---|---|
| Package request | 16 | 64 GB | 11.5 h | 1 | 96 |
| After `settings.py` | 16 | 64 GB | 72 h | 1 | 96 |

- **Source of these numbers.** Read from each job object through the console, and matching `submit_log.run` of GiT88bxzoZbZ, DvVfxf1LrCBi and llSFybyKXkbL.
- **No explicit `-p`.** The package request contains no `sbatch_args`, so the new rule applies to all four.
- **Fits a V100 node.** A cn-32/33 node has 96 CPUs, 1,546,786 MB of memory and 16 V100s. The 7-day MaxTime equals the existing 168 h cap.
- **Account.** `hlt` comes from the user's default account (AllowAccounts=hlt), as before; the engine does not pass `--account`.
- **Comment tag.** The old gpu_mem > 24 path set no `--comment`, so the new rule sets none either.

## Old vs new behaviour

| Job | Before | After |
|---|---|---|
| Training `run` task with gpu_mem > 24 (all P0 trainings) | `-p gpu_48gb` | `-p gpu_32gb` |
| Training `run` task with gpu_mem <= 24 | flex rule, sticky to the partition of its last Slurm job | `-p gpu_32gb`, no Slurm query |
| Resume of a training (history on gpu_48gb, sacct CANCELLED, still queued) | stayed on its old partition | `-p gpu_32gb` |
| Other training tasks (`plot`, `create_files`) | unchanged | unchanged (a GPU `plot` would still follow the gpu_mem rules) |
| Other GPU jobs, gpu_mem <= 24 | flex24 (gpu_24gb / gpu_48gb, `--comment=flex24`) | unchanged |
| Other GPU jobs, gpu_mem > 24 | gpu_48gb | unchanged |
| CPU jobs, mem > 180 GB | gpu_32gb | unchanged |
| Other CPU jobs | default partition | unchanged |

Nothing is routed to gpu_11gb or A100.

## Checks

### 1. Compile
`python -m py_compile settings.py` passes.

### 2. Existing unit tests, `gpu_route_test.py` (run with the sae python 3.11)
- **Original `settings.py`:** 33 tests, OK.
- **New `settings.py`:** 33 tests, 4 failures. All 4 assert the old routing of trainings:
  - `test_over_24_never_on_24gb`, line 184: training with gpu_mem 96 expected on gpu_48gb;
  - `test_training_72h_rule_kept`, line 223: expected gpu_48gb (the 72 h part itself holds);
  - `test_fix3_training_prefers_slurm_partition_of_last_job`, line 312;
  - `test_corrupt_files_do_not_crash`, line 338: the training line expects gpu_24gb.
- **Proposal.** Someone with the right scope should update these assertions to expect gpu_32gb.
- A plain `python` (3.6) run also gives an unrelated `capture_output` error. It does not occur with the sae python.

### 3. New-behaviour checks
Scratch file: `/var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/bb154ec2-4c82-453f-afb8-04f4de6c5e2e/scratchpad/v100_route_check.py`. All 4 tests pass:
- trainings with gpu_mem 8, 24, 32, 48 or 96 all go to gpu_32gb, with time 72 and no Slurm calls;
- a resume is not held by its sticky history (submit_log on gpu_48gb, a usage host on cn-508, sacct CANCELLED, the job still queued on gpu_48gb);
- time is capped at 168 h;
- all other job classes route exactly as before, and an explicit `-p` is honoured.

### 4. Hash neutrality
I listed the sorted `_sis_id()` of `tk.sis_graph.jobs()` for `config/sae_i6_p0.py` in the console (sae python), once before and once after the edit. Both lists have 164 ids, and `diff` finds them IDENTICAL.

### 5. Dry check of the P0 trainings
Method: in the console, `engine.get_rqmt(run_task, 1, update=True)` followed by the Slurm engine's `options()`. Nothing is submitted.

The k2lat training was found through the alias `sae/4a/training/k2lat_20_ma3000`: it is ReturnnTrainingJob.jcKXbLMDk4hl, which is not yet runnable. Its task list was built the same way as `Job._sis_tasks`, without caching.

All four jobs give the same result:

| Job | Arm | Before | After |
|---|---|---|---|
| GiT88bxzoZbZ | ctrl_20 | gpu_48gb | gpu_32gb |
| DvVfxf1LrCBi | ctrl_20_s1 | gpu_48gb | gpu_32gb |
| llSFybyKXkbL | ctrl_20_rc | gpu_48gb | gpu_32gb |
| jcKXbLMDk4hl | k2lat_20_ma3000 | gpu_48gb | gpu_32gb |

sbatch args after the edit:
`--mem=64G --gres=gpu:1 --cpus-per-task=16 --time=4320 --export=all --ntasks-per-node=1 -p gpu_32gb`

Before the edit the args were the same except `-p gpu_48gb`.

Slurm accepts this request: `sbatch --test-only` with these args answered "to start ... on nodes cn-32 in partition gpu_32gb".

### 6. Test wrapper
- `bash -n` passes.
- `sbatch --test-only` accepts it for cn-32 in gpu_32gb.
- A `diff` against the G0.V script shows only the intended changes (listed below).

## Test wrapper: `analysis/v100_bench/run_k2_tests.sh`

**Submit with:**
`bash /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis/v100_bench/run_k2_tests.sh`

Run outside Slurm, the script creates `/work/asr4/hwu/setups/librispeech-960/2026-09-24-unsupervised/analysis_out/v100_k2_tests` and then runs `sbatch` on itself. Inside Slurm it runs the tests.

**Slurm settings:** gpu_32gb, 1 GPU, account hlt, 8 CPUs, 48 GB, 1 h. The log goes to `analysis_out/v100_k2_tests/slurm-%j.out`.

**Test command.** The command is the one in G0.V's `/work/asr4/hwu/sae_i6_tests/gpu_2026-09-24/run_tests_gpu.sbatch`, the Slurm 4334384 run in `reports/exec_gpu_tests_2026-09-24.md`. It runs from the package dir:

`$E/bin/python -m pytest tests/ -rA -p no:cacheprovider --durations=25 --basetemp=$OUT/pytest_tmp_<id> --junitxml=$OUT/junit_<id>.xml | tee $OUT/pytest_<id>.log`

It also uses the same environment as G0.V:
- PYTHONPATH starts with the RETURNN training clone KQ3NuCaDE6QH;
- `SAE_ARTEFACT_DIR` is unset;
- the same k2 CUDA smoke preflight and the same RETURNN-clone check run first.

**Changes relative to G0.V:**
- partition, time (1 h), job name and output dir;
- the self-submit block;
- a preflight that exits 2 unless the compute capability is (7, 0).

**Expected result, from G0.V on L40S:**
- 506 passed, 11 skipped (7 artefact + 4 ffmpeg pin), 12 xfailed, 0 failed;
- no skip with reason "gpu:" or "k2:";
- `test_t1_8_gpu_parity` and `test_t119_cpu_cuda_parity_log_z_hlg_and_log_z_h[1.0|2.0]` PASSED.

## Undetermined and assumptions
- **Test selection.** The dispatch says "gpu- and k2-marked selection", but G0.V actually ran the full suite (`tests/`, no `-m` filter). SAE_i6_P0.md says "full suite". I reproduced the exact G0.V command, which is a superset of the gpu- and k2-marked tests. That keeps the counts comparable with 506/11/12. If only the marked subset is wanted, the change is to add `-m "gpu or k2"`.
- **Explicit `-p` in a config.** An explicit `-p` still overrides the training rule, as it overrides every rule today. None of the P0 trainings sets one.
- **gpu_mem.** gpu_mem stays 96 in the rqmt. Sisyphus does not pass it to Slurm (only `--gres=gpu:1`), so no limit check is involved.
- **Resuming after `scancel`.** Sisyphus keeps time 72 unless sacct reports TIMEOUT, and memory 64 GB unless sacct reports OUT_OF_MEMORY or usage is near the limit (then it doubles, which still fits a node).
- **Running manager.** The P0 manager that is running now still has the old `settings.py` loaded. The new routing applies only after the planned manager restart.

## Addendum: `gpu_route_test.py` updated (in scope on the coordinator's request)
The concern above is resolved. Four tests changed to the new policy; every other expectation is untouched.
- `test_over_24_never_on_24gb`: a training with gpu_mem 96 is expected on gpu_32gb.
- `test_training_72h_rule_kept`: expects 72 h and gpu_32gb.
- `test_corrupt_files_do_not_crash`: the training line expects gpu_32gb (the forward line still expects gpu_24gb).
- `test_fix3_training_prefers_slurm_partition_of_last_job` is renamed `test_training_resume_goes_to_v100_whatever_its_history`. It expects gpu_32gb in each case:
  - the job moved to gpu_48gb and was accounted there by sacct;
  - sacct is down and the usage host is on gpu_48gb;
  - the last `-p` is gpu_48gb and the job is still queued there;
  - gpu_mem 48.
  It also asserts that an explicit `-p gpu_48gb` wins, and that routing a training makes no Slurm call.

Results (sae python):
- with the new `settings.py`: 33 tests, OK;
- with the original `settings.py`: the same 4 tests fail. This shows the tests tell the two policies apart.
