# Review: P1 Task B per-chunk k2 backward, arms entry point, launches A and B (2026-09-25)

Reviewer: code-reviewer. Scope: `reverse_model/rt_chunked_backward.py`, `tests/test_rt_chunked_backward.py`,
`config/sae_i6_p1_ladder.py`, Launch A (GPU parity test) and Launch B (rt_r90 probe). Input report:
`reports/impl_p1_arms_chunked_backward_2026-09-25.md`. Nothing was edited, launched or committed.
Package path below: `P = recipe/i6_experiments/users/wu/experiments/unsupervised_asr`.

## Verdict: PASS_WITH_FIXES

The per-chunk backward reproduces the held score and gradient exactly on CPU. The comparison is not a
self-comparison. The arms' configs have exactly the intended deltas. Launch A can go now. Launch B may go
once the G1.M peak-memory read is fixed or the gate text is amended to match what is measured (F1). The
preconditions on the fits are met as of 15:00 today: all three fit jobs have `finished` and
`epoch.008.pt`, and the fits manager (pid 1677624) has exited. Only the P0 manager (pid 1646677) and a sis
console (1682122) are running.

## Findings (most severe first)

F1 `P/reverse_model/rt_chunked_backward.py:142` (same as held `P/model/lexlat_k2_train.py:555`) - the k2 step
calls `torch.cuda.reset_peak_memory_stats()` on every step after the stability read (`:138`) and after the
dense tensor is built. RETURNN's per-step `mem_usage:cuda:0` (`recipe/returnn/returnn/torch/engine.py:1618`,
`max_memory_allocated`, which RETURNN itself resets only at epoch start, `:325`) and
`lexlat_k2_peak_reserved_gib` (`:170`) therefore cover only the span from the k2 leg through backward and the
optimizer step. Transients that are freed before the reset are not measured. These are the forward
temporaries and, above all, the stability read: it runs once per sub-epoch at the reference rung
max_active 10000 (arm rung 1000), on 16 sequences, one per call. Activations still held at the reset are
counted, because the reset sets the peak to the current allocation. When it breaks: G1.M
(`SAE_i6_P1.md:105`, "peak allocated <= 40 GiB over the whole sub-epoch") can read PASS while the true
sub-epoch peak was above 40 GiB. The only guard is an OOM in the stability read itself, which is caught and
printed as "FAILED", so G1.M fails on "stability read completes". A peak between 40 and 46 GiB passes
unseen. Fix (hash-neutral, no score effect, in the non-frozen file only): before line 142, read
`torch.cuda.max_memory_allocated()` and emit it as a per-step monitor or print, e.g.
`lexlat_k2_pre_leg_peak_alloc_gib`. The value read at step n covers step n-1's leg and backward plus step n's
forward and stability read, so the max over steps of it and of `mem_usage` is the true peak. The tests'
monitor-set equality (`tests/test_rt_chunked_backward.py:96`) must then allow the extra key. Alternatively,
amend G1.M before the probe, disclosing what the read excludes.

F2 `SAE_i6_P1.md:104` - G1.M records "`nvidia-smi` on the node", but neither the config nor the job produces it.
`srun --jobid` does not work from this desktop. When it breaks: the G1.M record lacks one of its listed
fields. This is not a PASS criterion. Either drop it from the gate, or have the executor capture it by
another route (for example `ssh <node> nvidia-smi`, if permitted).

F3 `config/sae_i6_p1_ladder.py:95,66` and `settings.py` check_engine_limits - the probe is the full rt_r90 arm:
8 sub-epochs, with `time_rqmt` 11.5 raised to 72 h by settings.py. When it breaks: a G1.M miss that does not
crash the job (a peak above 40 GiB, or a stability read that prints FAILED) leaves it training on into
sub-epoch 2 and beyond, while the gate says "no arm launches". The executor must read G1.M at the end of
sub-epoch 1 and `scancel` on a miss. This is operational; it needs no code change.

F4 `config/sae_i6_p1_ladder.py:95` - `memory_log=(arm == PROBE_ARM)` sets `torch_log_memory_usage` only for
rt_r90. G1.M (`SAE_i6_P1.md:102`) requires the probe on rt_r100 "if it is run". When it breaks: if G1.L's
both-LIFT branch fires, rt_r100 runs without per-step peak memory, and G1.M cannot be read for it.
`post_config` is not hashed, so this can be fixed before any R100 launch.

F5 `config/sae_i6_p1_ladder.py:101-103,129` - the second-seed arm varies corruption seed 1, flat (theta init)
seed 1, RETURNN `random_seed` 1 and `random_seed_offset` 1000 (`ladder.SECOND_SEED`, `ladder.py:395`, the
reference recipe's switch). G1.L (`SAE_i6_P1.md:126`) names only "corruption seed 1 and theta init seed 1".
When it breaks: a seed disagreement cannot be attributed to the two named seeds alone. Record the full seed
set in G1.L before the builder is used; it is conditional and not launched now. The State line
(`SAE_i6_P1.md:21-22`) says the builder "needs a `seed=` argument in ladder.py", but the builder already
exists in the config (`second_seed_arm`, `P1_LADDER_SEED2`, which has no default).

Notes (no failure path): the meanings of `lexlat_k2_sec` and `lexlat_k2_peak_reserved_gib` now include the
per-chunk backwards; this is disclosed in the module docstring (`:41-42`). The stability read is not
exercised by the tests: the fixture's max_active of 10000 equals the reference rung, so the read returns nan.
The chunked class inherits the read unchanged and calls it at the same place as the held path, so this
leaves no gap in exactness.

## Check 1: exactness against the held path

Compared line by line with `P/model/lexlat_k2_train.py:514-585` (held step), `P/model/lexlat_k2.py:988-1053`
(`chunked_tot_scores`), and `P/model/train_step.py` (the call site, `mark_as_loss(scale=lam)`). Both frozen
files are untouched by the change.
- Per-sequence scale: `term = ((-l_lex / retained) * w).sum() / n_keep` with `scored = keep>0 & finite`, the
  held expression. `z_hlg_upstream` differentiates this same expression with respect to a zero leaf standing
  in for z_hlg, with `grad_outputs = lam` in the term dtype. This gives -lam*w_i/retained_i/n_keep, the
  held chain rule, through the same autograd ops.
- theta: the HLG leg's gradient accumulates into a detached leaf, one chunk at a time over disjoint slices
  (`tot.backward(g)` per chunk; rows whose tot is not finite get 0, as the held path's `where` gives). It is
  injected at the live `dense_in` by `_InjectGrad` (value 0; the backward returns grad * incoming/lam). The H
  leg stays on the graph as held. phi gets no k2 gradient in either path: the reverse model does not feed
  `dense_in`, as in the held path.
- Empty lattices, abort rule and marker: these are the held code with the same threshold, and
  `_abort_on_empty` is inherited. The marker is `lexlat_k2_ABORT.json` in the job cwd (`<job>/work/`).
- Monitors: `_monitors_from_reads` gets the same per-chunk reads (`_sizes`, `_expected_words_one` under
  no_grad). The tests assert equality on every monitor except the two timing monitors.
- Stability read: inherited, and called before the reset exactly as in the held path (`:138` vs `:551`).
- no_grad and eval: `need_grad` is False, so no upstream, no backward and no injection. This is tested.
- RETURNN scaling: the loss is float64 (`dp_log_q = log_q.double()`). `total_loss` multiplies it by lam with
  `torch.mul` in the loss dtype, so incoming/lam = 1.0 exactly. `active()` means lam > 0, so there is no
  division by zero. There is no AMP or grad scaler, `accum_grad_multiple_step` is 1, and
  `gradient_clip_global_norm` 5.0 acts on identical gradients.
- Is "0.0 deviation" plausible? Yes, on CPU: the same op chain, with each slice's gradient written once. It
  is not a self-comparison: the test patches the held `log_z_hlg` to raise inside the new runtime, and the
  held reference is computed by the unmodified class. On GPU, k2's backward scatter uses atomics, so a
  nonzero deviation within the test tolerances (log Z 1e-9, grad 1e-7) is expected and acceptable.
- Rerun on CPU (this review): `tests/test_rt_chunked_backward.py` gave 38 passed, 2 skipped (gpu); every
  recorded deviation was 0.000e+00. The neighbouring suites (test_model_lexlat_k2, test_model_blankfree,
  test_reverse_ladder, test_training_config) gave 139 passed, 2 skipped (gpu), 4 xfailed.
- Differing full-step totals between parametrized cases (chunk1-ep1 2.7038 vs chunk16-ep1 2.7653) come
  from an unseeded model init across tests. Held equals new within each case. This is not a defect.

## Check 2: selection through the hash-neutral epilog

`config/sae_i6_p1_ladder.py:59-62` asserts that the epilog is empty, sets `python_epilog = EPILOG`, and
asserts that `python_epilog_hash` is unchanged. The ReturnnConfig hash reads `python_epilog_hash`, not the
text. An independent dry build (STAGE=arms) gave arm jobs `9lD2HS2Mzgcl` (r70), `5CL3pQyyOvQt` (r80) and
`EexT85vdfx25` (r90), all with `-p gpu_48gb`. The dumped configs are byte-identical to the implementer's
`analysis_out/p1_ladder_rt_*_2026-09-25.returnn.config`. Each ends with the epilog import that rebinds
`train_step`. No arm work dir exists yet, so no config has been written.

Risk: a job with the same hash constructed without the epilog would reuse the first instance (in-process),
or would be the first to write `returnn.config` (across managers). Nothing builds these ids now: P0 does not
build rt arms, and no other entry point calls `ladder.build_rt`. The post-launch checks below close the risk.

## Check 3: single delta per arm

The r70 config follows the JUPITER node R recipe (`ladder.rt_train_config`/`build_rt`, `ladder.py:611-698`).
Compared with P0 k2lat, it differs only in schedule, reverse checkpoint, k2 rung/on-set/chunk, keep, lr,
num_epochs and model dir, plus the epilog; the flat init `0J9d6wjrkRYH` is the same. `lexlat_k2_chunk_seqs`
4 is `RT_K2_CHUNK_SEQS`, as in the recipe. r80 differs from r70 only in `reverse_checkpoint_path` (and
model dir). r90 differs from r80 only in the checkpoint and `torch_log_memory_usage = True` (post_config,
not hashed). The fit ids are `ruLnJFWyifwp` (r70), `6IkAAuzcBwBR` (r80) and `uO8wkodbR2uh` (r90), matching
the finished fit jobs.

## Check 4: reads

`_reads` calls `analysis.per.epoch_reads` with `config.common.READ_SPLIT` (dev-other) at
`ladder.KEEP_EPOCHS`, the same call as P0's `train_and_read`. The probe-stage graph contains the
rt_r90 ep1/2/4/8 theta extraction, dev-other posteriors and BlankfreeGreedyPer jobs
(`analysis_out/p1_ladder_jobs_probe_2026-09-25.tsv`).

## Launch A: GPU parity test (can go now)

```
sbatch -A hlt -p gpu_test_24gb --gres=gpu:1 -c 4 --mem=16G -t 0:30:00 -J p1_rt_parity -o /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/log/p1_rt_parity.%j.out --wrap 'S=/u/hwu/setups/librispeech-960/2026-09-24-unsupervised; export PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH PYTHONPATH=/work/asr4/hwu/setups$S/i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH/output/returnn:$S/recipe:$S/recipe/returnn:$S/sisyphus; cd $S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr && nvidia-smi && python -c "import sys,torch,k2;sys.exit(not(torch.cuda.is_available() and k2.with_cuda))" && python -m pytest -p no:cacheprovider -m gpu -rA -s tests/test_rt_chunked_backward.py'
```
The command runs on gpu_test_24gb (RTX 3090, sm_86, 1 h cap), never gpu_11gb or A100. It uses no
apptainer, puts the sae python first on PATH, and puts the pinned RETURNN clone first on PYTHONPATH, as in
the 2026-09-24 GPU run. Slurm output goes to `log/`. If CUDA or k2-CUDA is missing, the check exits
non-zero before pytest, because conftest would otherwise skip the tests silently.

PASS read: `2 passed`, no `SKIPPED`, and two `[record] cuda chunk {1,4}` lines within log Z 1e-9 and
grad 1e-7.

## Launch B: rt_r90 probe (after F1 and Launch A PASS; fits FINISHED and pid 1677624 gone, both true now)

```
cd /u/hwu/setups/librispeech-960/2026-09-24-unsupervised && P1_LADDER_STAGE=probe PATH=/work/asr4/hwu/conda/envs/sae/bin:$PATH nohup /work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis --log_level 30 m -r config/sae_i6_p1_ladder.py > log/sae_i6_p1_ladder.manager.log 2>&1 & echo $! > log/sae_i6_p1_ladder.manager.pid
```
Run one manager only. STAGE=probe includes the (finished) fit jobs, so no other manager may hold them. The
watcher is the fits' pattern with `SIS_LAUNCHER="/work/asr4/hwu/conda/envs/sae/bin/python sisyphus/sis"`
on `config/sae_i6_p1_ladder.py`.

Resources: one GPU on `-p gpu_48gb` (L40S), 16 CPU, 64 GB, 72 h via settings.py.

Checks after submission, in `work/i6_core/returnn/training/ReturnnTrainingJob.EexT85vdfx25/`:
- `output/returnn.config` ends with the `rt_chunked_backward import train_step` epilog and contains
  `torch_log_memory_usage = True`.
- `submit_log.run` shows `-p gpu_48gb`.
- `log.run.1` contains `rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = 4)`.

G1.M read at the end of sub-epoch 1, from `log.run.1`/RETURNN log:
- max over all sub-epoch-1 steps of `mem_usage:cuda:0` <= 40 GiB (plus the F1 pre-leg read, if added);
- `lexlat_k2_peak_reserved_gib`;
- `lexlat_k2: stability at sub-epoch 1: median ...` present, and not `FAILED`;
- no CUDA OOM, no k2 `Check failed` or int32/Array error;
- no `work/lexlat_k2_ABORT.json`;
- `epoch.001.pt` written;
- sec/step and sub-epoch elapsed.

On a miss, the executor runs `scancel` (F3). A pass continues as rt_r90 under the same hash
`EexT85vdfx25`, because the probe and arms stages build identical ids. The later STAGE=arms manager
replaces the probe manager (never two at once).

## Check 7: conditional builders

rt_r100 (`P1_LADDER_R100=1`) and the second seed (`P1_LADDER_SEED2`) are not built by default and not
launched. See F4 and F5 for their consistency with G1.M and G1.L.
