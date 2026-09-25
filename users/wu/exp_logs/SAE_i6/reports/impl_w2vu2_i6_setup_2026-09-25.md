# Implementer: i6 setup of the w2vu2 GAN reproduction, up to launch (2026-09-25)

DONE_WITH_CONCERNS. Everything up to the launch is prepared. Nothing was applied to a live file and
nothing was submitted. The graph dry-loads with 101 jobs. 49 of them are shared with the P0 and P1 graphs
and all 49 are finished, so a separate manager can run it. The six trainings route to gpu_32gb and the
GPU forwards and decodes route to gpu_48gb. The w2vu2 tests give 33 passed, 15 skipped and 0 failed.
Adding torchaudio to the sae env cannot change a live job, and the wheel loads against the env's torch.

The concerns:
- Both trainings probably run longer than their first 11.5 h leg on V100, so they will depend on
  resume.
- `settings.py` still needs the patch applied, and the w2vu env has not been built yet.
- The extras script has not been run. The two torchaudio CPU jobs fail until it is.

Scope: code 68b39418a, reports 547238952, merged locally as 692d6e55a. Branch
`haotian_cycle_consistency_unsupervised`. Nothing is committed; the dispatch did not ask for a commit.

## Files

| file | delta |
|---|---|
| `config/sae_i6_w2vu2.py` (new, setup-local, in no repository) | `py()` returns `{"w2vu2": config.w2vu2.py()}`, the style of `config/sae_i6_p0_supinit.py` |
| `analysis_out/w2vu2_settings.patch` (new, NOT applied) | (a) +6 lines after `# K2_PYTHON deliberately unset`: a comment and `W2VU_PYTHON = "/work/asr4/hwu/conda/envs/w2vu/bin/w2vu-python"`; (b) addendum: `check_engine_limits` gives `FairseqHydraTrainingJob` run tasks the 72 h floor that ReturnnTrainingJob run tasks already get (see the addendum below) |
| `analysis/w2vu_env_build/build.sbatch` (new, NOT submitted) | runs `CUDA_ARCH=70 GATE_CUDA=1 build_w2vu_env.sh /work/asr4/hwu/conda/envs/w2vu` on 1 V100 |
| `analysis/w2vu2_dry_load/dump_graph.py` (new) | the read-only graph dump used for tasks 4 and 5 |
| `analysis_out/w2vu2_graph_jobs_2026-09-25.json`, `analysis_out/w2vu2_overlap_p0_p1_2026-09-25.tsv` (new) | the evidence for tasks 4 and 5 |

## 1. settings.py patch

- **The only missing required key is `W2VU_PYTHON`.** The other keys the docstring lists are already in
  `settings.py`: `FFMPEG_BINARY`, `FFMPEG_PIN_ACCEPT`, `KENLM_BINARY_PATH`, `SAE_PYTHON` and `HF_HOME`.
- **`W2VU_FAIRSEQ_ROOT` is left unset** (it is optional). The graph then uses the sparse
  `CloneGitRepositoryJob.rnfyLkSbJUoz`, a mini task, and the compute nodes have internet access.
- **Checks:**
  - `patch -p1 --dry-run` against `settings.py` applies cleanly (sha256 743b4f7e... before).
  - The patched copy parses.
  - `gs.GPU_ROUTE_TRAIN` = `gpu_32gb`, read in the dry load. This is the constant that
    `I6_TRAIN_PARTITION_SETTING` names. All 6 trainings got `sbatch_args = ["-p", "gpu_32gb"]`.
- **For the reviewer, on applying it while the P0 and P1 managers run:**
  - The managers do not re-read `settings.py`.
  - Each new worker does re-read it. Nothing in the P0 or P1 code reads `W2VU_PYTHON`, and settings
    content is not hashed.

## 2. Env build sbatch

- **Resources:** `-A hlt -p gpu_32gb --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00`.
- **Log:** `log/w2vu_env_build.%j.out` (shared /u path). The chdir is `analysis/w2vu_env_build`.
- **Conda:** `CONDA_BIN=/work/asr4/hwu/conda/bin/mamba`.
- **Additions not in the package script:**
  - `unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV`.
  - `PIP_NO_CACHE_DIR=1`, which keeps several GB of CUDA wheels out of `$HOME/.cache/pip`.
  - A preflight that fails before any download if the prefix already exists, if mamba or g++ is
    missing, or if the GPU is not compute capability 7.0.
- **The script's own gate is unchanged.** It asserts `torch.cuda.is_available()` and `sm_70` through
  the wrapper.
- **Check run:** `bash -n` only.
- **After a failed build,** `/work/asr4/hwu/conda/envs/w2vu` must be removed before a retry.
- **Nodes:** cn-32 and cn-33 have 16 V100 each.

## 3. torchaudio in the sae env

**Live jobs.** No live code path imports torchaudio, conditionally or otherwise:
- RETURNN (`recipe/returnn`, 00171dfe2) has no `torchaudio` import. The only hits are in
  `.github/workflows`.
- In the sae env's site-packages, only `transformers` and `torch` mention torchaudio:
  - `transformers`: feature extractors and pipelines, some through `is_torchaudio_available()`.
  - `torch`: `utils/data/datapipes/utils/decoder.py`, only inside `audiohandler()`.
- I imported the live training closure in the sae python: `model.train_step`,
  `reverse_model.rt_chunked_backward`, and RETURNN's `torch.engine`, `frontend`, `datasets.hdf`,
  `datasets.meta`, `datasets.audio` and `torch.data.pipeline`.
  - It loads no transformers, torchaudio, lhotse or torchdata module.
  - It makes no `find_spec("torchaudio")` call.
  - The datapipes decoder module loads, but `audiohandler` is never called.
- The package's only transformers user is `w2v2/forward.py`. It is used only by
  `L15ForwardSeqOrderJob`, and all 7 of those are finished.
- The unfinished P0 and P1 classes are ReturnnTraining, ReturnnForwardJobV2, ExtractSubmodule and
  the analysis jobs. None of them imports it.
- `pip --no-deps` adds only `torchaudio/` and its dist-info.

**ABI.** The x86_64 PyPI wheel (`torchaudio-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl`, sha256
f8bd6935...) is `2.7.1+cu126` and pins `torch==2.7.1`.
- Its `.so` files use the CXX11 ABI (`__cxx11` symbols). The env's torch reports
  `_GLIBCXX_USE_CXX11_ABI=True` and CUDA 12.6.
- Loaded from a scratch unpack on `PYTHONPATH`, with the env untouched:
  - `_IS_TORCHAUDIO_EXT_AVAILABLE=True`.
  - The extras script's gate (kaldi MFCC, `compute_deltas`, sklearn 1.8.0 MiniBatchKMeans) passes on
    CPU.
  - `test_convert_split_frame_rule_and_layout`, the one test skipped for torchaudio, passes.
- The CUDA kernels were not exercised; no GPU was used.
- The reference env's combination ran on JUPITER GH200, i.e. the aarch64 wheel, not this x86_64
  binary.

## 4. Dry load

Command, run from the setup dir:
`SIS_W2VU_PYTHON=<wrapper> sis console config/sae_i6_w2vu2.py -s -c "exec(open(dump_graph.py))"`,
under the sae python. The dump reads job state only with `job_finished()`, never `_sis_finished()`,
because that can write the finished marker.

101 jobs. Status: F = finished, N = new (no job directory).

| class | count | status |
|---|---|---|
| FairseqHydraTrainingJob | 6 | 6 N |
| ReturnnForwardJobV2 | 18 | 7 F, 11 N |
| W2vu2GanPerJob | 10 | 10 N |
| W2vu2GeneratorCheckpointJob | 6 | 6 N |
| L15ForwardSeqOrderJob | 7 | 7 F |
| DownloadJob | 4 | 2 F, 2 N |
| CloneGitRepositoryJob | 2 | 1 F, 1 N |
| FairseqAudioManifestJob | 3 | 3 N |
| i6 LibriSpeech, bliss, ogg, g2p, lexicon and text-pipeline jobs | 20 | all F |
| 1 each: KenLMplz, CreateBinaryLM, TextToPhoneme, FairseqTextData, MfccKmeans, W2vu2FeatureData, W2vu2GanSelect, W2vu2GanPseudoLabel, FairseqCtcData, CtcPhoneDecode, CtcWordDecode, FlashlightLexicon, OggZipWordRefs | 13 | all N |
| 1 each: port inputs (FfmpegPinCheck, GoldPhones, LibriSpeechSplitIds, BlankfreeVadHdf, CollectOovWords, PhonemizeWithSil, 4 w2v2.units jobs) plus 2 DownloadHuggingFaceSnapshot | 12 | all F |

Routing of the new GPU tasks under `settings.py` (through `check_engine_limits`; sbatch options
rendered by the Slurm engine):
- **GAN, 5 seeds** (`fCcCEHaqxnyE K0RJEx1UUwcb qLeCqytt98iC RzVQARNBfUb2 gY9AdQ2aYl3B`): gpu_32gb (V100).
  `--mem=100G --gres=gpu:1 --cpus-per-task=8 --time=690`.
- **CTC student** (`SWK00drxbXSC`): gpu_32gb (V100). `--mem=60G --gres=gpu:4 --cpus-per-task=16
  --time=690`, on one node.
- **11 ReturnnForwardJobV2:** gpu_48gb (L40S), because `gpu_mem` 40 > 24; 2 h each.
- **CtcPhoneDecodeJob** (2 h) and **CtcWordDecodeJob** (`decode`, 11.5 h): gpu_48gb.
  - My gpu_48gb QoS cap is 5 GPUs, and the P0 and P1 trainings hold 4 of them now. The forwards will
    queue behind them, which costs time but is safe.
- **CPU tasks:** MfccKmeans (8 CPUs, 48 GB, 8 h) and W2vu2FeatureData (4 CPUs, 16 GB, 12 h) import
  torchaudio, so the extras script must run before these start. Also FairseqAudioManifest ×3,
  FairseqTextData, TextToPhoneme and KenLM.
- **Mini tasks on the desktop's LocalEngine:** the CloneGit, the 2 Downloads, FairseqCtcData, Select,
  PseudoLabel, the generator conversions and PER jobs.

## 5. Overlap with P0 and P1

- **P0:** the graph of `config/sae_i6_p0.py` has 164 jobs.
- **P1:** the graph of `config/sae_i6_p1_ladder.py` has 113 jobs, built with `P1_LADDER_STAGE=arms`,
  which I read from the live manager's environment.
- **Shared:** 49 jobs, all in both graphs, and **all 49 finished** (list in the `.tsv`).
- **w2vu2 only:** 52 jobs. None has a job directory and none is running.
- **Verdict:** no shared unfinished job, so a separate w2vu2 manager does not collide with pids 1646677
  and 1726329. No other manager pid in `log/` is alive.

## 6. Tests

sae python, CPU (`CUDA_VISIBLE_DEVICES=`), `SAE_ARTEFACT_DIR` unset, absolute
`PYTHONPATH=recipe:recipe/returnn:sisyphus`, all four `tests/test_w2vu2_*.py` including the slow ones:
- **33 passed, 15 skipped, 0 failed.** 14 of the skips are artefact tests. The 15th is the torchaudio
  importorskip, which passes with the scratch wheel (section 3).
- A first run with a relative `PYTHONPATH` failed one subprocess test with a `sisyphus` import error.
  That was a harness artefact, not a code fault.

## 7. Time on V100 and resume (before the addendum; superseded where the addendum says so)

**Time estimates.** These are not measurements. No GH200-to-V100 ratio for fairseq exists in this
campaign.
- **CTC student:** fp16 wav2vec2-large, compute-bound. I estimate 2.5-4.5x slower than GH200, i.e.
  **about 11-19 h** from the banked 4.25 h.
- **GAN seed:** 150k updates in 10.5 h is 0.25 s per update. That is mostly per-step overhead (the
  generator and discriminator are tiny) plus the validation LM scoring every 1000 updates. The
  fp32 GPU gap matters less, so I estimate 1-3x, i.e. **about 11-32 h** per seed. The 5 seeds run in
  parallel.
- The first hour's `hydra_train.log` gives the real rate.

**Time limits** (from `check_engine_limits` and the engine):
- Both trainings get **11.5 h** on the first leg. The 72 h floor applies only to ReturnnTrainingJob;
  `time` is capped at 168 h.
- On TIMEOUT, sacct reports `out_of_time`, and `update_engine_rqmt` doubles the time: **23 h, then
  46 h, then 92 h**. The explicit `-p gpu_32gb` is kept on every leg.
- `MAX_SUBMIT_RETRIES` = 3, and FairseqHydraTrainingJob has no `completed_fraction`. So at most
  4 submissions, **172.5 h** in total, then RETRY_ERROR.

**Resume.**
- `Task("run", resume="run")` reruns the same fairseq command. `restore_file` `checkpoint_last.pt`
  resolves against `checkpoint.save_dir`.
- The GAN saves every 1000 updates and the student every 2500, so up to that many updates are redone
  after each timeout.
- The student's DDP is fairseq's single-node spawn with a fresh port on each leg (no `distributed_port`
  or MASTER_* variables). The 4 GPUs come from one `--gres=gpu:4` allocation on a 16-GPU node, and
  `CUDA_VISIBLE_DEVICES` is kept by `DEFAULT_ENVIRONMENT_KEEP`.
- The worker clears `PYTHONPATH`, because it is not in KEEP. i6_core then sets
  `<fairseq_root>:<empty>` and the wrapper prepends its shim. `recipe/` holds no `fairseq`.

## Undetermined, and proposals

- **Resume versus a longer first leg.** A resumed run is not bit-reproducible: the data order changes
  and updates are redone. This is addressed by the addendum below, which gives a 72 h first leg on the
  coordinator's instruction.
- **The i6_core `plot` mini task.** It `eval`s the `[train]` and `[valid]` INFO lines, and it has
  never run on a GAN log (production used its own job). If it fails, the job is not finished and the
  graph stops at the selection.
- **The fairseq-shadow check** in the docstring (`<W2VU_PYTHON> -c 'import fairseq'` with the worker's
  `PYTHONPATH`) can run only after the env build and the clone job.
- **Order before launch:** apply the settings patch, build the env, run the extras script, then start
  the manager.

## Addendum (coordinator request): 72 h first leg for the fairseq trainings, and the `plot` task

### 72 h first leg, in `analysis_out/w2vu2_settings.patch` (still NOT applied)

**The change.** In `check_engine_limits`:
- A new flag `is_fairseq_train_run` is true when the task is `run` and the job is exactly
  `i6_core.fairseq.training.FairseqHydraTrainingJob`, the same exact-class test used for
  ReturnnTrainingJob.
- The existing floor now applies to both: `if is_train_run or is_fairseq_train_run:
  time = min(168, max(time, 72))`.
- Routing is unchanged. The ReturnnTrainingJob `-p GPU_ROUTE_TRAIN` branch still tests only
  `is_train_run`, and the fairseq trainings keep their explicit `-p gpu_32gb`. The docstring gained one
  sentence.
- The value is unhashed. It lives in `settings.py`, and FairseqHydraTrainingJob's hash does not include
  rqmt.

**Checks.**
- `patch -p1 --dry-run` is clean, and `settings.py` is byte-identical to the patch base.
- I dry-loaded both graphs with the patched copy through `SIS_GLOBAL_SETTINGS_FILE`, and diffed the
  dumps against the unpatched ones. The patched file was in effect: `gs.W2VU_PYTHON` was set even for
  P0.
  - **w2vu2:** 101 ids before and after, sets equal. Exactly 6 tasks changed: the run tasks of
    `SWK00drxbXSC` (the 4-GPU student) and of `fCcCEHaqxnyE`, `K0RJEx1UUwcb`, `qLeCqytt98iC`,
    `RzVQARNBfUb2` and `gY9AdQ2aYl3B` (the 5 GAN seeds). The only change is time 11.5 to 72; the
    partition stays gpu_32gb. The sbatch option becomes `--time=4320`.
  - **P0:** 164 ids before and after, sets equal. 0 changed tasks.
  - The comparison covers every non-mini task's final rqmt (cpu, mem, time, gpu, gpu_mem, partition).
    For flex24 tasks the partition was left out, because it depends on current availability.

**Resume under the floor** (from `engine.get_rqmt`; not exercised).
- The first leg is logged as 72. That differs from the recipe's 11.5, so `get_rqmt` takes the recipe
  value 11.5 and doubles it to 23 on TIMEOUT. The floor then lifts it back to 72.
- Every leg is therefore 72 h. With at most 4 submissions, the total is 288 h. This is the same
  behaviour as ReturnnTrainingJob.
- My V100 estimates (GAN 11-32 h, student 11-19 h) fit in one 72 h leg.

**Live managers.** Applying the patch while the P0 and P1 managers run affects only their workers and
submissions. The only jobs that change are FairseqHydraTrainingJob, which neither graph contains.

### i6_core `FairseqHydraTrainingJob.plot` (`recipe/i6_core/fairseq/training.py:331-450`)

**What it parses.**
- It walks `./outputs`, meaning every hydra run dir, including those of resumes.
- In each `hydra_train.log`, it takes every line that contains `[train][INFO]` or `[valid][INFO]`.
  These are fairseq's end-of-epoch and validation stats lines: `progress.print` under `rename_logger`
  with the tag `train` or the subset `valid`.
- It calls `eval(line[line.index("{"):])`, then `int(epoch_dict["epoch"])`, and `float(v)` for keys
  ending in `_loss` or `_accuracy`. Only `ValueError` from that block is caught. It reads `train_lr`
  if present.
- It then writes two SVGs with matplotlib (3.11.2 in the sae env). As a mini task it runs in the
  manager's sae python.
- The GAN's yaml has `log_format: json`, and fairseq's `format_stat` turns numbers into strings, so a
  stats line is a flat dict literal.
- The GAN logs `lr_generator` and `lr_discriminator`, not `lr`. So `train_lr` is absent and the lr
  plot is simply empty, which is not an error.

**Can it fail?**
- On the banked JUPITER logs, the port's wire report ran this parser over 984 GAN lines and 159 CTC
  lines with 0 failures (`exp_logs/SAE/reports/impl_gan_port_wire_2026-09-25.md`, "Checks run").
- The remaining failure modes, none observed:
  - A matching line that is truncated, for example cut by a TIMEOUT kill mid-write, raises
    `SyntaxError`. The 72 h leg makes kills rare.
  - A JSON `null`, `true` or `false` would raise `NameError` in `eval`.
  - A list-valued `_loss` stat would raise `TypeError`.
- None of these is caught.

**Is it on the path to the seed selection? Yes.**
- `Job.path_available()` returns `self._sis_finished()` (`sisyphus/job.py:1101-1109`), and a job is
  finished only when all its tasks are, including `plot`.
- `W2vu2GanSelectJob` reads every seed's `out_checkpoint_dir` and `out_fairseq_hydra_yaml`, and each
  seed's generator conversion reads `out_checkpoint_dir/checkpoint_best.pt`.
- So a failed `plot` on any seed blocks that seed's PER and the selection. Downstream of the
  selection, it blocks the pseudo-labels and the whole of 1d.
- A failed `plot` on the student blocks both 1d decodes. The checkpoints themselves would be intact.

**Making it harmless without a hash change.**
1. **Recommended: edit i6_core `plot`** in the shared checkout `recipe/i6_core`. Wrap the per-line
   parse, or the whole body, in `try/except Exception`: skip bad lines and log them.
   - It is hash-neutral, because `FairseqHydraTrainingJob.hash()` uses only command_line_args, the
     config, the python and the root, not code.
   - No other job in this setup uses FairseqHydraTrainingJob.
   - It is a recipe edit in a separate repo (`recipe/i6_core`, currently clean), so it needs the
     implementer and a review. Settings or config cannot reach the method: the worker loads the job
     from `job.save` and imports the unpatched class, and `worker_wrapper` cannot mask a task error,
     because the worker writes the error marker itself.
2. **Runbook, no code: if `plot` errors, repair it by hand.** After checking that `finished.run.1`
   exists and the checkpoints are complete, touch `finished.plot.1` in that job dir. The job then
   finishes and the graph proceeds. This is a manual write into a job dir, for the orchestrator to
   authorise.
3. **Not recommended:** override `tasks` on the job instance to drop `plot`, or subclass the job. The
   first depends on what `job.save` pickles; the second changes the hash.

None of these is implemented here. Option 1 is outside my assigned files.

**Checks for the addendum:** the dry loads and diffs above, and reading the code. No log was parsed on
i6, because none exists yet.
