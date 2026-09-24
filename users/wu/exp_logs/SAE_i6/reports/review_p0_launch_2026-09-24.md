# Code review: P0 launch on i6 (screen + full config), 2026-09-24

Status: DONE_WITH_CONCERNS.
- Screen launch (`config/sae_i6_p0_screen.py`, 75 jobs): PASS_WITH_NOTES. Nothing blocks it.
- Full release (`config/sae_i6_p0.py`, 139 jobs): FAIL until issue 1 is fixed. The fix is a single
  env install and moves no hash.

The launch changed while I was reviewing it: absolute IMPORT_PATHS, `FFMPEG_PIN_ACCEPT`, and the
amendments for ctrl_20_s1, the p0 read, report-only VAD counts and the staged configs. This verdict
covers the tree as of 16:50. I rebuilt both configs with the label set, under the sae python with
its bin/ first on PATH:
- screen: 75 jobs, full: 139 jobs. Every screen id is in the full graph.
- The ids match `exec_audio_label` (ctrl_20 `GiT88bxzoZbZ`, VAD `RLrgIh6lFv9m`, pin check `wYIhhTGwejQe`).
- The rqmt and partition of every task are identical to the strict (no-label) graph.

## Issues

1. **matplotlib is missing from the sae env.** Every ReturnnTrainingJob fails in its `plot` task.
   - Where: `recipe/i6_core/returnn/training.py:374` (the task), `:439` (`import matplotlib`).
   - The task is a mini_task, so it runs on the desktop LocalEngine under the sae python and raises
     ModuleNotFoundError. I checked: `import matplotlib` fails in the env.
   - Consequence: the job never gets its finished marker, so `out_learning_rates` never becomes
     available (`training.py:352`). As a result:
     - p0's `GetBestPtCheckpointJob` never runs (`reverse_model/p0.py:360`), and neither do
       `p0_per_read` and `selected_epoch` (`config/supervised_init.py`). The G0.R3 Tier-A read
       never materialises.
     - The gold-phi `learning_rates` output (the R3 Tier-B read) is never registered.
     - All 5 trainings end in ERROR.
   - Checkpoint reads still run, because `path_available` accepts existing files.
   - Screen: ctrl_20 turns ERROR only after sub-epoch 20. The screen reads are unaffected.
   - Fix: `mamba install -p /work/asr4/hwu/conda/envs/sae -c conda-forge --override-channels
     matplotlib-base`, and add it to `env/environment.yml`.
     - The dry run installs 20 new packages and upgrades or downgrades nothing.
     - No hash moves. After the fact, clearing the error re-runs only `plot`.

2. **The VAD count check under the label (now report-only; fixed in the tree).**
   - The original `inputs.py:139` passed `BANKED_VAD_COUNTS` unconditionally, so the VAD job would
     have raised under the label.
   - My probe, on 120 dev-other utterances: a different ffmpeg build (median SNR 69 dB) changed the
     kept count of 2 utterances, and -100 dB noise changed 4. An exact kept_frames total is
     therefore unattainable on other audio.
   - Remaining gap: report-only mode also silences `utterances` and `original_frames`, and those
     do not depend on the audio. The Vorbis decode length equals the FLAC length on 24 of 24 files.
     G0.R0 should read those two as exact and apply the 0.5 % only to kept_frames.

3. **Relative IMPORT_PATHS broke lazy imports inside tasks (fixed).**
   - The first pin check failed with `No module named 'i6_core'`: RecipeFinder resolves paths
     against `<job>/work`.
   - The absolute IMPORT_PATHS in `settings.py` keep the same meaning (`sisyphus/loader.py:247-268`).
     The rerun reached ffmpeg.

## Notes

4. **Never run two managers at once.** At the full release, stop the screen manager first, then
   start the full config. The 75 shared jobs would otherwise be submitted twice, including the
   desktop LocalEngine tasks.

5. **Freeze the runtime code.** RETURNN imports the package's `model/`, `training/` and
   `analysis/` code live from `recipe/`, unhashed (`training/config.py:365`,
   `make_local_package_copy=False`).
   - Other sessions are editing this checkout.
   - A resume after a TIMEOUT, which is likely on the L40S, would silently run any edit.
   - Commit before launch, and do not edit these modules until the P0 trainings end.

6. **Resume, checked statically only.**
   - The patch applies cleanly to 00171dfe (`git apply --check`).
   - RETURNN reseeds from (epoch, step) (`engine.py:1233-1235`).
   - The agg EMA buffers are in the state_dict (`model/agg.py:185-187`).
   - `keep_last_n: 1` plus keep 1/4/10/20 cover the resume and every read.
   - Sisyphus doubles the time after a TIMEOUT.
   - Not exercised at run time. The time rqmt is not hashed, so raising it in `check_engine_limits`
     for ReturnnTrainingJob would avoid the dependence.

7. **GPU memory.**
   - The reference whole step of k2 at rung 3000 was 31.77 GiB, over 9 sub-epoch-10 batches that
     included the longest-T batch (`SAE_i6_ref_lexicon.md:244`). That leaves about 12 GiB of margin
     on the L40S.
   - Every batch is bounded to B ≤ 128 and B·T ≤ 88,000 (`model/lattice.py:160-190`), and most
     train-clean-100 batches sit at B ≈ 128. So the reference probe covers the worst DP shape.
   - The ctrl screen cannot see the k2 on-set peak (sub-epoch 8). A probe before launch is not
     possible without a trained posterior.
   - Value-identical fallback: `LEXLAT_K2_CHUNK_SEQS` (`model/lexlat_k2.py:906-922`). It reaches
     RETURNN only through `DEFAULT_ENVIRONMENT_SET`, because CLEANUP_ENVIRONMENT strips it.

8. **black.** `ReturnnConfig` resolves black at graph build (`i6_core/returnn/config.py:149`), and
   the desktop's default PATH has no black. So the manager must be started with sae/bin first on
   PATH. The launch command has it; the watcher's `SIS_LAUNCHER` does not.

## OK
- **Routing.**
  - GPU jobs go to gpu_24gb (A10 or RTX 3090, sm_86) or gpu_48gb (L40S). Never gpu_11gb or A100.
  - The HLG job (200 GB) goes to gpu_32gb (7-day limit, per sinfo).
  - CPU jobs of 96 GB or less go to cpu_modern.
  - Every mini_task uses the default 1 CPU / 1 GB.
  - The 64 GB forwards cannot land on the 64000 MB RTX 3090 node, but they fit the A10 nodes.
- **Interpreters.**
  - All RETURNN jobs, BlissToOggZip, the HLG child, g2p and KenLM resolve to the sae env or the
    configured binaries.
  - Nothing falls back to `RETURNN_PYTHON_EXE`.
  - ffmpeg and KenLM carry RPATH, which wins over the desktop's CUDA-9.1 `LD_LIBRARY_PATH`.
- **Paths and flags.**
  - No `HF_HUB_OFFLINE` flag is set anywhere.
  - `/e/project` appears only in the default fallbacks, which settings.py overrides.
  - `--scratch-prefix` appears only in `gpu-check`, which the graph never runs.
- **Pin check.** It raises before the wrapper is written (`data/ffmpeg_pin.py:278-289`), so no
  encode can run.
  - My probe matched 0 of 40 recordings, with the i6 FLAC byte-identical to openslr's; the
    executor's run failed 2850/2864.
  - The label is hashed, and every downstream id moved.
- **Graph contents.** The graph holds ctrl_20, ctrl_20_s1 (seeds as in `SAE_4A_prepro.md:13`),
  k2lat_20_ma3000, gold phi and p0, with their reads. There are no x60, off4 or k2_word_lm jobs.
- **Hashes.** In the strict graph, the 111 earlier job ids are unchanged; only the P0-only JsRows
  id moved.

## Other differences from the reference, beyond GPU nondeterminism
- The audio generation label.
- The k-means codebook is refit with MKL on x86_64.
- CPU jobs run with OMP/MKL_NUM_THREADS=2.
- Worker-side `SimpleHDFWriter` and `pformat` come from `recipe/returnn` 5f752be4, not from the
  pinned clone.
- g2p training (i6_core defaults, 60 h rqmt) lies on ctrl_20's critical path.

## Side effects of this review
None in SETUP apart from this file and `__pycache__`. The probes ran in the scratchpad (console
graph dumps; encode/decode of 40 dev-other files; the openslr dev-other subset).
