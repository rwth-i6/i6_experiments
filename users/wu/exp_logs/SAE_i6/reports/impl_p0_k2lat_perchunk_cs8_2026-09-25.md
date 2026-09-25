# Implementation: chunk-8 variant of the P0 k2lat per-chunk live change, prepared, not applied (2026-09-25)

Role: implementer. S = `/u/hwu/setups/librispeech-960/2026-09-24-unsupervised`;
P = `S/recipe/i6_experiments/users/wu/experiments/unsupervised_asr`;
J = `S/work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl` (Slurm 4359832).
Nothing was launched, cancelled or installed. J's dir was only read (its config sha256 is still
`dff329b6...cdc`). No file of the chunk-16 change was modified: `config/sae_i6_p0.py` (sha `0f609369...`),
`analysis_out/k2lat_perchunk_returnn.config{,.orig,.diff}` and `k2lat_perchunk_install.sh` all have
their pre-task mtimes and hashes. Recipe code was not edited. Nothing was committed.

## Status: DONE_WITH_CONCERNS

The chunk-8 variant is J's chunk-16 config plus 3 lines: a blank line, a comment, and
`get_model = __import__("functools").partial(get_model, lexlat_k2_chunk_seqs=8)`.
All 164 P0 job ids are unchanged, including `jcKXbLMDk4hl`. With the pinned RETURNN and J's interpreter,
the config builds a model whose k2 runtime reports `chunk_seqs = 8`. That holds even when
`LEXLAT_K2_CHUNK_SEQS=16` is set. The runtime's spec is identical to probe B's route (the chunk-16 config
plus `LEXLAT_K2_CHUNK_SEQS=8`). The concerns are listed at the end.

## Files (all new; nothing existing was changed)

| file | delta |
|---|---|
| `S/analysis_out/k2lat_perchunk_cs8_sae_i6_p0.patch` | NOT applied. +15/-1 against the current `config/sae_i6_p0.py` (the chunk-16 version). It adds `K2LAT_CHUNK_SEQS = 8` and `_CHUNK_EPILOG`, sets `cfg.python_epilog = EPILOG + _CHUNK_EPILOG`, and adds one docstring paragraph. `patch --dry-run -p1 -d S` applies cleanly. |
| `S/analysis_out/k2lat_perchunk_returnn_cs8.config` | the new config for J. 12345 bytes, sha256 `b7a5f3af209d46740b2b43b2bbb8517f501a89b25cca84f09f819f65a344b8fa` |
| `S/analysis_out/k2lat_perchunk_returnn_cs8.config.orig` | a copy of J's current config (sha `dff329b6...cdc`, identical to the chunk-16 `.orig`) |
| `S/analysis_out/k2lat_perchunk_returnn_cs8.config.diff` | `diff -u` against `.orig`: +8 lines (the 5 chunk-16 epilog lines plus the 3 chunk lines) |
| `S/analysis_out/k2lat_perchunk_returnn_cs8.config.vs_cs16.diff` | `diff -u` against the chunk-16 config: +3 lines, nothing removed |
| `S/analysis_out/k2lat_perchunk_returnn_cs8_install.sh` | NOT run on J. A copy of `k2lat_perchunk_install.sh` with only the new file, its sha, the `_cs8` `.orig` and tmp names, and comments changed |
| `S/analysis_out/k2lat_perchunk_p0_jobids_after_cs8_2026-09-25.txt` | 164 job ids from the patched graph. Identical to `k2lat_perchunk_p0_jobids_after_2026-09-25.txt` |
| `S/analysis_out/k2lat_perchunk_cs8_dryrun_driver.py`, `..._path_check.py`, `..._test_k2lat.py` | the check scripts (read-only), copied for the reviewer |
| `S/analysis_out/k2lat_perchunk_cs8_checks_2026-09-25.log`, `k2lat_perchunk_test_k2lat_cs8_2026-09-25.log` | the check outputs |

The epilog text that the patch produces:
```
# P0 k2lat: sequences per k2 intersect call, the model kwarg lexlat_k2_chunk_seqs (unhashed epilog)
get_model = __import__("functools").partial(get_model, lexlat_k2_chunk_seqs=8)
```

## Why the epilog carries the kwarg, not the model's hashed kwargs

`config/sae_i6_p1_ladder.py` gets its chunk size through `ladder.rt_train_config`, which writes
`lexlat_k2_chunk_seqs` into the `get_model` partial. That partial is part of the hashed `python_prolog`
Collection (`training/config.py:359,433`; `training/jobs.py:30-38`). Writing 8 there would move J's hash.
So the patch uses the same kwarg, `SaeBlankfreeModelV1(lexlat_k2_chunk_seqs=...)`, and binds it in the
unhashed `python_epilog`. This is the mechanism the chunk-16 change already uses for `train_step`.
`functools.partial` of a partial merges the keywords, so the result is the same 27 hashed keywords plus
`lexlat_k2_chunk_seqs = 8` (28 in total, as the check shows). `__import__` adds no new global name to the
config namespace.

## Code path from the config to `rt_chunked_backward`

1. RETURNN execs the file. The epilog runs last, after `locals().update(**config)`, and rebinds
   `train_step` and then `get_model`.
2. `returnn/torch/engine.py:1242-1245` does `get_model_func = config.typed_value("get_model")`, then
   `get_model_func(epoch=, step=, device=, **fwd_compat)`. This calls
   `SaeBlankfreeModelV1(..., lexlat_k2_chunk_seqs=8)`.
3. `model/blankfree_model.py:543-547`: `beams["chunk_seqs"] = int(lexlat_k2_chunk_seqs)`, then
   `LexlatK2Runtime(..., **beams)`.
4. `model/lexlat_k2_train.py:286`: `chunk_seqs=K.resolve_chunk_seqs(chunk_seqs)`.
   `model/lexlat_k2.py:917-925`: an explicit value comes before `$LEXLAT_K2_CHUNK_SEQS`, which comes
   before `CHUNK_SEQS=16`. So `spec.chunk_seqs = 8`, and the model prints `k2 lexicon term: {... 'chunk_seqs': 8}`.
5. RETURNN's step calls the config's `train_step`, which is `rt_chunked_backward.train_step`
   (`rt_chunked_backward.py:317-327`). That calls `install_chunked_backward(model.lexlat_k2)`, which
   swaps the class in place and keeps the spec. It prints
   `rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = 8)`.
6. `ChunkedBackwardLexlatK2Runtime.log_z_hlg_chunked_backward` loops over
   `K.chunk_bounds(b, int(self.spec.chunk_seqs))` (`rt_chunked_backward.py:252`).
   The stability read uses its own `STABILITY_CHUNK_SEQS = 1` and is unaffected.

**Check (read-only, `analysis_out/k2lat_perchunk_cs8_path_check.py`).** It ran under `env -i` with the sae
python, the pinned RETURNN (`recipe/returnn`) and `CUDA_VISIBLE_DEVICES=`. It loads the file with
`returnn.config.Config.load_file`, makes RETURNN's `get_model` call (epoch 8), and follows steps 3-6.
Output is in `k2lat_perchunk_cs8_checks_2026-09-25.log`:

| config | env | get_model kwarg | spec.chunk_seqs | install print | chunk calls for 128 seqs |
|---|---|---|---|---|---|
| cs8 | unset | 8 | 8 | `chunk_seqs = 8` | 16 |
| cs8 | `LEXLAT_K2_CHUNK_SEQS=16` | 8 | 8 (the kwarg wins) | `chunk_seqs = 8` | 16 |
| chunk-16 | unset | absent | 16 | `chunk_seqs = 16` | 8 |
| chunk-16 | `LEXLAT_K2_CHUNK_SEQS=8` (probe B's route) | absent | 8 | `chunk_seqs = 8` | 16 |

The full `k2 lexicon term: {...}` descriptions of cs8 and chunk-16 differ only in `'chunk_seqs'`. The
description from cs8 with the env unset is byte-identical to the one from probe B's route. So probe B's
result transfers to this config. In every case, `train_step` resolves to `reverse_model.rt_chunked_backward`.
This shows that the file loads and builds the intended runtime. It does not show that J uses it; J's own
`installed (chunk_seqs = 8)` line after the restart does that.

## Job hash with the patched `sae_i6_p0.py`

Method: `sisyphus/sis --log_level 40 console --skip_config -s -c exec(<driver>)` from S, under the sae
python. The driver (`k2lat_perchunk_cs8_dryrun_driver.py`) imports a scratch copy of the file as the
module `config.sae_i6_p0`, then calls sisyphus's own `config_manager.load_config_file("config/sae_i6_p0.py")`.
This finds the module in `sys.modules` and runs its `py()`, which is the manager's load path. The driver
then lists `tk.sis_graph.jobs(update_graph=False)` and calls `returnn_config.write()`, the call
create_files makes, including black, for every ReturnnTrainingJob. No manager was started.

- Control (the current, unpatched file): 164 ids, identical to the chunk-16 after-list. The k2lat
  config is byte-identical to `analysis_out/k2lat_perchunk_returnn.config` (sha `36e7032e...`), so the
  driver reproduces the chunk-16 build.
- Patched file: 164 ids, identical to the control; `jcKXbLMDk4hl` is present. The k2lat epilog
  length is 408 (229 in chunk-16) and `python_epilog_hash` is `''`, unchanged; the patch's own assert
  also checks this. The five other training configs (ctrl_20, ctrl_20_s1, ctrl_20_rc and both
  supervised) are byte-identical between control and patched. The written k2lat config is the
  `_cs8.config` above.

## Exactness at chunk 8 (a scratch copy of the k2lat test; recipe tests untouched)

`analysis_out/k2lat_perchunk_cs8_test_k2lat.py` is `tests/test_rt_chunked_backward_k2lat.py` with 5
lines changed: `chunk_seqs` 16 -> 8 for both held and per-chunk paths, and the chunk-count and
default asserts adapted. It uses 40 sequences, so 5 chunks. Result: **6 passed**. At sub-epochs
8/9/10/11 and in the no_grad pass, max |dlogZ| = |dmonitor| = |dgrad log_q| = |dgrad param| = 0.000e+00.
The term is 0.0860062307299628, the same value as the chunk-16 run's. The existing recipe test covers
chunk sizes 1/2/3/16, not 8. The same fixture limits apply as in the chunk-16 report: rung 3000 prunes
nothing there (stability 0.0).

## Install and restart

These are identical to the chunk-16 report, section 3 (`reports/impl_p0_k2lat_perchunk_2026-09-25.md`),
with `bash S/analysis_out/k2lat_perchunk_returnn_cs8_install.sh` in step 3. The expected log lines
change to `'chunk_seqs': 8` and `installed (chunk_seqs = 8)`. On a mock copy of J (scratch), the script:
installs from the original and prints the 8-line diff and `INSTALLED`; says `already installed` on a
re-run; refuses with `ABORT: live config is neither ...` when the chunk-16 config is live; and refuses
when `epoch.008.pt` exists. The `_cs8.orig` was left unchanged.

## Concerns / undetermined

1. **16 -> 8 after a chunk-16 install is not covered.** The script accepts only J's original config
   (`dff329b6`), exactly as the chunk-16 script does, so it refuses if the chunk-16 config is already
   live. That happens if chunk 16 is installed first and chunk 8 is chosen later. Whether to allow that
   transition (a second accepted sha) is the orchestrator's decision. I did not add it, because it would
   make this variant differ in more than the chunk size.
2. **The delta is in the epilog, not in the prolog kwargs as in P1.** It is the same kwarg and the same
   model code path, but it is visible only as the epilog line in `returnn.config`. Any reader that takes
   the chunk size from the hashed `get_model` args (`training/jobs.py:get_model_args`) will not see it.
   J's log line is the record.
3. If the patch is applied and the P0 manager is restarted, the manager's in-memory config matches the
   installed file. Without a restart, the chunk-16 report's concern 1 (a `job.save` rewrite with the old
   config object) applies unchanged.
4. The patch is made against the chunk-16 `sae_i6_p0.py`. If that file changes during its review, the
   patch must be regenerated.
5. Before probe B's result is used for the switch, probe B's own `installed (chunk_seqs = N)` line in
   its `slurm-*.out` should confirm 8 (review SHOULD 2).
6. Disclosure for the phase file: the chunk size, the switch sub-epoch and the restart time are a
   runtime change of P0. They are not a change of arm constants.
