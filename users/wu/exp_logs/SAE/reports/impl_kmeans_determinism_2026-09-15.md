# Implementer: k-means determinism fix (1) + read-path net_args check (2), 2026-09-15

Follow-ups from `reports/review_companion_content_2026-09-15.md` sections 7 and 8.2.
Nothing launched, no manager touched, `settings.py` untouched.

Commit: **53d691a** on `haotian_modality_matching_jupiter` in `recipe/2025-10-speech-llm`
(two files staged by explicit path, not pushed). The two unrelated dirty files
(`config_sae_1g_v1.py` modified, `config_sae_3e1_d6_swap_cont_v1.py` untracked) were left alone.

## (1) FIX -- `sae/emc/mfcc_codes.py` + `test_mfcc_codes.py`

### Diagnosis (reproduced)

The suite as committed at fb97abb PASSES on an idle process and FAILS intermittently -- which is
why the two implementer reports and the review disagree. Driving `fit_kmeans(x, num_clusters=8,
seed=42)` 20 times in one process on the test's own fixture (1200 x 6):

| ambient threads | distinct results in 20 fits |
|---|---|
| `OMP_NUM_THREADS=1` | 1 |
| `OMP_NUM_THREADS=4` | 1 |
| `OMP_NUM_THREADS=8` | 2 |
| `OMP_NUM_THREADS=72` | 7 |
| unset (72 cores visible) | 5 |

The centroids had the same md5 in every one of those runs; what moved was the **inertia**
(6373.7134 ... 6373.7163), so the failing half of `assert np.array_equal(c1, c2) and i1 == i2` was
`i1 == i2`. Cause: sklearn reduces the inertia with an OpenMP float32 reduction whose summation
order follows the number of threads the runtime actually hands out, and that number varies with
machine load. It is not cosmetic: `MiniBatchKMeans` selects the best of `n_init = 3` BY that
inertia, so a wobble there can pick a different init and move the codebook itself. A fixed
`random_state` cannot prevent any of this.

### The fix

* New `deterministic_threads(threads=KMEANS_FIT_THREADS)` (`mfcc_codes.py`), a `threadpoolctl`
  `threadpool_limits` context manager; `KMEANS_FIT_THREADS = 1`.
* `fit_kmeans` builds `np.random.RandomState(int(seed))` itself (no stream carried in from an
  earlier call) and runs `km.fit(x)` inside that context.
* `assign_codes` runs its chunked argmin inside the same context -- the `block @ c.T` gemm is the
  only thread-sensitive step there, and a near-tie between two centroids must not be decided by
  how many threads the BLAS got. This is what makes the *codes*, not just the centroids,
  reproducible.
* 1, and not the job's `cpu_rqmt`: a fixed value above 1 is still at the mercy of the runtime
  handing out fewer threads (see the table: 8 already wobbles). Only a pin is environment-free.
* The fit conditions are now recorded where a reader of the codebook can see them:
  `params["fit_threads"]` in `codebook.npz` and the `threads=1` field of the `MiniBatchKMeans(...)`
  line in `kmeans.stats.txt`. The module docstring carries a REPRODUCIBILITY paragraph
  (pre-registration lives with the producing code).

### Cost, measured at the production shape (1,000,000 x 39, K = 64, n_init 3, max_iter 100)

| variant, 1 thread | wall time | bit-identical over 2 reps |
|---|---|---|
| `MiniBatchKMeans(batch 4096, n_init 3, max_iter 100)` -- **kept** | **0.9 s** | yes |
| full-batch `KMeans(n_init 3, max_iter 100)` | 31 s | yes |
| `assign_codes`, 2 M frames, default threads | 0.51 s | -- |
| `assign_codes`, 2 M frames, 1 thread | **0.32 s** | -- |

Against `MfccKMeansJob`'s 4 h / 8 CPU / 32 GB request both fits are free, and the pinned
assignment is *faster* than the threaded one at this shape (thread-pool overhead dominates a
39 x 64 gemm). Full-batch KMeans also reached a lower inertia on the synthetic probe
(8.43e6 vs 8.74e6), i.e. it is a strictly affordable upgrade -- but it is a **calibration**
decision, not a determinism fix, and the MiniBatch estimator is the setup's own convention
(`sae/build_units.py:173-174`) that the module docstring traces to. I kept MiniBatchKMeans and
left the option documented; switching it is the planner's call (nothing downstream has run, so it
costs nothing but the codebook itself).

### Test changes (`test_mfcc_codes.py`)

`test_kmeans_is_deterministic_at_a_fixed_seed_and_assign_matches_sklearn` now asserts, exactly:
(a) a fit at another seed *in between* does not move the second seed-42 fit (centroids, inertia
and `assign_codes` output); (b) the same under an ambient `threadpool_limits` of 1, 8 and 72 --
the inertia compared exactly, it is the sensitive quantity; (c) the same at the production
dimension and K (20,000 x 39, K = 64), which is the shape where the 3-way `n_init` selection the
job runs actually happens; the sklearn-own-fit oracle now runs under the same pin so its inertia
is comparable, and its inertia is compared too. The test prints
`kmeans seed-42 digest: ...` so two *processes* can be diffed. `test_constants` pins
`KMEANS_FIT_THREADS == 1`.

### Checks run

| check | result |
|---|---|
| whole suite twice in ONE process (driver calling all 8 tests twice) | exit **0**, both passes, identical digest |
| whole suite in a second process (`python -m ...test_mfcc_codes`) | exit **0** |
| whole suite in a third process with `OMP_NUM_THREADS=3` | exit **0** |
| digest across all 4 suite runs / 3 processes | **1 unique line**: `centroids b0c5262c386511f80b24d8966f3fc4a5  inertia 6373.70751953125  codes 3f611f48d2ece410d48a2c4a46d800cb | K64 centroids fd74e264e62cdbb37342e065cf56c425  inertia 662873.75` |
| `test_sae_emc` (`prefix_lm.model.train_steps`) | exit **0**, 9 PASS lines |
| `test_emc_train_jobs` (`sae.emc`) | exit **0**, 10 ok lines |

### Hash census -- NOTHING MOVED

`scripts/sae_4a_cons_census.py` (the fixed one), before = `git archive fb97abb` via
`SAE_CENSUS_SRC`, after = the live tree; `# src:` asserted on both sides.

| graph | jobs | verdict |
|---|---|---|
| s3 | 98 | identical id for id |
| phase | 1021 | identical id for id |
| rate (`ACTIVE_ARMS = ('lam3',)`) | 91 | identical id for id, `lam3 = ReturnnTrainingJob.DF6blPpto23t` |

Plus an all-arms build (same override machinery, `ACTIVE_ARMS` set to all six tags in a scratch
process, 415 jobs): the whole id list is identical before/after, and specifically
`MfccKMeansJob.wejLUflqBxxo`, `MfccCodesHdfJob.ovaQXsbT8s3E`, `MfccFeatureJob.Y8INRgLo8Avo`,
`lam3_ct ReturnnTrainingJob.DhsiFoHwBP5l`, and `lam1 WCh1fMyr88yu / lam3 DF6blPpto23t /
lam10 axf11lrsap5i / lam3_ent_a TWh8GPa8ItGi / lam3_ent_b HdP6u4MSWRpk` all unmoved.
**No new ids to report: the fix did not move the k-means job hash.** (Expected: sisyphus hashes
the constructor call, and the job signature, the frame grid and the feature definition are
untouched; the change is inside `run` / the helpers.) No `Mfcc*Job` dir exists on disk yet, so the
codebook has never been produced -- the old, non-reproducible fit was never banked.

## (2) CHECK ONLY -- does each per-sub-epoch read survive a `content_head.*` checkpoint?

Read-only; `emc_train_jobs.py` / `train_steps` / `definitions` are being edited by another
implementer, so the line numbers are those of the tree at 53d691a.

**Verdict: all five reads are fine for `lam3_ct`; no fix is needed.**

1. **`eval_jobs.posterior_dump` -- receives `net_args`, verified end to end.**
   `config_sae_4a_s3b_rate_v1.py:296-300` builds
   `content_net_args = {**init_jobs.RECOGNIZER_NET_ARGS, "content_k": CONTENT_K, "content_layer":
   content["content_layer"]}` inside the `lam3_ct` branch and passes it at `:364`
   (`net_args=content_net_args`; `None` at `:282` for every other arm).
   `emc_train_jobs.py:1540` declares the parameter, `:1565-1573` forwards it verbatim into
   `eval_jobs.posterior_dump(..., net_args=net_args)` (`:1571`).
   `eval_jobs.py:398` -> `:409` -> `build_posterior_forward_config` `:355-388`, where `:368`
   resolves `net_args = dict(net_args or RECOGNIZER_NET_ARGS)` and `:387` hands it to
   `_forward_serializer`, which binds it at `:331` as `PartialImport(..., code_object_path=
   "speech_llm.sae.emc.init_jobs.get_model", hashed_arguments=net_args)`.
   `init_jobs.get_model:458-476` keeps every kwarg `ConvRecognizer.__init__` declares --
   `content_k` and `content_layer` ARE declared (`recognizer.py:151-152`) so they are not dropped
   by the shim's filter -- and returns a recognizer with `content_head` (`recognizer.py:208`).
   The theta slice therefore loads into a model of exactly its own key set.
   *Side finding, worth knowing:* even without `net_args` the read would **not** have died.
   RETURNN loads the forward checkpoint with `load_state_dict(..., strict=False)`
   (`returnn/torch/engine.py:1017`) and raises only on MISSING keys (`:1186-1196`); unexpected
   keys are printed as a `Note` at `log.v4` (`:1022-1027`). So the review's failure path
   ("every sub-epoch read of lam3_ct would die on content_head.* as unexpected keys") would in
   fact have been a silent note, not a crash -- the strict-both-ways loader is
   `definitions.sae_emc._load_state` on the TRAINING side, not the read side. Passing `net_args`
   is still the right thing: it keeps the read's model identical to the trained one and keeps the
   unused head out of "silently ignored" territory.
2. **`eval_jobs.GreedyPerJob` (`eval_jobs.py:531`)** -- inputs are `posteriors_hdf` + `gold` +
   `split`. No checkpoint, no model construction. Unaffected.
3. **`emc_train_jobs.DecodeStatsJob` (`emc_train_jobs.py:1056`)** -- inputs are `posteriors_hdfs`
   + `split` + rate constants. No checkpoint. Unaffected.
4. **`s3_jobs.S3DerangementGapJob` (`s3_jobs.py:105`, constructed at
   `config_sae_4a_s3b_rate_v1.py:397-404`)** -- inputs are the `reverse.` slice, `hyps_json` (this
   checkpoint's own decodes, i.e. `GreedyPerJob.out_hyps`), `gold_json` (tag list only),
   `units_store`, `eta_npz`. `s3_jobs.py:169-175` builds `SegmentalReverseModel` from
   `reverse_kwargs` and loads the phi slice; the recognizer is never constructed, and
   `content_head.*` lives under `recognizer.`. Unaffected.
5. **`selection.UnsupervisedCheckpointSelectionJob` (`selection.py:192`,
   config `:410-416`)** -- inputs are the per-epoch posterior HDFs, the KenLM binary, the
   `learning_rates` file and `cv_loss_key`. No checkpoint. Unaffected.

Also on the path, and fine: **`emc_train_jobs.theta_checkpoint:1043-1051` /
`ExtractSubmoduleCheckpointJob:987-1042`** keeps every key under `prefix="recognizer."`
(`:1019`) and asserts only `len(picked) >= expect_min_keys` (`:1020-1023`), so `content_head.*`
survives into the theta slice, which is exactly what `net_args` then expects.
`test_sae_emc` (run above, exit 0) exercises both directions of that load, including
"a bare recognizer refuses it".

## Undetermined / left to the planner

* Whether to switch `MfccKMeansJob` to full-batch `KMeans` (31 s at the production shape, lower
  inertia). A calibration choice, not mine; the codebook has never been produced, so it is free
  now and expensive later.
* The `lam3_ct` posterior dumps hash differently from `lam3`'s by construction (`net_args` is a
  hashed argument of the serializer). That is intended and costs nothing: those reads have never
  run. `lam3`'s reads are unmoved -- confirmed by the rate census.
