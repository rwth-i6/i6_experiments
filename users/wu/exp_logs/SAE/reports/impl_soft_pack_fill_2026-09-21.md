# SAE 4a prior -- filling `LAM_SF_01` and building the four-arm pack

Dispatch: set `LAM_SF_01` from the finished `SfLamProbeJob.OHrcN9pEuXni`, change nothing else,
load the pack through `config/sae_4a_soft_pack.py`, census, commit the one file.  Branch
`haotian_modality_matching_jupiter`, parent `ff4f005` (the round-2 sf-arm commit).

## The change -- one constant, one comment

`recipe/2025-10-speech-llm/src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_soft_pack_v1.py`

    LAM_SF_01: Optional[float] = 0.0453938      # was None

The `#:` block above it now records the read the way `LAM_01` records the soft probe's: the source
job id, the quoted summary line **"median ratio |grad sf| / |grad l_tau| = 2.20294"**, the operating
point (three batches at ctrl_20 ep4 seed 0), the target row that produced the number
(`0.1 x l_tau -> lam_sf = 0.0453938`) and the 0.3 x row (0.136181) that no arm takes.  The
override's reason for a separate probe is kept.  Diff: 10 insertions, 6 deletions, all inside that
comment block and its constant.  Nothing else in the file, and no other file, is touched.

Provenance of the number, verbatim from
`work/speech_llm/sae/emc/sf_scorer_jobs/SfLamProbeJob.OHrcN9pEuXni/output/summary.txt`:

| target | lam_sf |
|---|---|
| 0.1 x l_tau | 0.0453938 |
| 0.3 x l_tau | 0.136181 |

with per-batch ratios 2.36821 / 2.18608 / 2.20294 (median 2.20294) and a GPU peak of 23.64 GiB.

## The assertion path

`_lam` was exercised directly (scratch script, file untouched):

* `_lam("soft_20") = _lam("soft_20_s1") = _lam("softshuf_20") = 0.326639` (key `LAM_01`),
  `_lam("sf_20") = 0.0453938` (key `LAM_SF_01`);
* with `LAM_SF_01` set back to `None` **in memory only**, the refusal still fires by name:
  `sf_20 takes LAM_SF_01, which is still None: run its gradient-norm probe
  (config/sae_4a_sf_lam_probe.py), read its summary.txt and fill LAM_SF_01 in ...`.

So the gate that blocked the build is intact and it is the filled value, not a weakened assert,
that lets the pack build.

## The pack

Loaded exactly as the shim does (`config_sae_4a_soft_pack_v1.py`, one graph per process, `tk`
after the setup-dir `chdir` so `settings.py` is the setup's).

* **`PackedBlankfreeTrainJob.MXKoywbfon8O`** -- the one training job, `speech_llm/sae/emc/blankfree_pack_jobs/`.
* **196 jobs** in the graph (the same count the round-2 placeholder load reported; the ids that
  depend on `LAM_SF_01` are now the real ones).
* arm names: the four expected -- `soft_20`, `soft_20_s1`, `sf_20`, `softshuf_20`.
  **Slot order is SORTED, not the `ARMS` declaration order**: `blankfree_pack_jobs` line 136 says
  the name "sorted, fixes the GPU the arm runs on", and `job.arms` is built from `sorted(arms)`, so
  the job's slot order is `['sf_20', 'soft_20', 'soft_20_s1', 'softshuf_20']`.  The dispatch's
  expected order is the config's `ARMS` dict order.  Same four arms, different listing; nothing to
  fix, but the GPU-slot order is the sorted one.

The weight reaches the arm: the written per-arm config carries `sf_lam = 0.0453938` plus
`sf_scorer / sf_num_samples / sf_sample_seed / sf_unigram_npz` on `sf_20` and no `soft_*` key; the
three soft arms carry `soft_lam = 0.326639` (and `soft_derangement_seed` on `softshuf_20`) and no
`sf_*` key.

## Census -- nothing banked moved

| graph | jobs | expected | result |
|---|---|---|---|
| prepro pack | 133 | 133 | unchanged |
| budget pack | 775 | 775 | unchanged |
| lexlat probes | 3 | 3 | unchanged |
| falsifier (ii) `sf_probe` | 3 | 3 | `d1NoUQJ2EXN5`, `HVHIlaUkIkVi`, `Y81PrZ6fWWKu` -- unchanged |
| soft lam probe | 1 | -- | `SoftLamProbeJob.wAJQ26T7iZzX` **unmoved** |
| sf lam probe | 1 | -- | `SfLamProbeJob.OHrcN9pEuXni` **unmoved** |

## Commit

`ecf846c` on `haotian_modality_matching_jupiter`, parent `ff4f005`, one staged path (the config).
Not pushed.  Three unrelated files were already modified in that checkout by someone else
(`config_sae_1g_v1.py`, `lexlat_k2.py`, `lexlat_k2_jobs.py`) plus one untracked file; none was
staged and none was touched.

## What this is and is not

Nothing has run.  A graph load and a census prove the constant is accepted, reaches the `sf_20`
arm's written config and moves no banked hash; they are not a result.  Nothing was launched.
