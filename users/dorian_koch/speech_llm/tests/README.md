# Standalone checks (login-node, no GPU, no Sisyphus manager)

Cheap guards for the classes of bug that are invisible in a loss curve — wrong channel/role mapping,
a silently-unmapped model tag, a node exclusion that never reaches SLURM, a config value that loads
as a string. Each exercises the **real** code (importing it, or exec'ing the real source) rather than
reimplementing the logic, so it fails when the thing it guards actually breaks.

Run from the setup root. `CUDA_HOME=/usr` is only needed because `settings.py` asserts it at import:

```bash
cd /home/tt201262/setups/2026-01-speech-llm
T=recipe/i6_experiments/users/dorian_koch/speech_llm/tests
for c in $T/check_*.py; do
  printf '%-32s ' "$(basename $c)"
  CUDA_HOME=/usr .venv/bin/python "$c" >/dev/null 2>&1 && echo PASS || echo FAIL
done
```

| Script | Guards |
|---|---|
| `check_fisher_windows.py` | `FisherToMoshiTrainData`: Fisher markup cleaning, agent-only alignments, in-window/ordered timestamps, role-keyed A/B→user/agent channel mapping, arrow row layout. Decisive check: **assistant RMS in-word vs out-of-word must be >2×** — if the channel mapping were inverted the words would land on the wrong track and this collapses to ~1. |
| `check_model_provenance.py` | Every `fdb_benchmark_py` tag in the recipe has an `FDB_MODEL_ORIGIN` entry, the map is well-formed, and an unknown tag is **rejected** rather than rendered unmarked. |
| `check_partition_routing.py` | Cluster-agnostic requirements (`requires`, `gpu_mem_gb`) route to a partition that actually provides them; `BROKEN_NODES` reaches `sbatch -x` on **every** job shape (the old `default_rqmt` mechanism silently dropped it for jobs setting their own `sbatch_args`); unsatisfiable requirements raise instead of mis-routing. |
| `check_mixed_loader.py` | Weighted on-the-fly corpus mixing: realised proportions match the weights, DDP shards are disjoint, a small corpus cycles rather than starving, and corpora whose HF feature types differ (`Audio()` vs plain struct) still decode identically from arrow. |
| `check_train_data_common.py` | The shared training-data primitives are **numerically identical** to the three per-architecture copies they replaced. Keeps the pre-consolidation implementations inlined as `_legacy_*` and diffs against them, so deliberately changing shared behaviour forces updating the legacy copy in the same commit rather than absorbing the change silently. Also asserts each architecture module still exports what its launcher imports. |
| `check_text_stream.py` | the three text-stream conventions, none of which a loss curve can show. **EPAD**: our interleaver never emitted Moshi's end-of-padding token (id 0) -- the one the paper says *"immediately triggers the start of speech"* -- so at the frame where it fires our target said PAD. **Clamp**: no word may be placed before the speaker makes a sound (59.2% of QA rows were a median 62 frames early). **Floor**: a word belongs in the frame that CONTAINS its onset, not the nearest one; rounding up puts the token one frame after its own audio for ~half of all words. The EPAD and floor halves are checked **against the upstream `Interleaver` itself, frame for frame**, because divergence from the reference *was* the bug -- and the rounding mode is asserted to FAIL the off-grid comparison, so the stronger test cannot pass vacuously. |
| `check_common_helpers.py` | `common.py`'s shared runtime plumbing plus the chatterbox worker's reproducibility-critical samplers: `pick_free_port` returns a genuinely bindable port, `run_worker_script` stringifies argv and honours `with_hf_home=False`, `available_speakers` is **sorted** (the seeded speaker draw is otherwise filesystem-order-dependent and the corpus stops being reproducible), and `silence_length_sampler` really resamples into its window. The worker imports torch/chatterbox at module level, so its real source is exec'd with those stubbed rather than copied. |
| `check_probe_rank_parallel.py` | The in-loop knowledge probe split across DDP ranks (G3) scores exactly what one rank would. `select_probe_entries` partitions the `max_n`-capped set by stride and tags every row with its **global** index (the "first N questions kept as audio" set is decided from that index alone); `merge_probe_results` over striped parts equals the single-worker computation -- summary and coherence **recomputed over the union** (a mean of per-rank means weights small shards wrongly), n-weighted truncation, summed audio, overlap refused -- regardless of rank or arrival order; probe audio files are named by global index so ranks write disjoint files into one step dir; `run_training` (driven for real over a toy model) calls `knowledge_fn` on a non-writer rank only under `knowledge_all_ranks`; and the launcher's retry counter and fatal raise sit as **siblings** of the `is_main` block after the gather, so every rank raises together and none reaches the next collective alone. |
| `check_gpu_fanout.py` | A job scales to however many GPUs the cluster hands it **without changing what it produces** (backlog G1; the next cluster's smallest GPU node is 4 cards). `settings.MIN_GPUS_PER_JOB` raises the request, `common.visible_gpus` reads what the job got, and `run_worker_script_per_gpu` runs one pinned worker per card in its own `gpu<k>/` cwd -- driven here with a real script under `CUDA_VISIBLE_DEVICES=0,1,2,3`: each worker sees its card and its argv, a failing worker fails the job **and stops the others** (a partial output is never merged), and one visible GPU is byte-for-byte the old single call. The arithmetic that keeps outputs identical is checked against the real thing: `contiguous_slice` == `datasets.Dataset.shard(contiguous=True)` (global row numbering, ids and per-item seeds unchanged), `nested_shard` partitions exactly the job's own Sisyphus shard (the offline driver needs no change), `merge_jsonl_parts` restores clip order and refuses overlaps, and `completed_fraction` sums every worker. Wiring: the five converted jobs call the helper, their workers take the flags, vLLM asks for tensor-parallel, and **training caps its ranks at the declared count** -- an inference job given more GPUs is free speed, a training run given more would silently quadruple its effective batch. |
| `check_loss_parity.py` | `duplex_ce_loss` at `weighted_mean=True` equals the reference `moshi_finetune.finetune.loss.compute_loss_with_mask` **exactly** (both halves, EPAD in the padding set as the reference's `train.py` passes it), and at the default it does **not**, by the stream-dependent factor the arithmetic predicts. The deviation (paper audit, 2026-09-09): ours divided each half by the COUNT of valid positions, the paper's eq. 7 and the reference divide by the SUM OF THE WEIGHTS -- so the audio term came out ~7x smaller and the text ~0.5-0.7x, i.e. the text stream weighed ~3.5-5x more against the audio than "the same importance to the text token and the combined audio tokens". Every finished run's docstring claimed fork parity for the *weights*; nobody looked at the denominator. Also asserts the launcher reads `loss_norm` unconditionally and forwards it at every `moshi_loss` call site, and that the presets emit no `loss_norm` (else every finished run re-hashes). |
| `check_checkpoint_guard.py` | An eval is never wired to a checkpoint that will never exist -- refused at **graph build**, on the login node, by `assert_checkpoint_will_exist` inside `ResolveOverlayCheckpoint.__init__` (the one place every checkpoint reference is minted; `FinetuneRun.weights`/`optimizer_state` call it too). Sisyphus waits on input *paths*, so an eval under a run that FINISHED without writing its step is not an error to the manager, it is `waiting` -- forever, with `error(0)`. Three sources of truth, strongest first: the disk (present = exists, whatever the cadence says), the run's `finished` marker (finished + absent = never; the message lists what the run did write), and for a pending run its `hparams` (`max_steps`/`save_every`) via `impossible_checkpoint_reason`, the same predicate `attach_knowledge_evals` applies to a whole track. Every rejection is paired with an acceptance on the same fixture, the real constructor is driven, and the `finetune.py` template default is asserted equal to `SAVE_EVERY`. |
| `check_finetune_configs.py` | Every `FinetuneAdapter` renders parseable YAML for a single corpus; adapters whose launcher cannot read `train_data_mix` **refuse** a mix at render time (a login-node mini_task) instead of dying after a GPU is allocated; numeric hparams load as numbers (YAML 1.1 parses `1e-06` as a *string*); `train_data_specs` rejects the `{path: weight}` dict form that loses the Sisyphus dependency edge. |

Run `check_fisher_windows.py` after touching `fisher_prep.py`, `check_model_provenance.py` after
adding a benchmark tag, `check_partition_routing.py` after editing `check_engine_limits`,
`check_train_data_common.py` after touching any `*_train_data.py`, `check_finetune_configs.py`
after touching `finetune.py` or adding an adapter, and `check_common_helpers.py` after touching
`common.py` or `chatterbox_inference.py`.

> **Note (2026-08-02).** The retired pytest suite `test_pipeline.py` was folded into
> `check_dialogue_templates.py` (template selection, generator JSON schema, markdown-fence
> stripping) and `check_common_helpers.py` (the rest). It was orphaned -- referenced by nothing and
> absent from this table -- though still passing. Two of its cases asserted against a **local copy**
> of the function under test, so they could not have failed; making them real required hoisting
> `silence_length_sampler` and `strip_dialogue_markdown_fence` out of the enclosing function bodies.
> Both migrated guards were verified to fail when the production code is broken.

The RL suite lives separately and needs the moshi_family venv (5 tests):

```bash
PYTHONPATH=recipe/speech_llm/full_duplex output/venv/moshi_family/bin/python \
  -m pytest recipe/speech_llm/full_duplex/moshi_family/rl/tests/
```
