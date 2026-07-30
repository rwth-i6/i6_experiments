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
| `check_finetune_configs.py` | Every `FinetuneAdapter` renders parseable YAML for a single corpus; adapters whose launcher cannot read `train_data_mix` **refuse** a mix at render time (a login-node mini_task) instead of dying after a GPU is allocated; numeric hparams load as numbers (YAML 1.1 parses `1e-06` as a *string*); `train_data_specs` rejects the `{path: weight}` dict form that loses the Sisyphus dependency edge. |

Run `check_fisher_windows.py` after touching `fisher_prep.py`, `check_model_provenance.py` after
adding a benchmark tag, `check_partition_routing.py` after editing `check_engine_limits`,
`check_train_data_common.py` after touching any `*_train_data.py`, and `check_finetune_configs.py`
after touching `finetune.py` or adding an adapter.

The RL suite lives separately and needs the moshi_family venv (5 tests):

```bash
PYTHONPATH=recipe/speech_llm/full_duplex output/venv/moshi_family/bin/python \
  -m pytest recipe/speech_llm/full_duplex/moshi_family/rl/tests/
```
