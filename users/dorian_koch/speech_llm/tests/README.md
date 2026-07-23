# Standalone checks (login-node, no GPU, no Sisyphus manager)

Cheap guards for the classes of bug that are invisible in a loss curve — wrong channel/role mapping,
a silently-unmapped model tag, a node exclusion that never reaches SLURM. Each exercises the **real**
code (importing it, or exec'ing the real source) rather than reimplementing the logic, so it fails
when the thing it guards actually breaks.

Run from the setup root. `CUDA_HOME=/usr` is only needed because `settings.py` asserts it at import:

```bash
cd /home/tt201262/setups/2026-01-speech-llm
T=recipe/i6_experiments/users/dorian_koch/speech_llm/tests
CUDA_HOME=/usr .venv/bin/python $T/check_fisher_windows.py
CUDA_HOME=/usr .venv/bin/python $T/check_model_provenance.py
.venv/bin/python $T/check_partition_routing.py
```

| Script | Guards |
|---|---|
| `check_fisher_windows.py` | `FisherToMoshiTrainData`: Fisher markup cleaning, agent-only alignments, in-window/ordered timestamps, role-keyed A/B→user/agent channel mapping, arrow row layout. Decisive check: **assistant RMS in-word vs out-of-word must be >2×** — if the channel mapping were inverted the words would land on the wrong track and this collapses to ~1. |
| `check_model_provenance.py` | Every `fdb_benchmark_py` tag in the recipe has an `FDB_MODEL_ORIGIN` entry, the map is well-formed, and an unknown tag is **rejected** rather than rendered unmarked. |
| `check_partition_routing.py` | Cluster-agnostic requirements (`requires`, `gpu_mem_gb`) route to a partition that actually provides them; `BROKEN_NODES` reaches `sbatch -x` on **every** job shape (the old `default_rqmt` mechanism silently dropped it for jobs setting their own `sbatch_args`); unsatisfiable requirements raise instead of mis-routing. |

Run `check_fisher_windows.py` after touching `fisher_prep.py`, `check_model_provenance.py` after adding
a benchmark tag, and `check_partition_routing.py` after editing `check_engine_limits`.
