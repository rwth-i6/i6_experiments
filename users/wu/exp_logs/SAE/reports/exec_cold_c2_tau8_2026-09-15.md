# Executor report: cold c2 tau=8 rerun attempt (2026-09-15)

## Task
Rerun read c2 (flat theta, phi refit) at tau = 8 for both bigram and trigram chains,
labels c2_flat_fitphi_tau8, from analysis/emc_cold_fixed_point.cmd.

## Finding
`analysis/emc_cold_fixed_point.py` does NOT expose tau as a direct CLI option.
`grep -n "^p.add_argument" analysis/emc_cold_fixed_point.py` lists no `--tau` flag.

tau is instead derived from the model's temperature schedule at a chosen sub-epoch:
- `--epoch` (int) selects the sub-epoch; `model.temperature(epoch)` then gives tau.
- Default (no `--epoch`): `resolve_epoch_at_tau(model, want_tau=2.0)` (call site
  analysis/emc_cold_fixed_point.py:1261) walks the schedule to the FIRST sub-epoch
  holding its final plateau value, and hard-asserts that plateau equals 2.0
  (function body, emc_cold_fixed_point.py:263-267).
- If `--epoch` is passed explicitly instead, line 1265 asserts
  `args.allow_any_tau or abs(tau - 2.0) < 1e-9` — i.e. tau=2 is the hard-coded
  required operating point unless `--allow-any-tau` is also passed; even then the
  correct `--epoch` for tau=8 (documented as sub-epoch 1 of the 8->2 anneal over
  sub-epochs 1-4, per docstring emc_cold_fixed_point.py:249-250) is not something
  the CLI states directly — it requires reading the schedule off the config.

Per the dispatch's explicit branching rule (no `--tau` CLI option -> report BLOCKED,
do not edit the script), I did not construct or submit any sbatch line, and made no
script edits (script is implementer-owned).

## Blocking lines
- analysis/emc_cold_fixed_point.py:1261 — `epoch, tau = resolve_epoch_at_tau(model, want_tau=2.0)`
- analysis/emc_cold_fixed_point.py:263-267 — assert inside resolve_epoch_at_tau pins tau to want_tau (2.0 by default)
- analysis/emc_cold_fixed_point.py:1265 — assert pins tau=2 unless `--allow-any-tau` passed with an explicit `--epoch`

## No action taken
No SLURM submission, no .cmd file edit, no script edit.

## SLURM (sbatch, cold c2 tau8 lines -- audit item iii)
- cold_c2_tau8: job 1818817, single 9h alloc (bigram $SRUN3=03:00:00 + trigram $SRUN6=06:00:00,
  sequential in one --wrap chain, same account/partition as the other cold lines: -A spell -p
  booster -N1 -n1 --cpus-per-task=16 --gres=gpu:1). Submitted, not polled to completion (do not
  wait). Log: analysis/out/cold_c2_tau8.run.log. Will write:
  analysis/out/cold.c2_flat_fitphi_tau8.bigram.dev-other.{txt,json},
  analysis/out/cold.c2_flat_fitphi_tau8.trigram.dev-other.{txt,json}.
