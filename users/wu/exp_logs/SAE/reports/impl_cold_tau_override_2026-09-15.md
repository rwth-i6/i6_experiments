# `--tau` override for `analysis/emc_cold_fixed_point.py` (2026-09-15)

Status: DONE.  Dispatch: add a `--tau FLOAT` option (default `None` = today's behaviour,
bit-identical) that sets the lattice temperature of every DP pass directly, print it in the header,
extend `--self-test`, and append the two c2/tau-8 lines to the `.cmd` file.  Nothing submitted.

## Files touched

### `/e/project1/spell/wu24/2026-07-13_unsupervised/analysis/emc_cold_fixed_point.py`
1. New helper `effective_tau(schedule_tau, override)` (right after `resolve_epoch_at_tau`):
   returns `float(schedule_tau)` when the override is `None`, else the override, asserting it is
   positive.  It is the single place the run's temperature is decided.
2. New flag `--tau` (`type=float, default=None`), added next to `--allow-any-tau`.
3. `main()`: the schedule read now lands in `schedule_tau` (`resolve_epoch_at_tau(..., want_tau=2.0)`
   for the default epoch, `model.temperature(epoch)` for `--epoch`), and
   `tau = effective_tau(schedule_tau, args.tau)` is what is handed to `run_cold_fixed_point` ->
   `run_pass` -> `lattice_forward_backward(temperature=tau)` (emc_target_vs_gold.py:847 is the only
   temperature use, so the override reaches every DP pass, including every `--rate-tilt` evaluation).
4. The `--epoch` guard at the old L1265 now reads
   `assert args.tau is not None or args.allow_any_tau or abs(schedule_tau - 2.0) < 1e-9`: with an
   explicit override the schedule's value at that sub-epoch no longer decides what runs, so it must
   not block e.g. `--epoch 1 --tau 8.0`.  The *default* epoch path keeps
   `resolve_epoch_at_tau(want_tau=2.0)` unchanged -- that assert is a guard on the CONFIG's schedule
   plateau (S3 anneals 8 -> 2), not on the run's tau, and it still holds for the tau-8 lines.
5. Header: only when `--tau` was given, one extra line
   `tau              : 8 (--tau override; schedule tau at this epoch = 2)`.
   Numbers are formatted with `:g` (the file's convention everywhere else), so the dispatch's
   illustrative `8.0` prints as `8`; that is the only deviation from the literal string.
6. Module docstring: four lines describing the flag.

No other behaviour changed.  Consequences that follow for free and were NOT hand-coded: the json's
`"tau"` field carries the override (`payload["tau"] = tau`), `banked_comparable` keys on `tau`, so a
tau-8 run is automatically declared not comparable to the banked tau-2 fixed point and asserts
nothing against it (`--banked-check` would exit 3, "NOT COMPARABLE"); the per-pass report line
"phi, the prior, tau = ... and alpha = ... are held fixed WITHIN a pass" prints the override.
Not added (outside the dispatch): a separate `tau_override` / `schedule_tau` field in the json --
provenance is currently only in the .txt header; `"tau"` itself is correct either way.

### `/e/project1/spell/wu24/2026-07-13_unsupervised/analysis/out/emc_cold_fixed_point.cmd`
Appended at the end (after the TOTAL block), lines 105-123: comment header
`# ---- (c2, tau 8 -- audit item iii) ...` plus the two srun lines copied verbatim from the existing
c2 lines with `--tau 8.0` inserted after `--fit-phi` and the labels/outs renamed.  They use the
file's own `$SRUN3 / $SRUN6 / $PY / $S3 / $O` variables, exactly as every other line does:

    $SRUN3 $PY analysis/emc_cold_fixed_point.py --job-dir $S3 --theta flat --phi s3init --fit-phi --tau 8.0 --fixed-point 30 --split dev-other --n-utts 300 --select stride --label c2_flat_fitphi_tau8_bigram --out $O.c2_flat_fitphi_tau8.bigram.dev-other.txt
    $SRUN6 $PY analysis/emc_cold_fixed_point.py --job-dir $S3 --theta flat --phi s3init --fit-phi --tau 8.0 --fixed-point 30 --split dev-other --n-utts 300 --select stride --ablate prior_trigram --batch-frames 10000 --label c2_flat_fitphi_tau8_trigram --out $O.c2_flat_fitphi_tau8.trigram.dev-other.txt

Everything else (job dir, theta, phi, refit, K = 30, 300 --select stride dev-other, band,
--batch-frames 10000 on the trigram chain, the two allocations) is the c2 lines'.

## Checks run

* `python analysis/emc_cold_fixed_point.py --self-test` (CPU, login node): exit 0,
  **14 checks passed** (13 before; the new one is block "(5b)"):
  `PASS  --tau: the default (None) hands the DP the schedule's tau and reproduces the tau = 2 pass
  BIT-IDENTICALLY, while --tau 8.0 changes the same pass's post_q (max |delta| = 0.2248); a
  non-positive --tau is refused`.
  The check runs the REAL `lattice_forward_backward` on the tiny fixture of check (5) at
  `effective_tau(2.0, None)` and at `effective_tau(2.0, 8.0)`; the first `post_q` is compared with
  `torch.equal` against the `temperature=2.0` pass check (5) already made (that is the "previous
  output exactly" reading: same tiny path, same inputs, bit-identical tensor), the second must move
  `post_q` by more than 1e-3 in max norm.  It also asserts `parse_args([]).tau is None`,
  `parse_args(["--tau", "8.0"]).tau == 8.0` and that `effective_tau(2.0, 0.0)` raises.
* Argparse acceptance of every line of the .cmd file (shell variables expanded, `parse_args` only,
  no data loaded): the two new lines give `tau=8.0`, `label=c2_flat_fitphi_tau8_{bigram,trigram}`,
  `ablate=['prior_trigram']` / `None`, `batch_frames=10000` / `None`, `fixed_point=30`, `n_utts=300`,
  `select=stride`, `theta=flat`, `phi=s3init`, `fit_phi=True`, and `--out` keeps the required
  `cold.` prefix; all 11 pre-existing lines still parse and give `tau is None`.
* `ast.parse` of the edited file: ok.

No GPU run, no submission.  The self-test exercises the flag on the tiny fixture path only: it does
not prove anything about what tau = 8 does to the 300-utterance fixed point.

## Left undetermined by the dispatch (assumptions, one line each)

* Label stem: the dispatch says "labels c2_flat_fitphi_tau8"; the file's convention appends the
  chain, so the labels are `c2_flat_fitphi_tau8_bigram` / `..._trigram` and the outputs
  `cold.c2_flat_fitphi_tau8.{bigram,trigram}.dev-other.txt`.
* "sbatch/srun lines": written in the srun form of the c2 lines they were copied from (the file's
  variables), not as a c5-style single `sbatch --wrap` chain; say so if the executor wants the
  wrapped form instead.
* The header number prints with `%g` (`8`, not `8.0`), consistent with every other number in the
  header.
