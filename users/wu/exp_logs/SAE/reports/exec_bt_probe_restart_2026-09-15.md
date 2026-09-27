# Executor: sae_4a_bt_probe manager restart -- 2026-09-15

## Cause of stall
Manager pid 2037231's manager.log was 0 bytes because no ERROR-level line was ever emitted, not
because the graph was untouched: it submitted the remaining runnable jobs, all 8 target BtProbeJob
hashes ran to completion on SLURM (timestamps 21:39-21:54, after the log file's 21:12 mtime), and
the manager thread then exited on `EOFError` at the standard non-interactive "All calculations are
done, print verbose overview (v), update outputs and alias (u), cancel (c)?" prompt (stdin is
`/dev/null` under nohup). This is a clean finished-graph exit, not a code defect.

## Per-arm state (all output/ dirs confirmed on disk: bt_probe.json, bt_probe.txt, best_per,
recognizer.round1.pt, recognizer.round2.pt)
FINISHED: BtProbeJob.1E6mfTahsKLZ, .3BFG7tATDxqI, .5owmxhkPXkdf, .91VDNBrLup2B, .96K3LFZ7brhJ,
.rNJbMa2sG9ia, .uXVX60ASzhKI, .ZXlljMDUea5m -- all 8/8.

## Restart
No BtProbeJob was ever in error state (no clear-jobs prompt seen). Started manager manually
(sis_managers.sh has no entry for this config, matches prior executor report):
`nohup sis_env/bin/python tools/sisyphus/sis --log_level 30 m -r config/sae_4a_bt_probe.py`,
pid 2267815. Graph read (skip_finished=True) showed zero runnable/waiting/error/queue_error work
both before and after this restart -- the new manager hit the same "all calculations are done"
EOFError immediately and exited; this is expected given nothing remains. `squeue -u $USER` shows no
bt_probe SLURM jobs pending.

## Conclusion
DONE: whole sae_4a_bt_probe graph is finished, all 8 targets have results on disk. No watcher
needed (no live work), no code defect found.
