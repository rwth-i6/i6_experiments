# executor report: sae_4a_infomax_pack manager restart 2026-09-21

## 1. Original manager (pid 1355841) exit cause
Not a traceback/error-prompt/kill. Log ends cleanly: cleared 4 BlankfreeDerangementGapJob
outputs (aughi/aug/entaug dev-clean/dev-other), then `All output calculated` at 02:57:25,
manager exited normally. ps -p 1355841 shows no such process (already gone).

## 2. node_d (Slurm 1912407)
sacct: 1912407_1 COMPLETED, Elapsed 08:40:12, End 2026-09-21T02:39:53, node jpbo-035-48.
Job dir work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.YtvrSez8z9Wf
has finished.tar.gz (archived 02:40, JOB_AUTO_CLEANUP). node_d genuinely finished, not a
process the manager needed to re-attach to.

## 3. Manager restart
Started fresh (nohup ... -io > log/sae_4a_infomax_pack.manager.20260921T010832Z.log).
It loaded the config, found nothing runnable for any requested Output, and hit the
interactive prompt:
`All calculations are done, print verbose overview (v), update outputs and alias (u), cancel (c)?`
Stdin was closed (nohup, no tty) -> EOFError, main thread exited immediately (~<1s).
No jobs were submitted (correct: no runnable work exists to submit). Per instructions,
stopped here on the prompt rather than forcing an answer.

## Cross-check of watcher's "runnable=2"
Directly checked on-disk state of the 4 ctrl_50-dependent BlankfreeGreedyPerJob dirs the
console flagged transiently as "waiting"/"runnable" across repeated get_jobs_by_status
calls (counts churned: finished:1/runnable:1, then waiting:4, then finished:304/waiting:7/
runnable:3/queue:1) -- all 4 already have finished.tar.gz. This matches the known
console-status-misreports-finished trap: one-shot get_jobs_by_status calls FINISHED jobs
"waiting"/"runnable". No genuinely unfinished work was found; the manager's own verdict
("All calculations are done") agrees.

## Node A (PackedBlankfreeTrainJob.ks7CbtlvpcIL, Slurm 1921418, manager 2945974)
Not touched.
