# Executor report: launch config/sae_4a_lexlat_e1.py (2026-09-21)

## Pre-launch graph check (read-only console, before manager start)
10 jobs total: 7 finished, 3 runnable.
- finished: FlatRecognizerInitJob, BlankfreeVadHdfJob, SampleLinesJob(prior), L15FeatureHdfJob(dev), L15FeatureHdfJob(train), CvHoldoutSplitJob, LibriSpeechSplitIdsJob
- runnable: LexlatEfficiencyProbeJob.r4Iaa72mU27T (GPU, alias sae/4a/lexlat_pack/e1_efficiency), PhoneNgramPriorJob.RtzbESkOedsT (CPU), LexlatWordCountsJob.X2YnVYfqN7aV (CPU)
No PackedBlankfreeTrainJob / BoundedBlankfreeTrainingJob / data job / PriorGapAnalysisJob runnable. Safe to launch.

LexiconTrieBuildJob.rlMsnTBSZXsB: found finished (finished.tar.gz) at
work/speech_llm/sae/emc/lexlat_jobs/LexiconTrieBuildJob.rlMsnTBSZXsB, but it is NOT an ancestor
of this config's graph (used only by LexlatCensusJob/LexlatEquivalenceProbeJob elsewhere); not
touched or at risk of rerun.

## Manager
Started: `nohup python tools/sisyphus/sis --log_level 20 m -r config/sae_4a_lexlat_e1.py -io`
Log: log/sae_4a_lexlat_e1.manager.20260921T044426Z.log
pid 2582573 (child under bash pid 2582135). Manager log at start shows queue(2) after
LexlatEfficiencyProbeJob and LexlatWordCountsJob submitted; PhoneNgramPriorJob resolved without
appearing in the live queue (likely already satisfied/shared hash). No prompt/EOFError.

## LexlatEfficiencyProbeJob.r4Iaa72mU27T
Dir: work/speech_llm/sae/emc/lexlat_train_jobs/LexlatEfficiencyProbeJob.r4Iaa72mU27T
submit_log.run: rqmt {cpu:16, mem:64.0, time:2.0, gpu:1, gpu_mem:96}, sbatch_args
[-A spell -p booster --exclusive], Slurm job 1923108_[1].
sacct as of report time: State=PENDING, Elapsed=00:00:00 (queued on booster, not yet started).

## Status
DONE (launch step complete); job PENDING in Slurm queue, not yet a result. No further action
needed from this executor; a later status check should re-run
`sacct -j 1923108 --format=JobID,State,Elapsed,ExitCode` and tail log.run.1 once it starts.
