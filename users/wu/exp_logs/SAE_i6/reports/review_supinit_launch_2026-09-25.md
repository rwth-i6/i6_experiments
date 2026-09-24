# Review: supervised-init launch (`config/sae_i6_p0_supinit.py`, second manager), 2026-09-25

**Verdict: PASS_WITH_NOTES.** The launch as written is a strict subset of the reviewed full P0 graph. It
has no delta and no race with the live screen manager. p0's G0.R3 read does not depend numerically on
which phone prior is chosen. The three notes are about sequencing *later* actions: the full launch, the
g2p fix and the prior decision. None of them affects this launch if the orchestrator follows them.

Evidence: scratchpad `/var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/bb154ec2-4c82-453f-afb8-04f4de6c5e2e/scratchpad/rev/`
(`dump2.py`, `sae_i6_p0*.json`, `cfg.py`/`cfg.log`, `route.py`). Everything was run in `sis console`
under the sae python with sae/bin first on PATH. Nothing was submitted or edited.

## 1. Delta: none (PASS)
- Graphs dumped from the console: supinit 65 jobs, full 164, screen 75. The full ids sha1 is 61a1419117f4
  and the screen's is db40328df82a, identical to the pins in `review_p0_full_launch_2026-09-24.md`. The
  package is unchanged since 51f4def2d (`git diff`/`git status` on `unsupervised_asr` are empty).
  settings.py is ea461de8, the round-4 PASS in `review_gpu_route_2026-09-24.md`.
- All 65 supinit ids are in the full graph. Per job, the input set (creator id and path) and the alias
  list are identical to the full graph's. All 20 registered outputs exist in the full graph with
  identical requirements: 11 supervised_init outputs, plus 9 feats/units stats outputs that
  `get_inputs()` registers, which the screen graph also registers. `sae/4a/lm/prior.npz` is not registered.
- Status: 53 finished, 1 runnable (DownloadHuggingFaceSnapshotJob.7tLASYdh10dO: gilkeyio MFA, train-clean-100,
  a sha256-pinned mini_task on the desktop; 969 GB free on /work/asr4/hwu), 11 waiting. No job is in
  error, and none of the 12 unfinished jobs has a work dir yet. The implementer's claims all hold.

## 2. Does p0 read the prior's VALUES? No.
- The prior enters p0's hash (`training/config.py:312`), and the model loads the file at construction
  (`model/emc_model.py:395`). The values go only into buffers: `prior_log_bi` and the agg `text_uni`/`text_bi`.
  The model then checks that `prior_log_bi` is finite (`blankfree_model.py:483`) and that its shape is
  the fixed (1681, 40). No RNG is drawn from those values, and the recognizer is overwritten by the flat init
  (FlatRecognizerInitJob seed 0, which has no prior ancestor).
- Loss: `p0_steps.train_step` reads only `model.recognizer`, `features` and `targets` (`p0_steps.py:61-81`).
  Optimiser: the reverse parameters get no gradient. Selection: `GetBestPtCheckpointJob` reads
  `dev_loss_blankfree_supervised`, which this step alone produces. RETURNN's keep_best_n=1 keeps the argmin
  epoch of each non-constant key (`returnn/engine/base.py:385-405`), so the selected checkpoint survives
  cleanup. Export keeps `recognizer.*` only. The dump builds a bare ConvRecognizer (`recognizer_only.py`).
  The PER job reads posteriors, dev-other features/originals and the MFA gold.
- So p0's PER is valid evidence whichever prior is chosen. Gold phi has no prior or g2p ancestor at all.

## 3. Resources and routing (PASS)
- `check_engine_limits` gives both trainings 1 GPU, 16 CPU, 64 GB, 72 h, `-p gpu_48gb`. The dump gets 4 CPU,
  24 GB, 2 h, flexible (gpu_48gb with flex24 at the time of the check). Nothing goes to gpu_11gb or A100,
  and none of these jobs imports k2. Slurm receives only `--gres=gpu:1`.
- Memory, from the code (no banked figure exists). p0: batch 88,000 frames and 128 seqs through a 1-layer conv plus
  a [B,U] DP, well under 2 GB. Gold phi: fp32, 8 utterances, [8,40,50,S+1] segment table, DP about 3 GB at
  worst. Both fit easily in 46 GB. Banked time: p0 15 min on one JUPITER GPU (`SAE/SAE_4A_lexlat.md:446`),
  gold phi a 4 h budget. The gpu_48gb cap: ctrl_20 1 + 2 trainings + the dump = 4 of 5.
- The dump -> PER chain already ran on i6 (ctrl_20 ep1: PER 0.855180), with the same non-checkpoint
  inputs as p0's read. All 17 PER jobs and 17 dev-other dumps share them.

## 4. Two managers (PASS now)
- No unfinished supinit job is in the screen graph. The 53 shared jobs are finished. The screen's live
  ctrl_20 (GiT88bxzoZbZ) is not in the supinit graph. Output and alias links that already exist with the
  same target are left alone (`sisyphus/graph.py:113-118`). The command starts plain, without `-co`.

## 5. Notes (sequencing of later actions)
1. `SAE_i6_P0.md:25-26`. The switch sequence stops only the screen manager before starting
   `config/sae_i6_p0.py`. All 12 supinit jobs are in the full graph, so the supinit manager must be stopped too.
   Otherwise two live managers can submit the gold phi, p0 or the dump twice.
2. `SAE_i6_P0.md:168-169` (the g2p fix: clear the g2p job and rerun down to the prior). The chain
   TrainG2PModelJob.pD4nbqFLWtbi -> ... -> PhoneNgramPriorJob.qJxXHgXLe31S is in both the screen and the supinit
   graphs. If it is cleared while both managers are live, both will submit it. p0 also opens `prior.npz`
   at every start or resume (`emc_model.py:395`), so if p0 starts while the prior dir is missing, it dies;
   if it is not yet set up, it waits hours for the chain. Apply the fix after p0 has finished, with one
   manager. The line at `SAE_i6_P0.md:209` ("do not read the prior") is true only for values: p0 does open the file.
3. Hash coupling. If the prior decision changes PhoneNgramPriorJob's hash (for example by importing JUPITER's lexicon),
   the full config builds a new p0 id and reruns p0, its dump and PER. That costs about 15 min of GPU and the
   number is the same. An in-place g2p fix keeps Y1vbqR6KeJSx.
- Operational: State's watcher covers pid 1583423 only, so the new manager needs its own.
