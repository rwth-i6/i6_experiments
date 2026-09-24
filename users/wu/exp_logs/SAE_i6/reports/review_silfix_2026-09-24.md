# Review: SIL-run fix (`sil_run_collapse`, arm `ctrl_20_rc`), 2026-09-24

Verdict: PASS_WITH_NOTES. I found no defects.
- On the default path, the train step, every read and every existing job id are unchanged, and I show this by construction and by measurement.
- With the option on, the model uses the run-collapse lattice at every place the lattice runs for this arm.
- The notes below are not code defects. They say what has not been shown yet and what the definition costs (3a).

Inputs reviewed:
- /work/asr4/hwu/tmp_dev/silfix.patch (worktree /work/asr4/hwu/tmp_dev/silfix);
- /work/asr4/hwu/tmp_dev/silfix_config.diff;
- reports/impl_silfix_2026-09-24.md.

Nothing was applied to the live tree. Evidence lives in the scratchpad
/var/tmp/claude-2764/-u-hwu-setups-librispeech-960-2026-09-24-unsupervised/e99790e1-d0a8-40e7-8845-f2cb4e1545a6/scratchpad
(written SP below).

PKG = users/wu/experiments/unsupervised_asr. Line numbers refer to the worktree.

## What the patch touches
The patch changes model/blankfree_model.py, training/config.py and training/arms.py, and adds tests. No read,
analysis, lattice or train-step code is touched.
- blankfree_model.py:444 adds the keyword `sil_run_collapse=False`. It is consumed by name and never forwarded to super.
- blankfree_model.py:503-506 rebuild `prior_history` from the blank-free `lattice_cfg`, but only when the option is on.
  `_use_matmul` there is the same job-start validation as emc_model.py:374 and returns a value that is not used.
- config.py:348-349 write `model_args["sil_run_collapse"]=True` only when the option is True.
- arms.py:148-155 define `ctrl_20_rc`, which is ctrl_20 plus `sil_run_collapse=True`.

## 1. Default path unchanged
- Bit identity. I ran one T2.1 train step under the live package and under the worktree package, at eps 1e-4 and 0.25
  (SP/bitid.py, bit_live.pt and bit_wt.pt). All 72 of 72 tensors are `torch.equal`: log Z, the total loss, every
  logged loss and 34 parameter gradients.
- Job ids. I built the graphs in shadow setups against the live tree and the worktree (SP/sh_live, SP/sh_wt, dump_ids.py):
  - sae_i6_p0 has 139 ids in both builds, and the two tsv files are byte-identical.
  - sae_i6_p0_screen has 75 ids in both builds.
  - These match the implementer's tsv files.
  - The live alias ctrl_20 points to ReturnnTrainingJob.GiT88bxzoZbZ in both builds.
- The ctrl_20 returnn.config is identical between the live and worktree packages, apart from the shadow path.
- Reads. None of them runs the lattice, so none can see the history:
  - the posterior dump uses recognizer_only.get_model with RECOGNIZER_NET_ARGS, not the training model_args
    (analysis/posterior.py:53 and :94), and it drops undeclared kwargs in any case;
  - the greedy PER read, the gaps (reverse model), JsRowsReadJob and PairedPerDeltaJob read decodes, reverse-model
    outputs and hyps only.
- Test suites (SP/pt_live, SP/pt_wt, junit):
  - live: 503 passed, 14 skipped, 12 xfailed;
  - worktree: 526 passed, 14 skipped, 12 xfailed, which matches the implementer's numbers;
  - no existing test changed outcome, and all 23 new tests passed;
  - the 14 skips are 7 artefact, 4 ffmpeg-pin and 3 gpu;
  - the two T1.6 strict xfails still xfail in the worktree, so the default path still carries the split:
    `test_t1_6_log_z_vs_run_collapse_oracle` and `test_t1_6_model_history_is_the_blankfree_one`.

## 2. Option-on correctness
The history differs only in `same_nonsil`, which is the only topology-dependent field (lattice.py:355-358).
- Under blankfree, `last(h)==k` is forbidden for every k, SIL included.
- f=0 is unreachable after the start, so this is exactly run collapse.
- The matmul tables (lattice.py:708) and the backward mask (:1197) read the same `hist`, so the forward and backward
  passes agree.

Oracle check. tests/lattice_oracle.py is independent of the package. It enumerates every one of the K^T frame paths.
- It uses `run_collapse` with SIL included.
- Durations lie in [d_min, D_k] and sum to S.
- The band in "code" mode is |s - 3t| <= W at every frame of the run.
- The trigram is BOS-padded.
- `sil_split=False` gives the definition of section 4.1 plus S2. I found it correct.

Existing tests against the oracle:
- T1.1-T1.5 already hold the DP with a blank-free history to this oracle, including T1.4c "sil_only_is_one_token".
- The new rc tests hold the model's step log Z to the run-collapse oracle and away from the split enumeration.

Consumer inventory, checked independently with grep. Every lattice call takes `model.prior_history`:
- train_step.py:90, which feeds l_tau, post_q, seg_post, expected tokens and the prior monitors;
- rate_term.py:256 and :298, the tilted passes;
- genmarg_steps.py:214, which ctrl_20_rc never reaches: genmarg.py:474-478 refuses the extra key, and genmarg is
  registered only in lexlat_v2.
- The history is rebuilt per call from that argument (`history or bigram_history`), so nothing is cached. It is not
  a buffer, so the state_dict and checkpoints are unchanged.

End-to-end through a real config (SP/cfg_probe.py):
- I exec'd the written ctrl_20 and ctrl_20_rc returnn.config from the worktree graph and called `get_model` with the
  fixture prior and eta.
- ctrl_20: the partial has no key; `sil_run_collapse` is False; the history equals the ctc build.
- ctrl_20_rc: the partial has True; `sil_run_collapse` is True; the history equals the blank-free build.
- One real `train_step` each makes two lattice calls (l_tau and fd), and both take `model.prior_history`.
- The state_dict is equal at the same seed.
- rc log Z is lower than ctrl on every utterance, by 0.0022, 0.0006 and 0.0055 nats. That is expected, because rc
  keeps a subset of the latents.

The two configs differ by one key. The returnn.config diff is only `"sil_run_collapse": True` plus the job's own
model path. rqmt is identical: cpu 16, gpu 1, gpu_mem 96, mem 64, time 11.5. rc uses SAE_PYTHON_EXE, because it has
no k2 key.

## 4. Config wiring
The rc graph has 164 jobs: all 139 old ids with the same aliases, plus 25 new ones.
- paired_delta(ctrl_rc, ctrl) has per_a = ctrl_20 as the baseline, so the delta is rc - ctrl, as the docstring says.
- It covers keep_epochs 1/4/10/20, and both arms keep all of them.
- JS runs in its own job "p0_rc", so the three-arm p0 job keeps its hash.
- The screen config is untouched.

## 3(a) Long SIL runs (answered separately from the verdict)
Confirmed. The band requires |s - 3t| <= 25 at every frame of a token's run, so one token covers at most 17
recognizer frames, about 1.02 s. The first token covers at most 28 units, and a trailing run at most 10 frames.
- Under rc, SIL->SIL is forbidden. A frame path whose SIL run is longer than that therefore has lattice weight 0.
- Its mass moves to paths that put a non-SIL frame inside the pause.
- That non-SIL token receives theta and phi gradient and counts in the rate term's E[N]. l_tau therefore pushes q
  toward breaking long pauses.
- Expected symptom: insertions inside pauses of more than 1 s. The frequency of such pauses is unmeasured on the
  retained frames. On the original audio, 2.2% of dev-other utterances contain one, which is an upper bound.
- z_zero is unchanged.

What the banked split did:
- It read such a run as several SIL tokens. Each paid P(SIL|.,SIL) and carried its own SIL segment.
- The rate term does not count SIL, but expected_tokens does.
- The split also added extra, non-unique readings to every SIL run, including short ones.

Cleaner alternative:
- Within the definition (section 4.1 run collapse plus the S2 band) there is none. These paths have zero weight by
  definition.
- The alternatives change the definition:
  - A SIL-only band and duration cap would keep readings unique. It needs W_sil and D_sil both raised, about 3 units
    per frame of pause, and the band axis is global, so it widens the DP.
  - The proposal "SIL->SIL only when the previous SIL has reached its maximum span" also leaves the definition: it
    reads B'(a) with extra SIL tokens, each paying P(SIL|.,SIL). Readings are unique only if "span" means the
    17-frame run length, which needs a run-length counter in the DP state (up to 16 extra values of f per history).
    If "span" means unit duration equal to D_sil, the cut frame stays free inside the band, so readings are not
    unique.

## 3(b) Reads
ctrl_20 has no genmarg read in P0, so ctrl_20_rc lacks nothing ctrl_20 has. It gets:
- the 4 PER reads, the posteriors and both gaps;
- the paired deltas at 1/4/10/20;
- its own JS row, in job p0_rc.

G0.RC can be read from:
- the paired deltas;
- the emitted rate, from per.json `phone_rate_original_hz` at ep20;
- the derangement gap at ep20;
- the step-1 log Z, from the training log.

## Notes (not defects)
1. Nothing has run at bed shapes or on a GPU. The G0.RC clause "step-1 log Z decreases" has been shown only at test
   shapes (above, and in the implementer's tests).
2. The T1.6 rc discrimination is thin: 3 of 4723 split latents, and in T2.1 a relative margin of 6e-10 against a
   tolerance of 1e-10. The DP-level oracle tests at longer runs and the real-config history check cover this.
