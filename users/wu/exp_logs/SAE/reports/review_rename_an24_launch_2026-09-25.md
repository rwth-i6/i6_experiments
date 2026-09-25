APPROVE_WITH_CONDITIONS

# Pre-launch review: AN-2 / AN-4 (SAE_4A_rename), 2026-09-25

Scope: commits 941e392d (build) and 996a6aad (R1) in recipe/2025-10-speech-llm; the config
`configs/config_sae_4a_rename_an24_v1.py`; the shim `config/sae_4a_rename_an24.py`; the build report
`reports/impl_rename_an24_2026-09-25.md`. I also read the phase file SAE_4A_rename.md (AN-2 at l.134-159,
AN-4 at l.166-179, R1 at l.94-112), the design review `reports/design_review_rename_2026-09-25.md` (A1-A12),
and the AN-0 R1 review `reports/review_rename_an0_r1_2026-09-25.md`, because of the shared H_LM, derangement
and phi conventions. The checkout has no uncommitted changes to these files. I fixed nothing.

Verdict: the AN-2 build does what the phase file registers, including R1. The AN-4 build computes the
registered quantities, but R1's "two forms everywhere lambda appears" also covers AN-4's derivative, and
nobody has decided that question. It must be decided before the reader's verdicts appear. The launch itself
is sound: 337 GPU forwards on gpupack at 0.5 h each, no unfinished job shared with a live graph, and nothing
trains.

## Conditions (findings, most severe first)

1. `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/rename_an24_jobs.py:1340` (also l.614). AN-4's LM
   term is -dS_lambda/dlambda at lambda = 1 +/- 0.05, computed in the plain form only.
   - What the text says: R1 (SAE_4A_rename.md:100) reads "Two lambda forms everywhere lambda appears". The
     build report (impl:133) says R1 was not applied to AN-4, and the phase file does not record that scope.
   - The plain form is not wrong. Its derivative is exactly the registered E_q[log P_LM]: the real check put
     FD against exact at 3.5e-4 relative.
   - The rate-neutral derivative is a different quantity: E_q[log P_LM] + H_LM x E_q[N_all], where N_all
     includes SIL tokens.
     - It shifts the LM term per frame by H_LM x (all-token rate), about 2.257 x 0.236 = 0.53 nats per frame
       at R1's probe rate.
     - It shifts D_LM (finals minus basin) by H_LM x (rate gap). A gap of 0.05 tokens per frame moves D_LM
       by 0.11, the same order as the paired S gap that P2 compares against.
     - P1b (LM per phone) also moves through the SIL-to-non-SIL token ratio.
     - So P2 and P1b can read differently depending on the form.
   - What breaks: An4ReadJob is a mini task. It runs as soon as the 50 anatomy forwards finish and prints
     P1/P2/P3 verdicts. If the form question is settled after those verdicts are visible, the choice is post
     hoc.
   - Remedy (before launch): record in the phase file either (a) "AN-4 is read in the plain form only,
     because the named quantity is E_q[log P_LM]", or (b) that a rate-neutral column is read beside it.
     Option (b) is a reader-only CPU change: anatomy.json already stores expected_tokens,
     expected_sil_tokens and expected_nonsil_tokens per utterance (l.618-619). No GPU forward changes under
     either choice, and a reader edit re-hashes only the reader.

2. `reports/impl_rename_an24_2026-09-25.md:71` (repeated at :143). The launch line there runs in the
   foreground, with no setsid, no nohup and no log file. Run as written, the manager dies with the shell.
   - The syntax itself is valid: `--config` is a global flag (tools/sisyphus/sisyphus/__main__.py:41).
   - Launch with the dispatch's command instead. It also parses: global --log_level, then `manager -r`,
     then the config as argv (__main__.py:93).
     `cd /e/project1/spell/wu24/2026-07-13_unsupervised && setsid nohup /e/project1/spell/wu24/env/sis_env/bin/python tools/sisyphus/sis --log_level 20 manager -r config/sae_4a_rename_an24.py < /dev/null >> log/sae_4a_rename_an24.manager.$(date +%Y%m%d_%H%M%S).log 2>&1 &`
     Do not add -co or -cio.

3. `recipe/.../configs/config_sae_4a_rename_an0_v1.py:40` (AN3 = False). The four stage-1 (a)
   PhiFromKeyInitJob ids built by this config are the same jobs AN-3 builds.
   - If AN3 is switched on while the an24 manager still holds those jobs unfinished, two managers run the
     same local mini task and write the same outputs.
   - Keep AN-3 off until they finish. They are CPU mini tasks, so the window is minutes after launch.

## What I checked (evidence)

1. **Registered readings.**
   - An2ReadJob (l.836-1190) quotes the AN-2 rule and R1 verbatim, in the docstring and in report.txt.
   - H2 (l.951) and the name clauses (l.958) follow the text:
     - SEES: d < min R - 0.01;
     - PREFERS: d > 0.01;
     - BLIND: min R <= d <= max R;
     - every clause that fires is reported, joined; MIXED if none fires.
   - The LAMBDA lead (l.974) is read on rate_neutral, and the plain lead and its verdict are printed beside
     it, as R1 requires:
     - KEEPS if lead > 0 at all four lambdas;
     - FAVOURS if the lead decreases strictly and lead(10) < 0;
     - otherwise NEITHER.
   - The random-rename null uses 3 full derangements of each stage-1 (a) key (seeds 1-3).
   - Each clause fires on planted rows: the unit test passes (re-run today, `PASS unit`). Each reader
     reproduces banked S on all 260 in the fixtures.
   - SEES and PREFERS can fire together. The phase's decision table handles the joined output.
2. **Rate-neutral shift.**
   - rate_neutral_offset (l.404) is (lam - 1)/lam x H_LM.
   - rate_neutral_get_model (l.411-433) adds it to the float64 prior_log_bi.
   - The lattice multiplies the table by prior_weight = lam (lattice.py:770-791, `scale * prior_weight *
     prior_log_bi`). The lattice therefore scores lam log P + (lam - 1) H_LM per token, SIL included; the
     lattice has no EOS term.
   - At lam = 1 the offset is 0, and the config builds one job for both forms.
   - H_LM = 2.257108384 comes from TrigramLogLossJob on the prior's own counted lines (text only). It matches
     AN-0's raw-count lm_rate_constant to 1e-9, which the AN-0 R1 review also confirmed.
   - The real check showed the shift reaching the `_setup` lattice input exactly.
3. **Reproduction at lambda = 1.**
   - The 21 reused forwards are the banked jobs themselves: the config asserts path equality, and each id is
     in the INPUT lists of the banked readers.
   - All 21 jsons have prior_weight as banked, anchor_weight 0, 285 utterances and 0 impossible.
   - The fixtures reproduce 3.207044 (gold key) and 3.299030.
4. **Derangement pools.**
   - The gold pool has 37 symbols, read from GoldUnitKeyJob.sLnMRRd2qO0t. OY and ZH are excluded (asserted
     in key_pool, l.164).
   - All 9 gold-key derangements equal those of the committed rename_emstep_jobs.derangement (d2bbd4c7),
     re-checked today.
   - The stage-1 (a) derangements use all 39 non-SIL symbols. Every symbol in stage-1 keys 1-4 holds units
     (none is empty), so the 39-symbol pool is R1's hold-units pool for those keys. AN-3 does not derange
     stage-1 phis, so the two phases share no derangement convention there.
5. **AN-4 and lambda.** Lambda enters only the LM finite difference (ANATOMY_LAMBDAS = 0.9, 0.95, 1,
   1.05, 1.1) and its +/- 0.1 check. That is condition 1. The channel term, H, the rates, the coverage, KL
   and MI all come from the lambda = 1 forward-backward.
6. **Labels and training.**
   - Labels enter only the readers: PER, key identity in AN-2, and the unit-symbol table's MI. There is no
     trainer in the graph.
   - PhiDerangeJob writes deranged copies for evaluation only; none becomes a candidate phi or table.
7. **SIL in tokens per frame.** AN-2 prints non-SIL tokens per frame, because the genmarg json has no SIL
   count. No registered reading needs SIL:
   - R1's "tokens per frame" is printed beside, not gated;
   - SIL fragmentation is ruled out by the prior (log P(SIL|SIL,SIL) = -14.49; SIL is 13.9 % of the text).
   - AN-4 reads SIL share from seg_post as registered.
8. **Launch.**
   - The graph test passes: 308 AN-2 forwards (21 reused + 287 new), 50 AN-4 forwards, 369 unfinished jobs.
   - 337 ReturnnForwardJobV2.run tasks route to gpupack with 1 GPU, 1 slot, 4 CPUs, 32 GB and 0.5 h.
     Everything else goes to the short (login-node) engine.
   - Shared unfinished jobs: 0 with keyinit (3 unfinished), keyarms (315), an0 (1) and an1 (0). The only live
     manager found was keyarms.
   - Time: the banked forwards took 24-54 s. The anatomy forward has not been measured; it is estimated at a
     few minutes (5 forward passes plus a manual no-grad forward-backward). Its GPU memory is a few GB of
     float64 [B, 40, 25, S] tensors at batch 88000 features, far under 96 GB.
   - gpupack packs up to 4 tasks per slot for asks of 2 h or less, so about 2.6 GPU-h of work occupies a
     handful of node allocations.
9. **Beyond the spec.**
   - The implementer's open choices are acceptable and none moves a gate:
     - full derangements with seeds 1-3;
     - P2 on group means of paired per-utterance differences;
     - P3 endpoint 0 -> 48 on the three keyinit basin trajectories, with Hungarian PER;
     - LM per phone = pooled E_q[LM] including SIL tokens' cost, divided by E[N_nonSIL], which matches
       unit_key.py's E10 convention;
     - FD tolerances that live only in the test.
   - Key identity is not computed in AN-4, and the AN-4 text does not ask for it.
   - Stage-2 arms are a hook that is off by default, matching "when finished".
   - I found no other delta.

## Notes (not findings)
- The rate-neutral form over-restores the token rate at high lambda (build check, 8 utterances: 0.198
  non-SIL tokens per frame at lambda 10, against 0.178 at 1). This follows from R1's design, not from the
  code. It matters for interpreting KEEPS at lambda = 10.
- `FORWARD_TIME_H` is applied through job.rqmt after construction (config l.189/198/266), outside the
  hashes. The reused finished jobs are unaffected.
