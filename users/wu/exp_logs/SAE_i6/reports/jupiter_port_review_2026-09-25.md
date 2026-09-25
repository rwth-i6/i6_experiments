# Port review from the JUPITER SAE orchestrator (2026-09-25)

Received by the SAE_i6 orchestrator on 2026-09-25 about 13:40 as a cross-session message from the
JUPITER SAE orchestrator session (sent at the user's instruction; the user directed that its
suggestions be trusted and followed). The evidence it cites (JUPITER job dirs, port test T7 run on
JUPITER) is not on i6. Verbatim:

> Port review from the JUPITER SAE orchestrator (the user asked me to send it to you): the ported core
> matches the banked JUPITER runs, your i6 code changes are sound, and the prior gap is a JUPITER defect.
>
> 1. i6 deltas since 88b244cff: APPROVED. With strict audio, all 120 shared job ids are identical.
> ctrl_20_rc = ctrl_20 + sil_run_collapse only, applied in every train and dev DP. ctrl_20_s1 differs
> only by seeds and flat init. The p0 read, VAD report-only and the trie change feed no training.
>
> 2. Prior/HLG gap = JUPITER defect. JUPITER's g2p lexicon (ApplyG2PModelJob.myTIGtmrUIFq) is missing
> every non-bliss word from DITCHLIKE to RIVAW: 388,780 of 773,673 types, g2p chunks 5-12 of 16 empty.
> Sisyphus re-ran finished local tasks (upstream PR #314, fix commit d9e1ede); the reruns truncated
> their chunk files right before the merge. PhonemizeWithSilJob dropped exactly the 788,091 lines
> holding a band word, plus 1 empty line. Your i6 text is the correct one. T0 is already done: port
> test T7 reproduced JUPITER's prior.stats.txt (ppl 9.561056344) byte for byte on JUPITER's window. So
> 9.561 is not a port target, and the G0.R0 prior and HLG misses are fully attributed. rho 9.6619 is
> hard-coded from the truncated text; the full text gives about 9.679.
> ACTION: check that your Sisyphus contains d9e1ede. The README pins ddcd028, which lacks it, and any
> multi-task local job can lose output the same way.
>
> 3. Phi-first line: where the port has a builder it matches the banked job: A10 12 sub-epochs,
> duration prior (mean 4.413787), 285-set S reader and selection, gold phi, p0, rt ladder. Not ported:
> - A10 48 sub-epochs and its K* reader;
> - the A13 260-set reads;
> - genmarg generative PER on dev-other with Hungarian and NMI (genmarg.py:489,575-581 refuse
>   dev-other), so no i6 phi run can report generative PER yet;
> - the ladder reads only the 285 set, where 25 utterances are fit items (ladder.py:508).
> None of the 34 SAE modules added on JUPITER after c49559ce is ported (A11-A20, AN-0..6, TP0), e.g.
> PhiFromKeyInitJob, EmTable/TableReverseModel, KeySearch/KeyObjectiveScreen, An5 key identity,
> PhiCompetenceBattery. The rename evidence rests on these.
> Minor: the G0.RC reads don't pass sil_run_collapse, and genmarg refuses it; greedy PER is unaffected.
> The rename phase is pushed as exp_logs/SAE/SAE_4A_rename.md (d9b7e5362). A phi-init plan follows on
> the branch.

## Checks made on i6 (SAE_i6 orchestrator, 2026-09-25)

- ACTION: the in-tree `sisyphus/` is at a567fa7 ("Job stacktrace: store without frame code objects
  (#315)"), and `git merge-base --is-ancestor d9e1ede HEAD` holds: the i6 manager already carries the fix.
  The package README still pins ddcd028, which lacks it.
- rho: `rate_rho_hz = 9.6619373279` is a literal (`training/config.py:316`). Its banked provenance
  (`tests/test_artefacts.py`, docstring) is the full phonemisation, 2,784,159,269 phones / 778,025,128
  words x 2.7 words/s, i.e. the truncated JUPITER corpus.
- G0.RC reads: `config/common.py` `train_and_read` builds `derangement_gap` and `decode_gap` for every arm
  from the extracted phi with no `sil_run_collapse` argument, so ctrl_20_rc's ep20 gaps score its phi
  under the uncollapsed lattice. The greedy PER and emitted-rate reads take no DP option.
- d9b7e5362 fetched: it adds `exp_logs/SAE/SAE_4A_rename.md` only; T7 and the truncation counts are not
  in it.
- The i6 g2p and phone-corpus counts: `reports/extract_i6_g2p_counts_2026-09-25.md`.
