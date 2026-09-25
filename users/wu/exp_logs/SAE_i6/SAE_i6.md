# SAE_i6 — unsupervised ASR through an exact-marginal cycle, on the i6 cluster

Continuation of phase 4A of the SAE campaign (JUPITER, 2026-07 to 2026-09-24), moved to i6 for
compute quota. Phase 4A is the whole of SAE_i6; the other SAE phases (speech-LLM GRPO etc.) are out
of scope. Basis: `SAE_i6_ref.md` (setup, protocol, banked numbers, constraints, established results)
and its topic files `SAE_i6_ref_*.md`. JUPITER logs: `exp_logs/SAE/` (frozen provenance).

## Objective

Pure unsupervised phone recognition without GANs: make the cycle (recognizer theta + semi-Markov
reverse model phi + frozen text prior, trained jointly through an exact marginal) take off from a
cold start on LibriSpeech train-clean-100, i.e. leave the content-free band (dev-other PER
0.83-0.91) toward the wav2vec-U 2.0 reference (0.214). Seeded (10 h supervised) refinement is a
disclosed parallel analysis, never cold-start progress.

## Constraints

`SAE_i6_ref.md` section 2 governs: label quarantine, no GAN, final or label-free checkpoints only,
paired speaker-clustered comparisons, one delta per arm with an own `ctrl_20`, uniform-sample
n-grams, N = 20 sub-epochs, float64 DP, pre-registered gates. i6: L40S 46 GB GPUs, 5 per user; V100 32 GB,
32 of them with no per-user cap. The V100 fits the reference batch and is 1.32x faster per step (`SAE_i6_P0.md`,
Results, cost screen). Trainings run on V100 from 2026-09-25 (user decision). A pack runs its arm AND its own control on the same GPU type.
Project rules (documents, commits, env): `CLAUDE.md` in the setup dir.

## Queue

1. **P0 port (ACTIVE)**, `SAE_i6_P0.md`: env, i6 settings, reproduce `ctrl_20`, `k2lat_20_ma3000`
   and the gold-phi init; verify every component and loss term (gates G0.R1-R3, G0.V).
2. After P0: resume the cold line from JUPITER's final state (`SAE_i6_ref_lexlat_v2.md`, "Final reads after
   the move" and "Open after the final reads", from the JUPITER logs at 7e7c38aee). Every registered run of
   JUPITER's phase is read and audited: no label-free phi lifts a random theta, while every label-built phi in the
   basin does. JUPITER's named next steps are the rename proposal (`exp_logs/SAE/SAE_4A_rename.md`) and the
   phi-init plan (item 4), both awaiting the user. What i6 runs next, and in what order, is decided after P0 with
   the user's priorities. Port scope (user, through the JUPITER orchestrator, 2026-09-25): core experiments only.
   - Ported and matching the banked jobs: ctrl_20 (with rc and s1), k2lat, word LM/trie/HLG/VAD, p0, the A10
     durinit recipe at 12 sub-epochs with its S reader, gold phi, the rt lift ladder.
   - Core fixes, before any phi run (P1 Task A, gate G1.G): (1) generative PER on the D4 dev-other set
     (direct, Hungarian, NMI), a standing read for every phi run; (2) ladder reads on the 260 set (the 285
     set holds 25 fit items); (3) the wave default durinit at 12 sub-epochs (`WAVE_*` is None).
   - On demand only, when a funded phi step needs them: PhiFromKeyInitJob with PhiDurinitSwapJob, the AN-5
     key-identity read.
   - Not ported (conclusions stay in the JUPITER logs): A10 at 48 with K*, A11/A12 table EM, A13-A20 readers,
     key search and J screens, AN-0..6, TP0, the A15-F battery, the inventory ceiling, statistic (b), node P,
     phi_c, k2lat_20_x60 (G4a.L2.3 closed CANNOT_TELL).
   Sources: `reports/jupiter_port_review_2026-09-25.md`, `exp_logs/SAE/SAE_4A_rename.md` (d9b7e5362).
3. **P1 completion and small extensions of the JUPITER framework (ACTIVE, user 2026-09-25, in parallel with P0)**,
   `SAE_i6_P1.md`. Task A: the core phi reads (Queue 2's fixes). Task B: JUPITER's L2-0 corrupted-phi lift ladder;
   rt_r70 reproduces, rt_r80 and rt_r90 place the lift boundary (JUPITER: rho*_lift = 0.7, r100 NO LIFT); a
   label-using diagnostic. Small label-using analyses and ports of existing JUPITER pieces join P1 as tasks.
4. **Candidate, unfunded, awaiting the user's OK:** the phi-init plan (a new label-free phi initialisation: named
   alphabet coarse-to-fine, SIL anchor, oracle split arms), `exp_logs/SAE/SAE_4A_phiinit.md` (origin 7711f1513,
   audited). A new cold-start method with its own objective and gate, so it becomes its own phase if funded.
5. **GAN reproduction (user, 2026-09-25):** wav2vec-U 2.0 through fairseq, pushed by the JUPITER orchestrator
   (68b39418a, merged locally 692d6e55a). Per the user through that orchestrator, it runs inside P0 under gate
   G0.GAN. It is a reference baseline, not a cold-start method arm, so the no-GAN constraint on the method
   (`SAE_i6_ref.md` section 2) stands.

## Phase pointers

- **P0 port and verification** — OPEN. ctrl_20 finished: G0.R1 PASS under the user's amendment (step-1 matching
  clauses are diagnostics); the branch is pushed. ctrl_20_s1, ctrl_20_rc and k2lat (per-chunk k2, chunk 2) run on
  V100. The GAN reproduction (G0.GAN), with intermediate CPU eval, is in its prep jobs — `SAE_i6_P0.md`.
- **P1 completion and small extensions of the JUPITER framework** — OPEN: Task A G1.G PASS (dev-other phone read
  reproduces JUPITER's gold row, 0.1960 vs 0.193), fix 3 after P0; Task B G1.M PASS, arms r70/r80/r90 running —
  `SAE_i6_P1.md`.
