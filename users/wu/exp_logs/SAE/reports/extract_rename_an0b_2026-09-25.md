# AN-0b extraction (2026-09-25)

Source: work/speech_llm/sae/emc/rename_escape_jobs/RenameEscapeJob.cw4BeJzrQ3U9/output/ (report.txt:1-1873, an0b.json, n_su.npz)

## 1. Verdict/reading line (report.txt:1)
"AN-0b READINGS: E-STEP CHECK does not fail (10 of 10 MAP paths re-added, max rel 6.40e-07 <= 0.0001); ESCAPE NOT THE BLOCK; SIL ESCAPE (descriptive; VALID)"

Setup line (report.txt:3): H_LM = 2.257108 nats/token, source PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz
report.txt:4: 300 train utterances (A12 selection, seed 0); tau 1.0; E-step emission 0.9 table + 0.1 uniform; type-level M-step (pseudo-count 0.001); weighted against GoldUnitKeyJob.sLnMRRd2qO0t/output/key.json
report.txt:5: reference derangements (gold-row split): {'5pair_s1': ['AH<->T','AO<->N','K<->OW','M<->Y','P<->S'], '1pair_s1': ['AH<->T']}

## 2. E-step check (report.txt: MAP paths block, ~lines 79-88)
All 10 MAP paths extracted (5 utterances x {gold_key, gold_key__5pair_s1} rows). Per-path rel diff between lattice score and independent re-add ranges 7.91e-08 to 6.40e-07 (all <= 1e-4). Bridge log Z tau-1 (matmul vs elementwise) rel values listed as 0.0e+00 to 4.2e-16 per path (all "bridge_rel_diff" <= 1e-4 per the report). Ties noted per path ("N segments, M with mass < 0.99 (ties)") — see report.txt lines ~79-88 for exact per-path M values (6,8,34,34,21,28,8,12,16,16).

Guard lines (report.txt:7-8):
"gold-row guard, as_is (VALID iff gold identity after >= 0.95): plain/1 VALID (0.9964), plain/4.4 VALID (0.9888), rate_neutral/1 VALID (0.9964), rate_neutral/4.4 VALID (0.9834)"
"gold-row guard, no_escape (VALID iff gold identity after >= 0.95): plain/1 VALID (0.9964), plain/4.4 VALID (0.9888), rate_neutral/1 VALID (0.9964), rate_neutral/4.4 VALID (0.9834)"

## 3. Per-cell/row/operator table (report.txt:10-33, verbatim columns: row | operator | cell | restore | rho | id_after | tok/fr | SIL | escape | re-keyed | guard)

row=gold_key:
- as_is plain/1: restore 0.0000, rho -0.0036, id_aft 0.9964, tok/fr 0.2356, SIL 0.0803, escape 0.0008, re-keyed 3, guard VALID
- as_is plain/4.4: restore 0.0000, rho -0.0112, id_aft 0.9888, tok/fr 0.1618, SIL 0.1067, escape 0.0236, re-keyed 10, guard VALID
- as_is rate_neutral/1: restore 0.0000, rho -0.0036, id_aft 0.9964, tok/fr 0.2356, SIL 0.0803, escape 0.0008, re-keyed 3, guard VALID (shared with plain/1)
- as_is rate_neutral/4.4: restore 0.0000, rho -0.0166, id_aft 0.9834, tok/fr 0.2524, SIL 0.1105, escape 0.0015, re-keyed 8, guard VALID
- no_escape plain/1: restore 0.0000, rho -0.0036, id_aft 0.9964, tok/fr 0.2356, SIL 0.0804, escape 0.0000, re-keyed 3, guard VALID
- no_escape plain/4.4: restore 0.0000, rho -0.0112, id_aft 0.9888, tok/fr 0.1632, SIL 0.1080, escape 0.0000, re-keyed 7, guard VALID
- no_escape rate_neutral/1: restore 0.0000, rho -0.0036, id_aft 0.9964, tok/fr 0.2356, SIL 0.0804, escape 0.0000, re-keyed 3, guard VALID (shared with plain/1)
- no_escape rate_neutral/4.4: restore 0.0000, rho -0.0166, id_aft 0.9834, tok/fr 0.2525, SIL 0.1106, escape 0.0000, re-keyed 8, guard VALID

row=gold_key__5pair_s1:
- as_is plain/1: restore 0.0000, rho -0.0016, id_aft 0.5916, tok/fr 0.2312, SIL 0.0786, escape 0.0045, re-keyed 8, guard VALID
- as_is plain/4.4: restore 0.0001, rho -0.0160, id_aft 0.5771, tok/fr 0.1367, SIL 0.1190, escape 0.0567, re-keyed 35, guard VALID
- as_is rate_neutral/1: restore 0.0000, rho -0.0016, id_aft 0.5916, tok/fr 0.2312, SIL 0.0786, escape 0.0045, re-keyed 8, guard VALID (shared with plain/1)
- as_is rate_neutral/4.4: restore 0.0053, rho +0.0053, id_aft 0.5985, tok/fr 0.2528, SIL 0.1184, escape 0.0055, re-keyed 15, guard VALID
- no_escape plain/1: restore 0.0000, rho -0.0016, id_aft 0.5916, tok/fr 0.2314, SIL 0.0788, escape 0.0000, re-keyed 8, guard VALID
- no_escape plain/4.4: restore 0.0000, rho -0.0181, id_aft 0.5750, tok/fr 0.1386, SIL 0.1234, escape 0.0000, re-keyed 29, guard VALID
- no_escape rate_neutral/1: restore 0.0000, rho -0.0016, id_aft 0.5916, tok/fr 0.2314, SIL 0.0788, escape 0.0000, re-keyed 8, guard VALID (shared with plain/1)
- no_escape rate_neutral/4.4: restore 0.0053, rho +0.0053, id_aft 0.5985, tok/fr 0.2530, SIL 0.1191, escape 0.0000, re-keyed 14, guard VALID

row=gold_key__1pair_s1 (descriptive, no reading):
- as_is plain/1: restore 0.0041, rho +0.0040, id_aft 0.8577, tok/fr 0.2336, SIL 0.0797, escape 0.0026, re-keyed 6, guard VALID
- as_is plain/4.4: restore 0.0028, rho -0.0141, id_aft 0.8397, tok/fr 0.1525, SIL 0.1092, escape 0.0296, re-keyed 18, guard VALID
- as_is rate_neutral/1: restore 0.0041, rho +0.0040, id_aft 0.8577, tok/fr 0.2336, SIL 0.0797, escape 0.0026, re-keyed 6, guard VALID (shared with plain/1)
- as_is rate_neutral/4.4: restore 0.0101, rho +0.0024, id_aft 0.8561, tok/fr 0.2524, SIL 0.1115, escape 0.0025, re-keyed 12, guard VALID
- no_escape plain/1: restore 0.0041, rho +0.0040, id_aft 0.8577, tok/fr 0.2336, SIL 0.0798, escape 0.0000, re-keyed 6, guard VALID
- no_escape plain/4.4: restore 0.0028, rho -0.0153, id_aft 0.8384, tok/fr 0.1537, SIL 0.1112, escape 0.0000, re-keyed 15, guard VALID
- no_escape rate_neutral/1: restore 0.0041, rho +0.0040, id_aft 0.8577, tok/fr 0.2336, SIL 0.0798, escape 0.0000, re-keyed 6, guard VALID (shared with plain/1)
- no_escape rate_neutral/4.4: restore 0.0074, rho -0.0003, id_aft 0.8534, tok/fr 0.2524, SIL 0.1119, escape 0.0000, re-keyed 11, guard VALID

## Swapped-unit mass split (report.txt:35-56; right/partner/SIL/masked-or-empty/other)
gold_key: escape symbols ['OY','ZH']; moved gold frame share 0.0000
- as_is plain/1 [5pair_s1]: right .808/partner .009/SIL .013/masked-empty .001/other .168 (170/173 units)
- as_is plain/1 [1pair_s1]: right .793/.020/.011/.001/.175 (62/64 units)
- as_is plain/4.4 [5pair_s1]: .636/.021/.039/.025/.278
- as_is plain/4.4 [1pair_s1]: .592/.045/.041/.026/.295
- as_is rate_neutral/4.4 [5pair_s1]: .722/.012/.049/.002/.215
- as_is rate_neutral/4.4 [1pair_s1]: .697/.025/.047/.002/.230
- no_escape plain/1 [5pair_s1]: .809/.009/.013/.000/.169
- no_escape plain/1 [1pair_s1]: .793/.020/.011/.000/.176
- no_escape plain/4.4 [5pair_s1]: .647/.022/.041/.000/.291
- no_escape plain/4.4 [1pair_s1]: .602/.047/.043/.000/.308
- no_escape rate_neutral/4.4 [5pair_s1]: .723/.012/.050/.000/.216
- no_escape rate_neutral/4.4 [1pair_s1]: .697/.025/.047/.000/.231

gold_key__5pair_s1: drawn ['AH<->T','AO<->N','K<->OW','M<->Y','P<->S']; escape ['OY','ZH']; moved gold frame share 0.4068
- as_is plain/1 [own]: right .027/partner .695/SIL .013/masked-empty .007/other .258 (170/173)
- as_is plain/4.4 [own]: .042/.415/.063/.069/.412
- as_is rate_neutral/4.4 [own]: .060/.488/.070/.009/.373
- no_escape plain/1 [own]: .028/.697/.014/.000/.261
- no_escape plain/4.4 [own]: .044/.434/.069/.000/.453
- no_escape rate_neutral/4.4 [own]: .060/.489/.071/.000/.379

gold_key__1pair_s1: drawn ['AH<->T']; escape ['OY','ZH']; moved gold frame share 0.1463
- as_is plain/1 [own]: .068/.647/.012/.007/.266 (62/64)
- as_is plain/4.4 [own]: .086/.422/.053/.037/.401
- as_is rate_neutral/4.4 [own]: .126/.479/.058/.004/.333
- no_escape plain/1 [own]: .069/.650/.012/.000/.269
- no_escape plain/4.4 [own]: .089/.435/.055/.000/.421
- no_escape rate_neutral/4.4 [own]: .126/.480/.058/.000/.335

## 4. MAP path excerpt (report.txt line ~90 onward): gold_key__5pair_s1 1723-141149-0038, rate_neutral/4.4, as-is
```
   0   6  SIL              | gold SIL(5) AE(1)
   6   5  AE               | gold AE(4) N(1)
  11   2  N   [row of AO]  | gold N(2)
  13   2  D                | gold D(2)
  15   5  HH               | gold HH(4) EH(1)
  20   3  EH               | gold EH(3)
  23   2  L                | gold N(2)
  25   8  P   [row of S]   | gold N(2) S(6)
  33   8  SIL              | gold S(4) SIL(2) IH(2)
  41   3  IH               | gold IH(2) T(1)
```
(first 10 of the segment list; report.txt continues beyond line 100)

## 5. Runtime and warnings
usage.run.1: used_time 0.057625200086169774 (fraction of requested time 0.5 h) -> ~3.46 minutes; requested_resources: cpu 4, gpu 1, mem 24.0 GB, time 0.5 h, engine gpupack, partition booster, --exclusive. host jpbo-012-01.jupiter.internal.
No "warn"/"error"/"traceback" strings found in log.run.1 (grep -in "warn|error|traceback" returned nothing).
