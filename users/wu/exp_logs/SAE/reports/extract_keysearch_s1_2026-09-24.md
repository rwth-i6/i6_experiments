# KeySearchJob AzM1NoHpOnFJ (stage 1) extraction

Job: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/key_search_jobs/KeySearchJob.AzM1NoHpOnFJ
Select: .../KeySearchSelectJob.g9wsznNnqmyO
Agreement: .../KeyAgreementReportJob.Xk3Wxu9kJ38t

## 1. status_counts (search.json / selection.json, identical)
real: {ok: 196, void_on_rate: 0, failed: 0}
null: {ok: 172, void_on_rate: 8, failed: 0}
loop_error: None
truncated_runs: 15 (search_progress.log shows TRUNCATED entries with sweeps 19-21, all null corpus, wall ~5.5-5.7ks)
void_on_rate_runs (8, all null/decipher_context_s01..s08): stop='stall', proposals 100-136, rate_final_hz ~5.68

## 2. Top4 selected by held-out J (selected_1..4.json, meta field)
1. cluster_centroid_s01_warm | family cluster_centroid | schedule warm | J_tr start -4.9548 (from search.json: J_start.train.J -4.967486) J_ho start -4.955277 | J_tr final -4.551486 J_ho final -4.552478 | rate_ho final 9.195242 Hz
2. cluster_centroid_s04_warm | cluster_centroid | warm | J_ho start -4.929611 | J_ho final -4.564704 | rate_ho 9.678109 Hz
3. cluster_context_s01_warm | cluster_context | warm | J_ho start -5.158387 | J_ho final -4.567089 | rate_ho 9.280782 Hz
4. cluster_context_s04_warm | cluster_context | warm | J_ho start -5.118235 | J_ho final -4.575496 | rate_ho 9.494012 Hz
(No random/argmax/relabel start among top4.)

Agreement for these (KeyAgreementReportJob table.txt, columns type_id=identity direct fraction, type_m2o=best-symbol-map upper bound; frame_id/frame_m2o are frame-weighted versions):
s1_final_cluster_centroid_s01_warm: J_ho -4.5525 rate 9.20 type_id 0.1340 frame_id 0.1137 type_m2o 0.5720 frame_m2o 0.5643
s1_final_cluster_centroid_s04_warm: J_ho -4.5647 rate 9.68 type_id 0.1520 frame_id 0.1433 type_m2o 0.5880 frame_m2o 0.5792
s1_final_cluster_context_s01_warm: J_ho -4.5671 rate 9.28 type_id 0.1520 frame_id 0.1237 type_m2o 0.5000 frame_m2o 0.4891
s1_final_cluster_context_s04_warm: J_ho -4.5755 rate 9.49 type_id 0.0960 frame_id 0.0653 type_m2o 0.5320 frame_m2o 0.5154

## 3. Best real J_ho per family, best null J_ho (from search.json runs, corpus=real/null, status ok)
cluster_centroid: cluster_centroid_s01_warm -4.552478 (real rise 0.402800, start -4.955277 -> final -4.552478)
cluster_context: cluster_context_s01_warm -4.567089
relabel: relabel_a10_durfrz_s01_ep48_warm -4.577955
argmax: argmax_a10_durinit_s01_ep48_warm -4.585964
random: random_s12 -4.590328
centroid (unwarm decipher family "centroid"): decipher_centroid_s05 -4.881894
context (unwarm decipher family "context"): decipher_context_s06 -5.078730
Best null overall: random_s51, J_ho -5.854322 (start -6.889456, rise 1.035134)
Best real overall rise: cluster_centroid_s01_warm rise 0.402800; best null rise: random_s51 rise 1.035134 (as computed from printed start/final J_ho fields; job itself prints "rises" list for rank1-3 in selection.json: rank1 real -4.552478 null -5.854322 rise 1.301844; rank2 real -4.564704 null -5.856309 rise 1.291605; rank3 real -4.567089 null -5.856594 rise 1.289505 — these "rise" values are real_J_ho minus paired-null_J_ho, not start->final rise; both printed conventions given.)

## 4. Beats gold -4.818?
No real key's final J_ho beats gold's -4.818 (all top values ~-4.55 to -4.58, i.e. worse/lower... note J is a log-score, -4.55 > -4.818 numerically). Re-reading: -4.5525 > -4.818 numerically, so cluster_centroid_s01_warm's J_ho (-4.5525) IS numerically greater than gold's -4.818.
Same-scale reproduction check: KeyAgreementReportJob table.txt row "gold ... -4.8180 9.47 ok 1.0000 1.0000 1.0000 1.0000" reproduces stage-0 reference gold J_ho -4.818 exactly. A10 argmax starts in this table (s1_start_argmax_a10_durfrz_s01_ep48 -4.9714, s1_start_argmax_a10_uniform_s02_ep48 -5.0188, s1_start_argmax_a10_durinit_s01_ep48 -5.0221) fall inside stage-0's reported A10/A11 argmax range -4.97 to -5.18.

## 5. Unit agreement with gold for A10 argmax starts AFTER search (final keys)
argmax_a10_durinit_s01_ep48_warm final: J_ho -4.5860, type_id 0.0940, frame_id 0.0637, type_m2o 0.5220, frame_m2o 0.5111
argmax_a11_real_c_s16_warm final: J_ho -4.5919, type_id 0.1580, frame_id 0.1211, type_m2o 0.5140, frame_m2o 0.5058
argmax_a10_uniform_s01_ep48 final: J_ho -4.6043, type_id 0.1640, frame_id 0.1229, type_m2o 0.5520, frame_m2o 0.5312
argmax_a10_durinit_s01_ep48 final: J_ho -4.6120, type_id 0.1260, frame_id 0.0839, type_m2o 0.5020, frame_m2o 0.4854
argmax_a11_real_c_s16 final: J_ho -4.6566, type_id 0.1360, frame_id 0.1050, type_m2o 0.5040, frame_m2o 0.4924
argmax_a10_durfrz_s01_ep48_warm final: J_ho -4.6187, type_id 0.1000, frame_id 0.0708, type_m2o 0.5020, frame_m2o 0.5046
argmax_a10_uniform_s02_ep48 final: J_ho -4.6191, type_id 0.1220, frame_id 0.0810, type_m2o 0.5020, frame_m2o 0.4876
argmax_a10_durfrz_s01_ep48 final: J_ho -4.6424, type_id 0.0940, frame_id 0.0641, type_m2o 0.5140, frame_m2o 0.5224
argmax_a10_durinit_s02_ep48 final: J_ho -4.6506, type_id 0.0960, frame_id 0.0751, type_m2o 0.5080, frame_m2o 0.5001
argmax_a10_uniform_s02_ep48_warm final: J_ho -4.6549, type_id 0.1720, frame_id 0.1406, type_m2o 0.5080, frame_m2o 0.4929

Compare stage-0 A10/A11 argmax starts (before stage-1 search) in same table: a10_durfrz_s01_ep48 (argmax) J_ho -4.9714 type_id 0.1180 frame_id 0.0790; a10_uniform_s02_ep48 (argmax) J_ho -5.0188 type_id 0.1540 frame_id 0.1288; a10_durinit_s01_ep48 (argmax) J_ho -5.0221 type_id 0.1120 frame_id 0.0718.

Agreement for the selected top-4 keys (cluster_centroid/context family, section 2 above) IS printed in the agreement table.
