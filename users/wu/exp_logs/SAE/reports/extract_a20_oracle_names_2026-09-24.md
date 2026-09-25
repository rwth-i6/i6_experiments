A20 -- does J reward the right names on the found partitions?  DISCLOSED LABEL-USING ANALYSIS (the gold key renames the keys); gates nothing registered
held-out: 260 utterances (137933 frames), A13 disjoint set; train side: 28254 utterances (15275716 frames); J = stage 0/1's key_search_jobs._j_pair (lm + emis + dur per frame; rate band [5.8, 14.49] Hz, VOID printed, not applied)
frame weights: train-side unit frame counts (15275716 frames), KeyAgreementReportJob's
(a) as found; (b) Hungarian 1:1 rename maximising frame-weighted identity with gold, partition unchanged; (c) each symbol -> its majority gold phone (frame-weighted); frame_id / frame_m2o = identity / many-to-one agreement with the gold key
rule: READING RULE (registered 2026-09-24, SAE_4A_lexlat_v2.md A20, fixed before any result), on the 4 selected keys, with d = J(b) - J(a) held out on the 260 set in nats per frame: NAMES VISIBLE if d > 0.01 (A7's floor) on at least 3 of the 4. J then rewards the right names on the found partitions, and a global rename move (an assignment over names under J, or S for phi) is the cost work. NAME-BLIND if d <= 0.01 on at least 3 of the 4. The trigram term then does not reward the right names on these partitions, a rename step would not help, and the cost work goes to the objective's name signal. MIXED otherwise. Reported beside, never read by the rule: each variant's term decomposition, J(c), the 1:1 identity agreement after (b) (it separates wrong names from merges), and the same rows for the A10 / A11 finals and the scale keys. The rule reads J as computed; a variant's VOID flag is printed, not applied.

key                                group         var      J_ho       lm     emis      dur   rate void   J-J(a) frame_id frame_m2o n_sym  same-scale |dJ(a)|
selected_1                         selected      (a)   -4.5525  -0.9484  -3.1290  -0.4750   9.20   ok  +0.0000   0.1137    0.5643    40  0.0e+00 vs stage-1 search.json
selected_1                         selected      (b)   -5.1258  -1.5218  -3.1290  -0.4750   9.20   ok  -0.5734   0.4652    0.5643    40
selected_1                         selected      (c)   -5.1292  -1.0838  -3.5819  -0.4635   8.02   ok  -0.5767   0.5643    0.5643    23
selected_2                         selected      (a)   -4.5647  -0.9483  -3.1364  -0.4800   9.68   ok  +0.0000   0.1433    0.5792    40  0.0e+00 vs stage-1 search.json
selected_2                         selected      (b)   -5.0006  -1.3842  -3.1364  -0.4800   9.68   ok  -0.4359   0.4730    0.5792    40
selected_2                         selected      (c)   -5.0390  -0.9609  -3.6098  -0.4683   8.95   ok  -0.4742   0.5792    0.5792    25
selected_3                         selected      (a)   -4.5671  -0.9457  -3.1450  -0.4764   9.28   ok  +0.0000   0.1237    0.4891    40  0.0e+00 vs stage-1 search.json
selected_3                         selected      (b)   -4.9889  -1.3675  -3.1450  -0.4764   9.28   ok  -0.4218   0.3943    0.4891    40
selected_3                         selected      (c)   -5.1413  -1.0678  -3.6121  -0.4614   8.43   ok  -0.5742   0.4891    0.4891    22
selected_4                         selected      (a)   -4.5755  -0.9754  -3.1223  -0.4779   9.49   ok  +0.0000   0.0653    0.5154    40  0.0e+00 vs stage-1 search.json
selected_4                         selected      (b)   -5.0104  -1.4102  -3.1223  -0.4779   9.49   ok  -0.4349   0.4284    0.5154    40
selected_4                         selected      (c)   -5.0649  -1.0688  -3.5297  -0.4664   8.94   ok  -0.4894   0.5154    0.5154    26
argmax_a10_durfrz_s01_ep48         argmax_final  (a)   -4.6424  -1.0014  -3.1668  -0.4742   9.18   ok  +0.0000   0.0641    0.5224    40  0.0e+00 vs stage-1 search.json
argmax_a10_durfrz_s01_ep48         argmax_final  (b)   -5.0355  -1.3944  -3.1668  -0.4742   9.18   ok  -0.3930   0.4267    0.5224    40
argmax_a10_durfrz_s01_ep48         argmax_final  (c)   -5.0867  -0.9970  -3.6283  -0.4615   8.57   ok  -0.4443   0.5224    0.5224    24
argmax_a10_durfrz_s02_ep48         argmax_final  (a)   -4.7108  -1.0105  -3.2188  -0.4815   9.00   ok  +0.0000   0.1367    0.4845    40  0.0e+00 vs stage-1 search.json
argmax_a10_durfrz_s02_ep48         argmax_final  (b)   -5.2687  -1.5684  -3.2188  -0.4815   9.00   ok  -0.5578   0.4003    0.4845    40
argmax_a10_durfrz_s02_ep48         argmax_final  (c)   -5.1618  -1.0847  -3.6099  -0.4672   8.22   ok  -0.4509   0.4845    0.4845    26
argmax_a10_durinit_s01_ep48        argmax_final  (a)   -4.6120  -0.9446  -3.1981  -0.4693   8.92   ok  +0.0000   0.0839    0.4854    40  0.0e+00 vs stage-1 search.json
argmax_a10_durinit_s01_ep48        argmax_final  (b)   -5.1044  -1.4370  -3.1981  -0.4693   8.92   ok  -0.4924   0.3890    0.4854    40
argmax_a10_durinit_s01_ep48        argmax_final  (c)   -5.2600  -1.0392  -3.7573  -0.4635   8.44   ok  -0.6480   0.4854    0.4854    19
argmax_a10_durinit_s02_ep48        argmax_final  (a)   -4.6506  -0.9565  -3.2128  -0.4813   8.98   ok  +0.0000   0.0751    0.5001    40  0.0e+00 vs stage-1 search.json
argmax_a10_durinit_s02_ep48        argmax_final  (b)   -5.2668  -1.5727  -3.2128  -0.4813   8.98   ok  -0.6162   0.3938    0.5001    40
argmax_a10_durinit_s02_ep48        argmax_final  (c)   -5.2689  -1.1416  -3.6548  -0.4725   8.53   ok  -0.6183   0.5001    0.5001    21
argmax_a10_uniform_s01_ep48        argmax_final  (a)   -4.6043  -0.9607  -3.1767  -0.4669   8.76   ok  +0.0000   0.1229    0.5312    40  0.0e+00 vs stage-1 search.json
argmax_a10_uniform_s01_ep48        argmax_final  (b)   -5.0374  -1.3937  -3.1767  -0.4669   8.76   ok  -0.4330   0.4688    0.5312    40
argmax_a10_uniform_s01_ep48        argmax_final  (c)   -5.1226  -1.1389  -3.5191  -0.4647   8.57   ok  -0.5183   0.5312    0.5312    26
argmax_a10_uniform_s02_ep48        argmax_final  (a)   -4.6191  -0.9491  -3.2005  -0.4695   9.00   ok  +0.0000   0.0810    0.4876    40  0.0e+00 vs stage-1 search.json
argmax_a10_uniform_s02_ep48        argmax_final  (b)   -5.1050  -1.4351  -3.2005  -0.4695   9.00   ok  -0.4860   0.4114    0.4876    40
argmax_a10_uniform_s02_ep48        argmax_final  (c)   -5.1121  -1.0099  -3.6316  -0.4706   8.40   ok  -0.4930   0.4876    0.4876    24
argmax_a11_real_c_s16              argmax_final  (a)   -4.6566  -0.9780  -3.2060  -0.4726   8.98   ok  +0.0000   0.1050    0.4924    40  0.0e+00 vs stage-1 search.json
argmax_a11_real_c_s16              argmax_final  (b)   -5.0821  -1.4036  -3.2060  -0.4726   8.98   ok  -0.4255   0.4042    0.4924    40
argmax_a11_real_c_s16              argmax_final  (c)   -5.2506  -1.0592  -3.7295  -0.4619   8.25   ok  -0.5941   0.4924    0.4924    22
gold                               scale         (a)   -4.8180  -0.9897  -3.3468  -0.4815   9.47   ok  +0.0000   1.0000    1.0000    38  0.0e+00 vs stage-0 table.json
gold                               scale         (b)   -4.8180  -0.9897  -3.3468  -0.4815   9.47   ok  +0.0000   1.0000    1.0000    38
gold                               scale         (c)   -4.8180  -0.9897  -3.3468  -0.4815   9.47   ok  +0.0000   1.0000    1.0000    38
K30_s1                             scale         (a)   -5.6022  -1.3991  -3.7141  -0.4889   9.53   ok  +0.0000   0.6929    0.7064    40  0.0e+00 vs stage-0 table.json
K30_s1                             scale         (b)   -5.6059  -1.4028  -3.7141  -0.4889   9.53   ok  -0.0037   0.6929    0.7064    40
K30_s1                             scale         (c)   -5.5862  -1.3630  -3.7317  -0.4915   9.53   ok  +0.0160   0.7064    0.7064    36
K30_s2                             scale         (a)   -5.6414  -1.4368  -3.7178  -0.4868   9.33   ok  +0.0000   0.6814    0.6960    40  0.0e+00 vs stage-0 table.json
K30_s2                             scale         (b)   -5.6414  -1.4368  -3.7178  -0.4868   9.33   ok  +0.0000   0.6814    0.6960    40
K30_s2                             scale         (c)   -5.6095  -1.3316  -3.7908  -0.4871   9.35   ok  +0.0319   0.6960    0.6960    35
K30_s3                             scale         (a)   -5.5866  -1.4469  -3.6490  -0.4907   9.70   ok  +0.0000   0.6963    0.7213    40  0.0e+00 vs stage-0 table.json
K30_s3                             scale         (b)   -5.5866  -1.4469  -3.6490  -0.4907   9.70   ok  +0.0000   0.6963    0.7213    40
K30_s3                             scale         (c)   -5.5321  -1.3442  -3.6912  -0.4967   9.59   ok  +0.0545   0.7213    0.7213    35
K30_s4                             scale         (a)   -5.6200  -1.4247  -3.7065  -0.4888   9.44   ok  +0.0000   0.7047    0.7207    40  0.0e+00 vs stage-0 table.json
K30_s4                             scale         (b)   -5.6200  -1.4247  -3.7065  -0.4888   9.44   ok  +0.0000   0.7047    0.7207    40
K30_s4                             scale         (c)   -5.5699  -1.3294  -3.7494  -0.4911   9.60   ok  +0.0502   0.7207    0.7207    33
K30_s5                             scale         (a)   -5.5983  -1.4109  -3.7027  -0.4847   9.60   ok  +0.0000   0.7031    0.7183    40  0.0e+00 vs stage-0 table.json
K30_s5                             scale         (b)   -5.5866  -1.3992  -3.7027  -0.4847   9.60   ok  +0.0117   0.7052    0.7183    40
K30_s5                             scale         (c)   -5.5485  -1.2997  -3.7627  -0.4862   9.69   ok  +0.0497   0.7183    0.7183    32
K70_s1                             scale         (a)   -6.3678  -1.8003  -4.0835  -0.4840   9.44   ok  +0.0000   0.3013    0.3639    40  0.0e+00 vs stage-0 table.json
K70_s1                             scale         (b)   -6.3468  -1.7792  -4.0835  -0.4840   9.44   ok  +0.0211   0.3071    0.3639    40
K70_s1                             scale         (c)   -6.2253  -1.4155  -4.3170  -0.4928   9.52   ok  +0.1425   0.3639    0.3639    26
K70_s2                             scale         (a)   -6.2992  -1.7598  -4.0562  -0.4832   9.60   ok  +0.0000   0.3052    0.3564    40  0.0e+00 vs stage-0 table.json
K70_s2                             scale         (b)   -6.3197  -1.7802  -4.0562  -0.4832   9.60   ok  -0.0204   0.3146    0.3564    40
K70_s2                             scale         (c)   -6.2031  -1.3765  -4.3293  -0.4973   9.52   ok  +0.0962   0.3564    0.3564    27
K70_s3                             scale         (a)   -6.3237  -1.8133  -4.0297  -0.4808   9.61   ok  +0.0000   0.3060    0.3615    40  0.0e+00 vs stage-0 table.json
K70_s3                             scale         (b)   -6.3265  -1.8192  -4.0297  -0.4776   9.64   ok  -0.0028   0.3183    0.3615    40
K70_s3                             scale         (c)   -6.2289  -1.4703  -4.2682  -0.4904   9.54   ok  +0.0948   0.3615    0.3615    28
K70_s4                             scale         (a)   -6.2819  -1.6967  -4.0936  -0.4915   9.30   ok  +0.0000   0.2921    0.3332    40  0.0e+00 vs stage-0 table.json
K70_s4                             scale         (b)   -6.2928  -1.7077  -4.0936  -0.4915   9.30   ok  -0.0110   0.2964    0.3332    40
K70_s4                             scale         (c)   -6.2277  -1.3829  -4.3373  -0.5074   9.14   ok  +0.0542   0.3332    0.3332    26
K70_s5                             scale         (a)   -6.3950  -1.9007  -4.0107  -0.4837   9.82   ok  +0.0000   0.2787    0.3431    40  0.0e+00 vs stage-0 table.json
K70_s5                             scale         (b)   -6.3102  -1.8158  -4.0107  -0.4837   9.82   ok  +0.0848   0.2907    0.3431    40
K70_s5                             scale         (c)   -6.3078  -1.3852  -4.3988  -0.5238   8.86   ok  +0.0872   0.3431    0.3431    22

selected keys, J(b) - J(a) held out (margin 0.01 nats/frame): selected_1 (cluster_centroid_s01_warm) -0.5734 <= 0.01; selected_2 (cluster_centroid_s04_warm) -0.4359 <= 0.01; selected_3 (cluster_context_s01_warm) -0.4218 <= 0.01; selected_4 (cluster_context_s04_warm) -0.4349 <= 0.01
selected keys, J(c) - J(a) held out: selected_1 -0.5767; selected_2 -0.4742; selected_3 -0.5742; selected_4 -0.4894
selected keys, distinct symbols after (c): selected_1 23; selected_2 25; selected_3 22; selected_4 26
scale: K30 seed mean J(a) held out -5.6097 (n=5)
scale: K70 seed mean J(a) held out -6.3335 (n=5)
scale: J(gold) held out -4.8180
same-scale check: 22 keys within 1e-06 of their stage-1 / stage-0 values; J(gold) rounds to -4.8180: True
counts: d > 0.01 on 0 of 4; d <= 0.01 on 4 of 4
VERDICT: NAME-BLIND

symbols (phi's axis order): 0:AA 1:AE 2:AH 3:AO 4:AW 5:AY 6:B 7:CH 8:D 9:DH 10:EH 11:ER 12:EY 13:F 14:G 15:HH 16:IH 17:IY 18:JH 19:K 20:L 21:M 22:N 23:NG 24:OW 25:OY 26:P 27:R 28:S 29:SH 30:T 31:TH 32:UH 33:UW 34:V 35:W 36:Y 37:Z 38:ZH 39:SIL

--- extractor notes ---
Source: work/speech_llm/sae/emc/key_oracle_names_jobs/KeyOracleNameReadJob.QD3S2xcmg0Wt/output/table.txt
(resolved via readlink -f from output alias)
