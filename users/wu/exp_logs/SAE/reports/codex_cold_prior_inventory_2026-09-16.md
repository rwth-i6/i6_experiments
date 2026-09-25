DONE
§1f L1 positional-unigram/skip-1..6/tri-skipgram matcher: PER 0.8580, M1 +0.0365, M2 +0.0466, fail - SAE_1f.md:281-295
§1f G9 H c1/c2/c3/c4/c5: -2.1121/-2.0716/-2.4407/-0.0586/-0.0891; all fail - SAE_1f.md:389-430,682-693
§1f GraphUnsupASR update-40000 full/uni+bi PER 1.6828/1.2409 - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/gua_jobs/GuaTrainJob.PZo12D74ij2M
§1g Baum-Welch seg12.5 pinned bigram: no real seed/control, no gate - SAE_1g.md:187-203
§1g selected ESPUM matched-4g Gaussian/table EM count-4 PER 0.8271/0.8165 - SAE_1g.md:1152-1183
S3 flat tau 8/5.04/3.17/2/2/2/2/2: PER ep4 0.8955, best 0.8387; G4a.3 fail - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_core/returnn/training/ReturnnTrainingJob.sBlPYBA1YcIQ
c2/c3 blank; c5 greedy 0.000 ph/s, PER 1.0; tau-8 c2 blank - SAE_4A.md:895-932
S3b trigram rate/BT/consistency fail; ep8 PER 0.868-0.894 - SAE_4A.md:1030-1132
CT closed fail, ep8 PER .887926/.869733/.905441/.927249 - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/pack_jobs/PackedEmcTrainJob.nineWO8G0tFD
OR warm phi GOLD-DERIVED reporting-only; frozen/joint ep8 PER .367813/.624076 - SAE_4A.md:1310-1375
S2d supervised-10h init, all four G4a.S2d fail - SAE_4A.md:1216-1279
Cold input: frozen wav2vec2-lv60 L15 50Hz 1024-d; enc50 K=500 codebook QuantizeStatesJob.FWpGhC941JMi/store PackUnitsJob.I0uzRMfUrKWC - SAE_4A.md:69-95
Text: T_phi 39 ARPAbet+SIL; bigram/trigram Witten-Bell built from PhonemizeWithSilJob.DbFgvZOGZQ8F - SAE_4A.md:66,103,181-187
Duration: p(d|k), d_min=2, D=25, D_sil=50, SIL repeat; c3 pinned gamma mean 5 frames - SAE_4A.md:80,912
rho: label-free T_phi phones/word * 2.7 words/s / 50Hz; rho 9.6619373279/s; gold dev rate 9.8/9.4 never enters loss - reports/impl_rate_term_2026-09-15.md:3; SAE_4A.md:934-955
Selector: weighted LM perplexity picks S3b lam3 ep4; S2d selector uses dev-clean+dev-other - SAE_4A.md:981,1272-1277
Gold/GAN flags: c1/c4/warm phi use PhoneTargetHdfJob.DXDTg3VoP47A <- SeedGoldPhonesJob.zii9E9tvr51e; init i uses GanPseudoLabelJob.xjn6QnNqwEEH; MFA/GoldPhonesJob are evaluation reads - SAE_4A.md:94-96,277,927,1455-1461
Pending freeze_reverse S2d: not launched; no additional budget - SAE_4A.md:24-28,1424-1431
Executed cold config lam3_tri: feature HDF L15FeatureHdfJob.e2athsQ218Og; units HDF UnitsHdfJob.n8jHkM77HafN; prior PhoneNgramPriorJob.TRPE0D5nF3bh; eta SpeakerEtaJob.U4etvcSpsQi4; flat checkpoint FlatRecognizerInitJob.21Kxgr5JLR3k - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/pack_jobs/PackedEmcTrainJob.byYMQmBNEpLZ/output/lam3_tri/returnn.config
Executed lam3_tri/CT configs contain rate_rho_hz 9.6619373279 and no `revert` match; MISSING (searched: Pack3/lam3_tri and Pack5/lam3_ct_tri returnn.config) - work/speech_llm/sae/emc/pack_jobs/PackedEmcTrainJob.byYMQmBNEpLZ/output/lam3_tri/returnn.config; work/speech_llm/sae/emc/pack_jobs/PackedEmcTrainJob.nineWO8G0tFD/output/lam3_ct_tri/returnn.config
Gold 9.8/s DecodeStatsJob `rate_band` is pre-existing; gate reads `phone_rate`, never `rate_in_band`; rho report says “Gold rate enters nothing” - SAE_4A.md:952-955; reports/impl_rate_term_2026-09-15.md:3
OR selection: weighted LM perplexity over epochs 4-8/5,567 combined dev utterances; descriptions say no gold labels or PER enter - reports/codex_pack6_endpoint_audit_2026-09-16.md:33
§1f unary OT uses frequency/position marginals; §1f unit-BPE function-word test uses top-20 types; §1g syllabic split is two-class; §1g 4-gram whole-phone-sequence decoder is blocked/unreadable and the sequence family is documented UNRESOLVED - SAE_1f.md:149-177,178-230; SAE_1g.md:150-182,528-610,1350-1352,1438-1452
T_phi orthographic input: `librispeech-lm-norm.txt.gz` from DownloadJob.g4jClO48cAvP; TextToPhonemeJob.THKMON3k9LJQ uses that file, MergeLexiconJob.qKaOAPqURCkK and ApplyG2PModelJob.myTIGtmrUIFq - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/TextToPhonemeJob.THKMON3k9LJQ
PhonemizeWithSilJob.DbFgvZOGZQ8F uses the same `librispeech-lm-norm.txt.gz`, `max_lines: None`, `seed: 0`, `sil_prob: 0.5`, `surround: True` - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/text/PhonemizeWithSilJob.DbFgvZOGZQ8F
No paired-ID/reference input is listed in either phonemization job info; MISSING (searched: both job `info` files) - work/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/TextToPhonemeJob.THKMON3k9LJQ/info; work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/text/PhonemizeWithSilJob.DbFgvZOGZQ8F/info
Canonical orthographic refs: LibriSpeechWordRefsJob.1EsLvSbyl06D input HF dataset TransformAndMapHuggingFaceDatasetJob.OYvh9012Pgkb and GoldPhonesJob.ZGSp0hxyd2YP, dev_split `dev`, output `word_refs.json` - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/word_decode/LibriSpeechWordRefsJob.1EsLvSbyl06D
Canonical existing word scorer: Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks, inputs `word_refs.json`, 4-gram ARPA, flashlight lexicon, splits dev-clean/dev-other/train, beam 500, lm_weight 2.0, word_score -1.0 - /e/scratch/spell/wu24/2026-07-13_unsupervised/work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/word_decode/Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks
WER collection uppercases and whitespace-splits both `hypo.word` and `word_refs.json`; train references are deliberately not extracted - recipe/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/word_decode.py:90-125,296-330
CER renderer/scorer: MISSING (searched: recipe/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/ and config/sae_1d_word_decode.py)
