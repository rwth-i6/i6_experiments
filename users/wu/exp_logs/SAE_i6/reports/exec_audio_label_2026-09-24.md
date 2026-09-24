# FFMPEG_PIN_ACCEPT label: P0 graph re-derivation (2026-09-24, nothing launched)

settings.py lines 25-26:
# pin check 2026-09-24: 2850/2864 dev-other recordings differ; reports/exec_pincheck2_2026-09-24.md
FFMPEG_PIN_ACCEPT = "x86_64-conda-ffmpeg-7.1.1-i6"

## Key jobs (config/sae_i6_p0.py)
- FfmpegPinCheckJob.wYIhhTGwejQe (alias corpora/LibriSpeech/ffmpeg_pin_check), accept_label='x86_64-conda-ffmpeg-7.1.1-i6'
- BlankfreeVadHdfJob.RLrgIh6lFv9m (sae/4a/data/vad): counts_report_only=True (hash_exclude default False, so the flag is hashed), out_counts_report=.../output/counts_vs_expected.json
- ReturnnTrainingJob.GiT88bxzoZbZ ctrl_20; DvVfxf1LrCBi ctrl_20_s1; jcKXbLMDk4hl k2lat_20_ma3000; Ac2eioZbRX7d supervised_goldphi/phi_init; Y1vbqR6KeJSx supervised_blankfree/training

## Accept path (data/ffmpeg_pin.py run())
With accept_label set, mismatches are written as MISMATCH lines plus an 'ACCEPTED under FFMPEG_PIN_ACCEPT=...' line in the report; the RuntimeError is raised only when accept_label is None; the ffmpeg wrapper is then written. So it re-encodes, records the mismatch as ACCEPTED and finishes.

## Counts
sae_i6_p0.out:COUNT 139 [('ApplyG2PModelJob', 1), ('AssignUnitsJob', 1), ('BlankfreeDecodeGapJob', 3), ('BlankfreeDerangementGapJob', 3), ('BlankfreeDurationPriorMeanJob', 1), ('BlankfreeGreedyPerJob', 13), ('BlankfreeSeedSupportJob', 1), ('BlankfreeVadHdfJob', 1), ('BlissChangeEncodingJob', 3), ('BlissLexiconToG2PLexiconJob', 1), ('BlissToOggZipJob', 3), ('CloneGitRepositoryJob', 1), ('CollectOovWordsJob', 1), ('CreateBinaryLMJob', 1), ('CvHoldoutSplitJob', 2), ('DownloadHuggingFaceSnapshotJob', 3), ('DownloadJob', 2), ('DownloadLibriSpeechCorpusJob', 3), ('DownloadLibriSpeechMetadataJob', 1), ('ExtractSubmoduleCheckpointJob', 16), ('FfmpegPinCheckJob', 1), ('FlatRecognizerInitJob', 2), ('GetBestPtCheckpointJob', 1), ('GoldPhonesJob', 1), ('JsRowsReadJob', 1), ('KenLMplzJob', 1), ('L15ForwardSeqOrderJob', 7), ('LexiconFromTextFileJob', 1), ('LexiconTrieBuildJob', 1), ('LexlatHLGBuildJob', 1), ('LibriSpeechCreateBlissCorpusJob', 3), ('LibriSpeechSplitIdsJob', 1), ('MergeLexiconJob', 1), ('MergeUnitsPklJob', 1), ('PackUnitsJob', 1), ('PairedPerDeltaJob', 5), ('PhoneNgramPriorJob', 1), ('PhoneTargetHdfJob', 1), ('PhonemizeWithSilJob', 1), ('PipelineJob', 1), ('QuantizeStatesJob', 1), ('ReturnnForwardJobV2', 26), ('ReturnnTrainingJob', 5), ('ReverseGapItemsJob', 6), ('SampleLinesJob', 1), ('SeedGoldPhonesJob', 1), ('SpeakerEtaJob', 1), ('SupervisedReverseDataJob', 1), ('TrainG2PModelJob', 1), ('WordWindowReplayJob', 1), ('WriteLexiconJob', 1)]
sae_i6_p0_screen.out:COUNT 75 [('ApplyG2PModelJob', 1), ('AssignUnitsJob', 1), ('BlankfreeDecodeGapJob', 1), ('BlankfreeDerangementGapJob', 1), ('BlankfreeDurationPriorMeanJob', 1), ('BlankfreeGreedyPerJob', 4), ('BlankfreeVadHdfJob', 1), ('BlissChangeEncodingJob', 3), ('BlissLexiconToG2PLexiconJob', 1), ('BlissToOggZipJob', 3), ('CloneGitRepositoryJob', 1), ('CollectOovWordsJob', 1), ('CvHoldoutSplitJob', 1), ('DownloadHuggingFaceSnapshotJob', 2), ('DownloadJob', 2), ('DownloadLibriSpeechCorpusJob', 3), ('DownloadLibriSpeechMetadataJob', 1), ('ExtractSubmoduleCheckpointJob', 5), ('FfmpegPinCheckJob', 1), ('FlatRecognizerInitJob', 1), ('GoldPhonesJob', 1), ('L15ForwardSeqOrderJob', 7), ('LexiconFromTextFileJob', 1), ('LibriSpeechCreateBlissCorpusJob', 3), ('LibriSpeechSplitIdsJob', 1), ('MergeLexiconJob', 1), ('MergeUnitsPklJob', 1), ('PackUnitsJob', 1), ('PhoneNgramPriorJob', 1), ('PhonemizeWithSilJob', 1), ('PipelineJob', 1), ('QuantizeStatesJob', 1), ('ReturnnForwardJobV2', 13), ('ReturnnTrainingJob', 1), ('ReverseGapItemsJob', 2), ('SampleLinesJob', 1), ('SpeakerEtaJob', 1), ('TrainG2PModelJob', 1), ('WriteLexiconJob', 1)]

## GPU jobs and routed partition
### sae_i6_p0_screen
GPU i6_core/returnn/forward/ReturnnForwardJobV2.5z8bUupJ1f8j run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.AjTxcYsLi4E6 run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.GEVPu8mAuhZk run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.NxXNQ8v87CCM run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.QO3M1P9dOc2o run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.UzlPyMEw1eR4 run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.gKVMu7OAH04P run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.kaertnfvKAjF run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.khVKNe5qaTxE run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.qH5zoc9J9nxK run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.vVoEfyB8FVzF run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/training/ReturnnTrainingJob.GiT88bxzoZbZ run gpu_mem 96 {'sbatch_args': ['-p', 'gpu_48gb']}
### sae_i6_p0
GPU i6_core/returnn/forward/ReturnnForwardJobV2.3dtjN0OvY5n3 run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.5z8bUupJ1f8j run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.AjTxcYsLi4E6 run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.GEVPu8mAuhZk run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.HkvgYqWPtkCP run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.NxXNQ8v87CCM run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.QO3M1P9dOc2o run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.SELLIQxMB489 run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.StxzO95AzdXW run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.UzlPyMEw1eR4 run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.XF7QYBMBn8zx run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.XtscZAKUvIT6 run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.YqzN7zA1uZoJ run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.gKVMu7OAH04P run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.iQpVGOrmOZ6z run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.kHRkbFf5Zoo6 run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.kaertnfvKAjF run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.khVKNe5qaTxE run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.qH5zoc9J9nxK run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/forward/ReturnnForwardJobV2.vVoEfyB8FVzF run gpu_mem 24 {'sbatch_args': ['-p', 'gpu_24gb']}
GPU i6_core/returnn/training/ReturnnTrainingJob.Ac2eioZbRX7d run gpu_mem 96 {'sbatch_args': ['-p', 'gpu_48gb']}
GPU i6_core/returnn/training/ReturnnTrainingJob.DvVfxf1LrCBi run gpu_mem 96 {'sbatch_args': ['-p', 'gpu_48gb']}
GPU i6_core/returnn/training/ReturnnTrainingJob.GiT88bxzoZbZ run gpu_mem 96 {'sbatch_args': ['-p', 'gpu_48gb']}
GPU i6_core/returnn/training/ReturnnTrainingJob.Y1vbqR6KeJSx run gpu_mem 96 {'sbatch_args': ['-p', 'gpu_48gb']}
GPU i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl run gpu_mem 96 {'sbatch_args': ['-p', 'gpu_48gb']}

## Subset: every screen job id is in the full config (comm -23 empty: 0 ids missing).
Note: console could not list tasks of PipelineJob.p4BOP5qZ6T1G (input DownloadJob.0UXAqd5DuQG7 lexicon not yet on disk); CPU job, expected before its upstream runs.

## Full job lists
### sae_i6_p0_screen
i6_core/audio/encoding/BlissChangeEncodingJob.7mCZ3BI2DF2K
i6_core/audio/encoding/BlissChangeEncodingJob.S4XyWVdo0Rcd
i6_core/audio/encoding/BlissChangeEncodingJob.aGpiabksK7sL
i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.4LL17D9Sz7NZ
i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.CeoY6kKnh8B3
i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.gWxOHrcGHjNZ
i6_core/datasets/librispeech/DownloadLibriSpeechMetadataJob.n7Yd9EbtVi13
i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.OjEfOC2QXh8l
i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.qlLkwjdH203i
i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.wjSkfzJS1Ge2
i6_core/g2p/apply/ApplyG2PModelJob.3eJqzOadjOqw
i6_core/g2p/convert/BlissLexiconToG2PLexiconJob.kqQmbqufPzZ3
i6_core/g2p/train/TrainG2PModelJob.pD4nbqFLWtbi
i6_core/lexicon/conversion/LexiconFromTextFileJob.mTRl42KFeZSx
i6_core/lexicon/modification/MergeLexiconJob.qKaOAPqURCkK
i6_core/lexicon/modification/WriteLexiconJob.3Ih8wASQiD3q
i6_core/returnn/forward/ReturnnForwardJobV2.5z8bUupJ1f8j
i6_core/returnn/forward/ReturnnForwardJobV2.AjTxcYsLi4E6
i6_core/returnn/forward/ReturnnForwardJobV2.GEVPu8mAuhZk
i6_core/returnn/forward/ReturnnForwardJobV2.JBwn6FADTq67
i6_core/returnn/forward/ReturnnForwardJobV2.NxXNQ8v87CCM
i6_core/returnn/forward/ReturnnForwardJobV2.QO3M1P9dOc2o
i6_core/returnn/forward/ReturnnForwardJobV2.UzlPyMEw1eR4
i6_core/returnn/forward/ReturnnForwardJobV2.gKVMu7OAH04P
i6_core/returnn/forward/ReturnnForwardJobV2.kaertnfvKAjF
i6_core/returnn/forward/ReturnnForwardJobV2.khVKNe5qaTxE
i6_core/returnn/forward/ReturnnForwardJobV2.qH5zoc9J9nxK
i6_core/returnn/forward/ReturnnForwardJobV2.tC64pEKgkkF2
i6_core/returnn/forward/ReturnnForwardJobV2.vVoEfyB8FVzF
i6_core/returnn/oggzip/BlissToOggZipJob.0N867USsZYd3
i6_core/returnn/oggzip/BlissToOggZipJob.6RIPlWd7awp7
i6_core/returnn/oggzip/BlissToOggZipJob.sohGDj24P4Qm
i6_core/returnn/training/ReturnnTrainingJob.GiT88bxzoZbZ
i6_core/text/processing/PipelineJob.p4BOP5qZ6T1G
i6_core/tools/download/DownloadJob.0UXAqd5DuQG7
i6_core/tools/download/DownloadJob.g4jClO48cAvP
i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/BlankfreeDecodeGapJob.C2mhWfaciCef
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/BlankfreeDerangementGapJob.3gZNfM04VHa8
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/ReverseGapItemsJob.QrX8dZR2QAFn
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/ReverseGapItemsJob.TngR92EpUvxY
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.6oSHcmvHPHWI
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.LbLZ8pK3RIm9
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.Q2jr3WHN2nAM
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.xGeNddosJKVH
i6_experiments/users/wu/experiments/unsupervised_asr/data/ffmpeg_pin/FfmpegPinCheckJob.wYIhhTGwejQe
i6_experiments/users/wu/experiments/unsupervised_asr/data/gold/GoldPhonesJob.i8Jk4dttLq7O
i6_experiments/users/wu/experiments/unsupervised_asr/data/hf_hub/DownloadHuggingFaceSnapshotJob.D3VATm9g18Ji
i6_experiments/users/wu/experiments/unsupervised_asr/data/hf_hub/DownloadHuggingFaceSnapshotJob.zFvdSUmBgg9x
i6_experiments/users/wu/experiments/unsupervised_asr/data/librispeech/LibriSpeechSplitIdsJob.N4iFaINQjlbn
i6_experiments/users/wu/experiments/unsupervised_asr/data/speaker/SpeakerEtaJob.Zoat6qkUPL8Q
i6_experiments/users/wu/experiments/unsupervised_asr/data/splits/CvHoldoutSplitJob.mIdi9Dy1b4Xy
i6_experiments/users/wu/experiments/unsupervised_asr/data/vad/BlankfreeVadHdfJob.RLrgIh6lFv9m
i6_experiments/users/wu/experiments/unsupervised_asr/lm/lexicon/CollectOovWordsJob.q0KCWART4B2t
i6_experiments/users/wu/experiments/unsupervised_asr/lm/phone_prior/PhoneNgramPriorJob.qJxXHgXLe31S
i6_experiments/users/wu/experiments/unsupervised_asr/lm/phone_text/PhonemizeWithSilJob.NpoY1pGJWNUJ
i6_experiments/users/wu/experiments/unsupervised_asr/lm/phone_text/SampleLinesJob.CrPgeKXsOosb
i6_experiments/users/wu/experiments/unsupervised_asr/reverse_model/duration_prior/BlankfreeDurationPriorMeanJob.bwtiONR8C9SP
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.7Ch56Tm88H4j
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.AeYkwegiVxex
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.EsAF3D4UDuND
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.POe6lJv0dp2L
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.SecjMAyOuF5b
i6_experiments/users/wu/experiments/unsupervised_asr/training/init/FlatRecognizerInitJob.0J9d6wjrkRYH
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.8OzaNlaFBf4f
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.8U3zdZO6r3Il
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.I1opFivO3Q1O
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.cKHnBFt2O0x7
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.dgcQ71ZcMuVL
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.hdBbL45bXO8y
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.toHk6JWhm6Fj
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/units/AssignUnitsJob.OqzzdiNBxGiO
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/units/MergeUnitsPklJob.EeHHIpp6q4ty
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/units/PackUnitsJob.73NZ2G1r7ai7
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/units/QuantizeStatesJob.sGIEoYoIGHpj
### sae_i6_p0
i6_core/audio/encoding/BlissChangeEncodingJob.7mCZ3BI2DF2K
i6_core/audio/encoding/BlissChangeEncodingJob.S4XyWVdo0Rcd
i6_core/audio/encoding/BlissChangeEncodingJob.aGpiabksK7sL
i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.4LL17D9Sz7NZ
i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.CeoY6kKnh8B3
i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.gWxOHrcGHjNZ
i6_core/datasets/librispeech/DownloadLibriSpeechMetadataJob.n7Yd9EbtVi13
i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.OjEfOC2QXh8l
i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.qlLkwjdH203i
i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.wjSkfzJS1Ge2
i6_core/g2p/apply/ApplyG2PModelJob.3eJqzOadjOqw
i6_core/g2p/convert/BlissLexiconToG2PLexiconJob.kqQmbqufPzZ3
i6_core/g2p/train/TrainG2PModelJob.pD4nbqFLWtbi
i6_core/lexicon/conversion/LexiconFromTextFileJob.mTRl42KFeZSx
i6_core/lexicon/modification/MergeLexiconJob.qKaOAPqURCkK
i6_core/lexicon/modification/WriteLexiconJob.3Ih8wASQiD3q
i6_core/lm/kenlm/CreateBinaryLMJob.4gueCR4UnpLG
i6_core/lm/kenlm/KenLMplzJob.S6nPpHFposo4
i6_core/returnn/forward/ReturnnForwardJobV2.3dtjN0OvY5n3
i6_core/returnn/forward/ReturnnForwardJobV2.5z8bUupJ1f8j
i6_core/returnn/forward/ReturnnForwardJobV2.9JarGCiChBWm
i6_core/returnn/forward/ReturnnForwardJobV2.AjTxcYsLi4E6
i6_core/returnn/forward/ReturnnForwardJobV2.BWRjMWacy9kB
i6_core/returnn/forward/ReturnnForwardJobV2.GEVPu8mAuhZk
i6_core/returnn/forward/ReturnnForwardJobV2.HkvgYqWPtkCP
i6_core/returnn/forward/ReturnnForwardJobV2.HroGhg9ZnpMo
i6_core/returnn/forward/ReturnnForwardJobV2.JBwn6FADTq67
i6_core/returnn/forward/ReturnnForwardJobV2.NxXNQ8v87CCM
i6_core/returnn/forward/ReturnnForwardJobV2.QO3M1P9dOc2o
i6_core/returnn/forward/ReturnnForwardJobV2.SELLIQxMB489
i6_core/returnn/forward/ReturnnForwardJobV2.StxzO95AzdXW
i6_core/returnn/forward/ReturnnForwardJobV2.UzlPyMEw1eR4
i6_core/returnn/forward/ReturnnForwardJobV2.XF7QYBMBn8zx
i6_core/returnn/forward/ReturnnForwardJobV2.XtscZAKUvIT6
i6_core/returnn/forward/ReturnnForwardJobV2.YqzN7zA1uZoJ
i6_core/returnn/forward/ReturnnForwardJobV2.gKVMu7OAH04P
i6_core/returnn/forward/ReturnnForwardJobV2.iQpVGOrmOZ6z
i6_core/returnn/forward/ReturnnForwardJobV2.kHRkbFf5Zoo6
i6_core/returnn/forward/ReturnnForwardJobV2.kaertnfvKAjF
i6_core/returnn/forward/ReturnnForwardJobV2.khVKNe5qaTxE
i6_core/returnn/forward/ReturnnForwardJobV2.oedUGMDNEVhL
i6_core/returnn/forward/ReturnnForwardJobV2.qH5zoc9J9nxK
i6_core/returnn/forward/ReturnnForwardJobV2.tC64pEKgkkF2
i6_core/returnn/forward/ReturnnForwardJobV2.vVoEfyB8FVzF
i6_core/returnn/oggzip/BlissToOggZipJob.0N867USsZYd3
i6_core/returnn/oggzip/BlissToOggZipJob.6RIPlWd7awp7
i6_core/returnn/oggzip/BlissToOggZipJob.sohGDj24P4Qm
i6_core/returnn/training/GetBestPtCheckpointJob.rpORzC4ZhN0p
i6_core/returnn/training/ReturnnTrainingJob.Ac2eioZbRX7d
i6_core/returnn/training/ReturnnTrainingJob.DvVfxf1LrCBi
i6_core/returnn/training/ReturnnTrainingJob.GiT88bxzoZbZ
i6_core/returnn/training/ReturnnTrainingJob.Y1vbqR6KeJSx
i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl
i6_core/text/processing/PipelineJob.p4BOP5qZ6T1G
i6_core/tools/download/DownloadJob.0UXAqd5DuQG7
i6_core/tools/download/DownloadJob.g4jClO48cAvP
i6_core/tools/git/CloneGitRepositoryJob.KQ3NuCaDE6QH
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/BlankfreeDecodeGapJob.5OfEn8C0ZQcm
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/BlankfreeDecodeGapJob.C2mhWfaciCef
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/BlankfreeDecodeGapJob.v5MVU29jS4Ss
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/BlankfreeDerangementGapJob.3gZNfM04VHa8
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/BlankfreeDerangementGapJob.HDyCkBA9wKVG
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/BlankfreeDerangementGapJob.f481S07qIIan
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/ReverseGapItemsJob.2q93tFdMUXz6
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/ReverseGapItemsJob.9gizbWTV7Pp3
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/ReverseGapItemsJob.A12c1bK7A1FP
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/ReverseGapItemsJob.QrX8dZR2QAFn
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/ReverseGapItemsJob.SJnmrmxZDUPR
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/gaps/ReverseGapItemsJob.TngR92EpUvxY
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/jsd/JsRowsReadJob.SMI7O7drKLyv
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/paired/PairedPerDeltaJob.BVDlYxEdQj6c
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/paired/PairedPerDeltaJob.JJWsDSp9fHbC
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/paired/PairedPerDeltaJob.JaRoClNvssLI
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/paired/PairedPerDeltaJob.aKzohAMACSlr
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/paired/PairedPerDeltaJob.fo78CC19PVd6
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.6oSHcmvHPHWI
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.E2HXcJcrVo7j
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.Fl0n11nud7Rr
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.LbLZ8pK3RIm9
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.MBheYPN2lp93
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.Q2jr3WHN2nAM
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.TVPzxTiROAkk
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.d1EjqZKwjftj
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.kfcsO5xkUNTG
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.sa5M4QGL1ALP
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.ty5cVj8CdTwe
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.vLOholE6QGTF
i6_experiments/users/wu/experiments/unsupervised_asr/analysis/per/BlankfreeGreedyPerJob.xGeNddosJKVH
i6_experiments/users/wu/experiments/unsupervised_asr/data/ffmpeg_pin/FfmpegPinCheckJob.wYIhhTGwejQe
i6_experiments/users/wu/experiments/unsupervised_asr/data/gold/GoldPhonesJob.i8Jk4dttLq7O
i6_experiments/users/wu/experiments/unsupervised_asr/data/gold/PhoneTargetHdfJob.foScplXsRIMc
i6_experiments/users/wu/experiments/unsupervised_asr/data/gold/SeedGoldPhonesJob.BUWKmEsK2zTk
i6_experiments/users/wu/experiments/unsupervised_asr/data/hf_hub/DownloadHuggingFaceSnapshotJob.7tLASYdh10dO
i6_experiments/users/wu/experiments/unsupervised_asr/data/hf_hub/DownloadHuggingFaceSnapshotJob.D3VATm9g18Ji
i6_experiments/users/wu/experiments/unsupervised_asr/data/hf_hub/DownloadHuggingFaceSnapshotJob.zFvdSUmBgg9x
i6_experiments/users/wu/experiments/unsupervised_asr/data/librispeech/LibriSpeechSplitIdsJob.N4iFaINQjlbn
i6_experiments/users/wu/experiments/unsupervised_asr/data/speaker/SpeakerEtaJob.Zoat6qkUPL8Q
i6_experiments/users/wu/experiments/unsupervised_asr/data/splits/CvHoldoutSplitJob.mIdi9Dy1b4Xy
i6_experiments/users/wu/experiments/unsupervised_asr/data/splits/CvHoldoutSplitJob.zYcc8EJsvdfV
i6_experiments/users/wu/experiments/unsupervised_asr/data/vad/BlankfreeVadHdfJob.RLrgIh6lFv9m
i6_experiments/users/wu/experiments/unsupervised_asr/lm/hlg/LexlatHLGBuildJob.avjHv1Xvjyqd
i6_experiments/users/wu/experiments/unsupervised_asr/lm/lexicon/CollectOovWordsJob.q0KCWART4B2t
i6_experiments/users/wu/experiments/unsupervised_asr/lm/phone_prior/PhoneNgramPriorJob.qJxXHgXLe31S
i6_experiments/users/wu/experiments/unsupervised_asr/lm/phone_text/PhonemizeWithSilJob.NpoY1pGJWNUJ
i6_experiments/users/wu/experiments/unsupervised_asr/lm/phone_text/SampleLinesJob.CrPgeKXsOosb
i6_experiments/users/wu/experiments/unsupervised_asr/lm/word_lm/LexiconTrieBuildJob.W0e4no47Crfu
i6_experiments/users/wu/experiments/unsupervised_asr/lm/word_window/WordWindowReplayJob.YLcmHAGAbZ1j
i6_experiments/users/wu/experiments/unsupervised_asr/reverse_model/duration_prior/BlankfreeDurationPriorMeanJob.bwtiONR8C9SP
i6_experiments/users/wu/experiments/unsupervised_asr/reverse_model/p0/BlankfreeSeedSupportJob.cMzBOdHlXVZN
i6_experiments/users/wu/experiments/unsupervised_asr/reverse_model/supervised_data/SupervisedReverseDataJob.onjA2xZUQBdx
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.2NKBMLYeYtwY
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.5brDlXJP3CS7
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.604qif5TJNoJ
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.7Ch56Tm88H4j
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.8FsqvkpOgwBQ
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.AeYkwegiVxex
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.EsAF3D4UDuND
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.F8wmgmAv7CHD
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.POe6lJv0dp2L
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.Qcl8dPZqTcXJ
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.SecjMAyOuF5b
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.p752NpK39bZW
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.psvDYM3qHQfE
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.v1bPX23lPGsH
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.w2ObSfSHWH0o
i6_experiments/users/wu/experiments/unsupervised_asr/training/checkpoints/ExtractSubmoduleCheckpointJob.yemA6RvEUnkI
i6_experiments/users/wu/experiments/unsupervised_asr/training/init/FlatRecognizerInitJob.0J9d6wjrkRYH
i6_experiments/users/wu/experiments/unsupervised_asr/training/init/FlatRecognizerInitJob.DMSwTLXT9MWG
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.8OzaNlaFBf4f
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.8U3zdZO6r3Il
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.I1opFivO3Q1O
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.cKHnBFt2O0x7
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.dgcQ71ZcMuVL
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.hdBbL45bXO8y
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/features/L15ForwardSeqOrderJob.toHk6JWhm6Fj
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/units/AssignUnitsJob.OqzzdiNBxGiO
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/units/MergeUnitsPklJob.EeHHIpp6q4ty
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/units/PackUnitsJob.73NZ2G1r7ai7
i6_experiments/users/wu/experiments/unsupervised_asr/w2v2/units/QuantizeStatesJob.sGIEoYoIGHpj
