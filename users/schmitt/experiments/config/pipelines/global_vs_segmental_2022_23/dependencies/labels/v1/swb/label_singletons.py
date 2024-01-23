from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.phonemes.phoneme_hmm_alignment import HMMPhoneme
# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.phonemes.phoneme_hmm_alignment import HMMPhoneme
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.bpe.bpe_rna_alignment import RNABPE
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.bpe.bpe_sil_rna_hmm_alignment import RNABPESil, RNABPESplitSil, RNABPESilBase
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.bpe.bpe_labels import BPELabels

RNA_BPE = RNABPE()
HMM_PHONEME = HMMPhoneme()
RNA_BPE_SIL_BASE = RNABPESilBase(ref_bpe=RNA_BPE, ref_phoneme=HMM_PHONEME)
RNA_BPE_SIL_TIME_RED_6 = RNABPESil(
  time_reduction=6, ref_bpe=RNA_BPE, ref_phoneme=HMM_PHONEME, ref_rna_bpe_sil_base=RNA_BPE_SIL_BASE)
RNA_BPE_SPLIT_SIL_TIME_RED_6 = RNABPESplitSil(ref_bpe_sil=RNA_BPE_SIL_TIME_RED_6)

BPE_LABELS = BPELabels()
