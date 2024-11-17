from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.bpe.bpe_labels import (
  LibrispeechBPE10025Labels,
  LibrispeechBPE10025LabelsWithSilence,
  LibrispeechBPE5048Labels,
  LibrispeechBPE1056Labels
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.sentencepiece.sentencepiece_labels import LibrispeechSP10240Labels
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.bpe.bpe_alignments import (
  LibrispeechBpe10025CtcAlignment,
  LibrispeechBpe10025CtcAlignmentEos,
  LibrispeechBpe1056AlignmentJointModel,
  LibrispeechBpe1056AlignmentSepModel,
  LibrispeechBpe1056AlignmentCtcModel,
  LibrispeechBpe5048AlignmentJointModel,
  LibrispeechBpe5048AlignmentSepModel,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.sentencepiece.sentencepiece_alignments import (
  LibrispeechSP10240AlignmentSepModel,
  LibrispeechSP10240AlignmentJointModel
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora


LIBRISPEECH_CORPUS = LibrispeechCorpora()

# BPE labels
LibrispeechBPE10025_LABELS = LibrispeechBPE10025Labels()
LibrispeechBPE10025_LABELS_WITH_SILENCE = LibrispeechBPE10025LabelsWithSilence(LibrispeechBPE10025_LABELS)
LibrispeechBPE10025_CTC_ALIGNMENT = LibrispeechBpe10025CtcAlignment()
LibrispeechBPE10025_CTC_ALIGNMENT_EOS = LibrispeechBpe10025CtcAlignmentEos(LibrispeechBPE10025_CTC_ALIGNMENT)

LibrispeechBPE1056_LABELS = LibrispeechBPE1056Labels()

LibrispeechBPE5048_LABELS = LibrispeechBPE5048Labels()

# BPE alignments
LibrispeechBPE1056_ALIGNMENT_JOINT_MODEL = LibrispeechBpe1056AlignmentJointModel()
LibrispeechBPE1056_ALIGNMENT_SEP_MODEL = LibrispeechBpe1056AlignmentSepModel()
LibrispeechBPE1056_CTC_ALIGNMENT = LibrispeechBpe1056AlignmentCtcModel()

LibrispeechBPE5048_ALIGNMENT_JOINT_MODEL = LibrispeechBpe5048AlignmentJointModel()
LibrispeechBPE5048_ALIGNMENT_SEP_MODEL = LibrispeechBpe5048AlignmentSepModel()

# sentencepiece labels
LibrispeechSP10240_LABELS = LibrispeechSP10240Labels()

# sentencepiece alignments
LibrispeechSP10240_ALIGNMENT_SEP_MODEL = LibrispeechSP10240AlignmentSepModel()
LibrispeechSP10240_ALIGNMENT_JOINT_MODEL = LibrispeechSP10240AlignmentJointModel()
