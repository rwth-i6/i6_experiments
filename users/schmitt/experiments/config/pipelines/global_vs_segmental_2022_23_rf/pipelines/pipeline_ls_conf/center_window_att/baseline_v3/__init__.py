from . import (
  alignment_augmentation,
  att_weight_feedback,
  baseline,
  blank_decoder_variants,
  optimizer_variants,
  label_decoder_variants,
  bpe1k,
)


def run_exps():
  # Done
  baseline.run_exps()
  # Done
  alignment_augmentation.run_exps()
  # Done
  att_weight_feedback.run_exps()
  # Running
  blank_decoder_variants.run_exps()
  # Done
  label_decoder_variants.run_exps()

  bpe1k.run_experiments()
