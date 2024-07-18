from . import (
  alignment_augmentation,
  att_weight_feedback,
  baseline,
  blank_decoder_variants,
  optimizer_variants,
  label_decoder_variants
)


def run_exps():
  # Done
  baseline.run_exps()
  # Running
  alignment_augmentation.run_exps()
  # Done
  att_weight_feedback.run_exps()
  # Done
  blank_decoder_variants.run_exps()
  # Running
  label_decoder_variants.run_exps()
