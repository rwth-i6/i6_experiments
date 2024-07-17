from . import (
  baseline,
  blank_decoder_variants,
  label_decoder_variants,
)


def run_exps():
  # Done
  baseline.run_exps()
  # Done
  blank_decoder_variants.run_exps()
  # Running
  label_decoder_variants.run_exps()
