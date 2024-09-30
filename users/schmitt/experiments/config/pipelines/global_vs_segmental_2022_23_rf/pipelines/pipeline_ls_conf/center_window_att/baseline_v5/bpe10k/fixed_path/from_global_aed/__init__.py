from . import (
  alignment_augmentation,
  baseline,
  blank_decoder_variants,
  transducer_w_att_random,
  transducer_w_att_gate,
  transducer_w_att
)


def run_exps():
  # Done
  alignment_augmentation.run_exps()
  # Done
  baseline.run_exps()
  # Done
  blank_decoder_variants.run_exps()
  # Done
  transducer_w_att_gate.run_exps()
  # Done
  transducer_w_att_random.run_exps()
  # Done
  transducer_w_att.run_exps()
