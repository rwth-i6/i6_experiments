"""Recognition (RecogDef, as consumed by zeyer's recog / recog_training_exp).

Two defs, both valid for the AED model:
  aed_beam_search  -- label-sync AED beam search (what the Loquacious base-v2 reference uses)
  ctc_beam_search  -- time-sync search on the top aux CTC head, much cheaper
Import the module you want; the names collide by design (both are called ``recog_def``).
"""
