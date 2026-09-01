"""
Sisyphus recipes for the packed-tensor Loquacious AED experiments.

  loquacious_aed_packed -- base-v2 (padded reference) + base-v2-packed, i.e. the reproduction of
                           base-v2-large-nFullEp4.0-nEp100-totalHours100k with and without the
                           packed-tensor path.
  llm_vocab             -- Qwen2 tokenizer handles + Loquacious vocab-usage measurements.
  vocab_usage           -- the ExtractVocabUsageJob those measurements use.
"""

# Alias/output prefix for every recipe in this package. get_setup_prefix_for_module walks up the
# module hierarchy, so defining it here covers modules that do not set their own.
__setup_root_prefix__ = "exp2026_08_27_aed_llm_vocab"
