"""English normalization from the Hugging Face Open ASR leaderboard.

We intentionally retain the updated upstream normalizer from revision
48219c6028db0517d704600d92f31edfc96e8c23. Published leaderboard WERs may
have been produced with an older normalizer and are not necessarily directly
comparable, even when the model predictions are identical.

For example, revision b67aef21b4d2df35a7e81b7da2d06808027258a2 (2026-08-25)
maps standalone "oh" to "0", whereas the updated version usually preserves
"oh". On the public Granite Speech 5.0 470M TurboCTC-NC test-other predictions,
the historical scorer gives 2.6059% WER (the published 2.61%) and the updated
scorer gives 2.6741%, using kaldialign 0.12.0 with compound merging in both
cases. This difference is due to normalization, not model inference.

Upstream history: https://github.com/huggingface/open_asr_leaderboard/commit/
14469179567f4b829f96ba38f2f4651c52acc5ce
"""

from .normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer

__all__ = ["EnglishTextNormalizer", "BasicMultilingualTextNormalizer"]
